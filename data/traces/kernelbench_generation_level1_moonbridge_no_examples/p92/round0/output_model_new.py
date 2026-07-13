import torch
import torch.nn as nn
import avelang
import avelang.language as al

TILE_SIZE: int = 256
LOG_TILE: int = 8


@avelang.jit
def exclusive_cumsum_dim1_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    N: al.i32,
    stride: al.i32,
    n_tiles: al.i32,
    TILE_SIZE: al.constexpr,
    LOG_TILE: al.constexpr,
):
    """Exclusive cumulative sum along dim=1.

    Computes the inclusive cumsum of the shifted row [0, x[row,0], x[row,1], ...]
    using a tiled Hillis-Steele scan with fp32 shared memory and bf16 output.
    """
    row = al.block_id(0)
    tid = al.thread_id(0)

    if row < N - 1:
        x_2d = al.make_tensor(x_ptr, al.bf16, al.make_layout((N, stride), (stride, 1)))
        out_2d = al.make_tensor(out_ptr, al.bf16, al.make_layout((N - 1, stride + 1), (stride + 1, 1)))

        smem_full = al.make_shared((TILE_SIZE + 2,), al.f32)
        tile_data = al.subview(smem_full, (2,), (TILE_SIZE,), (1,))

        if tid == 0:
            smem_full[0] = al.convert(0.0, al.f32)
        al.syncthreads()

        for tile in al.range(n_tiles):
            tile_start = tile * TILE_SIZE

            # Load tile: shifted[row, pos] where pos=0 is leading zero
            in_idx = tile_start + tid - 1
            if tile == 0 and tid == 0:
                tile_data[tid] = al.convert(0.0, al.f32)
            elif in_idx >= 0 and in_idx < stride:
                tile_data[tid] = al.convert(x_2d[row, in_idx], al.f32)
            else:
                tile_data[tid] = al.convert(0.0, al.f32)

            al.syncthreads()

            # Hillis-Steele inclusive prefix sum in fp32
            offset = 1
            for d in al.range(LOG_TILE):
                if tid >= offset:
                    tile_data[tid] = tile_data[tid] + tile_data[tid - offset]
                offset = offset * 2
                al.syncthreads()

            # Write output: inclusive[tid] + cross-tile running sum -> bf16
            running_val = smem_full[0]
            out_idx = tile_start + tid
            if out_idx < stride + 1:
                out_2d[row, out_idx] = al.convert(
                    tile_data[tid] + running_val, al.bf16
                )

            al.syncthreads()

            # Advance cross-tile running sum: add tile total, round to bf16
            if tid == 0:
                fp32_total = smem_full[0] + tile_data[TILE_SIZE - 1]
                bf16_total = al.convert(fp32_total, al.bf16)
                smem_full[0] = al.convert(bf16_total, al.f32)
            al.syncthreads()


@avelang.jit
def exclusive_cumsum_dim0_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    N: al.i32,
    stride: al.i32,
    n_tiles: al.i32,
    TILE_SIZE: al.constexpr,
    LOG_TILE: al.constexpr,
):
    """Exclusive cumulative sum along dim=0."""
    col = al.block_id(0)
    tid = al.thread_id(0)

    if col < stride:
        x_2d = al.make_tensor(x_ptr, al.bf16, al.make_layout((N, stride), (stride, 1)))
        out_2d = al.make_tensor(out_ptr, al.bf16, al.make_layout((N, stride), (stride, 1)))

        smem_full = al.make_shared((TILE_SIZE + 2,), al.f32)
        tile_data = al.subview(smem_full, (2,), (TILE_SIZE,), (1,))

        if tid == 0:
            smem_full[0] = al.convert(0.0, al.f32)
        al.syncthreads()

        for tile in al.range(n_tiles):
            tile_start = tile * TILE_SIZE

            in_idx = tile_start + tid - 1
            if tile == 0 and tid == 0:
                tile_data[tid] = al.convert(0.0, al.f32)
            elif in_idx >= 0 and in_idx < N - 1:
                tile_data[tid] = al.convert(x_2d[in_idx, col], al.f32)
            else:
                tile_data[tid] = al.convert(0.0, al.f32)

            al.syncthreads()

            offset = 1
            for d in al.range(LOG_TILE):
                if tid >= offset:
                    tile_data[tid] = tile_data[tid] + tile_data[tid - offset]
                offset = offset * 2
                al.syncthreads()

            running_val = smem_full[0]
            out_idx = tile_start + tid
            if out_idx < N:
                out_2d[out_idx, col] = al.convert(
                    tile_data[tid] + running_val, al.bf16
                )

            al.syncthreads()

            if tid == 0:
                fp32_total = smem_full[0] + tile_data[TILE_SIZE - 1]
                bf16_total = al.convert(fp32_total, al.bf16)
                smem_full[0] = al.convert(bf16_total, al.f32)
            al.syncthreads()


def avelang_exclusive_cumsum(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda, "Tensor must be on CUDA/HIP device."
    assert x.ndim == 2, "Expected 2D tensor."
    N, stride_val = x.shape
    x = x.contiguous()

    if dim == 1:
        out = torch.empty(N - 1, stride_val + 1, dtype=x.dtype, device=x.device)
        grid = (N - 1, 1, 1)
        block = (TILE_SIZE, 1, 1)
        n_tiles = (stride_val + 1 + TILE_SIZE - 1) // TILE_SIZE
        exclusive_cumsum_dim1_kernel[lambda: (grid, block)](
            x, out, N, stride_val, n_tiles, TILE_SIZE, LOG_TILE
        )
        return out
    elif dim == 0:
        out = torch.empty(N, stride_val, dtype=x.dtype, device=x.device)
        grid = (stride_val, 1, 1)
        block = (TILE_SIZE, 1, 1)
        n_tiles = (N + TILE_SIZE - 1) // TILE_SIZE
        exclusive_cumsum_dim0_kernel[lambda: (grid, block)](
            x, out, N, stride_val, n_tiles, TILE_SIZE, LOG_TILE
        )
        return out
    else:
        raise ValueError(f"Unsupported dim: {dim}")


class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        return avelang_exclusive_cumsum(x, self.dim)
