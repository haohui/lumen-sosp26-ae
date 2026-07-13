import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE: al.constexpr = 256


@avelang.jit
def reverse_cumsum_pass1_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    block_sums_ptr: al.Pointer(al.f32),
    batch_size: al.i32,
    N: al.i32,
    num_tiles: al.i32,
):
    """
    Pass 1: compute per-tile reverse cumulative sum using fp32 accumulation.
    Backward Hillis-Steele scan for direct reverse inclusive prefix.
    grid = (batch_size * num_tiles, 1, 1), block = (BLOCK_SIZE, 1, 1)
    """
    tid = al.thread_id(0)
    bid = al.block_id(0)

    batch_idx = bid // num_tiles
    tile_idx = bid - batch_idx * num_tiles

    if batch_idx < batch_size:
        smem = al.make_shared((BLOCK_SIZE,), al.f32)

        row_start = batch_idx * N
        tile_start = tile_idx * BLOCK_SIZE
        tile_end = tile_start + BLOCK_SIZE
        if tile_end > N:
            tile_end = N

        layout_1d = al.make_layout((batch_size * N,), (1,))
        x = al.make_tensor(x_ptr, al.bf16, layout_1d)

        local_val = al.convert(0.0, al.f32)
        for i in al.range(tile_start + tid, tile_end, BLOCK_SIZE):
            idx = row_start + i
            local_val = al.convert(x[idx], al.f32)

        smem[tid] = local_val
        al.syncthreads()

        # Backward inclusive scan: smem[tid] = sum of a[tid .. end of tile]
        if tid + 1 < BLOCK_SIZE:
            smem[tid] = smem[tid] + smem[tid + 1]
        al.syncthreads()
        if tid + 2 < BLOCK_SIZE:
            smem[tid] = smem[tid] + smem[tid + 2]
        al.syncthreads()
        if tid + 4 < BLOCK_SIZE:
            smem[tid] = smem[tid] + smem[tid + 4]
        al.syncthreads()
        if tid + 8 < BLOCK_SIZE:
            smem[tid] = smem[tid] + smem[tid + 8]
        al.syncthreads()
        if tid + 16 < BLOCK_SIZE:
            smem[tid] = smem[tid] + smem[tid + 16]
        al.syncthreads()
        if tid + 32 < BLOCK_SIZE:
            smem[tid] = smem[tid] + smem[tid + 32]
        al.syncthreads()
        if tid + 64 < BLOCK_SIZE:
            smem[tid] = smem[tid] + smem[tid + 64]
        al.syncthreads()
        if tid + 128 < BLOCK_SIZE:
            smem[tid] = smem[tid] + smem[tid + 128]
        al.syncthreads()

        local_rev = smem[tid]
        tile_total = smem[0]

        layout_out = al.make_layout((batch_size * N,), (1,))
        out = al.make_tensor(out_ptr, al.bf16, layout_out)

        for i in al.range(tile_start + tid, tile_end, BLOCK_SIZE):
            idx = row_start + i
            out[idx] = al.convert(local_rev, al.bf16)

        if tid == 0:
            layout_bs = al.make_layout((batch_size, num_tiles), (num_tiles, 1))
            bs = al.make_tensor(block_sums_ptr, al.f32, layout_bs)
            bs[batch_idx, tile_idx] = tile_total


@avelang.jit
def reverse_cumsum_pass2_kernel(
    out_ptr: al.Pointer(al.bf16),
    block_sums_ptr: al.Pointer(al.f32),
    batch_size: al.i32,
    N: al.i32,
    num_tiles: al.i32,
):
    """
    Pass 2: add cross-tile suffix sums using fp32 accumulation.
    grid = (batch_size, 1, 1), block = (BLOCK_SIZE, 1, 1)
    """
    tid = al.thread_id(0)
    bid = al.block_id(0)

    batch_idx = bid

    if batch_idx < batch_size:
        smem_bs = al.make_shared((BLOCK_SIZE,), al.f32)

        layout_bs = al.make_layout((batch_size, num_tiles), (num_tiles, 1))
        bs = al.make_tensor(block_sums_ptr, al.f32, layout_bs)

        local_bs_val = al.convert(0.0, al.f32)
        for t in al.range(tid, num_tiles, BLOCK_SIZE):
            local_bs_val = bs[batch_idx, t]
        smem_bs[tid] = local_bs_val
        al.syncthreads()

        # Backward scan: smem_bs[t] = sum of block_sums[t .. end]
        if tid + 1 < BLOCK_SIZE:
            smem_bs[tid] = smem_bs[tid] + smem_bs[tid + 1]
        al.syncthreads()
        if tid + 2 < BLOCK_SIZE:
            smem_bs[tid] = smem_bs[tid] + smem_bs[tid + 2]
        al.syncthreads()
        if tid + 4 < BLOCK_SIZE:
            smem_bs[tid] = smem_bs[tid] + smem_bs[tid + 4]
        al.syncthreads()
        if tid + 8 < BLOCK_SIZE:
            smem_bs[tid] = smem_bs[tid] + smem_bs[tid + 8]
        al.syncthreads()
        if tid + 16 < BLOCK_SIZE:
            smem_bs[tid] = smem_bs[tid] + smem_bs[tid + 16]
        al.syncthreads()
        if tid + 32 < BLOCK_SIZE:
            smem_bs[tid] = smem_bs[tid] + smem_bs[tid + 32]
        al.syncthreads()
        if tid + 64 < BLOCK_SIZE:
            smem_bs[tid] = smem_bs[tid] + smem_bs[tid + 64]
        al.syncthreads()
        if tid + 128 < BLOCK_SIZE:
            smem_bs[tid] = smem_bs[tid] + smem_bs[tid + 128]
        al.syncthreads()

        layout_out = al.make_layout((batch_size * N,), (1,))
        out = al.make_tensor(out_ptr, al.bf16, layout_out)

        row_start = batch_idx * N

        for i in al.range(tid, N, BLOCK_SIZE):
            tile_idx = i // BLOCK_SIZE
            idx = row_start + i

            # suffix = sum of block_sums[tile_idx+1 .. end] = smem_bs[tile_idx+1]
            suffix = smem_bs[tile_idx + 1]

            val = al.convert(out[idx], al.f32)
            out[idx] = al.convert(val + suffix, al.bf16)


def avelang_reverse_cumsum(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda, "Tensors must be on CUDA/HIP device."
    assert x.dtype == torch.bfloat16, "Input tensor must be bfloat16"
    assert x.ndim == 2, "Expected 2D input tensor"
    assert dim == 1, "Only dim=1 is supported"

    x_contig = x.contiguous()
    batch_size_val = x_contig.shape[0]
    N_val = x_contig.shape[1]

    num_tiles = (N_val + BLOCK_SIZE - 1) // BLOCK_SIZE

    out = torch.empty_like(x_contig)
    block_sums = torch.empty((batch_size_val, num_tiles), dtype=torch.float32, device=x.device)

    grid1 = (batch_size_val * num_tiles, 1, 1)
    reverse_cumsum_pass1_kernel[lambda: (grid1, (BLOCK_SIZE, 1, 1))](
        x_contig, out, block_sums, batch_size_val, N_val, num_tiles
    )

    grid2 = (batch_size_val, 1, 1)
    reverse_cumsum_pass2_kernel[lambda: (grid2, (BLOCK_SIZE, 1, 1))](
        out, block_sums, batch_size_val, N_val, num_tiles
    )

    return out


class ModelNew(nn.Module):
    """
    Optimized reverse cumulative sum along dim=1 using AveLang GPU kernels.
    Uses fp32 accumulation for numerical stability, matching the mathematical
    semantics of torch.cumsum with fp32-accumulated bf16 inputs.
    """
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_bf16 = x.to(dtype=torch.bfloat16, device=x.device).contiguous()
        result = avelang_reverse_cumsum(x_bf16, self.dim)
        return result.to(x.dtype)
