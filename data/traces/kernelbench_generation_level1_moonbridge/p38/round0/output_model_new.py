import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE: al.constexpr = 256
TILE_SIZE: al.constexpr = 4096


@avelang.jit
def l1norm_reduce_kernel(
    x_ptr: al.Pointer(al.bf16),
    partial_sum_ptr: al.Pointer(al.f32),
    batch_size: al.i32,
    dim: al.i32,
    num_tiles: al.i32,
):
    """
    Phase 1: compute partial sums of abs(x) per tile per row.
    Launch: grid = (num_tiles * batch_size, 1, 1), block = (BLOCK_SIZE, 1, 1)
    """
    tid = al.thread_id(0)
    bid = al.block_id(0)

    batch_idx = bid // num_tiles
    tile_idx = bid - batch_idx * num_tiles

    if batch_idx < batch_size:
        smem = al.make_shared((BLOCK_SIZE,), al.f32)

        base = batch_idx * dim
        tile_start = tile_idx * TILE_SIZE
        tile_end = tile_start + TILE_SIZE
        if tile_end > dim:
            tile_end = dim

        layout_in = al.make_layout((batch_size * dim,), (1,))
        x = al.make_tensor(x_ptr, al.bf16, layout_in)

        local_sum = al.convert(0.0, al.f32)

        for i in al.range(tile_start + tid, tile_end, BLOCK_SIZE):
            idx = base + i
            val = al.convert(x[idx], al.f32)
            local_sum = local_sum + al.abs(val)

        smem[tid] = local_sum
        al.syncthreads()

        if tid < 128:
            smem[tid] = smem[tid] + smem[tid + 128]
        al.syncthreads()
        if tid < 64:
            smem[tid] = smem[tid] + smem[tid + 64]
        al.syncthreads()
        if tid < 32:
            smem[tid] = smem[tid] + smem[tid + 32]
        al.syncthreads()
        if tid < 16:
            smem[tid] = smem[tid] + smem[tid + 16]
        al.syncthreads()
        if tid < 8:
            smem[tid] = smem[tid] + smem[tid + 8]
        al.syncthreads()
        if tid < 4:
            smem[tid] = smem[tid] + smem[tid + 4]
        al.syncthreads()
        if tid < 2:
            smem[tid] = smem[tid] + smem[tid + 2]
        al.syncthreads()
        if tid < 1:
            smem[tid] = smem[tid] + smem[tid + 1]

        if tid == 0:
            layout_ps = al.make_layout((batch_size, num_tiles), (num_tiles, 1))
            ps = al.make_tensor(partial_sum_ptr, al.f32, layout_ps)
            ps[batch_idx, tile_idx] = smem[0]


@avelang.jit
def l1norm_aggregate_kernel(
    partial_sum_ptr: al.Pointer(al.f32),
    mean_out_ptr: al.Pointer(al.f32),
    batch_size: al.i32,
    num_tiles: al.i32,
    dim: al.i32,
):
    """
    Phase 2: aggregate partial tile sums into per-row means.
    Launch: grid = (batch_size, 1, 1), block = (BLOCK_SIZE, 1, 1)
    """
    tid = al.thread_id(0)
    bid = al.block_id(0)

    if bid < batch_size:
        smem = al.make_shared((BLOCK_SIZE,), al.f32)

        layout_ps = al.make_layout((batch_size, num_tiles), (num_tiles, 1))
        ps = al.make_tensor(partial_sum_ptr, al.f32, layout_ps)

        local_sum = al.convert(0.0, al.f32)
        if tid < num_tiles:
            local_sum = ps[bid, tid]

        smem[tid] = local_sum
        al.syncthreads()

        if tid < 128:
            smem[tid] = smem[tid] + smem[tid + 128]
        al.syncthreads()
        if tid < 64:
            smem[tid] = smem[tid] + smem[tid + 64]
        al.syncthreads()
        if tid < 32:
            smem[tid] = smem[tid] + smem[tid + 32]
        al.syncthreads()
        if tid < 16:
            smem[tid] = smem[tid] + smem[tid + 16]
        al.syncthreads()
        if tid < 8:
            smem[tid] = smem[tid] + smem[tid + 8]
        al.syncthreads()
        if tid < 4:
            smem[tid] = smem[tid] + smem[tid + 4]
        al.syncthreads()
        if tid < 2:
            smem[tid] = smem[tid] + smem[tid + 2]
        al.syncthreads()
        if tid < 1:
            smem[tid] = smem[tid] + smem[tid + 1]

        if tid == 0:
            dim_f32 = al.convert(dim, al.f32)
            mean = smem[0] / dim_f32

            layout_out = al.make_layout((batch_size,), (1,))
            mo = al.make_tensor(mean_out_ptr, al.f32, layout_out)
            mo[bid] = mean


@avelang.jit
def l1norm_apply_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    mean_ptr: al.Pointer(al.f32),
    batch_size: al.i32,
    dim: al.i32,
    num_tiles: al.i32,
):
    """
    Phase 3: apply L1 normalization — divide each element by its row mean.
    Launch: grid = (num_tiles * batch_size, 1, 1), block = (BLOCK_SIZE, 1, 1)
    """
    tid = al.thread_id(0)
    bid = al.block_id(0)

    batch_idx = bid // num_tiles
    tile_idx = bid - batch_idx * num_tiles

    if batch_idx < batch_size:
        base = batch_idx * dim
        tile_start = tile_idx * TILE_SIZE
        tile_end = tile_start + TILE_SIZE
        if tile_end > dim:
            tile_end = dim

        layout_in = al.make_layout((batch_size * dim,), (1,))
        x = al.make_tensor(x_ptr, al.bf16, layout_in)

        layout_out = al.make_layout((batch_size * dim,), (1,))
        ot = al.make_tensor(out_ptr, al.bf16, layout_out)

        layout_mean = al.make_layout((batch_size,), (1,))
        mt = al.make_tensor(mean_ptr, al.f32, layout_mean)

        mean = mt[batch_idx]

        for i in al.range(tile_start + tid, tile_end, BLOCK_SIZE):
            idx = base + i
            x_val = al.convert(x[idx], al.f32)
            result = x_val / mean
            ot[idx] = al.convert(result, al.bf16)


def avelang_l1norm(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Tensors must be on CUDA/HIP device."

    x_bf16 = x.contiguous().to(dtype=torch.bfloat16)

    batch_size = x_bf16.shape[0]
    dim = x_bf16.shape[1]

    num_tiles = (dim + TILE_SIZE - 1) // TILE_SIZE

    # Phase 1: tile-level reduction
    partial_sum = torch.empty((batch_size, num_tiles), dtype=torch.float32, device=x.device)

    l1norm_reduce_kernel[lambda: ((batch_size * num_tiles, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_bf16, partial_sum, batch_size, dim, num_tiles
    )

    # Phase 2: aggregate across tiles into per-row means
    mean_out = torch.empty((batch_size,), dtype=torch.float32, device=x.device)

    l1norm_aggregate_kernel[lambda: ((batch_size, 1, 1), (BLOCK_SIZE, 1, 1))](
        partial_sum, mean_out, batch_size, num_tiles, dim
    )

    # Phase 3: apply normalization
    out = torch.empty_like(x_bf16)

    l1norm_apply_kernel[lambda: ((batch_size * num_tiles, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_bf16, out, mean_out, batch_size, dim, num_tiles
    )

    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_bf16 = x.to(dtype=torch.bfloat16, device=x.device).contiguous()
        result = avelang_l1norm(x_bf16)
        return result.to(x.dtype)
