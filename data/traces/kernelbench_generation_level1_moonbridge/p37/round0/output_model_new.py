import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE: al.constexpr = 256
TILE_SIZE: al.constexpr = 32768


@avelang.jit
def frob_norm_reduce_kernel(
    x_ptr: al.Pointer(al.bf16),
    partial_sq_ptr: al.Pointer(al.f32),
    num_tiles: al.i32,
    N: al.i32,
):
    """
    Pass 1: reduce each tile into sum-of-squares partial.
    Launch: grid = (num_tiles, 1, 1), block = (BLOCK_SIZE, 1, 1)
    """
    tid = al.thread_id(0)
    bid = al.block_id(0)

    if bid < num_tiles:
        smem_sq = al.make_shared((BLOCK_SIZE,), al.f32)

        tile_start = bid * TILE_SIZE
        tile_end = tile_start + TILE_SIZE
        if tile_end > N:
            tile_end = N

        layout_in = al.make_layout((N,), (1,))
        x = al.make_tensor(x_ptr, al.bf16, layout_in)

        local_sq = al.convert(0.0, al.f32)

        for i in al.range(tile_start + tid, tile_end, BLOCK_SIZE):
            val = al.convert(x[i], al.f32)
            local_sq = local_sq + val * val

        smem_sq[tid] = local_sq
        al.syncthreads()

        # tree reduction over shared memory
        if tid < 128:
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 128]
        al.syncthreads()
        if tid < 64:
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 64]
        al.syncthreads()
        if tid < 32:
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 32]
        al.syncthreads()
        if tid < 16:
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 16]
        al.syncthreads()
        if tid < 8:
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 8]
        al.syncthreads()
        if tid < 4:
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 4]
        al.syncthreads()
        if tid < 2:
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 2]
        al.syncthreads()
        if tid < 1:
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 1]

        if tid == 0:
            layout_ps = al.make_layout((num_tiles,), (1,))
            ps = al.make_tensor(partial_sq_ptr, al.f32, layout_ps)
            ps[bid] = smem_sq[0]


@avelang.jit
def frob_norm_aggregate_kernel(
    partial_sq_ptr: al.Pointer(al.f32),
    rnorm_ptr: al.Pointer(al.f32),
    num_tiles: al.i32,
):
    """
    Pass 2: aggregate partial tile results into global sum_sq, compute rnorm = 1/sqrt(sum).
    Launch: grid = (1, 1, 1), block = (BLOCK_SIZE, 1, 1)
    """
    tid = al.thread_id(0)

    smem_sq = al.make_shared((BLOCK_SIZE,), al.f32)

    layout_ps = al.make_layout((num_tiles,), (1,))
    ps = al.make_tensor(partial_sq_ptr, al.f32, layout_ps)

    total_sq = al.convert(0.0, al.f32)

    chunk_start = al.convert(0, al.i32)
    for _ in al.range(0, 512):
        if chunk_start >= num_tiles:
            break

        local_sq = al.convert(0.0, al.f32)
        idx = chunk_start + tid
        if idx < num_tiles:
            local_sq = ps[idx]

        smem_sq[tid] = local_sq
        al.syncthreads()

        if tid < 128:
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 128]
        al.syncthreads()
        if tid < 64:
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 64]
        al.syncthreads()
        if tid < 32:
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 32]
        al.syncthreads()
        if tid < 16:
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 16]
        al.syncthreads()
        if tid < 8:
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 8]
        al.syncthreads()
        if tid < 4:
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 4]
        al.syncthreads()
        if tid < 2:
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 2]
        al.syncthreads()
        if tid < 1:
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 1]

        if tid == 0:
            total_sq = total_sq + smem_sq[0]

        chunk_start = chunk_start + BLOCK_SIZE
        al.syncthreads()

    if tid == 0:
        rnorm = al.convert(1.0, al.f32) / al.sqrt(total_sq)
        layout_out = al.make_layout((1,), (1,))
        ro = al.make_tensor(rnorm_ptr, al.f32, layout_out)
        ro[0] = rnorm


@avelang.jit
def frob_norm_apply_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    rnorm_ptr: al.Pointer(al.f32),
    num_tiles: al.i32,
    N: al.i32,
):
    """
    Pass 3: apply normalization: out = x * rnorm.
    Launch: grid = (num_tiles, 1, 1), block = (BLOCK_SIZE, 1, 1)
    """
    tid = al.thread_id(0)
    bid = al.block_id(0)

    if bid < num_tiles:
        tile_start = bid * TILE_SIZE
        tile_end = tile_start + TILE_SIZE
        if tile_end > N:
            tile_end = N

        layout_in = al.make_layout((N,), (1,))
        x = al.make_tensor(x_ptr, al.bf16, layout_in)

        layout_out = al.make_layout((N,), (1,))
        out = al.make_tensor(out_ptr, al.bf16, layout_out)

        layout_rn = al.make_layout((num_tiles,), (1,))
        rn = al.make_tensor(rnorm_ptr, al.f32, layout_rn)
        zero = al.convert(0, al.i32)
        rnorm = rn[zero]

        for i in al.range(tile_start + tid, tile_end, BLOCK_SIZE):
            val = al.convert(x[i], al.f32)
            out[i] = al.convert(val * rnorm, al.bf16)


def avelang_frob_norm(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Tensors must be on CUDA/HIP device."

    x_bf16 = x.to(dtype=torch.bfloat16, device=x.device).contiguous()
    N = x_bf16.numel()

    num_tiles = (N + TILE_SIZE - 1) // TILE_SIZE

    # Pass 1: tile-level sum-of-squares reduction
    partial_sq = torch.empty((num_tiles,), dtype=torch.float32, device=x.device)

    frob_norm_reduce_kernel[lambda: ((num_tiles, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_bf16, partial_sq, num_tiles, N
    )

    # Pass 2: aggregate partials into global sum_sq, compute rnorm
    rnorm = torch.empty((1,), dtype=torch.float32, device=x.device)

    frob_norm_aggregate_kernel[lambda: ((1, 1, 1), (BLOCK_SIZE, 1, 1))](
        partial_sq, rnorm, num_tiles
    )

    # Pass 3: apply scaling to every element
    out = torch.empty_like(x_bf16)

    frob_norm_apply_kernel[lambda: ((num_tiles, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_bf16, out, rnorm, num_tiles, N
    )

    return out.to(x.dtype)


class ModelNew(nn.Module):
    """
    Optimized model that performs Frobenius norm normalization using AveLang DSL.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_frob_norm(x)
