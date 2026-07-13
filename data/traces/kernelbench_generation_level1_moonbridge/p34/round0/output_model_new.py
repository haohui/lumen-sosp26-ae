import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE: al.constexpr = 256
TILE_SIZE: al.constexpr = 4096
MAX_TILES_PER_ROUND: al.constexpr = 256


@avelang.jit
def instancenorm_reduce_kernel(
    x_ptr: al.Pointer(al.bf16),
    partial_sum_ptr: al.Pointer(al.f32),
    partial_sq_ptr: al.Pointer(al.f32),
    total_pairs: al.i32,
    N: al.i32,
    num_tiles: al.i32,
):
    """
    Phase 1: reduce each spatial tile into (sum, sum_sq) for each (N,C) pair.
    Launch: grid = (num_tiles * total_pairs, 1, 1), block = (BLOCK_SIZE, 1, 1)
    """
    tid = al.thread_id(0)
    bid = al.block_id(0)

    bc = bid // num_tiles
    tile_idx = bid - bc * num_tiles

    if bc < total_pairs:
        smem_sum = al.make_shared((BLOCK_SIZE,), al.f32)
        smem_sq = al.make_shared((BLOCK_SIZE,), al.f32)

        base = bc * N
        tile_start = tile_idx * TILE_SIZE
        tile_end = tile_start + TILE_SIZE
        if tile_end > N:
            tile_end = N

        layout_in = al.make_layout((total_pairs * N,), (1,))
        x = al.make_tensor(x_ptr, al.bf16, layout_in)

        local_sum = al.convert(0.0, al.f32)
        local_sq = al.convert(0.0, al.f32)

        for i in al.range(tile_start + tid, tile_end, BLOCK_SIZE):
            idx = base + i
            val = al.convert(x[idx], al.f32)
            local_sum = local_sum + val
            local_sq = local_sq + val * val

        smem_sum[tid] = local_sum
        smem_sq[tid] = local_sq
        al.syncthreads()

        if tid < 128:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 128]
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 128]
        al.syncthreads()
        if tid < 64:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 64]
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 64]
        al.syncthreads()
        if tid < 32:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 32]
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 32]
        al.syncthreads()
        if tid < 16:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 16]
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 16]
        al.syncthreads()
        if tid < 8:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 8]
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 8]
        al.syncthreads()
        if tid < 4:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 4]
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 4]
        al.syncthreads()
        if tid < 2:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 2]
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 2]
        al.syncthreads()
        if tid < 1:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 1]
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 1]

        if tid == 0:
            layout_ps = al.make_layout((total_pairs, num_tiles), (num_tiles, 1))
            ps = al.make_tensor(partial_sum_ptr, al.f32, layout_ps)
            psq = al.make_tensor(partial_sq_ptr, al.f32, layout_ps)
            ps[bc, tile_idx] = smem_sum[0]
            psq[bc, tile_idx] = smem_sq[0]


@avelang.jit
def instancenorm_aggregate_kernel(
    partial_sum_ptr: al.Pointer(al.f32),
    partial_sq_ptr: al.Pointer(al.f32),
    mean_out_ptr: al.Pointer(al.f32),
    rstd_out_ptr: al.Pointer(al.f32),
    total_pairs: al.i32,
    num_tiles: al.i32,
    N: al.i32,
):
    """
    Phase 2: aggregate partial tile results into per-(N,C) mean and rstd.
    Launch: grid = (total_pairs, 1, 1), block = (BLOCK_SIZE, 1, 1)
    """
    tid = al.thread_id(0)
    bid = al.block_id(0)

    if bid < total_pairs:
        smem_sum = al.make_shared((BLOCK_SIZE,), al.f32)
        smem_sq = al.make_shared((BLOCK_SIZE,), al.f32)

        layout_ps = al.make_layout((total_pairs, num_tiles), (num_tiles, 1))
        ps = al.make_tensor(partial_sum_ptr, al.f32, layout_ps)
        psq = al.make_tensor(partial_sq_ptr, al.f32, layout_ps)

        total_sum = al.convert(0.0, al.f32)
        total_sq = al.convert(0.0, al.f32)

        chunk_start = al.convert(0, al.i32)
        for _ in al.range(0, 16):
            if chunk_start >= num_tiles:
                break

            local_sum = al.convert(0.0, al.f32)
            local_sq = al.convert(0.0, al.f32)

            tile_idx = chunk_start + tid
            if tile_idx < num_tiles:
                local_sum = ps[bid, tile_idx]
                local_sq = psq[bid, tile_idx]

            smem_sum[tid] = local_sum
            smem_sq[tid] = local_sq
            al.syncthreads()

            if tid < 128:
                smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 128]
                smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 128]
            al.syncthreads()
            if tid < 64:
                smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 64]
                smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 64]
            al.syncthreads()
            if tid < 32:
                smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 32]
                smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 32]
            al.syncthreads()
            if tid < 16:
                smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 16]
                smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 16]
            al.syncthreads()
            if tid < 8:
                smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 8]
                smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 8]
            al.syncthreads()
            if tid < 4:
                smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 4]
                smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 4]
            al.syncthreads()
            if tid < 2:
                smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 2]
                smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 2]
            al.syncthreads()
            if tid < 1:
                smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 1]
                smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 1]

            if tid == 0:
                total_sum = total_sum + smem_sum[0]
                total_sq = total_sq + smem_sq[0]

            chunk_start = chunk_start + MAX_TILES_PER_ROUND
            al.syncthreads()

        if tid == 0:
            N_f32 = al.convert(N, al.f32)
            mean = total_sum / N_f32
            var = total_sq / N_f32 - mean * mean
            rstd = al.convert(1.0, al.f32) / al.sqrt(var + al.convert(1e-5, al.f32))

            layout_out = al.make_layout((total_pairs,), (1,))
            mo = al.make_tensor(mean_out_ptr, al.f32, layout_out)
            ro = al.make_tensor(rstd_out_ptr, al.f32, layout_out)
            mo[bid] = mean
            ro[bid] = rstd


@avelang.jit
def instancenorm_apply_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    mean_ptr: al.Pointer(al.f32),
    rstd_ptr: al.Pointer(al.f32),
    total_pairs: al.i32,
    N: al.i32,
    num_tiles: al.i32,
):
    """
    Phase 3: apply normalization: out = (x - mean) * rstd.
    Launch: grid = (num_tiles * total_pairs, 1, 1), block = (BLOCK_SIZE, 1, 1)
    """
    tid = al.thread_id(0)
    bid = al.block_id(0)

    bc = bid // num_tiles
    tile_idx = bid - bc * num_tiles

    if bc < total_pairs:
        base = bc * N
        tile_start = tile_idx * TILE_SIZE
        tile_end = tile_start + TILE_SIZE
        if tile_end > N:
            tile_end = N

        layout_in = al.make_layout((total_pairs * N,), (1,))
        x = al.make_tensor(x_ptr, al.bf16, layout_in)

        layout_out = al.make_layout((total_pairs * N,), (1,))
        ot = al.make_tensor(out_ptr, al.bf16, layout_out)

        layout_mean = al.make_layout((total_pairs,), (1,))
        mt = al.make_tensor(mean_ptr, al.f32, layout_mean)
        rt = al.make_tensor(rstd_ptr, al.f32, layout_mean)

        mean = mt[bc]
        rstd = rt[bc]

        for i in al.range(tile_start + tid, tile_end, BLOCK_SIZE):
            idx = base + i
            x_val = al.convert(x[idx], al.f32)
            normalized = (x_val - mean) * rstd
            ot[idx] = al.convert(normalized, al.bf16)


def avelang_instancenorm(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Tensors must be on CUDA/HIP device."
    assert x.dtype == torch.bfloat16, "Input tensor must be bfloat16"

    batch_size = x.shape[0]
    features = x.shape[1]
    dim1 = x.shape[2]
    dim2 = x.shape[3]
    N = dim1 * dim2

    x_contig = x.contiguous()
    num_tiles = (N + TILE_SIZE - 1) // TILE_SIZE
    total_pairs = batch_size * features

    # Phase 1: tile-level reduction
    partial_sum = torch.empty((total_pairs, num_tiles), dtype=torch.float32, device=x.device)
    partial_sq = torch.empty((total_pairs, num_tiles), dtype=torch.float32, device=x.device)

    instancenorm_reduce_kernel[lambda: ((total_pairs * num_tiles, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_contig, partial_sum, partial_sq, total_pairs, N, num_tiles
    )

    # Phase 2: aggregate across tiles
    mean_out = torch.empty((total_pairs,), dtype=torch.float32, device=x.device)
    rstd_out = torch.empty((total_pairs,), dtype=torch.float32, device=x.device)

    instancenorm_aggregate_kernel[lambda: ((total_pairs, 1, 1), (BLOCK_SIZE, 1, 1))](
        partial_sum, partial_sq, mean_out, rstd_out, total_pairs, num_tiles, N
    )

    # Phase 3: apply normalization
    out = torch.empty_like(x_contig)

    instancenorm_apply_kernel[lambda: ((total_pairs * num_tiles, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_contig, out, mean_out, rstd_out, total_pairs, N, num_tiles
    )

    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs Instance Normalization using AveLang DSL.
    """
    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        self.num_features = num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_bf16 = x.to(dtype=torch.bfloat16, device=x.device).contiguous()
        result = avelang_instancenorm(x_bf16)
        return result.to(x.dtype)
