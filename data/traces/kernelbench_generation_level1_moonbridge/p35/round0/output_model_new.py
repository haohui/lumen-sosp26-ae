import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE: al.constexpr = 256
TILE_SIZE: al.constexpr = 4096
MAX_TILES_PER_ROUND: al.constexpr = 256


@avelang.jit
def group_norm_reduce_kernel(
    x_ptr: al.Pointer(al.bf16),
    partial_sum_ptr: al.Pointer(al.f32),
    partial_sq_ptr: al.Pointer(al.f32),
    total_pairs: al.i32,
    pair_N: al.i32,
    num_tiles: al.i32,
):
    """
    Phase 1: reduce each tile of each (batch, group) pair into (sum, sum_sq).
    Launch: grid = (total_pairs * num_tiles, 1, 1), block = (BLOCK_SIZE, 1, 1)
    """
    tid = al.thread_id(0)
    bid = al.block_id(0)

    pair_idx = bid // num_tiles
    tile_idx = bid - pair_idx * num_tiles

    if pair_idx < total_pairs:
        smem_sum = al.make_shared((BLOCK_SIZE,), al.f32)
        smem_sq = al.make_shared((BLOCK_SIZE,), al.f32)

        pair_start = pair_idx * pair_N
        tile_start = tile_idx * TILE_SIZE
        tile_end = tile_start + TILE_SIZE
        if tile_end > pair_N:
            tile_end = pair_N

        layout_in = al.make_layout((total_pairs * pair_N,), (1,))
        x = al.make_tensor(x_ptr, al.bf16, layout_in)

        local_sum = al.convert(0.0, al.f32)
        local_sq = al.convert(0.0, al.f32)

        for i in al.range(tile_start + tid, tile_end, BLOCK_SIZE):
            idx = pair_start + i
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
            ps[pair_idx, tile_idx] = smem_sum[0]
            psq[pair_idx, tile_idx] = smem_sq[0]


@avelang.jit
def group_norm_aggregate_kernel(
    partial_sum_ptr: al.Pointer(al.f32),
    partial_sq_ptr: al.Pointer(al.f32),
    mean_out_ptr: al.Pointer(al.f32),
    rstd_out_ptr: al.Pointer(al.f32),
    total_pairs: al.i32,
    num_tiles: al.i32,
    pair_N: al.i32,
):
    """
    Phase 2: aggregate partial tile results into per-pair mean and rstd.
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
            eps = al.convert(1e-5, al.f32)
            pair_N_f32 = al.convert(pair_N, al.f32)
            mean = total_sum / pair_N_f32
            var = total_sq / pair_N_f32 - mean * mean
            rstd = al.convert(1.0, al.f32) / al.sqrt(var + eps)

            layout_out = al.make_layout((total_pairs,), (1,))
            mo = al.make_tensor(mean_out_ptr, al.f32, layout_out)
            ro = al.make_tensor(rstd_out_ptr, al.f32, layout_out)
            mo[bid] = mean
            ro[bid] = rstd


@avelang.jit
def group_norm_apply_kernel(
    x_ptr: al.Pointer(al.bf16),
    gamma_ptr: al.Pointer(al.bf16),
    beta_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    mean_ptr: al.Pointer(al.f32),
    rstd_ptr: al.Pointer(al.f32),
    total_pairs: al.i32,
    pair_N: al.i32,
    num_tiles: al.i32,
    G: al.i32,
    C_per_group: al.i32,
    HW: al.i32,
):
    """
    Phase 3: apply normalization and affine transform.
    Launch: grid = (total_pairs * num_tiles, 1, 1), block = (BLOCK_SIZE, 1, 1)
    """
    tid = al.thread_id(0)
    bid = al.block_id(0)

    pair_idx = bid // num_tiles
    tile_idx = bid - pair_idx * num_tiles

    if pair_idx < total_pairs:
        pair_start = pair_idx * pair_N
        tile_start = tile_idx * TILE_SIZE
        tile_end = tile_start + TILE_SIZE
        if tile_end > pair_N:
            tile_end = pair_N

        group_idx = pair_idx - (pair_idx // G) * G

        layout_in = al.make_layout((total_pairs * pair_N,), (1,))
        x = al.make_tensor(x_ptr, al.bf16, layout_in)

        layout_w = al.make_layout((G * C_per_group,), (1,))
        gamma = al.make_tensor(gamma_ptr, al.bf16, layout_w)
        beta = al.make_tensor(beta_ptr, al.bf16, layout_w)

        layout_out = al.make_layout((total_pairs * pair_N,), (1,))
        out = al.make_tensor(out_ptr, al.bf16, layout_out)

        layout_stat = al.make_layout((total_pairs,), (1,))
        mean_t = al.make_tensor(mean_ptr, al.f32, layout_stat)
        rstd_t = al.make_tensor(rstd_ptr, al.f32, layout_stat)

        mean_val = mean_t[pair_idx]
        rstd_val = rstd_t[pair_idx]

        for i in al.range(tile_start + tid, tile_end, BLOCK_SIZE):
            flat_idx = pair_start + i
            local_c = i // HW
            global_c = group_idx * C_per_group + local_c

            x_val = al.convert(x[flat_idx], al.f32)
            g_val = al.convert(gamma[global_c], al.f32)
            b_val = al.convert(beta[global_c], al.f32)

            normalized = (x_val - mean_val) * rstd_val
            result = normalized * g_val + b_val
            out[flat_idx] = al.convert(result, al.bf16)


def avelang_group_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_groups: int,
    eps: float,
) -> torch.Tensor:
    assert x.is_cuda, "Tensors must be on CUDA/HIP device."
    assert x.dtype == torch.bfloat16, "Input tensor must be bfloat16"

    N = x.shape[0]
    C = x.shape[1]
    H = x.shape[2]
    W = x.shape[3]
    G = num_groups
    C_per_group = C // G
    HW = H * W
    pair_N = C_per_group * HW

    x_contig = x.contiguous()
    w_bf16 = weight.contiguous().to(torch.bfloat16)
    b_bf16 = bias.contiguous().to(torch.bfloat16)

    total_pairs = N * G
    num_tiles = (pair_N + TILE_SIZE - 1) // TILE_SIZE

    # Phase 1: tile-level reduction
    partial_sum = torch.empty((total_pairs, num_tiles), dtype=torch.float32, device=x.device)
    partial_sq = torch.empty((total_pairs, num_tiles), dtype=torch.float32, device=x.device)

    group_norm_reduce_kernel[lambda: ((total_pairs * num_tiles, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_contig, partial_sum, partial_sq, total_pairs, pair_N, num_tiles
    )

    # Phase 2: aggregate across tiles
    mean_out = torch.empty((total_pairs,), dtype=torch.float32, device=x.device)
    rstd_out = torch.empty((total_pairs,), dtype=torch.float32, device=x.device)

    group_norm_aggregate_kernel[lambda: ((total_pairs, 1, 1), (BLOCK_SIZE, 1, 1))](
        partial_sum, partial_sq, mean_out, rstd_out, total_pairs, num_tiles, pair_N
    )

    # Phase 3: apply normalization
    out = torch.empty_like(x_contig)

    group_norm_apply_kernel[lambda: ((total_pairs * num_tiles, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_contig, w_bf16, b_bf16, out, mean_out, rstd_out,
        total_pairs, pair_N, num_tiles, G, C_per_group, HW
    )

    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs Group Normalization using AveLang DSL.
    """
    def __init__(self, num_features: int, num_groups: int):
        super(ModelNew, self).__init__()
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.gn.weight.data
        bias = self.gn.bias.data

        x_bf16 = x.to(dtype=torch.bfloat16, device=x.device).contiguous()

        result = avelang_group_norm(x_bf16, weight, bias, self.gn.num_groups, self.gn.eps)
        return result.to(x.dtype)
