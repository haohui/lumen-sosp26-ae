import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE: al.constexpr = 256
TILE_SIZE: al.constexpr = 4096
MAX_TILES_PER_ROUND: al.constexpr = 256
WGT_PER_CHAN: al.constexpr = 81


# =============================================================================
# Phase 1: 3D Convolution Kernel (with shared-memory weight cache)
# =============================================================================
@avelang.jit
def conv3d_kernel(
    inp_ptr: al.Pointer(al.bf16),
    wgt_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C_in: al.i32,
    C_out: al.i32,
    D: al.i32,
    H: al.i32,
    W: al.i32,
    D_out: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    K: al.i32,
    in_N_s: al.i32,
    in_C_s: al.i32,
    in_D_s: al.i32,
    in_H_s: al.i32,
    w_CO_s: al.i32,
    out_N_s: al.i32,
    out_CO_s: al.i32,
    out_D_s: al.i32,
    out_H_s: al.i32,
    total_in: al.i32,
    total_w: al.i32,
    total_out: al.i32,
    out_per_chan: al.i32,
):
    tid = al.thread_id(0)
    bid_n = al.block_id(0)
    bid_co = al.block_id(1)
    bid_sp = al.block_id(2)

    spatial_idx = bid_sp * BLOCK_SIZE + tid

    if bid_n < N and bid_co < C_out and spatial_idx < out_per_chan:
        d = spatial_idx // (H_out * W_out)
        rem = spatial_idx - d * (H_out * W_out)
        h = rem // W_out
        w = rem - h * W_out

        layout_in = al.make_layout((total_in,), (1,))
        inp = al.make_tensor(inp_ptr, al.bf16, layout_in)

        layout_w = al.make_layout((total_w,), (1,))
        wgt = al.make_tensor(wgt_ptr, al.bf16, layout_w)

        layout_b = al.make_layout((C_out,), (1,))
        bias = al.make_tensor(bias_ptr, al.bf16, layout_b)

        layout_out = al.make_layout((total_out,), (1,))
        ot = al.make_tensor(out_ptr, al.bf16, layout_out)

        # Load per-channel weight into shared memory
        smem_wgt = al.make_shared((WGT_PER_CHAN,), al.bf16)
        co_w_base = bid_co * w_CO_s
        if tid < WGT_PER_CHAN:
            smem_wgt[tid] = wgt[co_w_base + tid]
        al.syncthreads()

        acc = al.convert(0.0, al.f32)

        n_base = bid_n * in_N_s
        out_n_base = bid_n * out_N_s
        co_out_base = bid_co * out_CO_s

        K2 = K * K
        for ci in al.range(C_in):
            ci_in_off = ci * in_C_s
            ci_w_off = ci * K * K2
            for kd in al.range(K):
                kd_in_off = kd * in_D_s
                kd_w_off = kd * K2
                for kh in al.range(K):
                    kh_in_off = kh * in_H_s
                    kh_w_off = kh * K
                    for kw in al.range(K):
                        in_idx = n_base + ci_in_off + (d + kd) * in_D_s + (h + kh) * in_H_s + (w + kw)
                        w_flat = ci_w_off + kd_w_off + kh_w_off + kw
                        in_val = al.convert(inp[in_idx], al.f32)
                        w_val = al.convert(smem_wgt[w_flat], al.f32)
                        acc = acc + in_val * w_val

        b_val = al.convert(bias[bid_co], al.f32)
        acc = acc + b_val

        out_idx = out_n_base + co_out_base + d * out_D_s + h * out_H_s + w
        ot[out_idx] = al.convert(acc, al.bf16)


# =============================================================================
# Phase 2a: GroupNorm Tile-Level Reduction
# =============================================================================
@avelang.jit
def group_norm_reduce_kernel(
    data_ptr: al.Pointer(al.bf16),
    partial_sum_ptr: al.Pointer(al.f32),
    partial_sq_ptr: al.Pointer(al.f32),
    batch_size: al.i32,
    num_groups: al.i32,
    per_group_elems: al.i32,
    num_tiles: al.i32,
    total_data: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    total_per_batch = num_groups * num_tiles
    batch_idx = bid // total_per_batch
    rem = bid - batch_idx * total_per_batch
    group_idx = rem // num_tiles
    tile_idx = rem - group_idx * num_tiles

    if batch_idx < batch_size:
        smem_sum = al.make_shared((BLOCK_SIZE,), al.f32)
        smem_sq = al.make_shared((BLOCK_SIZE,), al.f32)

        layout_data = al.make_layout((total_data,), (1,))
        data = al.make_tensor(data_ptr, al.bf16, layout_data)

        tile_start = tile_idx * TILE_SIZE
        tile_end = tile_start + TILE_SIZE
        if tile_end > per_group_elems:
            tile_end = per_group_elems

        group_base = batch_idx * num_groups * per_group_elems + group_idx * per_group_elems

        local_sum = al.convert(0.0, al.f32)
        local_sq = al.convert(0.0, al.f32)

        for i in al.range(tile_start + tid, tile_end, BLOCK_SIZE):
            idx = group_base + i
            val = al.convert(data[idx], al.f32)
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
            total_rows = batch_size * num_groups
            layout_ps = al.make_layout((total_rows, num_tiles), (num_tiles, 1))
            ps = al.make_tensor(partial_sum_ptr, al.f32, layout_ps)
            psq = al.make_tensor(partial_sq_ptr, al.f32, layout_ps)
            out_row = batch_idx * num_groups + group_idx
            ps[out_row, tile_idx] = smem_sum[0]
            psq[out_row, tile_idx] = smem_sq[0]


# =============================================================================
# Phase 2b: GroupNorm Aggregate
# =============================================================================
@avelang.jit
def group_norm_aggregate_kernel(
    partial_sum_ptr: al.Pointer(al.f32),
    partial_sq_ptr: al.Pointer(al.f32),
    mean_out_ptr: al.Pointer(al.f32),
    rstd_out_ptr: al.Pointer(al.f32),
    batch_size: al.i32,
    num_groups: al.i32,
    num_tiles: al.i32,
    per_group_elems: al.i32,
    eps: al.f32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    total_rows = batch_size * num_groups
    if bid < total_rows:
        smem_sum = al.make_shared((BLOCK_SIZE,), al.f32)
        smem_sq = al.make_shared((BLOCK_SIZE,), al.f32)

        layout_ps = al.make_layout((total_rows, num_tiles), (num_tiles, 1))
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
            N_f32 = al.convert(per_group_elems, al.f32)
            mean = total_sum / N_f32
            var = total_sq / N_f32 - mean * mean
            rstd = al.convert(1.0, al.f32) / al.sqrt(var + eps)

            layout_m = al.make_layout((total_rows,), (1,))
            mo = al.make_tensor(mean_out_ptr, al.f32, layout_m)
            ro = al.make_tensor(rstd_out_ptr, al.f32, layout_m)
            mo[bid] = mean
            ro[bid] = rstd


# =============================================================================
# Phase 2c: GroupNorm Apply
# =============================================================================
@avelang.jit
def group_norm_apply_kernel(
    data_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    mean_ptr: al.Pointer(al.f32),
    rstd_ptr: al.Pointer(al.f32),
    batch_size: al.i32,
    num_groups: al.i32,
    ch_per_group: al.i32,
    C_out: al.i32,
    out_CO_s: al.i32,
    out_D_s: al.i32,
    out_H_s: al.i32,
    per_group_elems: al.i32,
    num_tiles: al.i32,
    total_data: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    total_per_batch = num_groups * num_tiles
    batch_idx = bid // total_per_batch
    rem = bid - batch_idx * total_per_batch
    group_idx = rem // num_tiles
    tile_idx = rem - group_idx * num_tiles

    if batch_idx < batch_size:
        tile_start = tile_idx * TILE_SIZE
        tile_end = tile_start + TILE_SIZE
        if tile_end > per_group_elems:
            tile_end = per_group_elems

        layout_data = al.make_layout((total_data,), (1,))
        data = al.make_tensor(data_ptr, al.bf16, layout_data)

        layout_out = al.make_layout((total_data,), (1,))
        ot = al.make_tensor(out_ptr, al.bf16, layout_out)

        layout_w = al.make_layout((C_out,), (1,))
        wt = al.make_tensor(weight_ptr, al.bf16, layout_w)
        bt = al.make_tensor(bias_ptr, al.bf16, layout_w)

        total_rows = batch_size * num_groups
        layout_m = al.make_layout((total_rows,), (1,))
        mt = al.make_tensor(mean_ptr, al.f32, layout_m)
        rt = al.make_tensor(rstd_ptr, al.f32, layout_m)

        mean_row = batch_idx * num_groups + group_idx
        mean_val = mt[mean_row]
        rstd_val = rt[mean_row]

        batch_base = batch_idx * C_out * out_CO_s
        group_ch_start = group_idx * ch_per_group

        for i in al.range(tile_start + tid, tile_end, BLOCK_SIZE):
            c_local = i // out_CO_s
            spat = i - c_local * out_CO_s
            c_global = group_ch_start + c_local
            d = spat // out_D_s
            rem_sp = spat - d * out_D_s
            h = rem_sp // out_H_s
            w = rem_sp - h * out_H_s

            data_idx = batch_base + c_global * out_CO_s + d * out_D_s + h * out_H_s + w

            x_val = al.convert(data[data_idx], al.f32)
            w_val = al.convert(wt[c_global], al.f32)
            b_val = al.convert(bt[c_global], al.f32)

            normed = (x_val - mean_val) * rstd_val
            result = normed * w_val + b_val
            ot[data_idx] = al.convert(result, al.bf16)


# =============================================================================
# Phase 3a: Mean Reduction - Partial Sums
# =============================================================================
@avelang.jit
def mean_reduce_kernel(
    data_ptr: al.Pointer(al.bf16),
    partial_sum_ptr: al.Pointer(al.f32),
    batch_size: al.i32,
    out_per_batch: al.i32,
    num_tiles: al.i32,
    total_data: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    batch_idx = bid // num_tiles
    tile_idx = bid - batch_idx * num_tiles

    if batch_idx < batch_size:
        smem_sum = al.make_shared((BLOCK_SIZE,), al.f32)

        tile_start = tile_idx * TILE_SIZE
        tile_end = tile_start + TILE_SIZE
        if tile_end > out_per_batch:
            tile_end = out_per_batch

        layout_data = al.make_layout((total_data,), (1,))
        data = al.make_tensor(data_ptr, al.bf16, layout_data)

        base = batch_idx * out_per_batch
        local_sum = al.convert(0.0, al.f32)

        for i in al.range(tile_start + tid, tile_end, BLOCK_SIZE):
            idx = base + i
            val = al.convert(data[idx], al.f32)
            local_sum = local_sum + val

        smem_sum[tid] = local_sum
        al.syncthreads()

        if tid < 128:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 128]
        al.syncthreads()
        if tid < 64:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 64]
        al.syncthreads()
        if tid < 32:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 32]
        al.syncthreads()
        if tid < 16:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 16]
        al.syncthreads()
        if tid < 8:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 8]
        al.syncthreads()
        if tid < 4:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 4]
        al.syncthreads()
        if tid < 2:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 2]
        al.syncthreads()
        if tid < 1:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 1]

        if tid == 0:
            layout_ps = al.make_layout((batch_size, num_tiles), (num_tiles, 1))
            ps = al.make_tensor(partial_sum_ptr, al.f32, layout_ps)
            ps[batch_idx, tile_idx] = smem_sum[0]


# =============================================================================
# Phase 3b: Mean Reduction - Aggregate
# =============================================================================
@avelang.jit
def mean_aggregate_kernel(
    partial_sum_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.bf16),
    batch_size: al.i32,
    num_tiles: al.i32,
    out_per_batch: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    if bid < batch_size:
        smem_sum = al.make_shared((BLOCK_SIZE,), al.f32)

        layout_ps = al.make_layout((batch_size, num_tiles), (num_tiles, 1))
        ps = al.make_tensor(partial_sum_ptr, al.f32, layout_ps)

        total_sum = al.convert(0.0, al.f32)

        chunk_start = al.convert(0, al.i32)
        for _ in al.range(0, 16):
            if chunk_start >= num_tiles:
                break

            local_sum = al.convert(0.0, al.f32)
            tile_idx = chunk_start + tid
            if tile_idx < num_tiles:
                local_sum = ps[bid, tile_idx]

            smem_sum[tid] = local_sum
            al.syncthreads()

            if tid < 128:
                smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 128]
            al.syncthreads()
            if tid < 64:
                smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 64]
            al.syncthreads()
            if tid < 32:
                smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 32]
            al.syncthreads()
            if tid < 16:
                smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 16]
            al.syncthreads()
            if tid < 8:
                smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 8]
            al.syncthreads()
            if tid < 4:
                smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 4]
            al.syncthreads()
            if tid < 2:
                smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 2]
            al.syncthreads()
            if tid < 1:
                smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 1]

            if tid == 0:
                total_sum = total_sum + smem_sum[0]

            chunk_start = chunk_start + MAX_TILES_PER_ROUND
            al.syncthreads()

        if tid == 0:
            N_f32 = al.convert(out_per_batch, al.f32)
            mean_val = total_sum / N_f32
            layout_out = al.make_layout((batch_size, 1), (1, 1))
            ot = al.make_tensor(out_ptr, al.bf16, layout_out)
            ot[bid, 0] = al.convert(mean_val, al.bf16)


# =============================================================================
# Host Wrapper Functions
# =============================================================================

def _prepare_bf16(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_conv3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    x_bf16 = _prepare_bf16(x)
    w_bf16 = _prepare_bf16(weight)
    b_bf16 = _prepare_bf16(bias)

    N, C_in, D, H, W = x_bf16.shape
    C_out, C_in_w, K, _, _ = w_bf16.shape

    D_out = D - K + 1
    H_out = H - K + 1
    W_out = W - K + 1

    out_per_chan = D_out * H_out * W_out

    in_N_s = C_in * D * H * W
    in_C_s = D * H * W
    in_D_s = H * W
    in_H_s = W
    w_CO_s = C_in * K * K * K
    out_N_s = C_out * D_out * H_out * W_out
    out_CO_s = D_out * H_out * W_out
    out_D_s = H_out * W_out
    out_H_s = W_out

    total_in = N * C_in * D * H * W
    total_w = C_out * C_in * K * K * K
    total_out = N * C_out * D_out * H_out * W_out

    num_spatial_tiles = (out_per_chan + BLOCK_SIZE - 1) // BLOCK_SIZE

    out = torch.empty((N, C_out, D_out, H_out, W_out), dtype=torch.bfloat16, device=x_bf16.device)

    grid = (N, C_out, num_spatial_tiles)
    conv3d_kernel[lambda: (grid, (BLOCK_SIZE, 1, 1))](
        x_bf16, w_bf16, b_bf16, out,
        N, C_in, C_out, D, H, W, D_out, H_out, W_out, K,
        in_N_s, in_C_s, in_D_s, in_H_s,
        w_CO_s,
        out_N_s, out_CO_s, out_D_s, out_H_s,
        total_in, total_w, total_out,
        out_per_chan,
    )
    return out


def avelang_group_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_groups: int,
) -> torch.Tensor:
    x_bf16 = _prepare_bf16(x)
    w_bf16 = _prepare_bf16(weight)
    b_bf16 = _prepare_bf16(bias)

    N, C, D, H, W = x_bf16.shape
    ch_per_group = C // num_groups
    out_CO_s = D * H * W
    out_D_s = H * W
    out_H_s = W
    per_group_elems = ch_per_group * out_CO_s
    total_data = N * C * D * H * W

    eps = 1e-5
    num_tiles = (per_group_elems + TILE_SIZE - 1) // TILE_SIZE
    total_groups = N * num_groups

    # Phase 2a: tile-level reduction
    partial_sum = torch.empty((total_groups, num_tiles), dtype=torch.float32, device=x_bf16.device)
    partial_sq = torch.empty((total_groups, num_tiles), dtype=torch.float32, device=x_bf16.device)

    group_norm_reduce_kernel[lambda: ((total_groups * num_tiles, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_bf16, partial_sum, partial_sq,
        N, num_groups, per_group_elems, num_tiles, total_data,
    )

    # Phase 2b: aggregate across tiles
    mean_out = torch.empty((total_groups,), dtype=torch.float32, device=x_bf16.device)
    rstd_out = torch.empty((total_groups,), dtype=torch.float32, device=x_bf16.device)

    group_norm_aggregate_kernel[lambda: ((total_groups, 1, 1), (BLOCK_SIZE, 1, 1))](
        partial_sum, partial_sq, mean_out, rstd_out,
        N, num_groups, num_tiles, per_group_elems, eps,
    )

    # Phase 2c: apply normalization
    out = torch.empty_like(x_bf16)

    group_norm_apply_kernel[lambda: ((total_groups * num_tiles, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_bf16, out, w_bf16, b_bf16, mean_out, rstd_out,
        N, num_groups, ch_per_group, C,
        out_CO_s, out_D_s, out_H_s,
        per_group_elems, num_tiles, total_data,
    )

    return out


def avelang_mean_reduce(x: torch.Tensor) -> torch.Tensor:
    x_bf16 = _prepare_bf16(x)

    N = x_bf16.shape[0]
    out_per_batch = x_bf16.numel() // N
    total_data = N * out_per_batch

    num_tiles = (out_per_batch + TILE_SIZE - 1) // TILE_SIZE

    partial_sum = torch.empty((N, num_tiles), dtype=torch.float32, device=x_bf16.device)

    mean_reduce_kernel[lambda: ((N * num_tiles, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_bf16, partial_sum,
        N, out_per_batch, num_tiles, total_data,
    )

    out = torch.empty((N, 1), dtype=torch.bfloat16, device=x_bf16.device)

    mean_aggregate_kernel[lambda: ((N, 1, 1), (BLOCK_SIZE, 1, 1))](
        partial_sum, out,
        N, num_tiles, out_per_batch,
    )

    return out


# =============================================================================
# ModelNew Entrypoint
# =============================================================================

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_groups = num_groups

        self.conv_weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size, kernel_size)
        )
        self.conv_bias = nn.Parameter(torch.empty(out_channels))
        self.gn_weight = nn.Parameter(torch.empty(out_channels))
        self.gn_bias = nn.Parameter(torch.empty(out_channels))

        nn.init.kaiming_uniform_(self.conv_weight, a=5 ** 0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.conv_weight)
        bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
        nn.init.uniform_(self.conv_bias, -bound, bound)
        nn.init.ones_(self.gn_weight)
        nn.init.zeros_(self.gn_bias)

    def forward(self, x):
        orig_dtype = x.dtype

        x_bf16 = x.to(dtype=torch.bfloat16)

        x_conv = avelang_conv3d(x_bf16, self.conv_weight, self.conv_bias)
        x_gn = avelang_group_norm(x_conv, self.gn_weight, self.gn_bias, self.num_groups)
        x_mean = avelang_mean_reduce(x_gn)

        result = x_mean.to(dtype=orig_dtype)
        return result.squeeze(-1)


batch_size = 128
in_channels = 3
out_channels = 24
D, H, W = 24, 32, 32
kernel_size = 3
num_groups = 8


def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, num_groups]
