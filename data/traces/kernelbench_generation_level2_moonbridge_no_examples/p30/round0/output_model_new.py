import torch
import torch.nn as nn
import avelang
import avelang.language as al

# =============================================================================
# GEMM Kernel: 64x64 tile, K=16, unrolled B reads
# =============================================================================

@avelang.jit
def gemm_kernel(
    a_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.f32),
    c_ptr: al.Pointer(al.bf16),
    M: al.i32,
    N: al.i32,
    K: al.i32,
):
    bx = al.block_id(0)
    by = al.block_id(1)
    tx = al.thread_id(0)
    ty = al.thread_id(1)

    m_start = bx * 64
    n_start = by * 64

    a_layout = al.make_layout((M, K), (K, 1))
    a = al.make_tensor(a_ptr, al.bf16, a_layout)
    b_layout = al.make_layout((N, K), (K, 1))
    b = al.make_tensor(b_ptr, al.bf16, b_layout)

    a_tile = al.make_shared((64, 16), al.bf16)
    b_tile = al.make_shared((16, 64), al.bf16)

    acc = al.make_local((4, 4), al.f32)
    for i in al.range(4):
        for j in al.range(4):
            acc[i, j] = al.convert(0.0, al.f32)

    for k_block in al.range(0, K, 16):
        # Load A tile: 64x16, 16x16 threads, 4 rows per thread
        for i in al.range(4):
            a_row = m_start + ty * 4 + i
            a_col = k_block + tx
            if a_row < M:
                if a_col < K:
                    a_tile[ty * 4 + i, tx] = a[a_row, a_col]

        # Load B tile: 16x64, 16x16 threads, 4 cols per thread
        for j in al.range(4):
            b_n_idx = n_start + tx * 4 + j
            b_k_idx = k_block + ty
            if b_n_idx < N:
                if b_k_idx < K:
                    b_tile[ty, tx * 4 + j] = b[b_n_idx, b_k_idx]

        al.syncthreads()

        # Unrolled compute: 1 A read per 4 B reads
        for k in al.range(16):
            for i in al.range(4):
                a_val = al.convert(a_tile[ty * 4 + i, k], al.f32)
                b0 = al.convert(b_tile[k, tx * 4 + 0], al.f32)
                b1 = al.convert(b_tile[k, tx * 4 + 1], al.f32)
                b2 = al.convert(b_tile[k, tx * 4 + 2], al.f32)
                b3 = al.convert(b_tile[k, tx * 4 + 3], al.f32)
                acc[i, 0] = acc[i, 0] + a_val * b0
                acc[i, 1] = acc[i, 1] + a_val * b1
                acc[i, 2] = acc[i, 2] + a_val * b2
                acc[i, 3] = acc[i, 3] + a_val * b3

        al.syncthreads()

    bias_layout = al.make_layout((N,), (1,))
    bias_view = al.make_tensor(bias_ptr, al.f32, bias_layout)
    c_layout = al.make_layout((M, N), (N, 1))
    c = al.make_tensor(c_ptr, al.bf16, c_layout)

    for i in al.range(4):
        global_row = m_start + ty * 4 + i
        if global_row < M:
            for j in al.range(4):
                global_col = n_start + tx * 4 + j
                if global_col < N:
                    result = acc[i, j] + bias_view[global_col]
                    c[global_row, global_col] = al.convert(result, al.bf16)


# =============================================================================
# GroupNorm + HardTanh fused kernel (training mode)
# =============================================================================

@avelang.jit
def groupnorm_hardtanh_kernel(
    x_ptr: al.Pointer(al.bf16),
    gamma_ptr: al.Pointer(al.f32),
    beta_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.bf16),
    num_samples: al.i32,
    C: al.i32,
    G: al.i32,
    CG: al.i32,
):
    sample_idx = al.block_id(0)
    group_idx = al.block_id(1)
    tid = al.thread_id(0)

    c_start = group_idx * CG

    x_layout = al.make_layout((num_samples, C), (C, 1))
    x = al.make_tensor(x_ptr, al.bf16, x_layout)
    gamma_layout = al.make_layout((C,), (1,))
    gamma = al.make_tensor(gamma_ptr, al.f32, gamma_layout)
    beta_layout = al.make_layout((C,), (1,))
    beta_view = al.make_tensor(beta_ptr, al.f32, beta_layout)
    out_layout = al.make_layout((num_samples, C), (C, 1))
    out = al.make_tensor(out_ptr, al.bf16, out_layout)

    val0 = al.convert(0.0, al.f32)
    val1 = al.convert(0.0, al.f32)
    c_idx0 = c_start + tid
    c_idx1 = c_start + tid + 256
    if c_idx0 < c_start + CG:
        val0 = al.convert(x[sample_idx, c_idx0], al.f32)
    if c_idx1 < c_start + CG:
        val1 = al.convert(x[sample_idx, c_idx1], al.f32)

    smem = al.make_shared((256,), al.f32)
    smem_sq = al.make_shared((256,), al.f32)
    val_sum = val0 + val1
    val_sq = val0 * val0 + val1 * val1
    smem[tid] = val_sum
    smem_sq[tid] = val_sq
    al.syncthreads()

    if tid < 128:
        smem[tid] = smem[tid] + smem[tid + 128]
        smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 128]
    al.syncthreads()
    if tid < 64:
        smem[tid] = smem[tid] + smem[tid + 64]
        smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 64]
    al.syncthreads()
    if tid < 32:
        smem[tid] = smem[tid] + smem[tid + 32]
        smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 32]
    al.syncthreads()
    if tid < 16:
        smem[tid] = smem[tid] + smem[tid + 16]
        smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 16]
    al.syncthreads()
    if tid < 8:
        smem[tid] = smem[tid] + smem[tid + 8]
        smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 8]
    al.syncthreads()
    if tid < 4:
        smem[tid] = smem[tid] + smem[tid + 4]
        smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 4]
    al.syncthreads()
    if tid < 2:
        smem[tid] = smem[tid] + smem[tid + 2]
        smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 2]
    al.syncthreads()
    if tid < 1:
        smem[tid] = smem[tid] + smem[tid + 1]
        smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 1]
    al.syncthreads()

    if tid == 0:
        sum_val = smem[0]
        sum_sq_val = smem_sq[0]
        mean_val = sum_val / al.convert(CG, al.f32)
        var_val = sum_sq_val / al.convert(CG, al.f32) - mean_val * mean_val
        eps = al.convert(0.00001, al.f32)
        inv_std_val = al.convert(1.0, al.f32) / al.sqrt(var_val + eps)
        smem[0] = mean_val
        smem[1] = inv_std_val
    al.syncthreads()

    mean_val = smem[0]
    inv_std_val = smem[1]

    hmin = al.convert(-2.0, al.f32)
    hmax = al.convert(2.0, al.f32)

    if c_idx0 < c_start + CG:
        norm_val = (val0 - mean_val) * inv_std_val
        result = norm_val * gamma[c_idx0] + beta_view[c_idx0]
        if result < hmin:
            result = hmin
        else:
            if result > hmax:
                result = hmax
        out[sample_idx, c_idx0] = al.convert(result, al.bf16)

    if c_idx1 < c_start + CG:
        norm_val = (val1 - mean_val) * inv_std_val
        result = norm_val * gamma[c_idx1] + beta_view[c_idx1]
        if result < hmin:
            result = hmin
        else:
            if result > hmax:
                result = hmax
        out[sample_idx, c_idx1] = al.convert(result, al.bf16)


# =============================================================================
# Host wrapper
# =============================================================================

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max

        self.gemm_layer = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)

    def forward(self, x):
        batch_size = x.shape[0]
        M_val = batch_size
        N_val = self.out_features
        K_val = self.in_features
        G_val = self.num_groups
        CG_val = self.out_features // self.num_groups

        x = x.contiguous()
        weight = self.gemm_layer.weight.data.contiguous()
        bias = self.gemm_layer.bias.data.float().contiguous()
        gamma = self.group_norm.weight.data.float().contiguous()
        beta_w = self.group_norm.bias.data.float().contiguous()

        x_bf16 = x.to(torch.bfloat16).contiguous()
        w_bf16 = weight.to(torch.bfloat16).contiguous()

        gemm_out = torch.empty(batch_size, self.out_features, dtype=torch.bfloat16, device=x.device)

        grid_m = (M_val + 63) // 64
        grid_n = (N_val + 63) // 64
        gemm_kernel[lambda: ((grid_m, grid_n, 1), (16, 16, 1))](
            x_bf16.data_ptr(),
            w_bf16.data_ptr(),
            bias.data_ptr(),
            gemm_out.data_ptr(),
            M_val,
            N_val,
            K_val,
        )

        out = torch.empty(batch_size, self.out_features, dtype=torch.bfloat16, device=x.device)

        groupnorm_hardtanh_kernel[lambda: ((M_val, G_val, 1), (256, 1, 1))](
            gemm_out.data_ptr(),
            gamma.data_ptr(),
            beta_w.data_ptr(),
            out.data_ptr(),
            M_val,
            N_val,
            G_val,
            CG_val,
        )

        return out
