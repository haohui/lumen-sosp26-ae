import torch
import torch.nn as nn
import avelang
import avelang.language as al


# ── Matmul kernel: C = A @ B + bias (BF16 I/O, FP32 accumulation) ───────────
# Tile: 32×32 output, 16 K-step, 256 threads, 4 output elements per thread

@avelang.jit
def matmul_bf16_kernel(
    a_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    c_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    M: al.i32,
    N: al.i32,
    K: al.i32,
):
    pid_m = al.block_id(0)
    pid_n = al.block_id(1)
    tid = al.thread_id(0)

    m_start = pid_m * 32
    n_start = pid_n * 32

    a_layout = al.make_layout((M, K), (K, 1))
    a = al.make_tensor(a_ptr, al.bf16, a_layout)
    b_layout = al.make_layout((K, N), (N, 1))
    b = al.make_tensor(b_ptr, al.bf16, b_layout)
    c_layout = al.make_layout((M, N), (N, 1))
    c = al.make_tensor(c_ptr, al.bf16, c_layout)
    bias_layout = al.make_layout((N,), (1,))
    bias_t = al.make_tensor(bias_ptr, al.bf16, bias_layout)

    a_shared = al.make_shared((32, 16), al.bf16)
    b_shared = al.make_shared((16, 32), al.bf16)

    acc = al.full((1, 4), 0.0, al.f32)

    for k_block in al.range(0, K, 16):
        a_idx0 = tid * 2
        a_r0 = a_idx0 // 16
        a_c0 = a_idx0 % 16
        a_idx1 = a_idx0 + 1
        a_r1 = a_idx1 // 16
        a_c1 = a_idx1 % 16
        if a_r0 < 32:
            a_shared[a_r0, a_c0] = a[m_start + a_r0, k_block + a_c0]
        if a_r1 < 32:
            a_shared[a_r1, a_c1] = a[m_start + a_r1, k_block + a_c1]

        b_idx0 = tid * 2
        b_r0 = b_idx0 // 32
        b_c0 = b_idx0 % 32
        b_idx1 = b_idx0 + 1
        b_r1 = b_idx1 // 32
        b_c1 = b_idx1 % 32
        if b_r0 < 16:
            b_shared[b_r0, b_c0] = b[k_block + b_r0, n_start + b_c0]
        if b_r1 < 16:
            b_shared[b_r1, b_c1] = b[k_block + b_r1, n_start + b_c1]

        al.syncthreads()

        for e in al.range(0, 4):
            lm = (tid * 4 + e) // 32
            ln = (tid * 4 + e) % 32
            if lm < 32 and ln < 32:
                for kk in al.range(0, 16):
                    a_val = al.convert(a_shared[lm, kk], al.f32)
                    b_val = al.convert(b_shared[kk, ln], al.f32)
                    acc[0, e] = acc[0, e] + a_val * b_val

        al.syncthreads()

    for e in al.range(0, 4):
        lm = (tid * 4 + e) // 32
        ln = (tid * 4 + e) % 32
        if lm < 32 and ln < 32:
            global_m = m_start + lm
            global_n = n_start + ln
            if global_m < M and global_n < N:
                result = acc[0, e] + al.convert(bias_t[global_n], al.f32)
                c[global_m, global_n] = al.convert(result, al.bf16)


# ── GroupNorm stats: per-(sample, group) mean and var in FP32 ────────────────
# Input: x (M, C) bf16 → internally upcast to FP32
# Output: mean (M, G) f32, var (M, G) f32
# Grid: (M, G), each block reduces Cg elements

@avelang.jit
def groupnorm_stats_kernel(
    x_ptr: al.Pointer(al.bf16),
    mean_ptr: al.Pointer(al.f32),
    var_ptr: al.Pointer(al.f32),
    M: al.i32,
    G: al.i32,
    Cg: al.i32,
    C: al.i32,
):
    row = al.block_id(0)
    gp  = al.block_id(1)
    tid = al.thread_id(0)

    x_layout = al.make_layout((M, C), (C, 1))
    x = al.make_tensor(x_ptr, al.bf16, x_layout)
    m_layout = al.make_layout((M, G), (G, 1))
    m_out = al.make_tensor(mean_ptr, al.f32, m_layout)
    v_layout = al.make_layout((M, G), (G, 1))
    v_out = al.make_tensor(var_ptr, al.f32, v_layout)

    channel_base = gp * Cg

    partial_sum = al.convert(0.0, al.f32)
    partial_sq  = al.convert(0.0, al.f32)

    for cg in al.range(tid, Cg, 256):
        channel = channel_base + cg
        val = al.convert(x[row, channel], al.f32)
        partial_sum = partial_sum + val
        partial_sq  = partial_sq  + val * val

    s_sum = al.make_shared((256,), al.f32)
    s_sq  = al.make_shared((256,), al.f32)
    s_sum[tid] = partial_sum
    s_sq[tid]  = partial_sq
    al.syncthreads()

    if tid < 128:
        s_sum[tid] = s_sum[tid] + s_sum[tid + 128]
        s_sq[tid]  = s_sq[tid]  + s_sq[tid + 128]
    al.syncthreads()
    if tid < 64:
        s_sum[tid] = s_sum[tid] + s_sum[tid + 64]
        s_sq[tid]  = s_sq[tid]  + s_sq[tid + 64]
    al.syncthreads()
    if tid < 32:
        s_sum[tid] = s_sum[tid] + s_sum[tid + 32]
        s_sq[tid]  = s_sq[tid]  + s_sq[tid + 32]
    al.syncthreads()

    loc_sum = s_sum[tid]
    loc_sq  = s_sq[tid]

    v16 = al.shuffle_down(loc_sum, 16, 32)
    loc_sum = loc_sum + v16
    v16_sq = al.shuffle_down(loc_sq, 16, 32)
    loc_sq  = loc_sq + v16_sq

    v8 = al.shuffle_down(loc_sum, 8, 32)
    loc_sum = loc_sum + v8
    v8_sq = al.shuffle_down(loc_sq, 8, 32)
    loc_sq  = loc_sq + v8_sq

    v4 = al.shuffle_down(loc_sum, 4, 32)
    loc_sum = loc_sum + v4
    v4_sq = al.shuffle_down(loc_sq, 4, 32)
    loc_sq  = loc_sq + v4_sq

    v2 = al.shuffle_down(loc_sum, 2, 32)
    loc_sum = loc_sum + v2
    v2_sq = al.shuffle_down(loc_sq, 2, 32)
    loc_sq  = loc_sq + v2_sq

    v1 = al.shuffle_down(loc_sum, 1, 32)
    loc_sum = loc_sum + v1
    v1_sq = al.shuffle_down(loc_sq, 1, 32)
    loc_sq  = loc_sq + v1_sq

    if tid == 0:
        cg_f = al.convert(Cg, al.f32)
        mean_val = loc_sum / cg_f
        eps_val = al.convert(1e-5, al.f32)
        var_val  = loc_sq / cg_f - mean_val * mean_val + eps_val
        m_out[row, gp] = mean_val
        v_out[row, gp] = var_val


# ── GroupNorm apply + LeakyReLU + 2x ─────────────────────────────────────────
# Input: x (M, C) bf16, mean (M, G) f32, var (M, G) f32, gamma/beta (C,) bf16
# Output: y (M, C) bf16

@avelang.jit
def groupnorm_apply_kernel(
    x_ptr: al.Pointer(al.bf16),
    y_ptr: al.Pointer(al.bf16),
    mean_ptr: al.Pointer(al.f32),
    var_ptr: al.Pointer(al.f32),
    gamma_ptr: al.Pointer(al.bf16),
    beta_ptr: al.Pointer(al.bf16),
    M: al.i32,
    C: al.i32,
    G: al.i32,
    Cg: al.i32,
):
    tid = al.thread_id(0)
    gid_global = al.block_id(0) * 256 + tid
    total = M * C
    if gid_global >= total:
        return

    x_layout = al.make_layout((M, C), (C, 1))
    x = al.make_tensor(x_ptr, al.bf16, x_layout)
    y_layout = al.make_layout((M, C), (C, 1))
    y = al.make_tensor(y_ptr, al.bf16, y_layout)
    m_layout = al.make_layout((M, G), (G, 1))
    mean = al.make_tensor(mean_ptr, al.f32, m_layout)
    v_layout = al.make_layout((M, G), (G, 1))
    var = al.make_tensor(var_ptr, al.f32, v_layout)
    gamma_layout = al.make_layout((C,), (1,))
    gamma = al.make_tensor(gamma_ptr, al.bf16, gamma_layout)
    beta_layout = al.make_layout((C,), (1,))
    beta = al.make_tensor(beta_ptr, al.bf16, beta_layout)

    row = gid_global // C
    col = gid_global % C
    gp = col // Cg

    x_val = al.convert(x[row, col], al.f32)
    one = al.convert(1.0, al.f32)
    inv_std = one / al.sqrt(var[row, gp])
    norm_val = (x_val - mean[row, gp]) * inv_std
    aff_val = norm_val * al.convert(gamma[col], al.f32) + al.convert(beta[col], al.f32)

    zero = al.convert(0.0, al.f32)
    slope = al.convert(0.01, al.f32)
    lr_val = al.convert(0.0, al.f32)
    if aff_val > zero:
        lr_val = aff_val
    else:
        lr_val = aff_val * slope

    y[row, col] = al.convert(lr_val + lr_val, al.bf16)


# ── Host wrappers ────────────────────────────────────────────────────────────

def _launch_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    M, K_sz = a.shape
    K2, N = b.shape
    assert K_sz == K2

    out = torch.empty(M, N, dtype=torch.bfloat16, device=a.device)
    a = a.contiguous()
    b = b.contiguous()
    bias = bias.contiguous()

    grid_m = (M + 31) // 32
    grid_n = (N + 31) // 32

    matmul_bf16_kernel[lambda: ((grid_m, grid_n, 1), (256, 1, 1))](
        a.data_ptr(), b.data_ptr(), out.data_ptr(), bias.data_ptr(),
        M, N, K_sz,
    )
    return out


def _launch_groupnorm(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    num_groups: int,
) -> torch.Tensor:
    M, C = x.shape
    G = num_groups
    Cg = C // G

    x = x.contiguous()
    mean = torch.empty(M, G, dtype=torch.float32, device=x.device)
    var = torch.empty(M, G, dtype=torch.float32, device=x.device)

    groupnorm_stats_kernel[lambda: ((M, G, 1), (256, 1, 1))](
        x.data_ptr(), mean.data_ptr(), var.data_ptr(),
        M, G, Cg, C,
    )

    out = torch.empty(M, C, dtype=torch.bfloat16, device=x.device)
    gamma_bf16 = gamma.contiguous()
    beta_bf16 = beta.contiguous()

    total_elems = M * C
    grid = (total_elems + 255) // 256

    groupnorm_apply_kernel[lambda: ((grid, 1, 1), (256, 1, 1))](
        x.data_ptr(), out.data_ptr(),
        mean.data_ptr(), var.data_ptr(),
        gamma_bf16.data_ptr(), beta_bf16.data_ptr(),
        M, C, G, Cg,
    )
    return out


# ── ModelNew ─────────────────────────────────────────────────────────────────

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_groups, eps=1e-5, negative_slope=0.01):
        super(ModelNew, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=hidden_size, eps=eps)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_groups = num_groups
        self.eps = eps
        self.negative_slope = negative_slope

    def forward(self, x):
        weight = self.fc.weight.data.contiguous()
        b_mat = weight.T.contiguous()
        bias = self.fc.bias.data

        x1 = _launch_matmul(x, b_mat, bias)

        gamma = self.gn.weight.data
        beta = self.gn.bias.data

        return _launch_groupnorm(x1, gamma, beta, self.num_groups)
