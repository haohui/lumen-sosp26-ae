import math
import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def fused_matmul_swish_bias_kernel(
    a_ptr: al.Pointer(al.f32),
    b_ptr: al.Pointer(al.f32),
    lin_bias_ptr: al.Pointer(al.f32),
    extra_bias_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.f32),
    M: al.i32,
    N: al.i32,
    K: al.i32,
):
    block_m = al.block_id(0)
    block_n = al.block_id(1)
    tx = al.thread_id(0)
    ty = al.thread_id(1)

    a_layout = al.make_layout((M, K), (K, 1))
    a = al.make_tensor(a_ptr, al.f32, a_layout)
    b_layout = al.make_layout((K, N), (N, 1))
    b = al.make_tensor(b_ptr, al.f32, b_layout)
    bias_layout = al.make_layout((N,), (1,))
    lb = al.make_tensor(lin_bias_ptr, al.f32, bias_layout)
    eb = al.make_tensor(extra_bias_ptr, al.f32, bias_layout)
    out_layout = al.make_layout((M, N), (N, 1))
    out = al.make_tensor(out_ptr, al.f32, out_layout)

    a_smem = al.make_shared((32, 32), al.f32)
    b_smem = al.make_shared((32, 32), al.f32)

    m_start = block_m * 32
    n_start = block_n * 32
    r0 = ty * 2
    r1 = ty * 2 + 1
    c0 = tx * 2
    c1 = tx * 2 + 1

    zero = al.convert(0.0, al.f32)
    acc00 = zero
    acc01 = zero
    acc10 = zero
    acc11 = zero

    for kk in al.range(0, K, 32):
        a_smem[r0, c0] = a[m_start + r0, kk + c0]
        a_smem[r0, c1] = a[m_start + r0, kk + c1]
        a_smem[r1, c0] = a[m_start + r1, kk + c0]
        a_smem[r1, c1] = a[m_start + r1, kk + c1]

        b_smem[r0, c0] = b[kk + r0, n_start + c0]
        b_smem[r0, c1] = b[kk + r0, n_start + c1]
        b_smem[r1, c0] = b[kk + r1, n_start + c0]
        b_smem[r1, c1] = b[kk + r1, n_start + c1]

        al.syncthreads()

        for k in al.range(32):
            a0 = a_smem[r0, k]
            a1 = a_smem[r1, k]
            b0 = b_smem[k, c0]
            b1 = b_smem[k, c1]
            acc00 = acc00 + a0 * b0
            acc01 = acc01 + a0 * b1
            acc10 = acc10 + a1 * b0
            acc11 = acc11 + a1 * b1

        al.syncthreads()

    n0 = n_start + c0
    n1 = n_start + c1

    acc00 = acc00 + lb[n0]
    acc01 = acc01 + lb[n1]
    acc10 = acc10 + lb[n0]
    acc11 = acc11 + lb[n1]

    # Swish activation: x * sigmoid(x) via tanh
    half = al.convert(0.5, al.f32)
    one = al.convert(1.0, al.f32)
    two = al.convert(2.0, al.f32)

    t00 = al.tanh(acc00 / two)
    t01 = al.tanh(acc01 / two)
    t10 = al.tanh(acc10 / two)
    t11 = al.tanh(acc11 / two)

    swish00 = acc00 * half * (t00 + one)
    swish01 = acc01 * half * (t01 + one)
    swish10 = acc10 * half * (t10 + one)
    swish11 = acc11 * half * (t11 + one)

    swish00 = swish00 + eb[n0]
    swish01 = swish01 + eb[n1]
    swish10 = swish10 + eb[n0]
    swish11 = swish11 + eb[n1]

    out[m_start + r0, n0] = swish00
    out[m_start + r0, n1] = swish01
    out[m_start + r1, n0] = swish10
    out[m_start + r1, n1] = swish11


@avelang.jit
def group_norm_kernel(
    in_ptr: al.Pointer(al.f32),
    gamma_ptr: al.Pointer(al.f32),
    beta_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.f32),
    M: al.i32,
    C: al.i32,
    G: al.i32,
):
    row = al.block_id(0)
    group = al.block_id(1)
    tid = al.thread_id(0)

    Cg = C // G
    c_start = group * Cg
    c_idx = c_start + tid

    io_layout = al.make_layout((M, C), (C, 1))
    inp = al.make_tensor(in_ptr, al.f32, io_layout)
    out = al.make_tensor(out_ptr, al.f32, io_layout)
    w_layout = al.make_layout((C,), (1,))
    gamma = al.make_tensor(gamma_ptr, al.f32, w_layout)
    beta = al.make_tensor(beta_ptr, al.f32, w_layout)

    x = inp[row, c_idx]

    s = x
    s = s + al.shuffle_down(s, 32, 64)
    s = s + al.shuffle_down(s, 16, 64)
    s = s + al.shuffle_down(s, 8, 64)
    s = s + al.shuffle_down(s, 4, 64)
    s = s + al.shuffle_down(s, 2, 64)
    s = s + al.shuffle_down(s, 1, 64)
    total = al.shuffle(s, 0, 64)
    mean = total / al.convert(64.0, al.f32)

    diff = x - mean
    diff_sq = diff * diff
    v = diff_sq
    v = v + al.shuffle_down(v, 32, 64)
    v = v + al.shuffle_down(v, 16, 64)
    v = v + al.shuffle_down(v, 8, 64)
    v = v + al.shuffle_down(v, 4, 64)
    v = v + al.shuffle_down(v, 2, 64)
    v = v + al.shuffle_down(v, 1, 64)
    var_total = al.shuffle(v, 0, 64)
    var = var_total / al.convert(64.0, al.f32)

    eps_val = al.convert(0.00001, al.f32)
    std = al.sqrt(var + eps_val)
    inv_std = al.convert(1.0, al.f32) / std
    y = diff * inv_std

    result = y * gamma[c_idx] + beta[c_idx]

    out[row, c_idx] = result


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)

    def forward(self, x):
        M = x.shape[0]
        K = self.matmul.in_features
        N = self.matmul.out_features
        G = self.group_norm.num_groups

        x_f32 = x.contiguous().to(torch.float32)
        w_f32 = self.matmul.weight.data.T.contiguous().to(torch.float32)
        lin_b_f32 = self.matmul.bias.data.contiguous().to(torch.float32)
        extra_b_f32 = self.bias.data.contiguous().to(torch.float32)
        gn_w_f32 = self.group_norm.weight.data.contiguous().to(torch.float32)
        gn_b_f32 = self.group_norm.bias.data.contiguous().to(torch.float32)

        mid = torch.empty(M, N, dtype=torch.float32, device=x.device)

        grid_m = M // 32
        grid_n = N // 32
        fused_matmul_swish_bias_kernel[lambda: ((grid_m, grid_n, 1), (16, 16, 1))](
            x_f32, w_f32, lin_b_f32, extra_b_f32, mid,
            M, N, K,
        )

        out = torch.empty(M, N, dtype=torch.float32, device=x.device)

        group_norm_kernel[lambda: ((M, G, 1), (64, 1, 1))](
            mid, gn_w_f32, gn_b_f32, out,
            M, N, G,
        )

        return out.to(torch.bfloat16)
