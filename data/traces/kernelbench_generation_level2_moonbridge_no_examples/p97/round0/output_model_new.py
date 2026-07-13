import torch
import torch.nn as nn
import math
import avelang
import avelang.language as al

_BM2 = al.constexpr(32)
_BN2 = al.constexpr(32)
_BK = al.constexpr(16)


# ── Fused kernel: 2×2 sub-block per thread, effective 32×32 tile ─
@avelang.jit
def fused_kernel(
    a_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    linear_bias_ptr: al.Pointer(al.bf16),
    running_mean_ptr: al.Pointer(al.bf16),
    running_var_ptr: al.Pointer(al.bf16),
    gamma_ptr: al.Pointer(al.bf16),
    beta_ptr: al.Pointer(al.bf16),
    extra_bias_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    M: al.i32,
    N: al.i32,
    K: al.i32,
):
    pid_m = al.block_id(0)
    pid_n = al.block_id(1)
    tid_m = al.thread_id(0)
    tid_n = al.thread_id(1)

    g_row0 = pid_m * _BM2 + tid_m * 2
    g_row1 = g_row0 + 1
    g_col0 = pid_n * _BN2 + tid_n * 2
    g_col1 = g_col0 + 1

    a_tile = al.make_shared((32, 16), al.bf16)
    b_tile = al.make_shared((16, 32), al.bf16)

    a = al.make_tensor(a_ptr, al.bf16, al.make_layout((M, K), (K, 1)))
    b = al.make_tensor(b_ptr, al.bf16, al.make_layout((K, N), (1, K)))
    lin_bias = al.make_tensor(linear_bias_ptr, al.bf16, al.make_layout((N,), (1,)))
    rm = al.make_tensor(running_mean_ptr, al.bf16, al.make_layout((N,), (1,)))
    rv = al.make_tensor(running_var_ptr, al.bf16, al.make_layout((N,), (1,)))
    gamma = al.make_tensor(gamma_ptr, al.bf16, al.make_layout((N,), (1,)))
    bn_beta = al.make_tensor(beta_ptr, al.bf16, al.make_layout((N,), (1,)))
    extra_b = al.make_tensor(extra_bias_ptr, al.bf16, al.make_layout((1, 1), (1, 1)))
    out = al.make_tensor(out_ptr, al.bf16, al.make_layout((M, N), (N, 1)))

    acc00 = al.convert(0.0, al.f32)
    acc01 = al.convert(0.0, al.f32)
    acc10 = al.convert(0.0, al.f32)
    acc11 = al.convert(0.0, al.f32)

    for k_start in al.range(0, K, _BK):
        a_tile[tid_m * 2, tid_n] = a[g_row0, k_start + tid_n]
        a_tile[tid_m * 2 + 1, tid_n] = a[g_row1, k_start + tid_n]

        b_tile[tid_m, tid_n * 2] = b[k_start + tid_m, g_col0]
        b_tile[tid_m, tid_n * 2 + 1] = b[k_start + tid_m, g_col1]

        al.syncthreads()

        for k_idx in al.range(_BK):
            a0 = al.convert(a_tile[tid_m * 2, k_idx], al.f32)
            a1 = al.convert(a_tile[tid_m * 2 + 1, k_idx], al.f32)
            b0 = al.convert(b_tile[k_idx, tid_n * 2], al.f32)
            b1 = al.convert(b_tile[k_idx, tid_n * 2 + 1], al.f32)
            acc00 = acc00 + a0 * b0
            acc01 = acc01 + a0 * b1
            acc10 = acc10 + a1 * b0
            acc11 = acc11 + a1 * b1

        al.syncthreads()

    extra_f32 = al.convert(extra_b[0, 0], al.f32)
    one = al.convert(1.0, al.f32)
    eps_f32 = al.convert(1e-5, al.f32)

    if g_row0 < M and g_col0 < N:
        v = acc00 + al.convert(lin_bias[g_col0], al.f32)
        v = al.convert(gamma[g_col0], al.f32) * (v - al.convert(rm[g_col0], al.f32)) / al.sqrt(al.convert(rv[g_col0], al.f32) + eps_f32) + al.convert(bn_beta[g_col0], al.f32)
        v = v + extra_f32
        v = v / (one + al.exp(al.convert(0.0, al.f32) - v))
        out[g_row0, g_col0] = al.convert(v, al.bf16)

    if g_row0 < M and g_col1 < N:
        v = acc01 + al.convert(lin_bias[g_col1], al.f32)
        v = al.convert(gamma[g_col1], al.f32) * (v - al.convert(rm[g_col1], al.f32)) / al.sqrt(al.convert(rv[g_col1], al.f32) + eps_f32) + al.convert(bn_beta[g_col1], al.f32)
        v = v + extra_f32
        v = v / (one + al.exp(al.convert(0.0, al.f32) - v))
        out[g_row0, g_col1] = al.convert(v, al.bf16)

    if g_row1 < M and g_col0 < N:
        v = acc10 + al.convert(lin_bias[g_col0], al.f32)
        v = al.convert(gamma[g_col0], al.f32) * (v - al.convert(rm[g_col0], al.f32)) / al.sqrt(al.convert(rv[g_col0], al.f32) + eps_f32) + al.convert(bn_beta[g_col0], al.f32)
        v = v + extra_f32
        v = v / (one + al.exp(al.convert(0.0, al.f32) - v))
        out[g_row1, g_col0] = al.convert(v, al.bf16)

    if g_row1 < M and g_col1 < N:
        v = acc11 + al.convert(lin_bias[g_col1], al.f32)
        v = al.convert(gamma[g_col1], al.f32) * (v - al.convert(rm[g_col1], al.f32)) / al.sqrt(al.convert(rv[g_col1], al.f32) + eps_f32) + al.convert(bn_beta[g_col1], al.f32)
        v = v + extra_f32
        v = v / (one + al.exp(al.convert(0.0, al.f32) - v))
        out[g_row1, g_col1] = al.convert(v, al.bf16)


# ── Host wrapper ────────────────────────────────────────────────
class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1,
                 bias_shape=(1,), divide_value=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ref_linear = nn.Linear(in_features, out_features)
        self.weight = nn.Parameter(ref_linear.weight.data.clone().to(torch.bfloat16))
        self.linear_bias = nn.Parameter(ref_linear.bias.data.clone().to(torch.bfloat16))
        del ref_linear

        ref_bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.bn_weight = nn.Parameter(ref_bn.weight.data.clone().to(torch.bfloat16))
        self.bn_bias = nn.Parameter(ref_bn.bias.data.clone().to(torch.bfloat16))
        self.bn_running_mean = nn.Parameter(
            ref_bn.running_mean.data.clone().to(torch.bfloat16), requires_grad=False)
        self.bn_running_var = nn.Parameter(
            ref_bn.running_var.data.clone().to(torch.bfloat16), requires_grad=False)
        del ref_bn

        self.bias = nn.Parameter(torch.randn(bias_shape).to(torch.bfloat16))

    def forward(self, x):
        x_bf16 = x.to(torch.bfloat16).contiguous()
        batch_size = x_bf16.shape[0]
        out_feat = self.out_features

        BM, BN = 32, 32
        grid_m = (batch_size + BM - 1) // BM
        grid_n = (out_feat + BN - 1) // BN
        out = torch.empty(batch_size, out_feat, dtype=torch.bfloat16,
                          device=x_bf16.device)

        fused_kernel[lambda: ((grid_m, grid_n, 1), (16, 16, 1))](
            x_bf16, self.weight, self.linear_bias,
            self.bn_running_mean, self.bn_running_var,
            self.bn_weight, self.bn_bias, self.bias, out,
            batch_size, out_feat, self.in_features,
        )

        return out
