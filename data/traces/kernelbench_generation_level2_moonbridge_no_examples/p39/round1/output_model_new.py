import torch
import torch.nn as nn
import avelang
import avelang.language as al

# ═══════════════════════════════════════════════════════════════════════
# Kernel 1 : tiled GEMM + bias + scale
#   BM=64  BN=64  BK=16  BDIM=256 (1D)
#   Each thread owns 4×4 sub-tile within a 64×64 block
# ═══════════════════════════════════════════════════════════════════════

@avelang.jit
def gemm_scale_bias_kernel(
    a_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.f32),
    scale_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.f32),
    M: al.i32,
    N: al.i32,
    K: al.i32,
):
    bid_m = al.block_id(1)
    bid_n = al.block_id(0)
    tid = al.thread_id(0)

    a_layout = al.make_layout((M, K), (K, 1))
    a = al.make_tensor(a_ptr, al.bf16, a_layout)
    b_layout = al.make_layout((N, K), (K, 1))
    b = al.make_tensor(b_ptr, al.bf16, b_layout)
    out_layout = al.make_layout((M, N), (N, 1))
    out = al.make_tensor(out_ptr, al.f32, out_layout)
    bias_layout = al.make_layout((N,), (1,))
    bias = al.make_tensor(bias_ptr, al.f32, bias_layout)
    scale_layout = al.make_layout((N,), (1,))
    scale = al.make_tensor(scale_ptr, al.f32, scale_layout)

    a_sh = al.make_shared((64, 16), al.bf16)
    b_sh = al.make_shared((64, 16), al.bf16)

    sixty4 = al.convert(64, al.i32)
    sixteen = al.convert(16, al.i32)
    four = al.convert(4, al.i32)
    two56 = al.convert(256, al.i32)

    m_begin = bid_m * sixty4
    n_begin = bid_n * sixty4

    local_m = (tid // sixteen) * four
    local_n = (tid - (tid // sixteen) * sixteen) * four

    zero_f32 = al.convert(0.0, al.f32)
    acc = al.make_local((4, 4), al.f32)
    for ii in al.range(0, 4):
        for jj in al.range(0, 4):
            acc[ii, jj] = zero_f32

    for k_block in al.range(0, K, 16):
        for idx in al.range(0, 4):
            gid = tid + idx * two56
            row_a = gid // sixteen
            col_a = gid - row_a * sixteen
            a_sh[row_a, col_a] = a[m_begin + row_a, k_block + col_a]

        for idx in al.range(0, 4):
            gid = tid + idx * two56
            row_b = gid // sixteen
            col_b = gid - row_b * sixteen
            b_sh[row_b, col_b] = b[n_begin + row_b, k_block + col_b]

        al.syncthreads()

        for k in al.range(0, 16):
            ak0 = al.convert(a_sh[local_m + 0, k], al.f32)
            ak1 = al.convert(a_sh[local_m + 1, k], al.f32)
            ak2 = al.convert(a_sh[local_m + 2, k], al.f32)
            ak3 = al.convert(a_sh[local_m + 3, k], al.f32)
            bk0 = al.convert(b_sh[local_n + 0, k], al.f32)
            bk1 = al.convert(b_sh[local_n + 1, k], al.f32)
            bk2 = al.convert(b_sh[local_n + 2, k], al.f32)
            bk3 = al.convert(b_sh[local_n + 3, k], al.f32)

            acc[0, 0] = acc[0, 0] + ak0 * bk0
            acc[0, 1] = acc[0, 1] + ak0 * bk1
            acc[0, 2] = acc[0, 2] + ak0 * bk2
            acc[0, 3] = acc[0, 3] + ak0 * bk3
            acc[1, 0] = acc[1, 0] + ak1 * bk0
            acc[1, 1] = acc[1, 1] + ak1 * bk1
            acc[1, 2] = acc[1, 2] + ak1 * bk2
            acc[1, 3] = acc[1, 3] + ak1 * bk3
            acc[2, 0] = acc[2, 0] + ak2 * bk0
            acc[2, 1] = acc[2, 1] + ak2 * bk1
            acc[2, 2] = acc[2, 2] + ak2 * bk2
            acc[2, 3] = acc[2, 3] + ak2 * bk3
            acc[3, 0] = acc[3, 0] + ak3 * bk0
            acc[3, 1] = acc[3, 1] + ak3 * bk1
            acc[3, 2] = acc[3, 2] + ak3 * bk2
            acc[3, 3] = acc[3, 3] + ak3 * bk3

        al.syncthreads()

    one_i32 = al.convert(1, al.i32)
    two_i32 = al.convert(2, al.i32)
    three_i32 = al.convert(3, al.i32)

    gm0 = m_begin + local_m
    gm1 = m_begin + local_m + one_i32
    gm2 = m_begin + local_m + two_i32
    gm3 = m_begin + local_m + three_i32
    gn0 = n_begin + local_n
    gn1 = n_begin + local_n + one_i32
    gn2 = n_begin + local_n + two_i32
    gn3 = n_begin + local_n + three_i32

    out[gm0, gn0] = (acc[0, 0] + bias[gn0]) * scale[gn0]
    out[gm0, gn1] = (acc[0, 1] + bias[gn1]) * scale[gn1]
    out[gm0, gn2] = (acc[0, 2] + bias[gn2]) * scale[gn2]
    out[gm0, gn3] = (acc[0, 3] + bias[gn3]) * scale[gn3]
    out[gm1, gn0] = (acc[1, 0] + bias[gn0]) * scale[gn0]
    out[gm1, gn1] = (acc[1, 1] + bias[gn1]) * scale[gn1]
    out[gm1, gn2] = (acc[1, 2] + bias[gn2]) * scale[gn2]
    out[gm1, gn3] = (acc[1, 3] + bias[gn3]) * scale[gn3]
    out[gm2, gn0] = (acc[2, 0] + bias[gn0]) * scale[gn0]
    out[gm2, gn1] = (acc[2, 1] + bias[gn1]) * scale[gn1]
    out[gm2, gn2] = (acc[2, 2] + bias[gn2]) * scale[gn2]
    out[gm2, gn3] = (acc[2, 3] + bias[gn3]) * scale[gn3]
    out[gm3, gn0] = (acc[3, 0] + bias[gn0]) * scale[gn0]
    out[gm3, gn1] = (acc[3, 1] + bias[gn1]) * scale[gn1]
    out[gm3, gn2] = (acc[3, 2] + bias[gn2]) * scale[gn2]
    out[gm3, gn3] = (acc[3, 3] + bias[gn3]) * scale[gn3]


# ═══════════════════════════════════════════════════════════════════════
# Kernel 2 : batchnorm eval mode
# ═══════════════════════════════════════════════════════════════════════

@avelang.jit
def batchnorm_eval_kernel(
    x_ptr: al.Pointer(al.f32),
    running_mean_ptr: al.Pointer(al.f32),
    running_var_ptr: al.Pointer(al.f32),
    gamma_ptr: al.Pointer(al.f32),
    beta_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.f32),
    M: al.i32,
    N: al.i32,
):
    tid = al.thread_id(0)
    cid = al.block_id(0)
    BDIM = al.convert(256, al.i32)

    x_layout = al.make_layout((M, N), (N, 1))
    x = al.make_tensor(x_ptr, al.f32, x_layout)
    out_layout = al.make_layout((M, N), (N, 1))
    out = al.make_tensor(out_ptr, al.f32, out_layout)

    ch_layout = al.make_layout((N,), (1,))
    rmean = al.make_tensor(running_mean_ptr, al.f32, ch_layout)
    rvar = al.make_tensor(running_var_ptr, al.f32, ch_layout)
    gamma = al.make_tensor(gamma_ptr, al.f32, ch_layout)
    beta = al.make_tensor(beta_ptr, al.f32, ch_layout)

    ch_rmean = rmean[cid]
    ch_rvar = rvar[cid]
    ch_gamma = gamma[cid]
    ch_beta = beta[cid]

    one = al.convert(1.0, al.f32)
    eps = al.convert(1e-5, al.f32)
    inv_std = one / al.sqrt(ch_rvar + eps)

    for i in al.range(tid, M, BDIM):
        val = x[i, cid]
        out[i, cid] = (val - ch_rmean) * inv_std * ch_gamma + ch_beta


# ═══════════════════════════════════════════════════════════════════════
# Host wrapper
# ═══════════════════════════════════════════════════════════════════════

_BDIM = 256
_NREDUCE = 256


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)

    def forward(self, x):
        w = self.gemm.weight.data
        bias_w = self.gemm.bias.data
        s = self.scale.data
        gamma = self.bn.weight.data
        beta = self.bn.bias.data
        running_mean = self.bn.running_mean
        running_var = self.bn.running_var

        M_int = x.shape[0]
        N_int = w.shape[0]
        K_int = w.shape[1]

        x_bf16 = x.contiguous()
        w_bf16 = w.contiguous()
        bias_f32 = bias_w.to(torch.float32).contiguous()
        s_f32 = s.to(torch.float32).contiguous()
        rmean_f32 = running_mean.to(torch.float32).contiguous()
        rvar_f32 = running_var.to(torch.float32).contiguous()
        gamma_f32 = gamma.to(torch.float32).contiguous()
        beta_f32 = beta.to(torch.float32).contiguous()

        device = x.device

        gemm_out = torch.empty((M_int, N_int), dtype=torch.float32, device=device)
        grid_m = (M_int + 63) // 64
        grid_n = (N_int + 63) // 64
        gemm_scale_bias_kernel[lambda: ((grid_n, grid_m, 1), (_BDIM, 1, 1))](
            x_bf16, w_bf16, bias_f32, s_f32, gemm_out, M_int, N_int, K_int,
        )

        out_f32 = torch.empty((M_int, N_int), dtype=torch.float32, device=device)
        batchnorm_eval_kernel[lambda: ((N_int, 1, 1), (_NREDUCE, 1, 1))](
            gemm_out, rmean_f32, rvar_f32, gamma_f32, beta_f32, out_f32,
            M_int, N_int,
        )

        return out_f32.to(torch.bfloat16)
