import torch
import torch.nn as nn
import avelang
import avelang.language as al


# ============================================================
# Kernel 1: Tiled MatMul + Swish activation + bias addition
# ============================================================

@avelang.jit
def fused_matmul_swish_bias_kernel(
    a_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    bias1_ptr: al.Pointer(al.bf16),
    bias2_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    N: al.i32,
    K: al.i32,
    M: al.i32,
    BLOCK_M: al.constexpr,
    BLOCK_N: al.constexpr,
    BLOCK_K: al.constexpr,
    BM_T: al.constexpr,
    BN_T: al.constexpr,
    TM: al.constexpr,
    TN: al.constexpr,
    BM_PER_THREAD: al.constexpr,
    BN_PER_THREAD: al.constexpr,
    BK_PER_THREAD: al.constexpr,
    BK_PER_THREAD_M: al.constexpr,
):
    a = al.make_tensor(a_ptr, al.bf16, al.make_layout((N, K), (K, 1)))
    b = al.make_tensor(b_ptr, al.bf16, al.make_layout((M, K), (K, 1)))
    out = al.make_tensor(out_ptr, al.bf16, al.make_layout((N, M), (M, 1)))
    bias1 = al.make_tensor(bias1_ptr, al.bf16, al.make_layout((M,), (1,)))
    bias2 = al.make_tensor(bias2_ptr, al.bf16, al.make_layout((M,), (1,)))

    block_m = al.block_id(0)
    block_n = al.block_id(1)
    tid_n = al.thread_id(0)
    tid_m = al.thread_id(1)

    m_start = block_m * BLOCK_M
    n_start = block_n * BLOCK_N

    As = al.make_shared((BLOCK_M, BLOCK_K), al.bf16)
    Bs = al.make_shared((BLOCK_N, BLOCK_K), al.bf16)

    acc = al.make_local((TM, TN), al.f32)
    zero = al.convert(0.0, al.f32)

    for tm in al.range(TM):
        for tn in al.range(TN):
            acc[tm, tn] = zero

    for k_start in al.range(al.convert(0, al.i32), K, BLOCK_K):
        for tm in al.range(BM_PER_THREAD):
            m_idx = tid_m * BM_PER_THREAD + tm
            for tk in al.range(BK_PER_THREAD):
                k_idx = tid_n * BK_PER_THREAD + tk
                gbl_m = m_start + m_idx
                gbl_k = k_start + k_idx
                if gbl_m < N and gbl_k < K:
                    As[m_idx, k_idx] = a[gbl_m, gbl_k]
                else:
                    As[m_idx, k_idx] = al.convert(0.0, al.bf16)

        for tn in al.range(BN_PER_THREAD):
            n_idx = tid_n * BN_PER_THREAD + tn
            for tk in al.range(BK_PER_THREAD_M):
                k_idx = tid_m * BK_PER_THREAD_M + tk
                gbl_n = n_start + n_idx
                gbl_k = k_start + k_idx
                if gbl_n < M and gbl_k < K:
                    Bs[n_idx, k_idx] = b[gbl_n, gbl_k]
                else:
                    Bs[n_idx, k_idx] = al.convert(0.0, al.bf16)

        al.syncthreads()

        for kk in al.range(BLOCK_K):
            for tm in al.range(TM):
                m = tid_m * TM + tm
                a_val = al.convert(As[m, kk], al.f32)
                for tn in al.range(TN):
                    n = tid_n * TN + tn
                    b_val = al.convert(Bs[n, kk], al.f32)
                    acc[tm, tn] = acc[tm, tn] + a_val * b_val

        al.syncthreads()

    one = al.convert(1.0, al.f32)

    for tm in al.range(TM):
        m = tid_m * TM + tm
        gbl_m = m_start + m
        for tn in al.range(TN):
            n = tid_n * TN + tn
            gbl_n = n_start + n

            if gbl_m < N and gbl_n < M:
                val = acc[tm, tn]
                val = val + al.convert(bias1[gbl_n], al.f32)
                val = al.convert(al.convert(val, al.bf16), al.f32)

                sigmoid_val = one / (one + al.exp(-val))
                sigmoid_val = al.convert(al.convert(sigmoid_val, al.bf16), al.f32)
                swish_val = val * sigmoid_val
                swish_val = al.convert(al.convert(swish_val, al.bf16), al.f32)

                val = swish_val + al.convert(bias2[gbl_n], al.f32)
                out[gbl_m, gbl_n] = al.convert(val, al.bf16)


# ============================================================
# Kernel 2: GroupNorm with shared-memory tree reduction
# ============================================================

@avelang.jit
def group_norm_kernel(
    inp_ptr: al.Pointer(al.bf16),
    gamma_ptr: al.Pointer(al.bf16),
    beta_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C: al.constexpr,
    channels_per_group: al.constexpr,
    num_groups: al.constexpr,
    groups_per_warp: al.constexpr,
):
    inp = al.make_tensor(inp_ptr, al.bf16, al.make_layout((N, C), (C, 1)))
    gamma = al.make_tensor(gamma_ptr, al.bf16, al.make_layout((C,), (1,)))
    beta = al.make_tensor(beta_ptr, al.bf16, al.make_layout((C,), (1,)))
    out = al.make_tensor(out_ptr, al.bf16, al.make_layout((N, C), (C, 1)))

    batch_idx = al.block_id(0)
    lane = al.thread_id(0)
    warp = al.thread_id(1)
    group_start = warp * groups_per_warp

    one = al.convert(1.0, al.f32)
    eps_val = al.convert(1e-5, al.f32)
    f_count = al.convert(al.convert(channels_per_group, al.i32), al.f64)

    for g in al.range(groups_per_warp):
        group_idx = group_start + g
        chan = group_idx * channels_per_group + lane

        val = al.convert(inp[batch_idx, chan], al.f32)

        # Warp reduction with FP64 accumulation for sum
        s = al.convert(val, al.f64)
        s = s + al.shuffle_down(s, 32, 64)
        s = s + al.shuffle_down(s, 16, 64)
        s = s + al.shuffle_down(s, 8, 64)
        s = s + al.shuffle_down(s, 4, 64)
        s = s + al.shuffle_down(s, 2, 64)
        s = s + al.shuffle_down(s, 1, 64)
        total = al.shuffle(s, 0, 64)
        mean_f64 = total / f_count
        mean = al.convert(mean_f64, al.f32)

        # Variance with FP64 accumulation
        diff = val - mean
        diff_sq = diff * diff
        v = al.convert(diff_sq, al.f64)
        v = v + al.shuffle_down(v, 32, 64)
        v = v + al.shuffle_down(v, 16, 64)
        v = v + al.shuffle_down(v, 8, 64)
        v = v + al.shuffle_down(v, 4, 64)
        v = v + al.shuffle_down(v, 2, 64)
        v = v + al.shuffle_down(v, 1, 64)
        var_total = al.shuffle(v, 0, 64)
        var_f64 = var_total / f_count
        var_val = al.convert(var_f64, al.f32)

        # Normalize and affine
        inv_std = one / al.sqrt(var_val + eps_val)
        norm_val = (val - mean) * inv_std

        gamma_val = al.convert(gamma[chan], al.f32)
        beta_val = al.convert(beta[chan], al.f32)
        result = norm_val * gamma_val + beta_val

        out[batch_idx, chan] = al.convert(result, al.bf16)


# ============================================================
# Host wrappers
# ============================================================

def avelang_fused_matmul_swish_bias(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias1: torch.Tensor,
    bias2: torch.Tensor,
) -> torch.Tensor:
    N, K = x.shape
    M = weight.shape[0]
    out = torch.empty(N, M, dtype=torch.bfloat16, device=x.device)

    BLOCK_M = 64
    BLOCK_N = 32
    BLOCK_K = 32
    BM_T = 8
    BN_T = 16
    TM = BLOCK_M // BM_T
    TN = BLOCK_N // BN_T
    BM_PER_THREAD = BLOCK_M // BM_T
    BN_PER_THREAD = BLOCK_N // BN_T
    BK_PER_THREAD = BLOCK_K // BN_T
    BK_PER_THREAD_M = BLOCK_K // BM_T

    grid = (
        (N + BLOCK_M - 1) // BLOCK_M,
        (M + BLOCK_N - 1) // BLOCK_N,
        1,
    )
    block = (BN_T, BM_T, 1)

    fused_matmul_swish_bias_kernel[lambda: (grid, block)](
        x.data_ptr(),
        weight.data_ptr(),
        bias1.data_ptr(),
        bias2.data_ptr(),
        out.data_ptr(),
        N, K, M,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        BM_T=BM_T,
        BN_T=BN_T,
        TM=TM,
        TN=TN,
        BM_PER_THREAD=BM_PER_THREAD,
        BN_PER_THREAD=BN_PER_THREAD,
        BK_PER_THREAD=BK_PER_THREAD,
        BK_PER_THREAD_M=BK_PER_THREAD_M,
    )

    return out


def avelang_group_norm(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    num_groups: int,
) -> torch.Tensor:
    N, C = x.shape
    out = torch.empty(N, C, dtype=torch.bfloat16, device=x.device)

    channels_per_group = C // num_groups
    groups_per_warp = num_groups // 4

    grid = (N, 1, 1)
    block = (64, 4, 1)

    group_norm_kernel[lambda: (grid, block)](
        x.data_ptr(),
        gamma.data_ptr(),
        beta.data_ptr(),
        out.data_ptr(),
        N,
        C=C,
        channels_per_group=channels_per_group,
        num_groups=num_groups,
        groups_per_warp=groups_per_warp,
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups

        self.matmul = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)

    def forward(self, x):
        orig_dtype = x.dtype

        if x.dtype != torch.bfloat16:
            x_bf16 = x.to(torch.bfloat16).contiguous()
        else:
            x_bf16 = x.contiguous()

        weight_bf16 = self.matmul.weight.data.to(torch.bfloat16).contiguous()
        bias1_bf16 = self.matmul.bias.data.to(torch.bfloat16).contiguous()
        bias2_bf16 = self.bias.data.to(torch.bfloat16).contiguous()

        mid_bf16 = avelang_fused_matmul_swish_bias(
            x_bf16, weight_bf16, bias1_bf16, bias2_bf16
        )

        gamma_bf16 = self.group_norm.weight.data.to(torch.bfloat16).contiguous()
        beta_bf16 = self.group_norm.bias.data.to(torch.bfloat16).contiguous()

        out_bf16 = avelang_group_norm(
            mid_bf16, gamma_bf16, beta_bf16, self.num_groups
        )

        return out_bf16.to(orig_dtype)


batch_size = 32768
in_features = 1024
out_features = 4096
num_groups = 64
bias_shape = (out_features,)


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, num_groups, bias_shape]
