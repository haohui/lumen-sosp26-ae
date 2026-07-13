import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 128
THREAD_M = 16
THREAD_N = 16
TM = BLOCK_M // THREAD_M
TN = BLOCK_N // THREAD_N


@avelang.jit
def fused_matmul_swish_scale_kernel(
    a_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.f32),
    c_ptr: al.Pointer(al.bf16),
    M: al.i32,
    N: al.i32,
    K: al.i32,
    stride_am: al.i32,
    stride_ak: al.i32,
    stride_bn: al.i32,
    stride_bk: al.i32,
    stride_cm: al.i32,
    stride_cn: al.i32,
    SCALE: al.constexpr,
):
    block_m = al.block_id(0)
    block_n = al.block_id(1)
    tid_m = al.thread_id(0)
    tid_n = al.thread_id(1)

    lay_a = al.make_layout((M, K), (stride_am, stride_ak))
    a = al.make_tensor(a_ptr, al.bf16, lay_a)
    lay_b = al.make_layout((N, K), (stride_bn, stride_bk))
    b = al.make_tensor(b_ptr, al.bf16, lay_b)
    lay_c = al.make_layout((M, N), (stride_cm, stride_cn))
    c = al.make_tensor(c_ptr, al.bf16, lay_c)
    lay_bias = al.make_layout((N,), (1,))
    bias = al.make_tensor(bias_ptr, al.f32, lay_bias)

    a_sh = al.make_shared((BLOCK_M, BLOCK_K), al.bf16)
    b_sh = al.make_shared((BLOCK_N, BLOCK_K), al.bf16)

    acc = al.make_local((TM, TN), al.f32)
    for i in al.range(TM):
        for j in al.range(TN):
            acc[i, j] = al.convert(0.0, al.f32)

    m_base = block_m * BLOCK_M
    n_base = block_n * BLOCK_N

    n_loads = (BLOCK_M * BLOCK_K) // (THREAD_M * THREAD_N)
    scale_f32 = al.convert(SCALE, al.f32)

    for kb in al.range(0, K, BLOCK_K):
        for li in al.range(n_loads):
            idx = (tid_m * THREAD_N + tid_n) * n_loads + li
            lm = idx // BLOCK_K
            lk = idx % BLOCK_K
            gm = m_base + lm
            gk = kb + lk
            if gm < M and gk < K:
                a_sh[lm, lk] = a[gm, gk]

        for li in al.range(n_loads):
            idx = (tid_m * THREAD_N + tid_n) * n_loads + li
            ln = idx // BLOCK_K
            lk = idx % BLOCK_K
            gn = n_base + ln
            gk = kb + lk
            if gn < N and gk < K:
                b_sh[ln, lk] = b[gn, gk]

        al.syncthreads()

        for k in al.range(BLOCK_K):
            gk = kb + k
            if gk < K:
                for i in al.range(TM):
                    av = al.convert(a_sh[tid_m * TM + i, k], al.f32)
                    for j in al.range(TN):
                        bv = al.convert(b_sh[tid_n * TN + j, k], al.f32)
                        acc[i, j] = acc[i, j] + av * bv

        al.syncthreads()

    one = al.convert(1.0, al.f32)
    zero = al.convert(0.0, al.f32)

    for i in al.range(TM):
        gm = m_base + tid_m * TM + i
        for j in al.range(TN):
            gn = n_base + tid_n * TN + j
            if gm < M and gn < N:
                result = acc[i, j] + bias[gn]
                neg_result = zero - result
                exp_val = al.exp(neg_result)
                sigmoid_val = one / (one + exp_val)
                result = result * sigmoid_val * scale_f32
                c[gm, gn] = al.convert(result, al.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        weight = self.matmul.weight
        bias = self.matmul.bias
        scale = self.scaling_factor

        M, K = x.shape
        N = weight.shape[0]

        x = x.contiguous()
        weight = weight.contiguous()

        x_bf16 = x.to(torch.bfloat16)
        w_bf16 = weight.to(torch.bfloat16)
        bias_f32 = bias.to(torch.float32)

        output = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)

        grid_m = (M + BLOCK_M - 1) // BLOCK_M
        grid_n = (N + BLOCK_N - 1) // BLOCK_N

        fused_matmul_swish_scale_kernel[lambda: ((grid_m, grid_n, 1), (THREAD_M, THREAD_N, 1))](
            x_bf16.data_ptr(),
            w_bf16.data_ptr(),
            bias_f32.data_ptr(),
            output.data_ptr(),
            M,
            N,
            K,
            x_bf16.stride(0),
            x_bf16.stride(1),
            w_bf16.stride(0),
            w_bf16.stride(1),
            output.stride(0),
            output.stride(1),
            scale,
        )

        return output
