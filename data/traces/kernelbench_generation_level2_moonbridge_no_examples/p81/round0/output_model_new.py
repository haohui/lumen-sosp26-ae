import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def gemm_fused_kernel(
    A_ptr: al.Pointer(al.bf16),
    B_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    C_ptr: al.Pointer(al.bf16),
    M: al.i64,
    N: al.i64,
    K: al.i64,
):
    pid_m = al.block_id(0)
    pid_n = al.block_id(1)
    tid_m = al.thread_id(0)
    tid_n = al.thread_id(1)

    m_base = pid_m * 32
    n_base = pid_n * 32

    A = al.make_tensor(A_ptr, al.bf16, al.make_layout((M, K), (K, 1)))
    B = al.make_tensor(B_ptr, al.bf16, al.make_layout((N, K), (K, 1)))
    C = al.make_tensor(C_ptr, al.bf16, al.make_layout((M, N), (N, 1)))
    bias_buf = al.make_tensor(bias_ptr, al.bf16, al.make_layout((N,), (1,)))

    A_tile = al.make_shared((32, 32), al.bf16)
    B_tile = al.make_shared((32, 32), al.bf16)

    acc00 = al.convert(0.0, al.f32)
    acc01 = al.convert(0.0, al.f32)
    acc10 = al.convert(0.0, al.f32)
    acc11 = al.convert(0.0, al.f32)

    zero_bf16 = al.convert(0.0, al.bf16)

    for k_tile in al.range(0, K, 32):
        row0 = m_base + tid_m * 2
        row1 = row0 + 1
        col0 = k_tile + tid_n * 2
        col1 = col0 + 1

        if row0 < M:
            if col0 < K:
                A_tile[tid_m * 2, tid_n * 2] = A[row0, col0]
            else:
                A_tile[tid_m * 2, tid_n * 2] = zero_bf16
            if col1 < K:
                A_tile[tid_m * 2, tid_n * 2 + 1] = A[row0, col1]
            else:
                A_tile[tid_m * 2, tid_n * 2 + 1] = zero_bf16
        else:
            A_tile[tid_m * 2, tid_n * 2] = zero_bf16
            A_tile[tid_m * 2, tid_n * 2 + 1] = zero_bf16

        if row1 < M:
            if col0 < K:
                A_tile[tid_m * 2 + 1, tid_n * 2] = A[row1, col0]
            else:
                A_tile[tid_m * 2 + 1, tid_n * 2] = zero_bf16
            if col1 < K:
                A_tile[tid_m * 2 + 1, tid_n * 2 + 1] = A[row1, col1]
            else:
                A_tile[tid_m * 2 + 1, tid_n * 2 + 1] = zero_bf16
        else:
            A_tile[tid_m * 2 + 1, tid_n * 2] = zero_bf16
            A_tile[tid_m * 2 + 1, tid_n * 2 + 1] = zero_bf16

        k0 = k_tile + tid_m * 2
        k1 = k0 + 1
        n0 = n_base + tid_n * 2
        n1 = n0 + 1

        if k0 < K:
            if n0 < N:
                B_tile[tid_m * 2, tid_n * 2] = B[n0, k0]
            else:
                B_tile[tid_m * 2, tid_n * 2] = zero_bf16
            if n1 < N:
                B_tile[tid_m * 2, tid_n * 2 + 1] = B[n1, k0]
            else:
                B_tile[tid_m * 2, tid_n * 2 + 1] = zero_bf16
        else:
            B_tile[tid_m * 2, tid_n * 2] = zero_bf16
            B_tile[tid_m * 2, tid_n * 2 + 1] = zero_bf16

        if k1 < K:
            if n0 < N:
                B_tile[tid_m * 2 + 1, tid_n * 2] = B[n0, k1]
            else:
                B_tile[tid_m * 2 + 1, tid_n * 2] = zero_bf16
            if n1 < N:
                B_tile[tid_m * 2 + 1, tid_n * 2 + 1] = B[n1, k1]
            else:
                B_tile[tid_m * 2 + 1, tid_n * 2 + 1] = zero_bf16
        else:
            B_tile[tid_m * 2 + 1, tid_n * 2] = zero_bf16
            B_tile[tid_m * 2 + 1, tid_n * 2 + 1] = zero_bf16

        al.syncthreads()

        for k in al.range(32):
            a0 = al.convert(A_tile[tid_m * 2, k], al.f32)
            a1 = al.convert(A_tile[tid_m * 2 + 1, k], al.f32)
            b0 = al.convert(B_tile[k, tid_n * 2], al.f32)
            b1 = al.convert(B_tile[k, tid_n * 2 + 1], al.f32)

            acc00 = al.convert(acc00 + a0 * b0, al.f32)
            acc01 = al.convert(acc01 + a0 * b1, al.f32)
            acc10 = al.convert(acc10 + a1 * b0, al.f32)
            acc11 = al.convert(acc11 + a1 * b1, al.f32)

        al.syncthreads()

    # Activation constants
    one_f32 = al.convert(1.0, al.f32)
    two_f32 = al.convert(2.0, al.f32)
    neg_one_f32 = al.convert(-1.0, al.f32)

    m0 = m_base + tid_m * 2
    m1 = m0 + 1
    n0 = n_base + tid_n * 2
    n1 = n0 + 1

    if m0 < M:
        if n0 < N:
            b = al.convert(bias_buf[n0], al.f32)
            val = al.convert(acc00 + b, al.f32)
            C[m0, n0] = al.convert(apply_activation(val, one_f32, two_f32, neg_one_f32), al.bf16)
        if n1 < N:
            b = al.convert(bias_buf[n1], al.f32)
            val = al.convert(acc01 + b, al.f32)
            C[m0, n1] = al.convert(apply_activation(val, one_f32, two_f32, neg_one_f32), al.bf16)
    if m1 < M:
        if n0 < N:
            b = al.convert(bias_buf[n0], al.f32)
            val = al.convert(acc10 + b, al.f32)
            C[m1, n0] = al.convert(apply_activation(val, one_f32, two_f32, neg_one_f32), al.bf16)
        if n1 < N:
            b = al.convert(bias_buf[n1], al.f32)
            val = al.convert(acc11 + b, al.f32)
            C[m1, n1] = al.convert(apply_activation(val, one_f32, two_f32, neg_one_f32), al.bf16)


@avelang.jit
def apply_activation(
    x: al.f32,
    one: al.f32,
    two: al.f32,
    neg_one: al.f32,
) -> al.f32:
    # Swish: x * sigmoid(x) = x / (1 + exp(-x))
    exp_neg = al.exp(al.convert(-x, al.f32))
    sigmoid = one / (one + exp_neg)
    y = al.convert(x * sigmoid, al.f32)

    # Divide by 2.0
    y = al.convert(y / two, al.f32)

    # Clamp [-1, 1]
    if y < neg_one:
        y = neg_one
    if y > one:
        y = one

    # Tanh
    y = al.tanh(y)

    # Clamp [-1, 1]
    if y < neg_one:
        y = neg_one
    if y > one:
        y = one

    return y


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        M, K = x.shape
        weight = self.gemm.weight
        bias = self.gemm.bias
        N = weight.shape[0]

        x_bf16 = x.to(torch.bfloat16).contiguous()
        w_bf16 = weight.data.to(torch.bfloat16).contiguous()
        if bias is not None:
            b_bf16 = bias.data.to(torch.bfloat16).contiguous()
        else:
            b_bf16 = torch.zeros(N, dtype=torch.bfloat16, device=x.device)

        out_bf16 = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)

        TILE_M = 32
        TILE_N = 32
        grid_m = (M + TILE_M - 1) // TILE_M
        grid_n = (N + TILE_N - 1) // TILE_N

        gemm_fused_kernel[lambda: ((grid_m, grid_n, 1), (16, 16, 1))](
            x_bf16, w_bf16, b_bf16, out_bf16, M, N, K
        )

        return out_bf16
