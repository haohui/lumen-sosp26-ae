import torch
import torch.nn as nn
import avelang
import avelang.language as al

DIVISOR_VAL = al.constexpr(10.0)


@avelang.jit
def matmul_gelu_fused_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    M: al.i32,
    N: al.i32,
    K: al.i32,
    BLOCK_M: al.constexpr,
    BLOCK_N: al.constexpr,
    BLOCK_K: al.constexpr,
):
    pid_m = al.block_id(0)
    pid_n = al.block_id(1)
    tid = al.thread_id(0)

    row_start = pid_m * BLOCK_M
    col_start = pid_n * BLOCK_N

    x = al.make_tensor(x_ptr, al.bf16, al.make_layout((M * K,), (1,)))
    w = al.make_tensor(w_ptr, al.bf16, al.make_layout((N * K,), (1,)))
    b = al.make_tensor(b_ptr, al.bf16, al.make_layout((N,), (1,)))
    out = al.make_tensor(out_ptr, al.bf16, al.make_layout((M * N,), (1,)))

    THREADS_M = BLOCK_M // 4
    THREADS_N = BLOCK_N // 4
    local_m = tid // THREADS_N
    local_n = tid % THREADS_N

    As = al.make_shared((BLOCK_M, BLOCK_K), al.bf16)
    Bs = al.make_shared((BLOCK_K, BLOCK_N), al.bf16)

    acc00 = al.convert(0.0, al.f32)
    acc01 = al.convert(0.0, al.f32)
    acc02 = al.convert(0.0, al.f32)
    acc03 = al.convert(0.0, al.f32)
    acc10 = al.convert(0.0, al.f32)
    acc11 = al.convert(0.0, al.f32)
    acc12 = al.convert(0.0, al.f32)
    acc13 = al.convert(0.0, al.f32)
    acc20 = al.convert(0.0, al.f32)
    acc21 = al.convert(0.0, al.f32)
    acc22 = al.convert(0.0, al.f32)
    acc23 = al.convert(0.0, al.f32)
    acc30 = al.convert(0.0, al.f32)
    acc31 = al.convert(0.0, al.f32)
    acc32 = al.convert(0.0, al.f32)
    acc33 = al.convert(0.0, al.f32)

    for k_block in al.range(0, K, BLOCK_K):
        k_start = k_block
        for i in al.range(4):
            src_row = row_start + local_m + i * THREADS_M
            src_col = k_start + local_n
            tgt_row = local_m + i * THREADS_M
            tgt_col = local_n
            if src_row < M and src_col < K:
                As[tgt_row, tgt_col] = x[src_row * K + src_col]
            else:
                As[tgt_row, tgt_col] = al.convert(0.0, al.bf16)

        for j in al.range(4):
            src_row = col_start + local_m + j * THREADS_N
            src_col = k_start + local_n
            tgt_row = local_n
            tgt_col = local_m + j * THREADS_N
            if src_row < N and src_col < K:
                Bs[tgt_row, tgt_col] = w[src_row * K + src_col]
            else:
                Bs[tgt_row, tgt_col] = al.convert(0.0, al.bf16)

        al.syncthreads()

        for kk in al.range(BLOCK_K):
            a0 = al.convert(As[local_m, kk], al.f32)
            a1 = al.convert(As[local_m + THREADS_M, kk], al.f32)
            a2 = al.convert(As[local_m + 2 * THREADS_M, kk], al.f32)
            a3 = al.convert(As[local_m + 3 * THREADS_M, kk], al.f32)
            b0 = al.convert(Bs[kk, local_n], al.f32)
            b1 = al.convert(Bs[kk, local_n + THREADS_N], al.f32)
            b2 = al.convert(Bs[kk, local_n + 2 * THREADS_N], al.f32)
            b3 = al.convert(Bs[kk, local_n + 3 * THREADS_N], al.f32)

            acc00 = acc00 + a0 * b0
            acc01 = acc01 + a0 * b1
            acc02 = acc02 + a0 * b2
            acc03 = acc03 + a0 * b3
            acc10 = acc10 + a1 * b0
            acc11 = acc11 + a1 * b1
            acc12 = acc12 + a1 * b2
            acc13 = acc13 + a1 * b3
            acc20 = acc20 + a2 * b0
            acc21 = acc21 + a2 * b1
            acc22 = acc22 + a2 * b2
            acc23 = acc23 + a2 * b3
            acc30 = acc30 + a3 * b0
            acc31 = acc31 + a3 * b1
            acc32 = acc32 + a3 * b2
            acc33 = acc33 + a3 * b3

        al.syncthreads()

    divisor = al.convert(DIVISOR_VAL, al.f32)
    sqrt_2_pi = al.convert(0.7978845608028654, al.f32)
    coeff = al.convert(0.044715, al.f32)
    half = al.convert(0.5, al.f32)
    one = al.convert(1.0, al.f32)

    out_m0 = row_start + local_m
    out_m1 = out_m0 + THREADS_M
    out_m2 = out_m0 + 2 * THREADS_M
    out_m3 = out_m0 + 3 * THREADS_M
    out_n0 = col_start + local_n
    out_n1 = out_n0 + THREADS_N
    out_n2 = out_n0 + 2 * THREADS_N
    out_n3 = out_n0 + 3 * THREADS_N

    if out_m0 < M:
        if out_n0 < N:
            v = acc00 + al.convert(b[out_n0], al.f32)
            v = v / divisor
            x3 = v * v * v
            gelu = half * v * (one + al.tanh(sqrt_2_pi * (v + coeff * x3)))
            out[out_m0 * N + out_n0] = al.convert(gelu, al.bf16)
        if out_n1 < N:
            v = acc01 + al.convert(b[out_n1], al.f32)
            v = v / divisor
            x3 = v * v * v
            gelu = half * v * (one + al.tanh(sqrt_2_pi * (v + coeff * x3)))
            out[out_m0 * N + out_n1] = al.convert(gelu, al.bf16)
        if out_n2 < N:
            v = acc02 + al.convert(b[out_n2], al.f32)
            v = v / divisor
            x3 = v * v * v
            gelu = half * v * (one + al.tanh(sqrt_2_pi * (v + coeff * x3)))
            out[out_m0 * N + out_n2] = al.convert(gelu, al.bf16)
        if out_n3 < N:
            v = acc03 + al.convert(b[out_n3], al.f32)
            v = v / divisor
            x3 = v * v * v
            gelu = half * v * (one + al.tanh(sqrt_2_pi * (v + coeff * x3)))
            out[out_m0 * N + out_n3] = al.convert(gelu, al.bf16)

        if out_m1 < M:
            if out_n0 < N:
                v = acc10 + al.convert(b[out_n0], al.f32)
                v = v / divisor
                x3 = v * v * v
                gelu = half * v * (one + al.tanh(sqrt_2_pi * (v + coeff * x3)))
                out[out_m1 * N + out_n0] = al.convert(gelu, al.bf16)
            if out_n1 < N:
                v = acc11 + al.convert(b[out_n1], al.f32)
                v = v / divisor
                x3 = v * v * v
                gelu = half * v * (one + al.tanh(sqrt_2_pi * (v + coeff * x3)))
                out[out_m1 * N + out_n1] = al.convert(gelu, al.bf16)
            if out_n2 < N:
                v = acc12 + al.convert(b[out_n2], al.f32)
                v = v / divisor
                x3 = v * v * v
                gelu = half * v * (one + al.tanh(sqrt_2_pi * (v + coeff * x3)))
                out[out_m1 * N + out_n2] = al.convert(gelu, al.bf16)
            if out_n3 < N:
                v = acc13 + al.convert(b[out_n3], al.f32)
                v = v / divisor
                x3 = v * v * v
                gelu = half * v * (one + al.tanh(sqrt_2_pi * (v + coeff * x3)))
                out[out_m1 * N + out_n3] = al.convert(gelu, al.bf16)
        if out_m2 < M:
            if out_n0 < N:
                v = acc20 + al.convert(b[out_n0], al.f32)
                v = v / divisor
                x3 = v * v * v
                gelu = half * v * (one + al.tanh(sqrt_2_pi * (v + coeff * x3)))
                out[out_m2 * N + out_n0] = al.convert(gelu, al.bf16)
            if out_n1 < N:
                v = acc21 + al.convert(b[out_n1], al.f32)
                v = v / divisor
                x3 = v * v * v
                gelu = half * v * (one + al.tanh(sqrt_2_pi * (v + coeff * x3)))
                out[out_m2 * N + out_n1] = al.convert(gelu, al.bf16)
            if out_n2 < N:
                v = acc22 + al.convert(b[out_n2], al.f32)
                v = v / divisor
                x3 = v * v * v
                gelu = half * v * (one + al.tanh(sqrt_2_pi * (v + coeff * x3)))
                out[out_m2 * N + out_n2] = al.convert(gelu, al.bf16)
            if out_n3 < N:
                v = acc23 + al.convert(b[out_n3], al.f32)
                v = v / divisor
                x3 = v * v * v
                gelu = half * v * (one + al.tanh(sqrt_2_pi * (v + coeff * x3)))
                out[out_m2 * N + out_n3] = al.convert(gelu, al.bf16)
        if out_m3 < M:
            if out_n0 < N:
                v = acc30 + al.convert(b[out_n0], al.f32)
                v = v / divisor
                x3 = v * v * v
                gelu = half * v * (one + al.tanh(sqrt_2_pi * (v + coeff * x3)))
                out[out_m3 * N + out_n0] = al.convert(gelu, al.bf16)
            if out_n1 < N:
                v = acc31 + al.convert(b[out_n1], al.f32)
                v = v / divisor
                x3 = v * v * v
                gelu = half * v * (one + al.tanh(sqrt_2_pi * (v + coeff * x3)))
                out[out_m3 * N + out_n1] = al.convert(gelu, al.bf16)
            if out_n2 < N:
                v = acc32 + al.convert(b[out_n2], al.f32)
                v = v / divisor
                x3 = v * v * v
                gelu = half * v * (one + al.tanh(sqrt_2_pi * (v + coeff * x3)))
                out[out_m3 * N + out_n2] = al.convert(gelu, al.bf16)
            if out_n3 < N:
                v = acc33 + al.convert(b[out_n3], al.f32)
                v = v / divisor
                x3 = v * v * v
                gelu = half * v * (one + al.tanh(sqrt_2_pi * (v + coeff * x3)))
                out[out_m3 * N + out_n3] = al.convert(gelu, al.bf16)


def avelang_fused(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    divisor: float,
) -> torch.Tensor:
    M, K = x.shape
    N = weight.shape[0]

    x_bf16 = x.to(torch.bfloat16).contiguous()
    w_bf16 = weight.to(torch.bfloat16).contiguous()
    b_bf16 = bias.to(torch.bfloat16).contiguous()
    out = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 16
    BLOCK_SIZE = 256

    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    matmul_gelu_fused_kernel[lambda: ((grid_m, grid_n, 1), (BLOCK_SIZE, 1, 1))](
        x_bf16,
        w_bf16,
        b_bf16,
        out,
        M, N, K,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, input_size, output_size, divisor):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.divisor = divisor
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.bias = nn.Parameter(torch.empty(output_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5.0 ** 0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1.0 / (fan_in ** 0.5) if fan_in > 0 else 0.0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return avelang_fused(x, self.weight, self.bias, self.divisor)
