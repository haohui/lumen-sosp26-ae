import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def fused_gemm_kernel(
    a_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.f32),
    c_ptr: al.Pointer(al.bf16),
    M: al.i32,
    N: al.i32,
    K: al.i32,
    multiplier: al.constexpr,
    negative_slope: al.constexpr,
):
    BM = 64
    BN = 64
    BK = 64

    block_m = al.block_id(1)
    block_n = al.block_id(0)
    tid = al.thread_id(0)

    row_quad = tid // 16
    col_quad = tid % 16

    row0 = block_m * BM + row_quad * 4
    col0 = block_n * BN + col_quad * 4

    one = al.convert(1, al.i32)

    a_layout = al.make_layout((M, K), (K, one))
    a = al.make_tensor(a_ptr, al.bf16, a_layout)
    b_layout = al.make_layout((N, K), (K, one))
    b = al.make_tensor(b_ptr, al.bf16, b_layout)
    bias_layout = al.make_layout((N,), (one,))
    bias = al.make_tensor(bias_ptr, al.f32, bias_layout)
    c_layout = al.make_layout((M, N), (N, one))
    c = al.make_tensor(c_ptr, al.bf16, c_layout)

    a_shared = al.make_shared((BM, BK), al.bf16)
    b_shared = al.make_shared((BK, BN), al.bf16)

    mult_f32 = al.convert(multiplier, al.f32)
    ns_f32 = al.convert(negative_slope, al.f32)
    zero_f32 = al.convert(0.0, al.f32)

    r0 = row0
    r1 = row0 + one
    r2 = row0 + one + one
    r3 = row0 + one + one + one

    c0 = col0
    c1 = col0 + one
    c2 = col0 + one + one
    c3 = col0 + one + one + one

    if r0 >= M or c0 >= N:
        return

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

    for k0 in al.range(0, K, BK):
        k_rem = K - k0
        k_tile = al.min(BK, k_rem)

        # Cooperative load A: (64, 64) = 4096 elements / 256 threads = 16/thread
        a_row = tid // 4
        a_col0 = (tid % 4) * 16
        a_global_row = block_m * BM + a_row
        for j in al.range(0, 16):
            a_col = a_col0 + j
            src_col = k0 + a_col
            if a_row < BM and a_col < BK and a_global_row < M and src_col < K:
                a_shared[a_row, a_col] = a[a_global_row, src_col]

        # Cooperative load B: (64, 64) = 4096 elements / 256 threads = 16/thread
        b_row = tid // 4
        b_col0 = (tid % 4) * 16
        for j in al.range(0, 16):
            b_col = b_col0 + j
            src_row = k0 + b_row
            global_col = block_n * BN + b_col
            if b_row < BK and b_col < BN and src_row < K and global_col < N:
                b_shared[b_row, b_col] = b[global_col, src_row]

        al.syncthreads()

        for k in al.range(0, k_tile):
            a0 = al.convert(a_shared[row_quad * 4, k], al.f32)
            a1 = al.convert(a_shared[row_quad * 4 + 1, k], al.f32)
            a2 = al.convert(a_shared[row_quad * 4 + 2, k], al.f32)
            a3 = al.convert(a_shared[row_quad * 4 + 3, k], al.f32)
            b0 = al.convert(b_shared[k, col_quad * 4], al.f32)
            b1 = al.convert(b_shared[k, col_quad * 4 + 1], al.f32)
            b2 = al.convert(b_shared[k, col_quad * 4 + 2], al.f32)
            b3 = al.convert(b_shared[k, col_quad * 4 + 3], al.f32)
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

    bias0 = bias[c0]
    bias1 = al.convert(0.0, al.f32)
    bias2 = al.convert(0.0, al.f32)
    bias3 = al.convert(0.0, al.f32)
    if c1 < N:
        bias1 = bias[c1]
    if c2 < N:
        bias2 = bias[c2]
    if c3 < N:
        bias3 = bias[c3]

    if r0 < M:
        if c0 < N:
            v = acc00 + bias0; v = v * mult_f32
            if v < zero_f32: v = v * ns_f32
            c[r0, c0] = al.convert(v, al.bf16)
        if c1 < N:
            v = acc01 + bias1; v = v * mult_f32
            if v < zero_f32: v = v * ns_f32
            c[r0, c1] = al.convert(v, al.bf16)
        if c2 < N:
            v = acc02 + bias2; v = v * mult_f32
            if v < zero_f32: v = v * ns_f32
            c[r0, c2] = al.convert(v, al.bf16)
        if c3 < N:
            v = acc03 + bias3; v = v * mult_f32
            if v < zero_f32: v = v * ns_f32
            c[r0, c3] = al.convert(v, al.bf16)

    if r1 < M:
        if c0 < N:
            v = acc10 + bias0; v = v * mult_f32
            if v < zero_f32: v = v * ns_f32
            c[r1, c0] = al.convert(v, al.bf16)
        if c1 < N:
            v = acc11 + bias1; v = v * mult_f32
            if v < zero_f32: v = v * ns_f32
            c[r1, c1] = al.convert(v, al.bf16)
        if c2 < N:
            v = acc12 + bias2; v = v * mult_f32
            if v < zero_f32: v = v * ns_f32
            c[r1, c2] = al.convert(v, al.bf16)
        if c3 < N:
            v = acc13 + bias3; v = v * mult_f32
            if v < zero_f32: v = v * ns_f32
            c[r1, c3] = al.convert(v, al.bf16)

    if r2 < M:
        if c0 < N:
            v = acc20 + bias0; v = v * mult_f32
            if v < zero_f32: v = v * ns_f32
            c[r2, c0] = al.convert(v, al.bf16)
        if c1 < N:
            v = acc21 + bias1; v = v * mult_f32
            if v < zero_f32: v = v * ns_f32
            c[r2, c1] = al.convert(v, al.bf16)
        if c2 < N:
            v = acc22 + bias2; v = v * mult_f32
            if v < zero_f32: v = v * ns_f32
            c[r2, c2] = al.convert(v, al.bf16)
        if c3 < N:
            v = acc23 + bias3; v = v * mult_f32
            if v < zero_f32: v = v * ns_f32
            c[r2, c3] = al.convert(v, al.bf16)

    if r3 < M:
        if c0 < N:
            v = acc30 + bias0; v = v * mult_f32
            if v < zero_f32: v = v * ns_f32
            c[r3, c0] = al.convert(v, al.bf16)
        if c1 < N:
            v = acc31 + bias1; v = v * mult_f32
            if v < zero_f32: v = v * ns_f32
            c[r3, c1] = al.convert(v, al.bf16)
        if c2 < N:
            v = acc32 + bias2; v = v * mult_f32
            if v < zero_f32: v = v * ns_f32
            c[r3, c2] = al.convert(v, al.bf16)
        if c3 < N:
            v = acc33 + bias3; v = v * mult_f32
            if v < zero_f32: v = v * ns_f32
            c[r3, c3] = al.convert(v, al.bf16)


def avelang_fused_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    multiplier: float,
    negative_slope: float,
) -> torch.Tensor:
    assert x.is_cuda, "Input must be on CUDA/HIP device."
    assert weight.is_cuda, "Weight must be on CUDA/HIP device."

    M, K_in = x.shape
    N, K_w = weight.shape
    assert K_in == K_w, f"Dimension mismatch: {K_in} vs {K_w}"

    x_bf16 = x.to(torch.bfloat16).contiguous()
    w_bf16 = weight.to(torch.bfloat16).contiguous()
    bias_f32 = bias.to(torch.float32).contiguous()

    out = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)

    BM = 64
    BN = 64

    grid_m = (M + BM - 1) // BM
    grid_n = (N + BN - 1) // BN

    fused_gemm_kernel[lambda: ((grid_n, grid_m, 1), (256, 1, 1))](
        x_bf16.data_ptr(),
        w_bf16.data_ptr(),
        bias_f32.data_ptr(),
        out.data_ptr(),
        M, N, K_in,
        multiplier,
        negative_slope,
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.multiplier = multiplier
        self.negative_slope = negative_slope

    def forward(self, x):
        return avelang_fused_gemm(
            x,
            self.gemm.weight,
            self.gemm.bias,
            self.multiplier,
            self.negative_slope,
        )
