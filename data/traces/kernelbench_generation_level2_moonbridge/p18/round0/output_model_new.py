import torch
import torch.nn as nn
import avelang
import avelang.language as al


BLOCK_SIZE: al.constexpr = 256
MMA_M: al.constexpr = 32
MMA_N: al.constexpr = 32
MMA_K: al.constexpr = 16


# ---------------------------------------------------------------------------
# GEMM kernel: computes C = A @ B + bias  where A is (M, K), B is (N, K)
# Output is (M, N) in bf16.
# Pattern adapted from verified level-1 GEMM kernel.
# ---------------------------------------------------------------------------

@avelang.jit
def gemm_bias_bf16_kernel(
    A_ptr: al.Pointer(al.bf16),
    B_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    C_ptr: al.Pointer(al.bf16),
    m: al.u32,
    n: al.u32,
    k: al.u32,
):
    lane = al.thread_id(0)
    lane_col = lane & 31
    lane_group = lane >> 5

    block_m = al.block_id(1) * MMA_M
    block_n = al.block_id(0) * MMA_N

    A_bf16 = al.make_tensor(A_ptr, al.bf16, al.make_layout((m, k), (k, 1)))
    B_bf16 = al.make_tensor(B_ptr, al.bf16, al.make_layout((n, k), (k, 1)))
    C_out = al.make_tensor(C_ptr, al.bf16, al.make_layout((m, n), (n, 1)))
    Bias = al.make_tensor(bias_ptr, al.bf16, al.make_layout((n,), (1,)))

    k_vecs = k >> 3
    packed_row_stride = k >> 1

    A_vec = al.view(
        A_bf16, al.i32, al.make_layout((m, k_vecs, 4), (packed_row_stride, 4, 1))
    )
    B_vec = al.view(
        B_bf16, al.i32, al.make_layout((n, k_vecs, 4), (packed_row_stride, 4, 1))
    )

    a_smem = al.make_shared((MMA_M * (MMA_K >> 3), MMA_K >> 2), al.i32)
    b_smem = al.make_shared((MMA_N * (MMA_K >> 3), MMA_K >> 2), al.i32)

    acc = al.full((16,), 0.0, al.f32)

    for kt in al.range(k // MMA_K):
        k_vec = kt * 2 + lane_group

        a_smem[lane] = A_vec[block_m + lane_col, k_vec]
        b_smem[lane] = B_vec[block_n + lane_col, k_vec]

        al.syncthreads()

        a_words = a_smem[lane]
        b_words = b_smem[lane]
        a_frag = al.view(a_words, al.Tensor((2, 2, 1), al.u32))
        b_frag = al.view(b_words, al.Tensor((2, 2, 1), al.u32))

        acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag[0], a_frag[0], acc)
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag[1], a_frag[1], acc)

        al.syncthreads()

    # Epilogue: write accumulator to shared memory, add bias, store to global
    c_smem = al.make_shared((MMA_M, MMA_N), al.f32)

    for r in al.range(16):
        row_offset = ((r >> 2) << 3) + lane_group * 4 + (r & 3)
        c_smem[lane_col, row_offset] = acc[r]

    al.syncthreads()

    store_row = lane >> 1
    col_base = (lane & 1) * 16

    for v in al.range(16):
        col = block_n + col_base + v
        bias_val = al.convert(Bias[col], al.f32)
        c_val = c_smem[store_row, col_base + v] + bias_val
        C_out[block_m + store_row, col] = al.convert(c_val, al.bf16)


# ---------------------------------------------------------------------------
# Row-sum reduction kernel
# ---------------------------------------------------------------------------

@avelang.jit
def row_sum_kernel(
    in_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    M: al.i32,
    N: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    if bid < M:
        smem = al.make_shared((BLOCK_SIZE,), al.f32)
        inp = al.make_tensor(in_ptr, al.bf16, al.make_layout((M, N), (N, 1)))

        partial = al.convert(0.0, al.f32)
        for j in al.range(tid, N, BLOCK_SIZE):
            partial = partial + al.convert(inp[bid, j], al.f32)

        smem[tid] = partial
        al.syncthreads()

        if tid < 128:
            smem[tid] = smem[tid] + smem[tid + 128]
        al.syncthreads()
        if tid < 64:
            smem[tid] = smem[tid] + smem[tid + 64]
        al.syncthreads()
        if tid < 32:
            smem[tid] = smem[tid] + smem[tid + 32]
        al.syncthreads()
        if tid < 16:
            smem[tid] = smem[tid] + smem[tid + 16]
        al.syncthreads()
        if tid < 8:
            smem[tid] = smem[tid] + smem[tid + 8]
        al.syncthreads()
        if tid < 4:
            smem[tid] = smem[tid] + smem[tid + 4]
        al.syncthreads()
        if tid < 2:
            smem[tid] = smem[tid] + smem[tid + 2]
        al.syncthreads()
        if tid < 1:
            smem[tid] = smem[tid] + smem[tid + 1]

        if tid == 0:
            out = al.make_tensor(out_ptr, al.bf16, al.make_layout((M, 1), (1, 1)))
            out[bid, 0] = al.convert(smem[0], al.bf16)


# ---------------------------------------------------------------------------
# Host-side helpers
# ---------------------------------------------------------------------------

def _prepare_bf16_cuda_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_linear_rowsum(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = _prepare_bf16_cuda_contiguous(x)
    weight_bf16 = _prepare_bf16_cuda_contiguous(weight)
    bias_bf16 = _prepare_bf16_cuda_contiguous(bias)

    m_val, k_val = x_bf16.shape
    n_val, weight_k = weight_bf16.shape
    if weight_k != k_val:
        raise ValueError(f"Weight/input K mismatch: x has K={k_val}, weight has K={weight_k}")
    if m_val % MMA_M != 0 or n_val % MMA_N != 0 or k_val % MMA_K != 0:
        raise ValueError(
            f"Expected m % {MMA_M} == 0, n % {MMA_N} == 0, k % {MMA_K} == 0 "
            f"(got m={m_val}, n={n_val}, k={k_val})"
        )

    # Phase 1: GEMM with bias -> intermediate output (M, N) bf16
    inter = torch.empty((m_val, n_val), device=x_bf16.device, dtype=torch.bfloat16)
    grid_gemm = (n_val // MMA_N, m_val // MMA_M, 1)
    gemm_bias_bf16_kernel[lambda: (grid_gemm, (64, 1, 1))](
        x_bf16, weight_bf16, bias_bf16, inter, m_val, n_val, k_val,
    )

    # Phase 2: Row-sum reduction -> output (M, 1) bf16
    out = torch.empty((m_val, 1), device=x_bf16.device, dtype=torch.bfloat16)
    grid_sum = (m_val, 1, 1)
    row_sum_kernel[lambda: (grid_sum, (BLOCK_SIZE, 1, 1))](
        inter, out, m_val, n_val,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs Linear + row-sum using two AveLang kernels:
    1. GEMM kernel (x @ W^T + bias) → intermediate (M, N) bf16 output
    2. Row-sum reduction kernel → final (M, 1) bf16 output

    This preserves the exact computation order of the reference Model.forward.
    """

    def __init__(self, in_features: int, out_features: int):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_linear_rowsum(x, self.linear.weight, self.linear.bias)


def get_inputs():
    batch_size = 1024
    in_features = 8192
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    in_features = 8192
    out_features = 8192
    return [in_features, out_features]
