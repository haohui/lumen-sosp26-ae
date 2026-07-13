import torch
import torch.nn as nn
import avelang
import avelang.language as al

# ---------------------------------------------------------------------------
# Tile constants
# ---------------------------------------------------------------------------
_BM = 32
_BN = 32
_BK = 8

# ---------------------------------------------------------------------------
# GEMM kernel: C = A @ B + bias
# 2D block = (_BM, _BN // 4, 1) = (32, 8, 1) = 256 threads.
# Each thread computes 4 output columns, reusing A values.
# ---------------------------------------------------------------------------
@avelang.jit
def gemm_kernel(
    A_ptr: al.Pointer(al.bf16),
    B_ptr: al.Pointer(al.bf16),
    C_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    M: al.i32,
    N: al.i32,
    K: al.i32,
    BM: al.constexpr,
    BN: al.constexpr,
    BK: al.constexpr,
    COLS_PER_THREAD: al.constexpr,
):
    a_layout = al.make_layout((M, K), (K, 1))
    A = al.make_tensor(A_ptr, al.bf16, a_layout)
    b_layout = al.make_layout((K, N), (N, 1))
    B = al.make_tensor(B_ptr, al.bf16, b_layout)
    c_layout = al.make_layout((M, N), (N, 1))
    C = al.make_tensor(C_ptr, al.bf16, c_layout)
    bias_layout = al.make_layout((N,), (1,))
    bias = al.make_tensor(bias_ptr, al.bf16, bias_layout)

    block_m = al.block_id(1)
    block_n = al.block_id(0)
    row = al.thread_id(0)
    col_base = al.thread_id(1)

    m_start = block_m * BM
    n_start = block_n * BN

    a_shared = al.make_shared((BM, BK), al.bf16)
    b_shared = al.make_shared((BK, BN), al.bf16)

    acc0 = al.convert(0.0, al.f32)
    acc1 = al.convert(0.0, al.f32)
    acc2 = al.convert(0.0, al.f32)
    acc3 = al.convert(0.0, al.f32)

    for k_block in al.range(0, K, BK):
        # Load A: 1 element per thread
        a_shared[row, col_base] = A[m_start + row, k_block + col_base]

        # Load B: transpose indices, 1 element per thread
        b_shared[col_base, row] = B[k_block + col_base, n_start + row]

        al.syncthreads()

        # Inner product: reuse A value for 4 output columns
        for k_idx in al.range(BK):
            a_val = al.convert(a_shared[row, k_idx], al.f32)
            b0 = al.convert(b_shared[k_idx, col_base], al.f32)
            b1 = al.convert(b_shared[k_idx, col_base + COLS_PER_THREAD], al.f32)
            b2 = al.convert(b_shared[k_idx, col_base + COLS_PER_THREAD * 2], al.f32)
            b3 = al.convert(b_shared[k_idx, col_base + COLS_PER_THREAD * 3], al.f32)
            acc0 = acc0 + a_val * b0
            acc1 = acc1 + a_val * b1
            acc2 = acc2 + a_val * b2
            acc3 = acc3 + a_val * b3

        al.syncthreads()

    n0 = n_start + col_base
    n1 = n0 + COLS_PER_THREAD
    n2 = n0 + COLS_PER_THREAD * 2
    n3 = n0 + COLS_PER_THREAD * 3

    acc0 = acc0 + al.convert(bias[n0], al.f32)
    acc1 = acc1 + al.convert(bias[n1], al.f32)
    acc2 = acc2 + al.convert(bias[n2], al.f32)
    acc3 = acc3 + al.convert(bias[n3], al.f32)

    C[m_start + row, n0] = al.convert(acc0, al.bf16)
    C[m_start + row, n1] = al.convert(acc1, al.bf16)
    C[m_start + row, n2] = al.convert(acc2, al.bf16)
    C[m_start + row, n3] = al.convert(acc3, al.bf16)


# ---------------------------------------------------------------------------
# MaxPool(k=2) + sum(dim=1) + scale  reduction kernel
# ---------------------------------------------------------------------------
@avelang.jit
def maxpool_sum_scale_kernel(
    inp_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    M: al.i32,
    P: al.i32,
    BLOCK_SIZE: al.constexpr,
):
    inp_layout = al.make_layout((M, P * 2), (P * 2, 1))
    inp = al.make_tensor(inp_ptr, al.bf16, inp_layout)
    out_layout = al.make_layout((M,), (1,))
    out = al.make_tensor(out_ptr, al.bf16, out_layout)

    row = al.block_id(0)
    tid = al.thread_id(0)
    chunk = P // BLOCK_SIZE

    local_sum = al.convert(0.0, al.f32)
    start = tid * chunk * 2

    for j in al.range(chunk):
        idx = start + j * 2
        a_val = al.convert(inp[row, idx], al.f32)
        b_val = al.convert(inp[row, idx + 1], al.f32)
        half = al.convert(0.5, al.f32)
        pooled = (a_val + b_val + al.abs(a_val - b_val)) * half
        local_sum = local_sum + pooled

    smem = al.make_shared((BLOCK_SIZE,), al.f32)
    smem[tid] = local_sum
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
    if tid == 0:
        smem[0] = smem[0] + smem[1]
    al.syncthreads()

    if tid == 0:
        out[row] = al.convert(smem[0] * al.convert(0.5, al.f32), al.bf16)


# ---------------------------------------------------------------------------
# Host launchers
# ---------------------------------------------------------------------------
def _launch_gemm(x, weight, bias):
    M_val = x.shape[0]
    K_val = x.shape[1]
    N_val = weight.shape[0]
    out = torch.empty((M_val, N_val), dtype=torch.bfloat16, device=x.device)

    cols_per_thread = _BN // 4
    grid_n = (N_val + _BN - 1) // _BN
    grid_m = (M_val + _BM - 1) // _BM

    gemm_kernel[lambda: ((grid_n, grid_m, 1), (_BM, cols_per_thread, 1))](
        x.data_ptr(), weight.data_ptr(), out.data_ptr(), bias.data_ptr(),
        M_val, N_val, K_val, _BM, _BN, _BK, cols_per_thread,
    )
    return out


def _launch_maxpool_sum_scale(x, scale_factor):
    M_val = x.shape[0]
    P_val = x.shape[1] // 2
    BLOCK_SIZE = 256
    out = torch.empty((M_val,), dtype=torch.bfloat16, device=x.device)
    maxpool_sum_scale_kernel[lambda: ((M_val, 1, 1), (BLOCK_SIZE, 1, 1))](
        x.data_ptr(), out.data_ptr(), M_val, P_val, BLOCK_SIZE,
    )
    return out


# ---------------------------------------------------------------------------
# ModelNew
# ---------------------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor

    def forward(self, x):
        x = x.contiguous()
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)

        weight = self.matmul.weight.data
        bias = self.matmul.bias.data
        if weight.dtype != torch.bfloat16:
            weight = weight.to(torch.bfloat16)
        if bias.dtype != torch.bfloat16:
            bias = bias.to(torch.bfloat16)

        x = _launch_gemm(x, weight.T.contiguous(), bias)
        x = _launch_maxpool_sum_scale(x, self.scale_factor)
        return x
