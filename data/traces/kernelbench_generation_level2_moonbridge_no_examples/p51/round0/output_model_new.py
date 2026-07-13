import torch
import torch.nn as nn
import avelang
import avelang.language as al

# ── Tile constants ──────────────────────────────────────────────────────────
BM = 256
BN = 256
BK = 32
BLOCK_SIZE = 256


# ── GEMM kernel ─────────────────────────────────────────────────────────────
# Tiled GEMM: C[M,N] = A[M,K] @ B[N,K]^T + bias[N]
# Each block computes a BM×BN output tile.  BM threads per block; each thread
# owns one row of the tile and accumulates BN f32 values in registers.
# Shared memory holds BM×BK (A tile) and BK×BN (B tile, transposed).


@avelang.jit
def gemm_kernel(
    a_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.f32),
    c_ptr: al.Pointer(al.bf16),
    M: al.i32,
    N: al.i32,
    K: al.i32,
    BM: al.constexpr,
    BN: al.constexpr,
    BK: al.constexpr,
):
    pid_m = al.block_id(0)
    pid_n = al.block_id(1)
    tid = al.thread_id(0)

    m_start = pid_m * BM
    n_start = pid_n * BN

    a_sh = al.make_shared((BM, BK), al.bf16)
    b_sh = al.make_shared((BK, BN), al.bf16)

    acc = al.make_local((BN,), al.f32)
    for j in al.range(BN):
        acc[j] = al.convert(0.0, al.f32)

    a = al.make_tensor(a_ptr, al.bf16, al.make_layout((M, K), (K, 1)))
    b = al.make_tensor(b_ptr, al.bf16, al.make_layout((N, K), (K, 1)))
    bias = al.make_tensor(bias_ptr, al.f32, al.make_layout((N,), (1,)))
    c = al.make_tensor(c_ptr, al.bf16, al.make_layout((M, N), (N, 1)))

    for k_block in al.range(0, K, BK):
        # Cooperative load A tile [BM, BK]
        for idx in al.range(tid, BM * BK, BM):
            i = idx // BK
            j = idx % BK
            a_sh[i, j] = a[m_start + i, k_block + j]

        # Cooperative load B tile [BK, BN] (transposed from row-major B[N,K])
        for idx in al.range(tid, BN * BK, BM):
            i = idx // BN
            j = idx % BN
            b_sh[i, j] = b[n_start + j, k_block + i]

        al.syncthreads()

        # Compute: this thread owns one row of the output tile
        row = tid
        if row < BM:
            for j in al.range(BN):
                dot = al.convert(0.0, al.f32)
                for k in al.range(BK):
                    dot = dot + al.convert(a_sh[row, k], al.f32) * al.convert(b_sh[k, j], al.f32)
                acc[j] = acc[j] + dot

        al.syncthreads()

    # Store with bias
    row = tid
    if row < BM:
        global_row = m_start + row
        if global_row < M:
            for j in al.range(BN):
                global_col = n_start + j
                if global_col < N:
                    c[global_row, global_col] = al.convert(acc[j] + bias[global_col], al.bf16)


# ── Post-process kernel ─────────────────────────────────────────────────────
# Fuses: subtract → global-avg-pool (per-row mean) → logsumexp (identity on
# size-1 dim) → GELU (tanh approx) → broadcast residual add to original.
# One block per row, BLOCK_SIZE threads cooperate on the reduction.


@avelang.jit
def postprocess_kernel(
    gemm_out_ptr: al.Pointer(al.bf16),
    subtract_ptr: al.Pointer(al.f32),
    original_x_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    M: al.i32,
    N: al.i32,
    K: al.i32,
    BLOCK_SIZE: al.constexpr,
):
    pid = al.block_id(0)
    tid = al.thread_id(0)
    row = pid

    if row >= M:
        return

    gemm_out = al.make_tensor(gemm_out_ptr, al.bf16, al.make_layout((M, N), (N, 1)))
    subtract = al.make_tensor(subtract_ptr, al.f32, al.make_layout((N,), (1,)))
    original_x = al.make_tensor(original_x_ptr, al.bf16, al.make_layout((M, K), (K, 1)))
    output = al.make_tensor(output_ptr, al.bf16, al.make_layout((M, K), (K, 1)))

    partial = al.make_shared((BLOCK_SIZE,), al.f32)

    # Step 1: per-thread partial sum of (gemm_out[row, :] - subtract)
    my_sum = al.convert(0.0, al.f32)
    for j in al.range(tid, N, BLOCK_SIZE):
        val = al.convert(gemm_out[row, j], al.f32) - subtract[j]
        my_sum = my_sum + val
    partial[tid] = my_sum
    al.syncthreads()

    # Step 2: tree reduction to get total row sum
    if tid < 128:
        partial[tid] = partial[tid] + partial[tid + 128]
    al.syncthreads()
    if tid < 64:
        partial[tid] = partial[tid] + partial[tid + 64]
    al.syncthreads()
    if tid < 32:
        partial[tid] = partial[tid] + partial[tid + 32]
    al.syncthreads()
    if tid < 16:
        partial[tid] = partial[tid] + partial[tid + 16]
    al.syncthreads()
    if tid < 8:
        partial[tid] = partial[tid] + partial[tid + 8]
    al.syncthreads()
    if tid < 4:
        partial[tid] = partial[tid] + partial[tid + 4]
    al.syncthreads()
    if tid < 2:
        partial[tid] = partial[tid] + partial[tid + 2]
    al.syncthreads()
    if tid < 1:
        partial[tid] = partial[tid] + partial[tid + 1]
    al.syncthreads()

    # Step 3: mean, logsumexp (identity on size-1 dim), GELU
    total_sum = partial[0]
    count = al.convert(N, al.f32)
    mean_val = total_sum / count

    # LogSumExp over size-1 dimension is identity: log(exp(x)) = x
    lse_val = mean_val

    # GELU tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    x3 = lse_val * lse_val * lse_val
    sqrt_2_div_pi = al.convert(0.7978845608028654, al.f32)
    coeff = al.convert(0.044715, al.f32)
    inner = sqrt_2_div_pi * (lse_val + coeff * x3)
    tanh_val = al.tanh(inner)
    half = al.convert(0.5, al.f32)
    one = al.convert(1.0, al.f32)
    gelu_val = half * lse_val * (one + tanh_val)

    # Step 4: broadcast residual add
    for j in al.range(tid, K, BLOCK_SIZE):
        out_val = gelu_val + al.convert(original_x[row, j], al.f32)
        output[row, j] = al.convert(out_val, al.bf16)


# ── Host wrapper ────────────────────────────────────────────────────────────


def avelang_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    subtract: torch.Tensor,
) -> torch.Tensor:
    """Run the full pipeline: GEMM → subtract → mean → logsumexp → GELU → residual add."""
    M, K_in = x.shape
    N = weight.shape[0]

    # Convert to BF16 for GPU compute; keep bias/subtract in FP32 for accumulation
    x_bf16 = x.to(torch.bfloat16).contiguous()
    w_bf16 = weight.to(torch.bfloat16).contiguous()
    b_f32 = bias.to(torch.float32).contiguous()
    s_f32 = subtract.data.to(torch.float32).contiguous()

    gemm_out = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)

    grid_gemm = ((M + BM - 1) // BM, (N + BN - 1) // BN, 1)
    block_gemm = (BM, 1, 1)

    gemm_kernel[lambda: (grid_gemm, block_gemm)](
        x_bf16, w_bf16, b_f32, gemm_out,
        M, N, K_in,
        BM, BN, BK,
    )

    output = torch.empty(M, K_in, dtype=torch.bfloat16, device=x.device)

    grid_post = (M, 1, 1)
    block_post = (BLOCK_SIZE, 1, 1)

    postprocess_kernel[lambda: (grid_post, block_post)](
        gemm_out, s_f32, x_bf16, output,
        M, N, K_in,
        BLOCK_SIZE,
    )

    return output.to(x.dtype)


# ── ModelNew ─────────────────────────────────────────────────────────────────


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.subtract = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        return avelang_forward(x, self.gemm.weight, self.gemm.bias, self.subtract)
