import torch
import torch.nn as nn
import avelang
import avelang.language as al

# ---------------------------------------------------------------------------
# Tiled GEMM kernel with shared memory
#   C[M,N] = A[M,K] @ B[N,K]^T   all bf16, FP32 accumulation
#
#   TILE_M=64, TILE_N=64, TILE_K=32
#   Block = (16, 16) = 256 threads
#   Shared: As[64,32], Bs[32,64]  bf16
# ---------------------------------------------------------------------------
@avelang.jit
def gemm_kernel(
    A_ptr: al.Pointer(al.bf16),
    B_ptr: al.Pointer(al.bf16),
    C_ptr: al.Pointer(al.bf16),
    M: al.i32,
    K: al.i32,
    N: al.i32,
    num_k_steps: al.i32,
):
    ti = al.thread_id(0)
    tj = al.thread_id(1)
    block_m = al.block_id(0)
    block_n = al.block_id(1)

    A = al.make_tensor(A_ptr, al.bf16, al.make_layout((M, K), (K, 1)))
    B = al.make_tensor(B_ptr, al.bf16, al.make_layout((N, K), (K, 1)))
    C = al.make_tensor(C_ptr, al.bf16, al.make_layout((M, N), (N, 1)))

    As = al.make_shared((64, 32), al.bf16)
    Bs = al.make_shared((32, 64), al.bf16)

    # 4x4 sub-tile per thread: 16 FP32 accumulators in local memory
    acc = al.make_local((4, 4), al.f32)
    for li in al.range(4):
        for lj in al.range(4):
            acc[li, lj] = al.convert(0.0, al.f32)

    k_off = al.convert(0, al.i32)

    for _k_step in al.range(num_k_steps):
        # -- Cooperative load of A tile into As[64,32] --
        # Thread (ti,tj) loads 8 elements covering:
        #   (ti,tj), (ti+16,tj), (ti+32,tj), (ti+48,tj),
        #   (ti,tj+16), (ti+16,tj+16), (ti+32,tj+16), (ti+48,tj+16)
        a_row0 = block_m * 64 + ti
        a_col0 = k_off + tj

        if a_row0 < M:
            if a_col0 < K:
                As[ti, tj] = A[a_row0, a_col0]
            else:
                As[ti, tj] = al.convert(0.0, al.bf16)
        else:
            As[ti, tj] = al.convert(0.0, al.bf16)

        a_row1 = a_row0 + 16
        if a_row1 < M:
            if a_col0 < K:
                As[ti + 16, tj] = A[a_row1, a_col0]
            else:
                As[ti + 16, tj] = al.convert(0.0, al.bf16)
        else:
            As[ti + 16, tj] = al.convert(0.0, al.bf16)

        a_row2 = a_row0 + 32
        if a_row2 < M:
            if a_col0 < K:
                As[ti + 32, tj] = A[a_row2, a_col0]
            else:
                As[ti + 32, tj] = al.convert(0.0, al.bf16)
        else:
            As[ti + 32, tj] = al.convert(0.0, al.bf16)

        a_row3 = a_row0 + 48
        if a_row3 < M:
            if a_col0 < K:
                As[ti + 48, tj] = A[a_row3, a_col0]
            else:
                As[ti + 48, tj] = al.convert(0.0, al.bf16)
        else:
            As[ti + 48, tj] = al.convert(0.0, al.bf16)

        a_col1 = a_col0 + 16
        if a_row0 < M:
            if a_col1 < K:
                As[ti, tj + 16] = A[a_row0, a_col1]
            else:
                As[ti, tj + 16] = al.convert(0.0, al.bf16)
        else:
            As[ti, tj + 16] = al.convert(0.0, al.bf16)

        if a_row1 < M:
            if a_col1 < K:
                As[ti + 16, tj + 16] = A[a_row1, a_col1]
            else:
                As[ti + 16, tj + 16] = al.convert(0.0, al.bf16)
        else:
            As[ti + 16, tj + 16] = al.convert(0.0, al.bf16)

        if a_row2 < M:
            if a_col1 < K:
                As[ti + 32, tj + 16] = A[a_row2, a_col1]
            else:
                As[ti + 32, tj + 16] = al.convert(0.0, al.bf16)
        else:
            As[ti + 32, tj + 16] = al.convert(0.0, al.bf16)

        if a_row3 < M:
            if a_col1 < K:
                As[ti + 48, tj + 16] = A[a_row3, a_col1]
            else:
                As[ti + 48, tj + 16] = al.convert(0.0, al.bf16)
        else:
            As[ti + 48, tj + 16] = al.convert(0.0, al.bf16)

        # -- Cooperative load of B tile into Bs[32,64] --
        # Thread (ti,tj) loads 8 elements covering:
        #   (ti,tj), (ti+16,tj),
        #   (ti,tj+16), (ti+16,tj+16),
        #   (ti,tj+32), (ti+16,tj+32),
        #   (ti,tj+48), (ti+16,tj+48)
        b_col0 = block_n * 64 + tj
        b_row0 = k_off + ti

        if b_col0 < N:
            if b_row0 < K:
                Bs[ti, tj] = B[b_col0, b_row0]
            else:
                Bs[ti, tj] = al.convert(0.0, al.bf16)
        else:
            Bs[ti, tj] = al.convert(0.0, al.bf16)

        b_row1 = b_row0 + 16
        if b_col0 < N:
            if b_row1 < K:
                Bs[ti + 16, tj] = B[b_col0, b_row1]
            else:
                Bs[ti + 16, tj] = al.convert(0.0, al.bf16)
        else:
            Bs[ti + 16, tj] = al.convert(0.0, al.bf16)

        b_col1 = b_col0 + 16
        if b_col1 < N:
            if b_row0 < K:
                Bs[ti, tj + 16] = B[b_col1, b_row0]
            else:
                Bs[ti, tj + 16] = al.convert(0.0, al.bf16)
        else:
            Bs[ti, tj + 16] = al.convert(0.0, al.bf16)

        if b_col1 < N:
            if b_row1 < K:
                Bs[ti + 16, tj + 16] = B[b_col1, b_row1]
            else:
                Bs[ti + 16, tj + 16] = al.convert(0.0, al.bf16)
        else:
            Bs[ti + 16, tj + 16] = al.convert(0.0, al.bf16)

        b_col2 = b_col0 + 32
        if b_col2 < N:
            if b_row0 < K:
                Bs[ti, tj + 32] = B[b_col2, b_row0]
            else:
                Bs[ti, tj + 32] = al.convert(0.0, al.bf16)
        else:
            Bs[ti, tj + 32] = al.convert(0.0, al.bf16)

        if b_col2 < N:
            if b_row1 < K:
                Bs[ti + 16, tj + 32] = B[b_col2, b_row1]
            else:
                Bs[ti + 16, tj + 32] = al.convert(0.0, al.bf16)
        else:
            Bs[ti + 16, tj + 32] = al.convert(0.0, al.bf16)

        b_col3 = b_col0 + 48
        if b_col3 < N:
            if b_row0 < K:
                Bs[ti, tj + 48] = B[b_col3, b_row0]
            else:
                Bs[ti, tj + 48] = al.convert(0.0, al.bf16)
        else:
            Bs[ti, tj + 48] = al.convert(0.0, al.bf16)

        if b_col3 < N:
            if b_row1 < K:
                Bs[ti + 16, tj + 48] = B[b_col3, b_row1]
            else:
                Bs[ti + 16, tj + 48] = al.convert(0.0, al.bf16)
        else:
            Bs[ti + 16, tj + 48] = al.convert(0.0, al.bf16)

        al.syncthreads()

        # -- Compute 4x4 sub-tile --
        for k in al.range(32):
            a0 = al.convert(As[ti * 4 + 0, k], al.f32)
            a1 = al.convert(As[ti * 4 + 1, k], al.f32)
            a2 = al.convert(As[ti * 4 + 2, k], al.f32)
            a3 = al.convert(As[ti * 4 + 3, k], al.f32)
            b0 = al.convert(Bs[k, tj * 4 + 0], al.f32)
            b1 = al.convert(Bs[k, tj * 4 + 1], al.f32)
            b2 = al.convert(Bs[k, tj * 4 + 2], al.f32)
            b3 = al.convert(Bs[k, tj * 4 + 3], al.f32)
            acc[0, 0] = acc[0, 0] + a0 * b0
            acc[0, 1] = acc[0, 1] + a0 * b1
            acc[0, 2] = acc[0, 2] + a0 * b2
            acc[0, 3] = acc[0, 3] + a0 * b3
            acc[1, 0] = acc[1, 0] + a1 * b0
            acc[1, 1] = acc[1, 1] + a1 * b1
            acc[1, 2] = acc[1, 2] + a1 * b2
            acc[1, 3] = acc[1, 3] + a1 * b3
            acc[2, 0] = acc[2, 0] + a2 * b0
            acc[2, 1] = acc[2, 1] + a2 * b1
            acc[2, 2] = acc[2, 2] + a2 * b2
            acc[2, 3] = acc[2, 3] + a2 * b3
            acc[3, 0] = acc[3, 0] + a3 * b0
            acc[3, 1] = acc[3, 1] + a3 * b1
            acc[3, 2] = acc[3, 2] + a3 * b2
            acc[3, 3] = acc[3, 3] + a3 * b3

        al.syncthreads()

        k_off = k_off + 32

    # -- Write 4x4 sub-tile to global memory --
    for li in al.range(4):
        g_row = block_m * 64 + ti * 4 + li
        if g_row < M:
            for lj in al.range(4):
                g_col = block_n * 64 + tj * 4 + lj
                if g_col < N:
                    C[g_row, g_col] = al.convert(acc[li, lj], al.bf16)


# ---------------------------------------------------------------------------
# Bias + sigmoid kernel   (2-D grid)
# ---------------------------------------------------------------------------
@avelang.jit
def bias_sigmoid_kernel(
    in_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    M: al.i32,
    N: al.i32,
):
    row = al.block_id(0) * al.block_dim(0) + al.thread_id(0)
    col = al.block_id(1) * al.block_dim(1) + al.thread_id(1)

    inp = al.make_tensor(in_ptr, al.bf16, al.make_layout((M, N), (N, 1)))
    bias = al.make_tensor(bias_ptr, al.bf16, al.make_layout((N,), (1,)))
    out = al.make_tensor(out_ptr, al.bf16, al.make_layout((M, N), (N, 1)))

    if row < M:
        if col < N:
            val = al.convert(inp[row, col], al.f32) + al.convert(bias[col], al.f32)
            neg_val = al.convert(0.0, al.f32) - val
            exp_neg = al.exp(neg_val)
            one = al.convert(1.0, al.f32)
            sig = one / (one + exp_neg)
            out[row, col] = al.convert(sig, al.bf16)


# ---------------------------------------------------------------------------
# Bias + logsumexp kernel   (one block per row)
# ---------------------------------------------------------------------------
@avelang.jit
def bias_logsumexp_kernel(
    in_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    M: al.i32,
    N: al.i32,
):
    row = al.block_id(0)
    tid = al.thread_id(0)
    bdim = al.block_dim(0)

    inp = al.make_tensor(in_ptr, al.bf16, al.make_layout((M, N), (N, 1)))
    bias = al.make_tensor(bias_ptr, al.bf16, al.make_layout((N,), (1,)))
    out = al.make_tensor(out_ptr, al.bf16, al.make_layout((M,), (1,)))

    smem = al.make_shared((256,), al.f32)

    # Phase 1: max reduction
    my_max = al.convert(0.0, al.f32)
    if row < M:
        if tid < N:
            val0 = al.convert(inp[row, tid], al.f32) + al.convert(bias[tid], al.f32)
            my_max = val0
        for j in al.range(tid + bdim, N, bdim):
            val = al.convert(inp[row, j], al.f32) + al.convert(bias[j], al.f32)
            if val > my_max:
                my_max = val

    smem[tid] = my_max
    al.syncthreads()

    if tid < 128:
        if smem[tid + 128] > smem[tid]:
            smem[tid] = smem[tid + 128]
    al.syncthreads()
    if tid < 64:
        if smem[tid + 64] > smem[tid]:
            smem[tid] = smem[tid + 64]
    al.syncthreads()
    if tid < 32:
        if smem[tid + 32] > smem[tid]:
            smem[tid] = smem[tid + 32]
    al.syncthreads()
    if tid < 16:
        if smem[tid + 16] > smem[tid]:
            smem[tid] = smem[tid + 16]
    al.syncthreads()
    if tid < 8:
        if smem[tid + 8] > smem[tid]:
            smem[tid] = smem[tid + 8]
    al.syncthreads()
    if tid < 4:
        if smem[tid + 4] > smem[tid]:
            smem[tid] = smem[tid + 4]
    al.syncthreads()
    if tid < 2:
        if smem[tid + 2] > smem[tid]:
            smem[tid] = smem[tid + 2]
    al.syncthreads()
    if tid < 1:
        if smem[1] > smem[0]:
            smem[0] = smem[1]
    al.syncthreads()

    row_max = smem[0]

    # Phase 2: sum of exp(inp + bias - row_max)
    my_sum = al.convert(0.0, al.f32)
    if row < M:
        for j in al.range(tid, N, bdim):
            val = al.convert(inp[row, j], al.f32) + al.convert(bias[j], al.f32)
            diff = val - row_max
            my_sum = my_sum + al.exp(diff)

    smem[tid] = my_sum
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
        smem[0] = smem[0] + smem[1]
    al.syncthreads()

    if tid == 0:
        if row < M:
            total_sum = smem[0]
            result = al.log(total_sum) + row_max
            out[row] = al.convert(result, al.bf16)


# ---------------------------------------------------------------------------
# Host helpers
# ---------------------------------------------------------------------------

def _run_gemm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    M, K = A.shape
    N, _ = B.shape
    out = torch.empty((M, N), dtype=torch.bfloat16, device=A.device)

    TILE_M = 64
    TILE_N = 64
    TILE_K = 32

    num_k_steps = (K + TILE_K - 1) // TILE_K
    grid_m = (M + TILE_M - 1) // TILE_M
    grid_n = (N + TILE_N - 1) // TILE_N

    gemm_kernel[lambda: ((grid_m, grid_n, 1), (16, 16, 1))](
        A.contiguous(), B.contiguous(), out,
        M, K, N,
        num_k_steps,
    )
    return out


def _run_bias_sigmoid(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    out = torch.empty_like(x)

    BLOCK_M = 64
    BLOCK_N = 4
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    bias_sigmoid_kernel[lambda: ((grid_m, grid_n, 1), (BLOCK_M, BLOCK_N, 1))](
        x.contiguous(), bias.contiguous(), out, M, N,
    )
    return out


def _run_bias_logsumexp(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    out = torch.empty((M,), dtype=torch.bfloat16, device=x.device)

    bias_logsumexp_kernel[lambda: ((M, 1, 1), (256, 1, 1))](
        x.contiguous(), bias.contiguous(), out, M, N,
    )
    return out


# ---------------------------------------------------------------------------
# ModelNew
# ---------------------------------------------------------------------------

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ModelNew, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.to(dtype=torch.bfloat16).contiguous()
        device = x.device

        w1 = self.linear1.weight.data.to(device=device, dtype=torch.bfloat16)
        b1 = self.linear1.bias.data.to(device=device, dtype=torch.bfloat16)
        w2 = self.linear2.weight.data.to(device=device, dtype=torch.bfloat16)
        b2 = self.linear2.bias.data.to(device=device, dtype=torch.bfloat16)

        h1 = _run_gemm(x, w1)
        h1 = _run_bias_sigmoid(h1, b1)
        h2 = _run_gemm(h1, w2)
        out = _run_bias_logsumexp(h2, b2)

        return out
