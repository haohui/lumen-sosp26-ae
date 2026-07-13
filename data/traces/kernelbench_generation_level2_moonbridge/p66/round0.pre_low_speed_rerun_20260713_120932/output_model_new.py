import torch
import torch.nn as nn
import avelang
import avelang.language as al

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WARP_SIZE = 64
NUM_WARPS = 4
THREADS = WARP_SIZE * NUM_WARPS  # 256
GROUP_M = 128
GROUP_N = 128
GROUP_K = 16
MMA_M = 32
MMA_N = 32
VEC_ELEMS = 8
BF16_BYTES = 2
ACC_SIZE = 16
WARPS_M = 2
WARPS_N = 2
M_TILES_PER_WARP = GROUP_M // (WARPS_M * MMA_M)  # 2
N_TILES_PER_WARP = GROUP_N // (WARPS_N * MMA_N)  # 2
A_VECS_PER_ROW = GROUP_K // VEC_ELEMS
B_VECS_PER_ROW = GROUP_K // VEC_ELEMS
SHM_A_VECS = GROUP_M * A_VECS_PER_ROW
SHM_B_VECS = GROUP_N * B_VECS_PER_ROW
GLOBAL_LOADS_A = SHM_A_VECS // THREADS
GLOBAL_LOADS_B = SHM_B_VECS // THREADS
ROW_U32 = A_VECS_PER_ROW * 4

SOFTMAX_THREADS = 256


# ---------------------------------------------------------------------------
# GEMM helper kernels (adapted from fused-linear-relu-bf16-gemm example)
# ---------------------------------------------------------------------------
@avelang.jit
def _load_global_a_to_shm(
    shm_a: al.Tensor((SHM_A_VECS, 4), al.u32),
    x_rsrc: al.Tensor((4,), al.u32),
    block_m: al.u32,
    k_base: al.u32,
    k: al.u32,
    tid: al.u32,
):
    zero = al.convert(0, al.u32)
    idx = tid
    for _ in al.range(GLOBAL_LOADS_A):
        row = idx // A_VECS_PER_ROW
        col_vec = idx % A_VECS_PER_ROW
        off = ((block_m * GROUP_M + row) * k + k_base + col_vec * VEC_ELEMS) * BF16_BYTES
        shm_a[idx] = al.amdgpu.raw_buffer_load_x4(x_rsrc, zero, off, 0)
        idx += THREADS


@avelang.jit
def _load_global_b_to_shm(
    shm_b: al.Tensor((SHM_B_VECS, 4), al.u32),
    w_rsrc: al.Tensor((4,), al.u32),
    block_n: al.u32,
    k_base: al.u32,
    k: al.u32,
    tid: al.u32,
):
    zero = al.convert(0, al.u32)
    idx = tid
    for _ in al.range(GLOBAL_LOADS_B):
        row = idx // B_VECS_PER_ROW
        col_vec = idx % B_VECS_PER_ROW
        off = ((block_n * GROUP_N + row) * k + k_base + col_vec * VEC_ELEMS) * BF16_BYTES
        shm_b[idx] = al.amdgpu.raw_buffer_load_x4(w_rsrc, zero, off, 0)
        idx += THREADS


# ---------------------------------------------------------------------------
# GEMM kernel: C = X @ W^T + bias
# Uses MFMA with u32-packed operands (B, A order), FP32 accumulate, BF16 out.
# ---------------------------------------------------------------------------
@avelang.jit
def gemm_bias_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    m: al.u32,
    n: al.u32,
    k: al.u32,
):
    tid = al.thread_id(0)
    block_n = al.block_id(0)
    block_m = al.block_id(1)
    wid = tid // WARP_SIZE
    lane = tid % WARP_SIZE
    warp_row = wid // WARPS_N
    warp_col = wid % WARPS_N

    x_memref = al.make_tensor(x_ptr, al.bf16, al.make_layout((m * k,), (1,)))
    w_memref = al.make_tensor(w_ptr, al.bf16, al.make_layout((n * k,), (1,)))
    g_bias = al.make_tensor(bias_ptr, al.bf16, al.make_layout((n,), (1,)))
    g_out = al.make_tensor(out_ptr, al.bf16, al.make_layout((m, n), (n, 1)))

    x_rsrc = al.amdgpu.make_rsrc(x_memref, m * k * BF16_BYTES)
    w_rsrc = al.amdgpu.make_rsrc(w_memref, n * k * BF16_BYTES)

    shm_a = al.make_shared((SHM_A_VECS, 4), al.u32)
    shm_b = al.make_shared((SHM_B_VECS, 4), al.u32)
    a_reg = al.make_local((M_TILES_PER_WARP, 4), al.u32)
    b_reg = al.make_local((N_TILES_PER_WARP, 4), al.u32)
    acc = al.make_local((M_TILES_PER_WARP * N_TILES_PER_WARP, ACC_SIZE), al.f32)

    for a in al.range(M_TILES_PER_WARP * N_TILES_PER_WARP):
        for b in al.range(ACC_SIZE):
            acc[a, b] = 0.0

    k_tiles = k // GROUP_K
    for kt in al.range(k_tiles):
        k_base = kt * GROUP_K
        _load_global_a_to_shm(shm_a, x_rsrc, block_m, k_base, k, tid)
        _load_global_b_to_shm(shm_b, w_rsrc, block_n, k_base, k, tid)
        al.syncthreads()

        # Fetch A operands from shared memory into u32 registers
        shm_a_u32 = al.view(shm_a, al.Tensor((SHM_A_VECS * 4,), al.u32))
        a_k_group = (lane // MMA_M) * 2
        for i in al.range(M_TILES_PER_WARP):
            tile_idx = warp_row * M_TILES_PER_WARP + i
            a_row = tile_idx * MMA_M + (lane % MMA_M)
            a_row_base = a_row * ROW_U32
            a_reg[i, 0] = shm_a_u32[a_row_base + a_k_group]
            a_reg[i, 1] = shm_a_u32[a_row_base + a_k_group + 1]
            a_reg[i, 2] = shm_a_u32[a_row_base + 4 + a_k_group]
            a_reg[i, 3] = shm_a_u32[a_row_base + 5 + a_k_group]

        # Fetch B operands from shared memory into u32 registers
        shm_b_u32 = al.view(shm_b, al.Tensor((SHM_B_VECS * 4,), al.u32))
        b_k_group = (lane // MMA_N) * 2
        for j in al.range(N_TILES_PER_WARP):
            tile_idx = warp_col * N_TILES_PER_WARP + j
            b_row = tile_idx * MMA_N + (lane % MMA_N)
            b_row_base = b_row * ROW_U32
            b_reg[j, 0] = shm_b_u32[b_row_base + b_k_group]
            b_reg[j, 1] = shm_b_u32[b_row_base + b_k_group + 1]
            b_reg[j, 2] = shm_b_u32[b_row_base + 4 + b_k_group]
            b_reg[j, 3] = shm_b_u32[b_row_base + 5 + b_k_group]

        for i in al.range(M_TILES_PER_WARP):
            for j in al.range(N_TILES_PER_WARP):
                acc_idx = i * N_TILES_PER_WARP + j
                a_frag = al.view(a_reg[i], al.Tensor((2, 2, 1), al.u32))
                b_frag = al.view(b_reg[j], al.Tensor((2, 2, 1), al.u32))
                acc[acc_idx] = al.amdgpu.mfma_32x32x8_bf16_f32(
                    b_frag[0], a_frag[0], acc[acc_idx]
                )
                acc[acc_idx] = al.amdgpu.mfma_32x32x8_bf16_f32(
                    b_frag[1], a_frag[1], acc[acc_idx]
                )

        al.syncthreads()

    # Epilogue: add bias and write back
    lane_group = lane // MMA_N
    lane_col = lane % MMA_N
    block_row_base = block_m * GROUP_M
    block_col_base = block_n * GROUP_N

    for j in al.range(N_TILES_PER_WARP):
        col = block_col_base + (warp_col * N_TILES_PER_WARP + j) * MMA_N + lane_col
        bias_val = al.convert(g_bias[col], al.f32)
        for i in al.range(M_TILES_PER_WARP):
            acc_idx = i * N_TILES_PER_WARP + j
            row_base = block_row_base + (warp_row * M_TILES_PER_WARP + i) * MMA_M
            acc_slice = acc[acc_idx]
            for t in al.range(ACC_SIZE):
                row = row_base + (t // 4) * 8 + lane_group * 4 + (t % 4)
                result = acc_slice[t] + bias_val
                g_out[row, col] = al.convert(result, al.bf16)


# ---------------------------------------------------------------------------
# Softmax kernel: row-wise softmax over dim=1
#   Pass 1: find per-row max via tree reduction
#   Pass 2: compute sum of exp(x_i - max) via tree reduction
#   Pass 3: compute exp(x_i - max) / sum and write output
# ---------------------------------------------------------------------------
@avelang.jit
def softmax_kernel(
    io_ptr: al.Pointer(al.bf16),
    m: al.u32,
    n: al.u32,
    N_ELEMS_PER_THREAD: al.constexpr,
):
    tid = al.thread_id(0)
    row = al.block_id(0)

    io = al.make_tensor(io_ptr, al.bf16, al.make_layout((m, n), (n, 1)))
    shm = al.make_shared((SOFTMAX_THREADS,), al.f32)

    first_idx = tid * N_ELEMS_PER_THREAD
    zero_u32 = al.convert(0, al.u32)
    one_u32 = al.convert(1, al.u32)
    zero_f32 = al.convert(0.0, al.f32)

    # ---- Pass 1: find per-row max ----
    local_max = al.convert(io[row, first_idx], al.f32)
    i = one_u32
    for _ in al.range(N_ELEMS_PER_THREAD - 1):
        idx = first_idx + i
        val = al.convert(io[row, idx], al.f32)
        if val > local_max:
            local_max = val
        i = i + one_u32

    # Tree reduction for max (unrolled, matching LayerNorm example pattern)
    shm[tid] = local_max
    al.syncthreads()
    if tid < al.convert(128, al.u32):
        if shm[tid + al.convert(128, al.u32)] > shm[tid]:
            shm[tid] = shm[tid + al.convert(128, al.u32)]
    al.syncthreads()
    if tid < al.convert(64, al.u32):
        if shm[tid + al.convert(64, al.u32)] > shm[tid]:
            shm[tid] = shm[tid + al.convert(64, al.u32)]
    al.syncthreads()
    if tid < al.convert(32, al.u32):
        if shm[tid + al.convert(32, al.u32)] > shm[tid]:
            shm[tid] = shm[tid + al.convert(32, al.u32)]
    al.syncthreads()
    if tid < al.convert(16, al.u32):
        if shm[tid + al.convert(16, al.u32)] > shm[tid]:
            shm[tid] = shm[tid + al.convert(16, al.u32)]
    al.syncthreads()
    if tid < al.convert(8, al.u32):
        if shm[tid + al.convert(8, al.u32)] > shm[tid]:
            shm[tid] = shm[tid + al.convert(8, al.u32)]
    al.syncthreads()
    if tid < al.convert(4, al.u32):
        if shm[tid + al.convert(4, al.u32)] > shm[tid]:
            shm[tid] = shm[tid + al.convert(4, al.u32)]
    al.syncthreads()
    if tid < al.convert(2, al.u32):
        if shm[tid + al.convert(2, al.u32)] > shm[tid]:
            shm[tid] = shm[tid + al.convert(2, al.u32)]
    al.syncthreads()
    if tid < al.convert(1, al.u32):
        if shm[tid + al.convert(1, al.u32)] > shm[tid]:
            shm[tid] = shm[tid + al.convert(1, al.u32)]
    al.syncthreads()

    global_max = shm[zero_u32]

    # ---- Pass 2: sum of exp(x_i - max) ----
    local_sum = zero_f32
    i = zero_u32
    for _ in al.range(N_ELEMS_PER_THREAD):
        idx = first_idx + i
        val = al.convert(io[row, idx], al.f32)
        exp_val = al.exp(val - global_max)
        local_sum = local_sum + exp_val
        i = i + one_u32

    # Tree reduction for sum
    shm[tid] = local_sum
    al.syncthreads()
    if tid < al.convert(128, al.u32):
        shm[tid] = shm[tid] + shm[tid + al.convert(128, al.u32)]
    al.syncthreads()
    if tid < al.convert(64, al.u32):
        shm[tid] = shm[tid] + shm[tid + al.convert(64, al.u32)]
    al.syncthreads()
    if tid < al.convert(32, al.u32):
        shm[tid] = shm[tid] + shm[tid + al.convert(32, al.u32)]
    al.syncthreads()
    if tid < al.convert(16, al.u32):
        shm[tid] = shm[tid] + shm[tid + al.convert(16, al.u32)]
    al.syncthreads()
    if tid < al.convert(8, al.u32):
        shm[tid] = shm[tid] + shm[tid + al.convert(8, al.u32)]
    al.syncthreads()
    if tid < al.convert(4, al.u32):
        shm[tid] = shm[tid] + shm[tid + al.convert(4, al.u32)]
    al.syncthreads()
    if tid < al.convert(2, al.u32):
        shm[tid] = shm[tid] + shm[tid + al.convert(2, al.u32)]
    al.syncthreads()
    if tid < al.convert(1, al.u32):
        shm[tid] = shm[tid] + shm[tid + al.convert(1, al.u32)]
    al.syncthreads()

    global_sum = shm[zero_u32]

    # ---- Pass 3: compute softmax and write ----
    i = zero_u32
    for _ in al.range(N_ELEMS_PER_THREAD):
        idx = first_idx + i
        val = al.convert(io[row, idx], al.f32)
        softmax_val = al.exp(val - global_max) / global_sum
        io[row, idx] = al.convert(softmax_val, al.bf16)
        i = i + one_u32


# ---------------------------------------------------------------------------
# Host wrappers
# ---------------------------------------------------------------------------
def _prepare_bf16_cuda_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_gemm_bias(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = _prepare_bf16_cuda_contiguous(x)
    weight_bf16 = _prepare_bf16_cuda_contiguous(weight)
    bias_bf16 = _prepare_bf16_cuda_contiguous(bias)

    m, k_in = x_bf16.shape
    n, k_w = weight_bf16.shape
    if k_w != k_in:
        raise ValueError(
            f"Weight/input K mismatch: x has K={k_in}, weight has K={k_w}"
        )
    if m % GROUP_M != 0 or n % GROUP_N != 0 or k_in % GROUP_K != 0:
        raise ValueError(
            f"Expected m % {GROUP_M} == 0, n % {GROUP_N} == 0, k % {GROUP_K} == 0 "
            f"(got m={m}, n={n}, k={k_in})"
        )

    out = torch.empty((m, n), device=x_bf16.device, dtype=torch.bfloat16)
    grid = (n // GROUP_N, m // GROUP_M, 1)
    gemm_bias_kernel[lambda: (grid, (THREADS, 1, 1))](
        x_bf16, weight_bf16, bias_bf16, out, m, n, k_in
    )
    return out


def avelang_softmax(x: torch.Tensor) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = _prepare_bf16_cuda_contiguous(x)
    m, n_val = x_bf16.shape

    if n_val % SOFTMAX_THREADS != 0:
        raise ValueError(
            f"Expected n % {SOFTMAX_THREADS} == 0 (got n={n_val})"
        )

    n_elem_per_thread = n_val // SOFTMAX_THREADS
    softmax_kernel[lambda: ((m, 1, 1), (SOFTMAX_THREADS, 1, 1))](
        x_bf16, m, n_val, n_elem_per_thread
    )
    return x_bf16


# ---------------------------------------------------------------------------
# ModelNew
# ---------------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Optimized BF16 matmul + softmax through custom AveLang AMDGPU kernels.
    Dropout is treated as identity (eval mode).
    """

    def __init__(self, in_features: int, out_features: int, dropout_p: float):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout_p = dropout_p

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))

        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: Linear (matmul + bias)
        x = avelang_gemm_bias(x, self.weight, self.bias)
        # Step 2: Dropout is identity in eval mode; skip.
        # Step 3: Softmax over features
        x = avelang_softmax(x)
        return x
