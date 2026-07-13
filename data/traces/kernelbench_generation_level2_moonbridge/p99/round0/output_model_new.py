import torch
import torch.nn as nn
import avelang
import avelang.language as al

# ── GEMM tiling constants ──────────────────────────────────────────────
WARP_SIZE = 64
NUM_WARPS = 4
THREADS = WARP_SIZE * NUM_WARPS
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
M_TILES_PER_WARP = GROUP_M // (WARPS_M * MMA_M)
N_TILES_PER_WARP = GROUP_N // (WARPS_N * MMA_N)
A_VECS_PER_ROW = GROUP_K // VEC_ELEMS
B_VECS_PER_ROW = GROUP_K // VEC_ELEMS
SHM_A_VECS = GROUP_M * A_VECS_PER_ROW
SHM_B_VECS = GROUP_N * B_VECS_PER_ROW
GLOBAL_LOADS_A = SHM_A_VECS // THREADS
GLOBAL_LOADS_B = SHM_B_VECS // THREADS
ROW_U32 = A_VECS_PER_ROW * 4

# ── GELU constants ─────────────────────────────────────────────────────
GELU_SQRT_2_PI = 0.7978845608028654   # sqrt(2 / pi)
GELU_COEFF = 0.044715

# ── Softmax constants ──────────────────────────────────────────────────
BLOCK_SIZE_SM = 256


# ═══════════════════════════════════════════════════════════════════════
# GEMM helper kernels
# ═══════════════════════════════════════════════════════════════════════

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


@avelang.jit
def _fetch_mfma_operand_32x32x16(
    ret: al.Tensor((2, 2), al.u32),
    shm: al.Tensor((SHM_A_VECS, 4), al.u32),
    tile_idx: al.u32,
    lane: al.u32,
):
    shm_u32 = al.view(shm, al.Tensor((SHM_A_VECS * 4,), al.u32))
    row = tile_idx * MMA_M + (lane % MMA_M)
    k_group_u32 = (lane // MMA_M) * 2
    row_base = row * ROW_U32

    ret[0, 0] = shm_u32[row_base + k_group_u32]
    ret[0, 1] = shm_u32[row_base + k_group_u32 + 1]
    ret[1, 0] = shm_u32[row_base + 4 + k_group_u32]
    ret[1, 1] = shm_u32[row_base + 5 + k_group_u32]


# ═══════════════════════════════════════════════════════════════════════
# Fused Linear + GELU GEMM kernel
# ═══════════════════════════════════════════════════════════════════════

@avelang.jit
def linear_gelu_bf16_kernel(
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
    a_reg = al.make_local((M_TILES_PER_WARP, 2, 2), al.u32)
    b_reg = al.make_local((N_TILES_PER_WARP, 2, 2), al.u32)
    acc = al.make_local((M_TILES_PER_WARP * N_TILES_PER_WARP, ACC_SIZE), al.f32)

    for i in al.range(M_TILES_PER_WARP * N_TILES_PER_WARP):
        for j in al.range(ACC_SIZE):
            acc[i, j] = 0

    k_tiles = k // GROUP_K
    for kt in al.range(k_tiles):
        k_base = kt * GROUP_K
        _load_global_a_to_shm(shm_a, x_rsrc, block_m, k_base, k, tid)
        _load_global_b_to_shm(shm_b, w_rsrc, block_n, k_base, k, tid)
        al.syncthreads()

        for i in al.range(M_TILES_PER_WARP):
            _fetch_mfma_operand_32x32x16(a_reg[i], shm_a, warp_row * M_TILES_PER_WARP + i, lane)
        for j in al.range(N_TILES_PER_WARP):
            _fetch_mfma_operand_32x32x16(b_reg[j], shm_b, warp_col * N_TILES_PER_WARP + j, lane)

        for i in al.range(M_TILES_PER_WARP):
            for j in al.range(N_TILES_PER_WARP):
                acc_idx = i * N_TILES_PER_WARP + j
                acc[acc_idx] = al.amdgpu.mfma_32x32x8_bf16_f32(a_reg[i, 0], b_reg[j, 0], acc[acc_idx])
                acc[acc_idx] = al.amdgpu.mfma_32x32x8_bf16_f32(a_reg[i, 1], b_reg[j, 1], acc[acc_idx])

        al.syncthreads()

    # GELU constants
    gelu_factor = al.convert(0.5, al.f32)
    gelu_sqrt2pi = al.convert(GELU_SQRT_2_PI, al.f32)
    gelu_coeff = al.convert(GELU_COEFF, al.f32)
    one = al.convert(1.0, al.f32)

    lane_group = lane // MMA_N
    lane_col = lane % MMA_N
    block_row_base = block_m * GROUP_M
    block_col_base = block_n * GROUP_N

    for j in al.range(N_TILES_PER_WARP):
        col = block_col_base + (warp_col * N_TILES_PER_WARP + j) * MMA_N + lane_col
        bias = al.convert(g_bias[col], al.f32)
        for i in al.range(M_TILES_PER_WARP):
            acc_idx = i * N_TILES_PER_WARP + j
            row_base = block_row_base + (warp_row * M_TILES_PER_WARP + i) * MMA_M
            for t in al.range(ACC_SIZE):
                row = row_base + (t // 4) * 8 + lane_group * 4 + (t % 4)
                result = acc[acc_idx, t] + bias
                # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                x3 = result * result * result
                inner = gelu_sqrt2pi * (result + gelu_coeff * x3)
                gelu = gelu_factor * result * (one + al.tanh(inner))
                g_out[row, col] = al.convert(gelu, al.bf16)


# ═══════════════════════════════════════════════════════════════════════
# Softmax kernel (along dim=1)
# ═══════════════════════════════════════════════════════════════════════

@avelang.jit
def softmax_kernel(
    in_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    M: al.i32,
    N: al.i32,
):
    tid = al.thread_id(0)
    row = al.block_id(0)

    if row >= M:
        return

    smem = al.make_shared((BLOCK_SIZE_SM * 2,), al.f32)

    layout_in = al.make_layout((M * N,), (1,))
    in_t = al.make_tensor(in_ptr, al.bf16, layout_in)

    # ── Step 1: find row max ─────────────────────────────────────────
    row_max = al.convert(in_t[row * N + tid], al.f32)
    for i in al.range(tid + BLOCK_SIZE_SM, N, BLOCK_SIZE_SM):
        val = al.convert(in_t[row * N + i], al.f32)
        if val > row_max:
            row_max = val

    smem[tid] = row_max
    al.syncthreads()

    # Reduction tree for max: 256 → 128 → 64 → 32 → 16 → 8 → 4 → 2 → 1
    if tid < 128:
        other = smem[tid + 128]
        if other > smem[tid]:
            smem[tid] = other
    al.syncthreads()
    if tid < 64:
        other = smem[tid + 64]
        if other > smem[tid]:
            smem[tid] = other
    al.syncthreads()
    if tid < 32:
        other = smem[tid + 32]
        if other > smem[tid]:
            smem[tid] = other
    al.syncthreads()
    if tid < 16:
        other = smem[tid + 16]
        if other > smem[tid]:
            smem[tid] = other
    al.syncthreads()
    if tid < 8:
        other = smem[tid + 8]
        if other > smem[tid]:
            smem[tid] = other
    al.syncthreads()
    if tid < 4:
        other = smem[tid + 4]
        if other > smem[tid]:
            smem[tid] = other
    al.syncthreads()
    if tid < 2:
        other = smem[tid + 2]
        if other > smem[tid]:
            smem[tid] = other
    al.syncthreads()
    if tid < 1:
        other = smem[tid + 1]
        if other > smem[tid]:
            smem[tid] = other
    al.syncthreads()

    row_max = smem[0]

    # ── Step 2: compute sum of exp(x - max) ──────────────────────────
    local_sum = al.convert(0.0, al.f32)
    for i in al.range(tid, N, BLOCK_SIZE_SM):
        val = al.convert(in_t[row * N + i], al.f32)
        local_sum = local_sum + al.exp(val - row_max)

    smem[BLOCK_SIZE_SM + tid] = local_sum
    al.syncthreads()

    # Reduction tree for sum: 256 → ... → 1
    if tid < 128:
        smem[BLOCK_SIZE_SM + tid] = smem[BLOCK_SIZE_SM + tid] + smem[BLOCK_SIZE_SM + tid + 128]
    al.syncthreads()
    if tid < 64:
        smem[BLOCK_SIZE_SM + tid] = smem[BLOCK_SIZE_SM + tid] + smem[BLOCK_SIZE_SM + tid + 64]
    al.syncthreads()
    if tid < 32:
        smem[BLOCK_SIZE_SM + tid] = smem[BLOCK_SIZE_SM + tid] + smem[BLOCK_SIZE_SM + tid + 32]
    al.syncthreads()
    if tid < 16:
        smem[BLOCK_SIZE_SM + tid] = smem[BLOCK_SIZE_SM + tid] + smem[BLOCK_SIZE_SM + tid + 16]
    al.syncthreads()
    if tid < 8:
        smem[BLOCK_SIZE_SM + tid] = smem[BLOCK_SIZE_SM + tid] + smem[BLOCK_SIZE_SM + tid + 8]
    al.syncthreads()
    if tid < 4:
        smem[BLOCK_SIZE_SM + tid] = smem[BLOCK_SIZE_SM + tid] + smem[BLOCK_SIZE_SM + tid + 4]
    al.syncthreads()
    if tid < 2:
        smem[BLOCK_SIZE_SM + tid] = smem[BLOCK_SIZE_SM + tid] + smem[BLOCK_SIZE_SM + tid + 2]
    al.syncthreads()
    if tid < 1:
        smem[BLOCK_SIZE_SM + tid] = smem[BLOCK_SIZE_SM + tid] + smem[BLOCK_SIZE_SM + tid + 1]
    al.syncthreads()

    row_sum = smem[BLOCK_SIZE_SM]

    # ── Step 3: write softmax output ─────────────────────────────────
    layout_out = al.make_layout((M, N), (N, 1))
    out_t = al.make_tensor(out_ptr, al.bf16, layout_out)

    for i in al.range(tid, N, BLOCK_SIZE_SM):
        val = al.convert(in_t[row * N + i], al.f32)
        result = al.exp(val - row_max) / row_sum
        out_t[row, i] = al.convert(result, al.bf16)


# ═══════════════════════════════════════════════════════════════════════
# Host wrapper
# ═══════════════════════════════════════════════════════════════════════

def _prepare_bf16_cuda_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_linear_gelu_softmax(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = _prepare_bf16_cuda_contiguous(x)
    weight_bf16 = _prepare_bf16_cuda_contiguous(weight)
    bias_bf16 = _prepare_bf16_cuda_contiguous(bias)

    m, k = x_bf16.shape
    n, weight_k = weight_bf16.shape
    if weight_k != k:
        raise ValueError(
            f"Weight/input K mismatch: x has K={k}, weight has K={weight_k}"
        )
    if m % GROUP_M != 0 or n % GROUP_N != 0 or k % GROUP_K != 0:
        raise ValueError(
            f"Expected m % {GROUP_M} == 0, n % {GROUP_N} == 0, k % {GROUP_K} == 0 "
            f"(got m={m}, n={n}, k={k})"
        )

    # Step 1: Linear + GELU
    gelu_out = torch.empty((m, n), device=x_bf16.device, dtype=torch.bfloat16)
    grid = (n // GROUP_N, m // GROUP_M, 1)
    linear_gelu_bf16_kernel[lambda: (grid, (THREADS, 1, 1))](
        x_bf16, weight_bf16, bias_bf16, gelu_out, m, n, k
    )

    # Step 2: Softmax along dim=1
    sm_out = torch.empty_like(gelu_out)
    softmax_kernel[lambda: ((m, 1, 1), (BLOCK_SIZE_SM, 1, 1))](
        gelu_out, sm_out, m, n
    )

    return sm_out


# ═══════════════════════════════════════════════════════════════════════
# ModelNew entrypoint
# ═══════════════════════════════════════════════════════════════════════

class ModelNew(nn.Module):
    """
    Optimized BF16 linear + GELU + softmax with custom AveLang AMDGPU kernels.
    """

    def __init__(self, in_features: int, out_features: int):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))

        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_linear_gelu_softmax(x, self.weight, self.bias)
