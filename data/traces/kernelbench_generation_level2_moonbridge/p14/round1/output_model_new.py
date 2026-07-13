import torch
import torch.nn as nn
import avelang
import avelang.language as al

# ---------------------------------------------------------------------------
# Problem dimensions
# ---------------------------------------------------------------------------
batch_size = 1024
input_size = 8192
hidden_size = 8192
scaling_factor = 1.5

# ---------------------------------------------------------------------------
# GEMM tile parameters
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
A_VECS_PER_ROW = GROUP_K // VEC_ELEMS  # 2
B_VECS_PER_ROW = GROUP_K // VEC_ELEMS  # 2
SHM_A_VECS = GROUP_M * A_VECS_PER_ROW  # 256
SHM_B_VECS = GROUP_N * B_VECS_PER_ROW  # 256
GLOBAL_LOADS_A = SHM_A_VECS // THREADS  # 1
GLOBAL_LOADS_B = SHM_B_VECS // THREADS  # 1
ROW_U32 = A_VECS_PER_ROW * 4  # 8

# ---------------------------------------------------------------------------
# Reduction block size
# ---------------------------------------------------------------------------
REDUCE_BLOCK_SIZE: al.constexpr = 256


# ===========================================================================
# GEMM helper kernels
# ===========================================================================
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
    ret: al.Tensor((2, 4), al.bf16),
    shm: al.Tensor((SHM_A_VECS, 4), al.u32),
    tile_idx: al.u32,
    lane: al.u32,
):
    ret_u32 = al.view(ret, al.Tensor((4,), al.u32))
    shm_u32 = al.view(shm, al.Tensor((SHM_A_VECS * 4,), al.u32))
    row = tile_idx * MMA_M + (lane % MMA_M)
    k_group_u32 = (lane // MMA_M) * 2
    row_base = row * ROW_U32

    ret_u32[0] = shm_u32[row_base + k_group_u32]
    ret_u32[1] = shm_u32[row_base + k_group_u32 + 1]
    ret_u32[2] = shm_u32[row_base + 4 + k_group_u32]
    ret_u32[3] = shm_u32[row_base + 5 + k_group_u32]


# ===========================================================================
# GEMM + divide-by-2 kernel
# ===========================================================================
@avelang.jit
def gemm_div2_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
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
    g_out = al.make_tensor(out_ptr, al.bf16, al.make_layout((m, n), (n, 1)))

    x_rsrc = al.amdgpu.make_rsrc(x_memref, m * k * BF16_BYTES)
    w_rsrc = al.amdgpu.make_rsrc(w_memref, n * k * BF16_BYTES)

    shm_a = al.make_shared((SHM_A_VECS, 4), al.u32)
    shm_b = al.make_shared((SHM_B_VECS, 4), al.u32)
    a_reg = al.make_local((M_TILES_PER_WARP, 2, 4), al.bf16)
    b_reg = al.make_local((N_TILES_PER_WARP, 2, 4), al.bf16)
    acc = al.make_local((M_TILES_PER_WARP * N_TILES_PER_WARP, ACC_SIZE), al.f32)

    for i in al.range(M_TILES_PER_WARP * N_TILES_PER_WARP):
        for j in al.range(ACC_SIZE):
            acc[i, j] = 0.0

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
                a0 = al.view(a_reg[i, 0], al.Tensor((2,), al.u32))
                b0 = al.view(b_reg[j, 0], al.Tensor((2,), al.u32))
                acc[acc_idx] = al.amdgpu.mfma_32x32x8_bf16_f32(a0, b0, acc[acc_idx])
                a1 = al.view(a_reg[i, 1], al.Tensor((2,), al.u32))
                b1 = al.view(b_reg[j, 1], al.Tensor((2,), al.u32))
                acc[acc_idx] = al.amdgpu.mfma_32x32x8_bf16_f32(a1, b1, acc[acc_idx])

        al.syncthreads()

    # Epilogue: divide by 2.0
    div_val = al.convert(2.0, al.f32)
    lane_group = lane // MMA_N
    lane_col = lane % MMA_N
    block_row_base = block_m * GROUP_M
    block_col_base = block_n * GROUP_N

    for j in al.range(N_TILES_PER_WARP):
        col = block_col_base + (warp_col * N_TILES_PER_WARP + j) * MMA_N + lane_col
        for i in al.range(M_TILES_PER_WARP):
            acc_idx = i * N_TILES_PER_WARP + j
            row_base = block_row_base + (warp_row * M_TILES_PER_WARP + i) * MMA_M
            for t in al.range(ACC_SIZE):
                row = row_base + (t // 4) * 8 + lane_group * 4 + (t % 4)
                result = acc[acc_idx, t] / div_val
                g_out[row, col] = al.convert(result, al.bf16)


# ===========================================================================
# Row-wise sum reduction (BF16 output, matching reference torch.sum)
# ===========================================================================
@avelang.jit
def row_sum_kernel(
    in_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    m: al.u32,
    n: al.u32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    if bid < m:
        smem = al.make_shared((REDUCE_BLOCK_SIZE,), al.f32)

        layout_in = al.make_layout((m, n), (n, 1))
        g_in = al.make_tensor(in_ptr, al.bf16, layout_in)

        local_sum = al.convert(0.0, al.f32)
        for i in al.range(tid, n, REDUCE_BLOCK_SIZE):
            local_sum = local_sum + al.convert(g_in[bid, i], al.f32)

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
        if tid < 1:
            smem[tid] = smem[tid] + smem[tid + 1]

        if tid == 0:
            sum_bf16 = al.convert(smem[0], al.bf16)
            layout_out = al.make_layout((m,), (1,))
            g_out = al.make_tensor(out_ptr, al.bf16, layout_out)
            g_out[bid] = sum_bf16


# ===========================================================================
# Scale kernel (BF16 multiplication, matching reference semantics)
# ===========================================================================
@avelang.jit
def scale_kernel(
    in_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    m: al.u32,
    scale: al.f32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    if bid < m:
        layout_in = al.make_layout((m,), (1,))
        g_in = al.make_tensor(in_ptr, al.bf16, layout_in)
        layout_out = al.make_layout((m,), (1,))
        g_out = al.make_tensor(out_ptr, al.bf16, layout_out)

        if tid == 0:
            scale_bf16 = al.convert(scale, al.bf16)
            g_out[bid] = g_in[bid] * scale_bf16


# ===========================================================================
# Host wrappers
# ===========================================================================
def _prepare_bf16_cuda_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_gemm_div2_sum_scale(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = _prepare_bf16_cuda_contiguous(x)
    weight_bf16 = _prepare_bf16_cuda_contiguous(weight)

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

    # Step 1: GEMM + divide-by-2 → BF16 intermediate
    intermediate = torch.empty((m, n), device=x_bf16.device, dtype=torch.bfloat16)
    grid = (n // GROUP_N, m // GROUP_M, 1)
    gemm_div2_kernel[lambda: (grid, (THREADS, 1, 1))](
        x_bf16, weight_bf16, intermediate, m, n, k
    )

    # Step 2: row-wise sum → BF16 output (matching torch.sum on BF16)
    sum_out = torch.empty((m,), device=x_bf16.device, dtype=torch.bfloat16)
    reduce_grid = (m, 1, 1)
    row_sum_kernel[lambda: (reduce_grid, (REDUCE_BLOCK_SIZE, 1, 1))](
        intermediate, sum_out, m, n
    )

    # Step 3: scaling in BF16 (matching reference x * scaling_factor)
    out = torch.empty((m,), device=x_bf16.device, dtype=torch.bfloat16)
    scale_kernel[lambda: (reduce_grid, (REDUCE_BLOCK_SIZE, 1, 1))](
        sum_out, out, m, float(scale)
    )

    return out.view(m, 1)


# ===========================================================================
# ModelNew
# ===========================================================================
class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.scaling_factor = scaling_factor
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))

    def forward(self, x):
        return avelang_gemm_div2_sum_scale(x, self.weight, self.scaling_factor)


# ---------------------------------------------------------------------------
# Input contract (must match input_model.py)
# ---------------------------------------------------------------------------
def get_inputs():
    return [torch.rand(batch_size, input_size)]


def get_init_inputs():
    return [input_size, hidden_size, scaling_factor]
