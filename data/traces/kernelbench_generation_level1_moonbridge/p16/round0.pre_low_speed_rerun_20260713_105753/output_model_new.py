import torch
import torch.nn as nn
import avelang
import avelang.language as al


M = 1024 * 2
K = 4096 * 2
N = 2048 * 2

WARP_SIZE = 64
NUM_WARPS = 4
THREADS = WARP_SIZE * NUM_WARPS
GROUP_M = 128
GROUP_N = 128
GROUP_K = 32
MMA_M = 32
MMA_N = 32
VEC_ELEMS = 8
BF16_BYTES = 2
ACC_SIZE = 16
WARPS_M = 2
WARPS_N = 2
M_TILES_PER_WARP = GROUP_M // (WARPS_M * MMA_M)
N_TILES_PER_WARP = GROUP_N // (WARPS_N * MMA_N)
NUM_16K_HALVES = GROUP_K // 16  # 2 for GROUP_K=32
A_VECS_PER_ROW = GROUP_K // VEC_ELEMS
B_VECS_PER_ROW = GROUP_K // VEC_ELEMS
SHM_A_VECS = GROUP_M * A_VECS_PER_ROW
SHM_B_VECS = GROUP_N * B_VECS_PER_ROW
GLOBAL_LOADS_A = SHM_A_VECS // THREADS
GLOBAL_LOADS_B = SHM_B_VECS // THREADS
ROW_U32 = A_VECS_PER_ROW * 4
HALF16_STRIDE_U32 = 16 * BF16_BYTES // 4  # 16 bf16 elements = 8 u32 per 16K block
VEC_U32 = VEC_ELEMS * BF16_BYTES // 4


@avelang.jit
def _load_global_a_to_shm(
    shm_a: al.Tensor((SHM_A_VECS, 4), al.u32),
    a_rsrc: al.Tensor((4,), al.u32),
    block_m: al.u32,
    k_base: al.u32,
    k_stride: al.u32,
    tid: al.u32,
):
    zero = al.convert(0, al.u32)
    idx = tid
    for _ in al.range(GLOBAL_LOADS_A):
        row = idx // A_VECS_PER_ROW
        col_vec = idx % A_VECS_PER_ROW
        off = ((block_m * GROUP_M + row) * k_stride + k_base + col_vec * VEC_ELEMS) * BF16_BYTES
        shm_a[idx] = al.amdgpu.raw_buffer_load_x4(a_rsrc, zero, off, 0)
        idx += THREADS


@avelang.jit
def _load_global_b_to_shm(
    shm_b: al.Tensor((SHM_B_VECS, 4), al.u32),
    b_rsrc: al.Tensor((4,), al.u32),
    block_n: al.u32,
    k_base: al.u32,
    k_stride: al.u32,
    tid: al.u32,
):
    zero = al.convert(0, al.u32)
    idx = tid
    for _ in al.range(GLOBAL_LOADS_B):
        row = idx // B_VECS_PER_ROW
        col_vec = idx % B_VECS_PER_ROW
        off = ((block_n * GROUP_N + row) * k_stride + k_base + col_vec * VEC_ELEMS) * BF16_BYTES
        shm_b[idx] = al.amdgpu.raw_buffer_load_x4(b_rsrc, zero, off, 0)
        idx += THREADS


@avelang.jit
def _fetch_mfma_operand_32x32x16_v2(
    ret: al.Tensor((2, 4), al.bf16),
    shm: al.Tensor((SHM_A_VECS, 4), al.u32),
    tile_idx: al.u32,
    lane: al.u32,
    half16_idx: al.u32,
):
    ret_u32 = al.view(ret, al.Tensor((4,), al.u32))
    shm_u32 = al.view(shm, al.Tensor((SHM_A_VECS * 4,), al.u32))
    row = tile_idx * MMA_M + (lane % MMA_M)
    k_group_u32 = (lane // MMA_M) * 2
    row_base = row * ROW_U32 + half16_idx * HALF16_STRIDE_U32

    ret_u32[0] = shm_u32[row_base + k_group_u32]
    ret_u32[1] = shm_u32[row_base + k_group_u32 + 1]
    ret_u32[2] = shm_u32[row_base + VEC_U32 + k_group_u32]
    ret_u32[3] = shm_u32[row_base + VEC_U32 + k_group_u32 + 1]


@avelang.jit
def gemm_bf16_kernel(
    a_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
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

    a_memref = al.make_tensor(a_ptr, al.bf16, al.make_layout((m * k,), (1,)))
    b_memref = al.make_tensor(b_ptr, al.bf16, al.make_layout((n * k,), (1,)))
    g_out = al.make_tensor(out_ptr, al.bf16, al.make_layout((m, n), (n, 1)))

    a_rsrc = al.amdgpu.make_rsrc(a_memref, m * k * BF16_BYTES)
    b_rsrc = al.amdgpu.make_rsrc(b_memref, n * k * BF16_BYTES)

    shm_a = al.make_shared((SHM_A_VECS, 4), al.u32)
    shm_b = al.make_shared((SHM_B_VECS, 4), al.u32)
    a_reg = al.make_local((M_TILES_PER_WARP, NUM_16K_HALVES, 2, 4), al.bf16)
    b_reg = al.make_local((N_TILES_PER_WARP, NUM_16K_HALVES, 2, 4), al.bf16)
    acc = al.make_local((M_TILES_PER_WARP * N_TILES_PER_WARP, ACC_SIZE), al.f32)

    for i in al.range(M_TILES_PER_WARP * N_TILES_PER_WARP):
        for j in al.range(ACC_SIZE):
            acc[i, j] = 0

    k_tiles = k // GROUP_K
    for kt in al.range(k_tiles):
        k_base = kt * GROUP_K
        _load_global_a_to_shm(shm_a, a_rsrc, block_m, k_base, k, tid)
        _load_global_b_to_shm(shm_b, b_rsrc, block_n, k_base, k, tid)
        al.syncthreads()

        for i in al.range(M_TILES_PER_WARP):
            _fetch_mfma_operand_32x32x16_v2(a_reg[i, 0], shm_a, warp_row * M_TILES_PER_WARP + i, lane, 0)
            _fetch_mfma_operand_32x32x16_v2(a_reg[i, 1], shm_a, warp_row * M_TILES_PER_WARP + i, lane, 1)
        for j in al.range(N_TILES_PER_WARP):
            _fetch_mfma_operand_32x32x16_v2(b_reg[j, 0], shm_b, warp_col * N_TILES_PER_WARP + j, lane, 0)
            _fetch_mfma_operand_32x32x16_v2(b_reg[j, 1], shm_b, warp_col * N_TILES_PER_WARP + j, lane, 1)

        for i in al.range(M_TILES_PER_WARP):
            for j in al.range(N_TILES_PER_WARP):
                acc_idx = i * N_TILES_PER_WARP + j
                # 16K block 0: halves 0 and 1
                a_low0 = al.view(a_reg[i, 0, 0], al.Tensor((2,), al.u32))
                b_low0 = al.view(b_reg[j, 0, 0], al.Tensor((2,), al.u32))
                a_high0 = al.view(a_reg[i, 0, 1], al.Tensor((2,), al.u32))
                b_high0 = al.view(b_reg[j, 0, 1], al.Tensor((2,), al.u32))
                acc[acc_idx] = al.amdgpu.mfma_32x32x8_bf16_f32(a_low0, b_low0, acc[acc_idx])
                acc[acc_idx] = al.amdgpu.mfma_32x32x8_bf16_f32(a_high0, b_high0, acc[acc_idx])
                # 16K block 1: halves 0 and 1
                a_low1 = al.view(a_reg[i, 1, 0], al.Tensor((2,), al.u32))
                b_low1 = al.view(b_reg[j, 1, 0], al.Tensor((2,), al.u32))
                a_high1 = al.view(a_reg[i, 1, 1], al.Tensor((2,), al.u32))
                b_high1 = al.view(b_reg[j, 1, 1], al.Tensor((2,), al.u32))
                acc[acc_idx] = al.amdgpu.mfma_32x32x8_bf16_f32(a_low1, b_low1, acc[acc_idx])
                acc[acc_idx] = al.amdgpu.mfma_32x32x8_bf16_f32(a_high1, b_high1, acc[acc_idx])

        al.syncthreads()

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
                g_out[row, col] = al.convert(acc[acc_idx, t], al.bf16)


def _prepare_bf16_cuda_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute C = A^T @ B where A is (K, M) and B is (K, N).

    Pre-transposes both inputs on the host so the kernel sees them as
    (M, K) and (N, K) row-major matrices, then computes the standard
    GEMM C = A_T @ B_T^T.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    a_t = a.T.contiguous()
    b_t = b.T.contiguous()

    a_bf16 = _prepare_bf16_cuda_contiguous(a_t)
    b_bf16 = _prepare_bf16_cuda_contiguous(b_t)

    m_val, k_val = a_bf16.shape
    n_val, bk_val = b_bf16.shape
    if bk_val != k_val:
        raise ValueError(
            f"K mismatch: A_T has K={k_val}, B_T has K={bk_val}"
        )
    if m_val % GROUP_M != 0 or n_val % GROUP_N != 0 or k_val % GROUP_K != 0:
        raise ValueError(
            f"Expected m % {GROUP_M} == 0, n % {GROUP_N} == 0, k % {GROUP_K} == 0 "
            f"(got m={m_val}, n={n_val}, k={k_val})"
        )

    out = torch.empty((m_val, n_val), device=a_bf16.device, dtype=torch.bfloat16)
    grid = (n_val // GROUP_N, m_val // GROUP_M, 1)
    gemm_bf16_kernel[lambda: (grid, (THREADS, 1, 1))](
        a_bf16, b_bf16, out, m_val, n_val, k_val
    )
    return out


class ModelNew(nn.Module):
    """Optimized BF16 GEMM kernel computing C = matmul(A.T, B)."""
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return avelang_gemm(A, B)


def get_inputs():
    A = torch.rand(K, M)
    B = torch.rand(K, N)
    return [A, B]


def get_init_inputs():
    return []
