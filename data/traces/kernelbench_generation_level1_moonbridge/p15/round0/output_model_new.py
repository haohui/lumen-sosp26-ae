import torch
import torch.nn as nn
import avelang
import avelang.language as al

# Tile configuration
GROUP_M = 128
GROUP_N = 128
GROUP_K = 32
WARP_SIZE = 64
NUM_WARPS = 4
THREADS = WARP_SIZE * NUM_WARPS
MMA_M = 32
MMA_N = 32
MMA_K = 8
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
K_BATCHES = GROUP_K // (MMA_K * 2)


@avelang.jit
def _load_global_a_to_shm(
    shm_a: al.Tensor((SHM_A_VECS, 4), al.u32),
    a_rsrc: al.Tensor((4,), al.u32),
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
        shm_a[idx] = al.amdgpu.raw_buffer_load_x4(a_rsrc, zero, off, 0)
        idx += THREADS


@avelang.jit
def _load_global_b_to_shm(
    shm_b: al.Tensor((SHM_B_VECS, 4), al.u32),
    b_rsrc: al.Tensor((4,), al.u32),
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
        shm_b[idx] = al.amdgpu.raw_buffer_load_x4(b_rsrc, zero, off, 0)
        idx += THREADS


@avelang.jit
def _fetch_mfma_operand_32x32x16(
    ret: al.Tensor((2, 4), al.bf16),
    shm: al.Tensor((SHM_A_VECS, 4), al.u32),
    tile_idx: al.u32,
    lane: al.u32,
    k_batch: al.u32,
):
    ret_u32 = al.view(ret, al.Tensor((4,), al.u32))
    shm_u32 = al.view(shm, al.Tensor((SHM_A_VECS * 4,), al.u32))
    row = tile_idx * MMA_M + (lane % MMA_M)
    k_group_u32 = (lane // MMA_M) * 2
    row_base = row * ROW_U32 + k_batch * 8

    ret_u32[0] = shm_u32[row_base + k_group_u32]
    ret_u32[1] = shm_u32[row_base + k_group_u32 + 1]
    ret_u32[2] = shm_u32[row_base + 4 + k_group_u32]
    ret_u32[3] = shm_u32[row_base + 5 + k_group_u32]


@avelang.jit
def tril_matmul_kernel(
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
    a_reg = al.make_local((M_TILES_PER_WARP, K_BATCHES, 2, 4), al.bf16)
    b_reg = al.make_local((N_TILES_PER_WARP, K_BATCHES, 2, 4), al.bf16)
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

        for kb in al.range(K_BATCHES):
            for i in al.range(M_TILES_PER_WARP):
                _fetch_mfma_operand_32x32x16(a_reg[i, kb], shm_a, warp_row * M_TILES_PER_WARP + i, lane, kb)
            for j in al.range(N_TILES_PER_WARP):
                _fetch_mfma_operand_32x32x16(b_reg[j, kb], shm_b, warp_col * N_TILES_PER_WARP + j, lane, kb)

        for kb in al.range(K_BATCHES):
            for i in al.range(M_TILES_PER_WARP):
                for j in al.range(N_TILES_PER_WARP):
                    acc_idx = i * N_TILES_PER_WARP + j
                    a_op0 = al.view(a_reg[i, kb, 0], al.Tensor((2,), al.u32))
                    b_op0 = al.view(b_reg[j, kb, 0], al.Tensor((2,), al.u32))
                    a_op1 = al.view(a_reg[i, kb, 1], al.Tensor((2,), al.u32))
                    b_op1 = al.view(b_reg[j, kb, 1], al.Tensor((2,), al.u32))
                    acc[acc_idx] = al.amdgpu.mfma_32x32x8_bf16_f32(a_op0, b_op0, acc[acc_idx])
                    acc[acc_idx] = al.amdgpu.mfma_32x32x8_bf16_f32(a_op1, b_op1, acc[acc_idx])

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
                result = acc[acc_idx, t]
                if row >= col:
                    g_out[row, col] = al.convert(result, al.bf16)


def _prepare_bf16_cuda_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_tril_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    A_bf16 = _prepare_bf16_cuda_contiguous(A)
    B_bf16 = _prepare_bf16_cuda_contiguous(B)

    m, k_a = A_bf16.shape
    k_b, n = B_bf16.shape
    if k_a != k_b:
        raise ValueError(f"K dimension mismatch: A has K={k_a}, B has K={k_b}")
    k = k_a

    if m % GROUP_M != 0 or n % GROUP_N != 0 or k % GROUP_K != 0:
        raise ValueError(
            f"Expected m % {GROUP_M} == 0, n % {GROUP_N} == 0, k % {GROUP_K} == 0 "
            f"(got m={m}, n={n}, k={k})"
        )

    B_t = B_bf16.T.contiguous()

    out = torch.zeros((m, n), device=A_bf16.device, dtype=torch.bfloat16)

    grid = (n // GROUP_N, m // GROUP_M, 1)
    tril_matmul_kernel[lambda: (grid, (THREADS, 1, 1))](
        A_bf16, B_t, out, m, n, k
    )
    return out


M = 4096


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        return avelang_tril_matmul(A, B)


def get_inputs():
    A = torch.rand(M, M)
    B = torch.rand(M, M)
    A = torch.tril(A)
    B = torch.tril(B)
    return [A, B]


def get_init_inputs():
    return []
