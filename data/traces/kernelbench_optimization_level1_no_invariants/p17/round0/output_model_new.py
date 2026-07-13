import torch
import torch.nn as nn

import avelang
import avelang.language as al


M = 2048
K = 8192
N = 4096

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
def _fetch_mfma_operand_u32(
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


@avelang.jit
def gemm_kernel(
    a_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    c_ptr: al.Pointer(al.bf16),
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
    c_out = al.make_tensor(c_ptr, al.bf16, al.make_layout((m, n), (n, 1)))

    a_rsrc = al.amdgpu.make_rsrc(a_memref, m * k * BF16_BYTES)
    b_rsrc = al.amdgpu.make_rsrc(b_memref, n * k * BF16_BYTES)

    shm_a = al.make_shared((SHM_A_VECS, 4), al.u32)
    shm_b = al.make_shared((SHM_B_VECS, 4), al.u32)
    a_reg = al.make_local((M_TILES_PER_WARP, 2, 2), al.u32)
    b_reg = al.make_local((N_TILES_PER_WARP, 2, 2), al.u32)
    acc = al.make_local((M_TILES_PER_WARP * N_TILES_PER_WARP, ACC_SIZE), al.f32)

    for i in al.range(M_TILES_PER_WARP * N_TILES_PER_WARP):
        for j in al.range(ACC_SIZE):
            acc[i, j] = al.convert(0.0, al.f32)

    k_tiles = k // GROUP_K
    for kt in al.range(k_tiles):
        k_base = kt * GROUP_K
        _load_global_a_to_shm(shm_a, a_rsrc, block_m, k_base, k, tid)
        _load_global_b_to_shm(shm_b, b_rsrc, block_n, k_base, k, tid)
        al.syncthreads()

        for i in al.range(M_TILES_PER_WARP):
            _fetch_mfma_operand_u32(a_reg[i], shm_a, warp_row * M_TILES_PER_WARP + i, lane)
        for j in al.range(N_TILES_PER_WARP):
            _fetch_mfma_operand_u32(b_reg[j], shm_b, warp_col * N_TILES_PER_WARP + j, lane)

        for i in al.range(M_TILES_PER_WARP):
            for j in al.range(N_TILES_PER_WARP):
                acc_idx = i * N_TILES_PER_WARP + j
                a_vec0 = al.view(a_reg[i, 0], al.Tensor((2,), al.u32))
                a_vec1 = al.view(a_reg[i, 1], al.Tensor((2,), al.u32))
                b_vec0 = al.view(b_reg[j, 0], al.Tensor((2,), al.u32))
                b_vec1 = al.view(b_reg[j, 1], al.Tensor((2,), al.u32))
                c_vec = al.view(acc[acc_idx], al.Tensor((ACC_SIZE,), al.f32))
                r1 = al.amdgpu.mfma_32x32x8_bf16_f32(a_vec0, b_vec0, c_vec)
                r2 = al.amdgpu.mfma_32x32x8_bf16_f32(a_vec1, b_vec1, r1)
                r_view = al.view(r2, al.Tensor((ACC_SIZE,), al.f32))
                for t in al.range(ACC_SIZE):
                    acc[acc_idx, t] = r_view[t]

        al.syncthreads()

    # Writeback: use the MFMA accumulator invariant mapping
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
                if row < m and col < n:
                    c_out[row, col] = al.convert(acc[acc_idx, t], al.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if tuple(A.shape) != (2048, 8192) or tuple(B.shape) != (4096, 8192):
            raise RuntimeError('Shape/dtype fallback is intentionally disabled for the AveLang optimized kernel.')
        A_bf16 = A.contiguous()
        B_bf16 = B.contiguous()

        M_val = A_bf16.shape[0]
        K_val = A_bf16.shape[1]
        N_val = B_bf16.shape[0]
        C = torch.empty((M_val, N_val), device=A.device, dtype=A.dtype)

        grid_n = N_val // GROUP_N
        grid_m = M_val // GROUP_M

        gemm_kernel[lambda: ((grid_n, grid_m, 1), (THREADS, 1, 1))](
            A_bf16, B_bf16, C,
            M_val, N_val, K_val,
        )  # type: ignore[arg-type]
        return C
