import torch
import torch.nn as nn
import avelang
import avelang.language as al


# ── Problem dimensions ─────────────────────────────────────────────────
M = 2048
K = 1048576

# ── Tile constants ─────────────────────────────────────────────────────
WARP_SIZE = 64
NUM_WARPS = 4
THREADS = WARP_SIZE * NUM_WARPS
WARPS_M = 2
WARPS_N = 2
MMA_M = 32
MMA_N = 32
GROUP_M = WARPS_M * MMA_M          # 64
GROUP_K = 256
VEC_ELEMS = 8
BF16_BYTES = 2
ACC_SIZE = 16
M_TILES_PER_WARP = GROUP_M // (WARPS_M * MMA_M)  # 1
A_VECS_PER_ROW = GROUP_K // VEC_ELEMS             # 32
B_VECS_PER_ROW = GROUP_K // VEC_ELEMS             # 32
SHM_A_VECS = GROUP_M * A_VECS_PER_ROW             # 2048
SHM_B_VECS = 1 * B_VECS_PER_ROW                   # 32
ROW_U32_A = A_VECS_PER_ROW * 4                    # 128
GLOBAL_LOADS_A = SHM_A_VECS // THREADS             # 8
MFMA_ROUNDS = GROUP_K // 16                        # 16


# ── GEMV kernel ────────────────────────────────────────────────────────

@avelang.jit
def gemv_kernel(
    A_ptr: al.Pointer(al.bf16),
    B_ptr: al.Pointer(al.bf16),
    C_ptr: al.Pointer(al.bf16),
):
    tid = al.thread_id(0)
    block_id_m = al.block_id(0)
    wid = tid // WARP_SIZE
    lane = tid % WARP_SIZE
    warp_row = wid // WARPS_N
    warp_col = wid % WARPS_N
    lane_col = lane % MMA_N
    lane_group = lane // MMA_N

    # Tensor views
    A_memref = al.make_tensor(A_ptr, al.bf16, al.make_layout((M, K), (K, 1)))
    B_memref = al.make_tensor(B_ptr, al.bf16, al.make_layout((K, 1), (1, K)))
    C_out = al.make_tensor(C_ptr, al.bf16, al.make_layout((M, 1), (1, M)))

    # Buffer resources
    A_rsrc = al.amdgpu.make_rsrc(A_memref, M * K * BF16_BYTES - 1)
    B_rsrc = al.amdgpu.make_rsrc(B_memref, K * BF16_BYTES - 1)

    # Shared memory
    shm_a = al.make_shared((SHM_A_VECS, 4), al.u32)
    shm_b = al.make_shared((SHM_B_VECS, 4), al.u32)

    # Registers
    a_reg = al.make_local((2, 4), al.bf16)
    b_reg = al.make_local((2, 4), al.bf16)
    acc = al.make_local((ACC_SIZE,), al.f32)
    for i in al.range(ACC_SIZE):
        acc[i] = al.convert(0.0, al.f32)

    zero = al.convert(0, al.u32)
    k_tiles = K // GROUP_K

    for kt in al.range(k_tiles):
        k_base = kt * GROUP_K

        # Load A: 2048 vectors, 8 per thread
        idx_a = tid
        for _ in al.range(GLOBAL_LOADS_A):
            a_row = idx_a // A_VECS_PER_ROW
            a_col = idx_a % A_VECS_PER_ROW
            a_off = ((block_id_m * GROUP_M + a_row) * K + k_base + a_col * VEC_ELEMS) * BF16_BYTES
            shm_a[idx_a] = al.amdgpu.raw_buffer_load_x4(A_rsrc, zero, a_off, 0)
            idx_a = idx_a + THREADS

        # Load B: 32 vectors, threads 0-31 load
        if tid < SHM_B_VECS:
            b_off = (k_base + tid * VEC_ELEMS) * BF16_BYTES
            shm_b[tid] = al.amdgpu.raw_buffer_load_x4(B_rsrc, zero, b_off, 0)

        al.syncthreads()

        shm_a_flat = al.view(shm_a, al.Tensor((SHM_A_VECS * 4,), al.u32))
        shm_b_flat = al.view(shm_b, al.Tensor((SHM_B_VECS * 4,), al.u32))
        a_row_in_block = warp_row * MMA_M + lane_col
        kg = lane_group * 2

        for rnd in al.range(MFMA_ROUNDS):
            kg_offset = rnd * 8
            a_row_base = a_row_in_block * ROW_U32_A

            a_reg_u32 = al.view(a_reg, al.Tensor((4,), al.u32))
            a_reg_u32[0] = shm_a_flat[a_row_base + kg_offset + kg]
            a_reg_u32[1] = shm_a_flat[a_row_base + kg_offset + kg + 1]
            a_reg_u32[2] = shm_a_flat[a_row_base + kg_offset + 4 + kg]
            a_reg_u32[3] = shm_a_flat[a_row_base + kg_offset + 5 + kg]

            b_reg_u32 = al.view(b_reg, al.Tensor((4,), al.u32))
            b_reg_u32[0] = shm_b_flat[kg_offset + kg]
            b_reg_u32[1] = shm_b_flat[kg_offset + kg + 1]
            b_reg_u32[2] = shm_b_flat[kg_offset + 4 + kg]
            b_reg_u32[3] = shm_b_flat[kg_offset + 5 + kg]

            acc = al.amdgpu.mfma_32x32x8_bf16_f32(
                al.view(a_reg[0], al.Tensor((2,), al.u32)),
                al.view(b_reg[0], al.Tensor((2,), al.u32)),
                acc,
            )
            acc = al.amdgpu.mfma_32x32x8_bf16_f32(
                al.view(a_reg[1], al.Tensor((2,), al.u32)),
                al.view(b_reg[1], al.Tensor((2,), al.u32)),
                acc,
            )

        al.syncthreads()

    # Output writeback
    if warp_col == 0:
        if lane_col == 0:
            global_row_base = block_id_m * GROUP_M + warp_row * MMA_M
            for t in al.range(ACC_SIZE):
                row = global_row_base + (t // 4) * 8 + lane_group * 4 + (t % 4)
                C_out[row, 0] = al.convert(acc[t], al.bf16)


# ── ModelNew ───────────────────────────────────────────────────────────

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if A.shape[0] != M or A.shape[1] != K or B.shape[0] != K or B.shape[1] != 1:
            raise RuntimeError(f'Shape mismatch: expected A=({M},{K}), B=({K},1), got A={tuple(A.shape)}, B={tuple(B.shape)}')
        A = A.contiguous()
        B = B.contiguous()
        if A.dtype != torch.bfloat16:
            A = A.to(torch.bfloat16)
        if B.dtype != torch.bfloat16:
            B = B.to(torch.bfloat16)
        C = torch.empty((M, 1), device=A.device, dtype=torch.bfloat16)
        grid = (M // GROUP_M, 1, 1)
        gemv_kernel[lambda: (grid, (THREADS, 1, 1))](A, B, C)
        return C
