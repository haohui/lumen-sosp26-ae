import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_M = 64
BLOCK_K = 128


@avelang.jit
def gemv_mfma_kernel(
    A_ptr: al.Pointer(al.bf16),
    B_ptr: al.Pointer(al.bf16),
    C_ptr: al.Pointer(al.bf16),
    M: al.i32,
    K: al.i32,
    stride_A_m: al.i32,
    stride_C_m: al.i32,
):
    tid = al.thread_id(0)
    block_m = al.block_id(0)

    warp_id = tid // 64
    warp_m = warp_id // 2
    warp_n = warp_id % 2
    lane_id = tid % 64

    m_offset = block_m * BLOCK_M + warp_m * 32

    A_layout = al.make_layout((M, K), (stride_A_m, 1))
    A = al.make_tensor(A_ptr, al.bf16, A_layout)

    B_layout = al.make_layout((K, 1), (1, 1))
    B = al.make_tensor(B_ptr, al.bf16, B_layout)

    C_layout = al.make_layout((M, 1), (stride_C_m, 1))
    C = al.make_tensor(C_ptr, al.bf16, C_layout)

    # Double-buffered shared memory
    A_lds_0 = al.make_shared((BLOCK_M, BLOCK_K), al.bf16)
    B_lds_0 = al.make_shared((BLOCK_K,), al.bf16)
    A_lds_1 = al.make_shared((BLOCK_M, BLOCK_K), al.bf16)
    B_lds_1 = al.make_shared((BLOCK_K,), al.bf16)

    acc = al.make_local((1, 16), al.f32)
    for ii in al.range(16):
        acc[0, ii] = al.convert(0.0, al.f32)

    a_buf0 = al.make_local((4,), al.bf16)
    b_buf0 = al.make_local((4,), al.bf16)
    a_buf1 = al.make_local((4,), al.bf16)
    b_buf1 = al.make_local((4,), al.bf16)

    a_k_off = (lane_id // 32) * 4
    b_j = lane_id % 8
    a_row = warp_m * 32 + (lane_id if lane_id < 32 else lane_id - 32)

    local_row_a = tid // 4
    local_col_a = (tid % 4) * 32

    # ============================================================
    # Prologue: load tile 0 into buf0
    # ============================================================
    for jj in al.range(32):
        A_lds_0[local_row_a, local_col_a + jj] = A[
            block_m * BLOCK_M + local_row_a, local_col_a + jj
        ]
    if warp_n == 0:
        if warp_m == 0:
            for ld in al.range(2):
                idx = lane_id * 2 + ld
                if idx < BLOCK_K:
                    B_lds_0[idx] = B[idx, 0]
    al.syncthreads()

    # ============================================================
    # Main pipeline: process tile pairs
    # kk is the K-offset of the current odd tile
    # Part A: load tile[kk] → buf1, compute tile[kk-BLOCK_K] from buf0
    # Part B: load tile[kk+BLOCK_K] → buf0, compute tile[kk] from buf1
    #
    # epi_buf tracks which buffer holds the uncomputed tile after
    # the loop: 0 = buf0, 1 = buf1
    # ============================================================
    two_blk = BLOCK_K + BLOCK_K
    epi_buf = al.convert(0, al.i32)

    for kk in al.range(BLOCK_K, K, two_blk):
        # --- Part A ---
        for jj in al.range(32):
            A_lds_1[local_row_a, local_col_a + jj] = A[
                block_m * BLOCK_M + local_row_a, kk + local_col_a + jj
            ]
        if warp_n == 0 and warp_m == 0:
            for ld in al.range(2):
                idx = lane_id * 2 + ld
                if idx < BLOCK_K:
                    B_lds_1[idx] = B[kk + idx, 0]
        al.syncthreads()

        # Compute buf0: K-loop unrolled by 2 (step 16)
        for k_sub in al.range(0, BLOCK_K, 16):
            for jj in al.range(4):
                a_buf0[jj] = A_lds_0[a_row, k_sub + a_k_off + jj]
            b_val = B_lds_0[k_sub + b_j]
            for jj in al.range(4):
                b_buf0[jj] = b_val
            acc[0] = al.amdgpu.mfma_32x32x8_bf16_f32(
                al.view(a_buf0, al.Tensor((2,), al.u32)),
                al.view(b_buf0, al.Tensor((2,), al.u32)),
                acc[0],
            )
            for jj in al.range(4):
                a_buf1[jj] = A_lds_0[a_row, k_sub + 8 + a_k_off + jj]
            b_val = B_lds_0[k_sub + 8 + b_j]
            for jj in al.range(4):
                b_buf1[jj] = b_val
            acc[0] = al.amdgpu.mfma_32x32x8_bf16_f32(
                al.view(a_buf1, al.Tensor((2,), al.u32)),
                al.view(b_buf1, al.Tensor((2,), al.u32)),
                acc[0],
            )

        # --- Part B (if another even tile exists) ---
        kk2 = kk + BLOCK_K
        if kk2 < K:
            for jj in al.range(32):
                A_lds_0[local_row_a, local_col_a + jj] = A[
                    block_m * BLOCK_M + local_row_a, kk2 + local_col_a + jj
                ]
            if warp_n == 0 and warp_m == 0:
                for ld in al.range(2):
                    idx = lane_id * 2 + ld
                    if idx < BLOCK_K:
                        B_lds_0[idx] = B[kk2 + idx, 0]
            al.syncthreads()

            # Compute buf1: K-loop unrolled by 2
            for k_sub in al.range(0, BLOCK_K, 16):
                for jj in al.range(4):
                    a_buf0[jj] = A_lds_1[a_row, k_sub + a_k_off + jj]
                b_val = B_lds_1[k_sub + b_j]
                for jj in al.range(4):
                    b_buf0[jj] = b_val
                acc[0] = al.amdgpu.mfma_32x32x8_bf16_f32(
                    al.view(a_buf0, al.Tensor((2,), al.u32)),
                    al.view(b_buf0, al.Tensor((2,), al.u32)),
                    acc[0],
                )
                for jj in al.range(4):
                    a_buf1[jj] = A_lds_1[a_row, k_sub + 8 + a_k_off + jj]
                b_val = B_lds_1[k_sub + 8 + b_j]
                for jj in al.range(4):
                    b_buf1[jj] = b_val
                acc[0] = al.amdgpu.mfma_32x32x8_bf16_f32(
                    al.view(a_buf1, al.Tensor((2,), al.u32)),
                    al.view(b_buf1, al.Tensor((2,), al.u32)),
                    acc[0],
                )

            epi_buf = al.convert(0, al.i32)
        else:
            epi_buf = al.convert(1, al.i32)

    # ============================================================
    # Epilogue: compute the uncomputed tile
    # epi_buf == 0 → tile in buf0 (loaded by Part B, never computed)
    # epi_buf == 1 → tile in buf1 (loaded by Part A, Part B skipped)
    # ============================================================
    if epi_buf == 0:
        for k_sub in al.range(0, BLOCK_K, 16):
            for jj in al.range(4):
                a_buf0[jj] = A_lds_0[a_row, k_sub + a_k_off + jj]
            b_val = B_lds_0[k_sub + b_j]
            for jj in al.range(4):
                b_buf0[jj] = b_val
            acc[0] = al.amdgpu.mfma_32x32x8_bf16_f32(
                al.view(a_buf0, al.Tensor((2,), al.u32)),
                al.view(b_buf0, al.Tensor((2,), al.u32)),
                acc[0],
            )
            for jj in al.range(4):
                a_buf1[jj] = A_lds_0[a_row, k_sub + 8 + a_k_off + jj]
            b_val = B_lds_0[k_sub + 8 + b_j]
            for jj in al.range(4):
                b_buf1[jj] = b_val
            acc[0] = al.amdgpu.mfma_32x32x8_bf16_f32(
                al.view(a_buf1, al.Tensor((2,), al.u32)),
                al.view(b_buf1, al.Tensor((2,), al.u32)),
                acc[0],
            )
    else:
        for k_sub in al.range(0, BLOCK_K, 16):
            for jj in al.range(4):
                a_buf0[jj] = A_lds_1[a_row, k_sub + a_k_off + jj]
            b_val = B_lds_1[k_sub + b_j]
            for jj in al.range(4):
                b_buf0[jj] = b_val
            acc[0] = al.amdgpu.mfma_32x32x8_bf16_f32(
                al.view(a_buf0, al.Tensor((2,), al.u32)),
                al.view(b_buf0, al.Tensor((2,), al.u32)),
                acc[0],
            )
            for jj in al.range(4):
                a_buf1[jj] = A_lds_1[a_row, k_sub + 8 + a_k_off + jj]
            b_val = B_lds_1[k_sub + 8 + b_j]
            for jj in al.range(4):
                b_buf1[jj] = b_val
            acc[0] = al.amdgpu.mfma_32x32x8_bf16_f32(
                al.view(a_buf1, al.Tensor((2,), al.u32)),
                al.view(b_buf1, al.Tensor((2,), al.u32)),
                acc[0],
            )

    al.syncthreads()

    # Write output
    if warp_n == 0:
        if lane_id == 0 or lane_id == 32:
            row0 = m_offset
            if lane_id == 32:
                row0 = m_offset + 4
            col = 0
            for acc_idx in al.range(16):
                row = row0 + 8 * (acc_idx // 4) + (acc_idx % 4)
                if row < M:
                    C[row, col] = al.convert(acc[0, acc_idx], al.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        M_val = A.shape[0]
        K_val = A.shape[1]

        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((M_val, 1), device=A.device, dtype=A.dtype)

        grid_m = (M_val + BLOCK_M - 1) // BLOCK_M

        gemv_mfma_kernel[
            lambda: ((grid_m, 1, 1), (256, 1, 1))
        ](A, B, C, M_val, K_val, A.stride(0), C.stride(0))

        return C
