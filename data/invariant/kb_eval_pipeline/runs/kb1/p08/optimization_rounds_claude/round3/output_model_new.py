import torch
import torch.nn as nn

import substrate
import substrate.language as S


M = 8205
K = 2949
N = 5921


@substrate.jit
def gemm_mfma_kernel(
    A: S.Tensor((M, K), S.bf16),
    B: S.Tensor((K, N), S.bf16),
    C: S.Tensor((M, N), S.bf16),
):
    bx = S.block_id(0)
    by = S.block_id(1)
    tid = S.thread_id(0)

    warp_id = tid // 64
    lane_id = tid % 64

    warp_m = warp_id // 2
    warp_n = warp_id % 2

    block_m_start = by * 64
    block_n_start = bx * 64
    warp_m_start = block_m_start + warp_m * 32
    warp_n_start = block_n_start + warp_n * 32

    # Create resource descriptor for A with range (size in bytes)
    # When range is set, OOB loads return 0
    rsrc_A = S.amdgpu.make_rsrc(A, M * K * 2)

    # Double buffering with 2 buffers
    A_lds = S.make_shared((4, 2, 32, 8), S.bf16)
    B_lds = S.make_shared((4, 2, 8, 32), S.bf16)

    # Accumulator
    acc = S.full((16,), 0.0, S.f32)

    # MFMA swizzle inverse:
    row_a = lane_id % 32
    col_a = (lane_id // 32) * 4

    col_b = lane_id % 32
    row_b = (lane_id // 32) * 4

    # K = 2949 = 368 * 8 + 5
    # Round up to 369 K-chunks

    # Prologue: load first K chunk into buffer 0
    k_base = 0
    # Load A using raw_buffer_load_x4 with range - loads 8 bf16 values
    # For col_a=0: loads cols 0-7, uses 0-3
    # For col_a=4: loads cols 4-11, uses 4-7
    byte_offset_A = ((warp_m_start + row_a) * K + k_base + col_a) * 2
    data_A = S.amdgpu.raw_buffer_load_x4(rsrc_A, byte_offset_A, 0, 0)
    data_A_bf16 = S.view(data_A, S.Tensor((8,), S.bf16))
    for e in S.range(4):
        A_lds[warp_id, 0, row_a, col_a + e] = data_A_bf16[e]

    # Load B using tensor indexing
    for e in S.range(4):
        B_lds[warp_id, 0, row_b + e, col_b] = B[k_base + row_b + e, warp_n_start + col_b]

    S.syncthreads()

    # Main loop with double buffering and K unrolled by 2
    for k_iter in S.range(185):
        buf_first = (k_iter * 2) % 2
        buf_second = 1 - buf_first

        # First K chunk from buf_first
        a_frag1 = S.make_local((4,), S.bf16)
        for e in S.range(4):
            a_frag1[e] = A_lds[warp_id, buf_first, row_a, col_a + e]

        b_frag1 = S.make_local((4,), S.bf16)
        for e in S.range(4):
            b_frag1[e] = B_lds[warp_id, buf_first, row_b + e, col_b]

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1, b_frag1, acc)

        # Prefetch second K chunk into buf_second
        # Use raw_buffer_load_x4 for A - no OOB branch needed (range returns 0 for OOB)
        k_second = (k_iter * 2 + 1) * 8
        byte_offset_A = ((warp_m_start + row_a) * K + k_second + col_a) * 2
        data_A = S.amdgpu.raw_buffer_load_x4(rsrc_A, byte_offset_A, 0, 0)
        data_A_bf16 = S.view(data_A, S.Tensor((8,), S.bf16))
        for e in S.range(4):
            A_lds[warp_id, buf_second, row_a, col_a + e] = data_A_bf16[e]

        # Load B - still need OOB handling for non-contiguous access
        for e in S.range(4):
            if k_second + row_b + e < K:
                B_lds[warp_id, buf_second, row_b + e, col_b] = B[k_second + row_b + e, warp_n_start + col_b]
            else:
                B_lds[warp_id, buf_second, row_b + e, col_b] = S.convert(0.0, S.bf16)

        S.syncthreads()

        # Second K chunk from buf_second
        # No OOB branch needed for LDS read - data is either valid or zero
        k_second_idx = k_iter * 2 + 1
        a_frag2 = S.make_local((4,), S.bf16)
        for e in S.range(4):
            a_frag2[e] = A_lds[warp_id, buf_second, row_a, col_a + e]

        b_frag2 = S.make_local((4,), S.bf16)
        for e in S.range(4):
            b_frag2[e] = B_lds[warp_id, buf_second, row_b + e, col_b]

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag2, b_frag2, acc)

        # Prefetch next iteration's first K chunk
        # Use raw_buffer_load_x4 for A - no OOB branch needed (range returns 0 for OOB)
        k_next = (k_iter * 2 + 2) * 8
        byte_offset_A = ((warp_m_start + row_a) * K + k_next + col_a) * 2
        data_A = S.amdgpu.raw_buffer_load_x4(rsrc_A, byte_offset_A, 0, 0)
        data_A_bf16 = S.view(data_A, S.Tensor((8,), S.bf16))
        for e in S.range(4):
            A_lds[warp_id, buf_first, row_a, col_a + e] = data_A_bf16[e]

        # Load B - still need OOB handling for non-contiguous access
        for e in S.range(4):
            if k_next + row_b + e < K:
                B_lds[warp_id, buf_first, row_b + e, col_b] = B[k_next + row_b + e, warp_n_start + col_b]
            else:
                B_lds[warp_id, buf_first, row_b + e, col_b] = S.convert(0.0, S.bf16)

        S.syncthreads()

    # Write results
    for acc_idx in S.range(16):
        col = warp_n_start + (lane_id % 32)
        row = warp_m_start + 8 * (acc_idx // 4) + 4 * (lane_id // 32) + (acc_idx % 4)
        if row < M and col < N:
            C[row, col] = S.convert(acc[acc_idx], S.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if tuple(A.shape) != (8205, 2949) or tuple(B.shape) != (2949, 5921):
            return torch.matmul(A, B)
        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((8205, 5921), device=A.device, dtype=A.dtype)

        grid_m = (M + 63) // 64
        grid_n = (N + 63) // 64

        gemm_mfma_kernel[lambda: ((grid_n, grid_m, 1), (256, 1, 1))](A, B, C)
        return C
