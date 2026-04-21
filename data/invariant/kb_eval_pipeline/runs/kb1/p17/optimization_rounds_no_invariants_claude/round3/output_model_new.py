import torch
import torch.nn as nn

import substrate
import substrate.language as S

M = 2048
K = 8192
N = 4096

TILE_M = 32
TILE_N = 32
TILE_K = 16  # Process K=16 per iteration (2 MFMA ops)

WARP_SIZE = 64
NUM_WARPS = 4  # 2x2 warp grid


@substrate.jit
def gemm_mfma_kernel(
    A: S.Tensor((2048, 8192), S.bf16),
    B: S.Tensor((4096, 8192), S.bf16),
    C: S.Tensor((2048, 4096), S.bf16),
):
    block_m = S.block_id(0)
    block_n = S.block_id(1)
    thread_id = S.thread_id(0)

    warp_id = thread_id // WARP_SIZE
    lane = thread_id % WARP_SIZE

    warp_m = warp_id % 2
    warp_n = warp_id // 2

    row_base = block_m * (TILE_M * 2) + warp_m * TILE_M
    col_base = block_n * (TILE_N * 2) + warp_n * TILE_N

    # Accumulator for 32x32 output, 16 f32 per lane
    acc = S.full((16,), 0.0, S.f32)

    # Create buffer resources with range set to tensor size in bytes
    # When range is set, raw_buffer_load_x4 returns 0 for OOB elements
    # and raw_buffer_store discards OOB writes - no explicit OOB branches needed
    rsrc_a = S.amdgpu.make_rsrc(A, M * K * 2)
    rsrc_b = S.amdgpu.make_rsrc(B, N * K * 2)

    num_k_iters = K // TILE_K  # 512 iterations

    # MFMA lane layout
    a_row = lane % 32
    b_col = lane % 32

    a_global_row = row_base + a_row
    b_global_col = col_base + b_col

    # Double-buffered LDS for software pipelining
    # Each buffer holds K=8 worth of data per warp
    A_shared_0 = S.make_shared((NUM_WARPS, WARP_SIZE, 8), S.bf16)
    A_shared_1 = S.make_shared((NUM_WARPS, WARP_SIZE, 8), S.bf16)
    B_shared_0 = S.make_shared((NUM_WARPS, WARP_SIZE, 8), S.bf16)
    B_shared_1 = S.make_shared((NUM_WARPS, WARP_SIZE, 8), S.bf16)

    # Register fragments
    a_frag = S.make_local((4,), S.bf16)
    b_frag = S.make_local((4,), S.bf16)

    # Main loop unrolled by 2
    # Each iteration processes 2 K-tiles (K=32 total, 4 MFMA ops)
    # No explicit OOB branches - range in buffer descriptor handles OOB automatically:
    # hardware returns 0 for OOB loads, discards OOB stores
    for k_iter in S.range(0, num_k_iters, 2):
        # === First K-tile (k_iter) ===
        k_base = k_iter * TILE_K

        # Load first K=8 half into buffer 0
        # Range in rsrc ensures OOB loads return 0, eliminating need for OOB branches
        k_start = k_base
        a_offset = a_global_row * K + k_start
        a_data_8 = S.amdgpu.raw_buffer_load_x4(rsrc_a, a_offset * 2, 0, 0)
        a_data_8_bf16 = S.view(a_data_8, S.Tensor((8,), S.bf16))
        for i in S.range(8):
            A_shared_0[warp_id, lane, i] = a_data_8_bf16[i]

        b_offset = b_global_col * K + k_start
        b_data_8 = S.amdgpu.raw_buffer_load_x4(rsrc_b, b_offset * 2, 0, 0)
        b_data_8_bf16 = S.view(b_data_8, S.Tensor((8,), S.bf16))
        for i in S.range(8):
            B_shared_0[warp_id, lane, i] = b_data_8_bf16[i]

        # Load second K=8 half into buffer 1
        k_start_1 = k_base + 8
        a_offset_1 = a_global_row * K + k_start_1
        a_data_8_1 = S.amdgpu.raw_buffer_load_x4(rsrc_a, a_offset_1 * 2, 0, 0)
        a_data_8_bf16_1 = S.view(a_data_8_1, S.Tensor((8,), S.bf16))
        for i in S.range(8):
            A_shared_1[warp_id, lane, i] = a_data_8_bf16_1[i]

        b_offset_1 = b_global_col * K + k_start_1
        b_data_8_1 = S.amdgpu.raw_buffer_load_x4(rsrc_b, b_offset_1 * 2, 0, 0)
        b_data_8_bf16_1 = S.view(b_data_8_1, S.Tensor((8,), S.bf16))
        for i in S.range(8):
            B_shared_1[warp_id, lane, i] = b_data_8_bf16_1[i]

        S.syncthreads()

        # Compute first MFMA from buffer 0
        for i in S.range(4):
            idx = (lane // 32) * 4 + i
            a_frag[i] = A_shared_0[warp_id, lane, idx]
            b_frag[i] = B_shared_0[warp_id, lane, idx]
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag, b_frag, acc)

        # Compute second MFMA from buffer 1
        for i in S.range(4):
            idx = (lane // 32) * 4 + i
            a_frag[i] = A_shared_1[warp_id, lane, idx]
            b_frag[i] = B_shared_1[warp_id, lane, idx]
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag, b_frag, acc)

        # === Second K-tile (k_iter + 1) ===
        k_base_2 = (k_iter + 1) * TILE_K

        # Load first K=8 half into buffer 0
        k_start_2 = k_base_2
        a_offset_2 = a_global_row * K + k_start_2
        a_data_8_2 = S.amdgpu.raw_buffer_load_x4(rsrc_a, a_offset_2 * 2, 0, 0)
        a_data_8_bf16_2 = S.view(a_data_8_2, S.Tensor((8,), S.bf16))
        for i in S.range(8):
            A_shared_0[warp_id, lane, i] = a_data_8_bf16_2[i]

        b_offset_2 = b_global_col * K + k_start_2
        b_data_8_2 = S.amdgpu.raw_buffer_load_x4(rsrc_b, b_offset_2 * 2, 0, 0)
        b_data_8_bf16_2 = S.view(b_data_8_2, S.Tensor((8,), S.bf16))
        for i in S.range(8):
            B_shared_0[warp_id, lane, i] = b_data_8_bf16_2[i]

        # Load second K=8 half into buffer 1
        k_start_3 = k_base_2 + 8
        a_offset_3 = a_global_row * K + k_start_3
        a_data_8_3 = S.amdgpu.raw_buffer_load_x4(rsrc_a, a_offset_3 * 2, 0, 0)
        a_data_8_bf16_3 = S.view(a_data_8_3, S.Tensor((8,), S.bf16))
        for i in S.range(8):
            A_shared_1[warp_id, lane, i] = a_data_8_bf16_3[i]

        b_offset_3 = b_global_col * K + k_start_3
        b_data_8_3 = S.amdgpu.raw_buffer_load_x4(rsrc_b, b_offset_3 * 2, 0, 0)
        b_data_8_bf16_3 = S.view(b_data_8_3, S.Tensor((8,), S.bf16))
        for i in S.range(8):
            B_shared_1[warp_id, lane, i] = b_data_8_bf16_3[i]

        S.syncthreads()

        # Compute third MFMA from buffer 0
        for i in S.range(4):
            idx = (lane // 32) * 4 + i
            a_frag[i] = A_shared_0[warp_id, lane, idx]
            b_frag[i] = B_shared_0[warp_id, lane, idx]
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag, b_frag, acc)

        # Compute fourth MFMA from buffer 1
        for i in S.range(4):
            idx = (lane // 32) * 4 + i
            a_frag[i] = A_shared_1[warp_id, lane, idx]
            b_frag[i] = B_shared_1[warp_id, lane, idx]
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag, b_frag, acc)

    # Write output
    for acc_idx in S.range(16):
        col = lane % 32
        row = 8 * (acc_idx // 4) + 4 * (lane // 32) + (acc_idx % 4)

        out_row = row_base + row
        out_col = col_base + col

        C[out_row, out_col] = S.convert(acc[acc_idx], S.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if tuple(A.shape) != (2048, 8192) or tuple(B.shape) != (4096, 8192):
            return torch.matmul(A, B.T)
        A2 = A.contiguous()
        B2 = B.contiguous()
        C = torch.empty((2048, 4096), device=A.device, dtype=A.dtype)

        grid_m = (M + TILE_M * 2 - 1) // (TILE_M * 2)
        grid_n = (N + TILE_N * 2 - 1) // (TILE_N * 2)

        gemm_mfma_kernel[lambda: ((grid_m, grid_n, 1), (WARP_SIZE * NUM_WARPS, 1, 1))](A2, B2, C)
        return C
