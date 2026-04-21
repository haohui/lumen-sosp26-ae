import torch
import torch.nn as nn

import substrate
import substrate.language as S


M = 2048
K = 8192
N = 4096
TILE_M = 64
TILE_N = 64
TILE_K = 8


@substrate.jit
def gemm_mfma_kernel(
    A: S.Tensor((M, K), S.bf16),
    B_T: S.Tensor((N, K), S.bf16),
    C: S.Tensor((M, N), S.bf16),
):
    bx = S.block_id(0)
    by = S.block_id(1)
    lane = S.thread_id(0)

    warp_id = lane // 64
    lane_in_warp = lane % 64

    warp_row = warp_id // 2
    warp_col = warp_id % 2

    tile_row_base = by * TILE_M + warp_row * 32
    tile_col_base = bx * TILE_N + warp_col * 32

    acc = S.full((16,), 0.0, S.f32)

    # Create resource descriptors with range for OOB handling
    # Range is in bytes - when set, OOB loads return 0, OOB stores are discarded
    # This eliminates the need for explicit bounds checking in the loop
    rsrc_A = S.amdgpu.make_rsrc(A, M * K * 2)
    rsrc_B_T = S.amdgpu.make_rsrc(B_T, N * K * 2)
    rsrc_C = S.amdgpu.make_rsrc(C, M * N * 2)

    # Precompute swizzle indices
    a_row = lane_in_warp % 32
    a_col_group = lane_in_warp // 32
    global_a_row = tile_row_base + a_row

    b_col = lane_in_warp % 32
    b_k_group = lane_in_warp // 32
    global_b_col = tile_col_base + b_col

    num_k_tiles = K // TILE_K

    # Software pipelined main loop with K-loop unroll by 2
    # Process 2 K-tiles per iteration to minimize branching
    # Use raw_buffer_load_x4 with range for OOB handling
    for k_tile in S.range(0, num_k_tiles, 2):
        # First K-tile
        k_base_0 = k_tile * TILE_K
        a_offset_0 = global_a_row * K + k_base_0 + a_col_group * 4
        # raw_buffer_load_x4 loads 8 bf16, OOB elements return 0
        a_data_0 = S.amdgpu.raw_buffer_load_x4(rsrc_A, a_offset_0 * 2, 0, 0)
        a_vals_0 = S.view(a_data_0, S.Tensor((8,), S.bf16))
        # Extract first 4 bf16 elements for MFMA
        a_frag_0 = S.make_local((4,), S.bf16)
        for i in S.range(4):
            a_frag_0[i] = a_vals_0[i]

        b_offset_0 = global_b_col * K + k_base_0 + b_k_group * 4
        b_data_0 = S.amdgpu.raw_buffer_load_x4(rsrc_B_T, b_offset_0 * 2, 0, 0)
        b_vals_0 = S.view(b_data_0, S.Tensor((8,), S.bf16))
        b_frag_0 = S.make_local((4,), S.bf16)
        for i in S.range(4):
            b_frag_0[i] = b_vals_0[i]

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag_0, b_frag_0, acc)

        # Second K-tile
        k_base_1 = (k_tile + 1) * TILE_K
        a_offset_1 = global_a_row * K + k_base_1 + a_col_group * 4
        a_data_1 = S.amdgpu.raw_buffer_load_x4(rsrc_A, a_offset_1 * 2, 0, 0)
        a_vals_1 = S.view(a_data_1, S.Tensor((8,), S.bf16))
        a_frag_1 = S.make_local((4,), S.bf16)
        for i in S.range(4):
            a_frag_1[i] = a_vals_1[i]

        b_offset_1 = global_b_col * K + k_base_1 + b_k_group * 4
        b_data_1 = S.amdgpu.raw_buffer_load_x4(rsrc_B_T, b_offset_1 * 2, 0, 0)
        b_vals_1 = S.view(b_data_1, S.Tensor((8,), S.bf16))
        b_frag_1 = S.make_local((4,), S.bf16)
        for i in S.range(4):
            b_frag_1[i] = b_vals_1[i]

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag_1, b_frag_1, acc)

    # Write results
    # The range in rsrc_C ensures OOB writes are discarded
    for acc_group in S.range(4):
        base_idx = acc_group * 4
        row_base = tile_row_base + 8 * acc_group + 4 * (lane_in_warp // 32)
        col = tile_col_base + (lane_in_warp % 32)

        for i in S.range(4):
            row = row_base + i
            bf16_val = S.convert(acc[base_idx + i], S.bf16)
            C[row, col] = bf16_val


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if tuple(A.shape) != (8192, 2048) or tuple(B.shape) != (4096, 8192):
            return torch.matmul(A.T, B.T)

        A_T = A.transpose(-2, -1).contiguous()
        B_T = B

        C = torch.empty((2048, 4096), device=A.device, dtype=A.dtype)

        grid = (N // TILE_N, M // TILE_M, 1)
        block = (256, 1, 1)

        gemm_mfma_kernel[lambda: (grid, block)](A_T, B_T, C)
        return C
