import torch
import torch.nn as nn

import substrate
import substrate.language as S


M = 4096
N = 4096

# Tile sizes for 4-wave kernel (2x2 warp grid)
TILE_M = 64
TILE_N = 64

# Range in bytes for OOB handling
A_RANGE_BYTES = M * 2  # M elements * 2 bytes per bf16
C_RANGE_BYTES = M * N * 2  # M*N elements * 2 bytes per bf16


@substrate.jit
def diag_left_mfma_kernel(
    A: S.Tensor((4096,), S.bf16),
    B_T: S.Tensor((4096, 4096), S.bf16),  # Transposed B: (N, M) = (4096, 4096)
    C: S.Tensor((4096, 4096), S.bf16),
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
    rsrc_A = S.amdgpu.make_rsrc(A, A_RANGE_BYTES)
    rsrc_B_T = S.amdgpu.make_rsrc(B_T, M * N * 2)
    rsrc_C = S.amdgpu.make_rsrc(C, C_RANGE_BYTES)

    # LDS for double buffering B fragments
    lds_B = S.make_shared((2, 256, 4), S.bf16)

    # A swizzle: A(i, j) -> lane = i + (j/4)*32, element = j%4
    # Inverse: row = lane % 32, col_group = lane // 32
    a_row = lane_in_warp % 32
    a_col_group = lane_in_warp // 32

    b_col_idx = lane_in_warp % 32
    b_k_group = lane_in_warp // 32
    b_T_row = tile_col_base + b_col_idx

    # ========================================
    # Software pipelining with double buffering
    # Using raw_buffer_load with range for OOB handling
    # ========================================

    # Prologue: Load B chunks 0 and 1 into LDS
    k_start_0 = tile_row_base + b_k_group * 4
    b_offset_0 = (b_T_row * 4096 + k_start_0) * 2
    b_data_0 = S.amdgpu.raw_buffer_load_x2(rsrc_B_T, b_offset_0, 0, 0)
    b_frag_0 = S.view(b_data_0, S.Tensor((4,), S.bf16))
    for e in S.range(4):
        lds_B[0, lane, e] = b_frag_0[e]

    k_start_1 = tile_row_base + 8 + b_k_group * 4
    b_offset_1 = (b_T_row * 4096 + k_start_1) * 2
    b_data_1 = S.amdgpu.raw_buffer_load_x2(rsrc_B_T, b_offset_1, 0, 0)
    b_frag_1 = S.view(b_data_1, S.Tensor((4,), S.bf16))
    for e in S.range(4):
        lds_B[1, lane, e] = b_frag_1[e]

    S.syncthreads()

    # ========================================
    # Process chunks 0, 1, 2, 3 with double buffering
    # Load A using raw_buffer_load with range for OOB handling
    # ========================================

    # Load A value using raw_buffer_load_x2 (loads 4 bf16 values)
    # Align to 8-byte boundary for efficiency
    a_idx = tile_row_base + a_row
    a_aligned_idx = (a_idx // 4) * 4  # Round down to multiple of 4
    a_offset = a_aligned_idx * 2  # Byte offset
    a_data = S.amdgpu.raw_buffer_load_x2(rsrc_A, a_offset, 0, 0)
    a_vec = S.view(a_data, S.Tensor((4,), S.bf16))
    # Extract the element we need (a_row might not be aligned)
    a_elem_idx = a_idx - a_aligned_idx
    a_val = a_vec[a_elem_idx]

    # Chunk 0
    a_frag_0 = S.make_local((4,), S.bf16)
    for e in S.range(4):
        k_idx_0 = a_col_group * 4 + e
        if a_row == k_idx_0:
            a_frag_0[e] = a_val
        else:
            a_frag_0[e] = S.convert(0.0, S.bf16)

    b_frag_lds_0 = S.make_local((4,), S.bf16)
    for e in S.range(4):
        b_frag_lds_0[e] = lds_B[0, lane, e]

    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag_0, b_frag_lds_0, acc)

    # Chunk 1
    a_frag_1 = S.make_local((4,), S.bf16)
    for e in S.range(4):
        k_idx_1 = 8 + a_col_group * 4 + e
        if a_row == k_idx_1:
            a_frag_1[e] = a_val
        else:
            a_frag_1[e] = S.convert(0.0, S.bf16)

    b_frag_lds_1 = S.make_local((4,), S.bf16)
    for e in S.range(4):
        b_frag_lds_1[e] = lds_B[1, lane, e]

    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag_1, b_frag_lds_1, acc)

    # Now load chunks 2, 3 into LDS
    k_start_2 = tile_row_base + 16 + b_k_group * 4
    b_offset_2 = (b_T_row * 4096 + k_start_2) * 2
    b_data_2 = S.amdgpu.raw_buffer_load_x2(rsrc_B_T, b_offset_2, 0, 0)
    b_frag_2 = S.view(b_data_2, S.Tensor((4,), S.bf16))
    for e in S.range(4):
        lds_B[0, lane, e] = b_frag_2[e]

    k_start_3 = tile_row_base + 24 + b_k_group * 4
    b_offset_3 = (b_T_row * 4096 + k_start_3) * 2
    b_data_3 = S.amdgpu.raw_buffer_load_x2(rsrc_B_T, b_offset_3, 0, 0)
    b_frag_3 = S.view(b_data_3, S.Tensor((4,), S.bf16))
    for e in S.range(4):
        lds_B[1, lane, e] = b_frag_3[e]

    S.syncthreads()

    # Chunk 2
    a_frag_2 = S.make_local((4,), S.bf16)
    for e in S.range(4):
        k_idx_2 = 16 + a_col_group * 4 + e
        if a_row == k_idx_2:
            a_frag_2[e] = a_val
        else:
            a_frag_2[e] = S.convert(0.0, S.bf16)

    b_frag_lds_2 = S.make_local((4,), S.bf16)
    for e in S.range(4):
        b_frag_lds_2[e] = lds_B[0, lane, e]

    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag_2, b_frag_lds_2, acc)

    # Chunk 3
    a_frag_3 = S.make_local((4,), S.bf16)
    for e in S.range(4):
        k_idx_3 = 24 + a_col_group * 4 + e
        if a_row == k_idx_3:
            a_frag_3[e] = a_val
        else:
            a_frag_3[e] = S.convert(0.0, S.bf16)

    b_frag_lds_3 = S.make_local((4,), S.bf16)
    for e in S.range(4):
        b_frag_lds_3[e] = lds_B[1, lane, e]

    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag_3, b_frag_lds_3, acc)

    # Write output using raw_buffer_store with range for OOB handling
    for acc_idx in S.range(16):
        col = tile_col_base + (lane_in_warp % 32)
        row = tile_row_base + 8 * (acc_idx // 4) + 4 * (lane_in_warp // 32) + (acc_idx % 4)
        C[row, col] = S.convert(acc[acc_idx], S.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if tuple(A.shape) != (4096,) or tuple(B.shape) != (4096, 4096):
            return torch.diag(A) @ B
        A = A.contiguous()
        B_T = B.transpose(-2, -1).contiguous()
        C = torch.empty((4096, 4096), device=B.device, dtype=B.dtype)

        grid = (N // TILE_N, M // TILE_M, 1)
        block = (256, 1, 1)

        diag_left_mfma_kernel[lambda: (grid, block)](A, B_T, C)
        return C
