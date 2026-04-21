import torch
import torch.nn as nn

import substrate
import substrate.language as S


M = 2048
K = 8192
N = 4096

# Tile sizes for 4-warps (2x2 warp grid)
TILE_M = 64
TILE_N = 64
# Use TILE_K = 8 for fine-grained overlap (single MFMA K-dimension)
TILE_K = 8
WARP_SIZE = 64
NUM_WARPS = 4
THREADS = WARP_SIZE * NUM_WARPS


def kernel_launch_config():
    grid_m = (M + TILE_M - 1) // TILE_M
    grid_n = (N + TILE_N - 1) // TILE_N
    return (grid_n, grid_m, 1), (THREADS, 1, 1)


@substrate.jit
def gemm_mfma_kernel(
    A: S.Tensor((2048, 8192), S.bf16),
    B: S.Tensor((8192, 4096), S.bf16),
    C: S.Tensor((2048, 4096), S.bf16),
):
    """GEMM kernel with software pipelining, double buffering, K-loop unroll by 2,
    and OOB handling via range parameter in buffer operations."""
    bx = S.block_id(0)
    by = S.block_id(1)
    tid = S.thread_id(0)
    lane = tid % WARP_SIZE
    warp_id = tid // WARP_SIZE

    warp_m = warp_id // 2
    warp_n = warp_id % 2

    tile_m_base = by * TILE_M + warp_m * 32
    tile_n_base = bx * TILE_N + warp_n * 32

    # Create buffer resources with range parameter (in bytes) for OOB handling
    # When range is set, OOB loads return 0 and OOB stores are discarded
    rsrc_A = S.amdgpu.make_rsrc(A, M * K * 2)
    rsrc_B = S.amdgpu.make_rsrc(B, K * N * 2)

    # Double buffered shared memory for fine-grained overlap
    # Each buffer holds TILE_M x TILE_K of A and TILE_K x TILE_N of B
    A_shared_0 = S.make_shared((TILE_M, TILE_K), S.bf16)
    A_shared_1 = S.make_shared((TILE_M, TILE_K), S.bf16)
    B_shared_0 = S.make_shared((TILE_K, TILE_N), S.bf16)
    B_shared_1 = S.make_shared((TILE_K, TILE_N), S.bf16)

    # Fragment storage for MFMA inputs
    A_frag_lds = S.make_shared((NUM_WARPS, WARP_SIZE, 2), S.u32)
    B_frag_lds = S.make_shared((NUM_WARPS, WARP_SIZE, 2), S.u32)

    A_frag_tensor = S.view(A_frag_lds, S.Tensor((NUM_WARPS, WARP_SIZE, 2), S.u32))
    B_frag_tensor = S.view(B_frag_lds, S.Tensor((NUM_WARPS, WARP_SIZE, 2), S.u32))

    acc = S.full((16,), 0.0, S.f32)

    num_k_tiles = K // TILE_K  # 1024

    # Prologue: Load first K-tile into buffer 0
    k_base_0 = 0
    # Load A tile
    load_idx_a = tid % 64
    row_a = load_idx_a
    global_row_a = by * TILE_M + row_a
    byte_offset_a = (global_row_a * K + k_base_0) * 2
    data_a = S.amdgpu.raw_buffer_load_x4(rsrc_A, byte_offset_a, 0, 0)
    data_a_bf16 = S.view(data_a, S.Tensor((8,), S.bf16))
    for i in S.range(8):
        A_shared_0[row_a, i] = data_a_bf16[i]

    # Load B tile
    load_idx_b = tid % 64
    row_b = load_idx_b // 8
    col_start_b = (load_idx_b % 8) * 8
    global_row_b = k_base_0 + row_b
    byte_offset_b = (global_row_b * N + bx * TILE_N + col_start_b) * 2
    data_b = S.amdgpu.raw_buffer_load_x4(rsrc_B, byte_offset_b, 0, 0)
    data_b_bf16 = S.view(data_b, S.Tensor((8,), S.bf16))
    for i in S.range(8):
        B_shared_0[row_b, col_start_b + i] = data_b_bf16[i]

    S.syncthreads()

    # Main loop: unrolled by 2 K-tiles for reduced branching
    # Process 2 K-tiles per iteration
    # OOB access is handled by range parameter in make_rsrc - no explicit guards needed
    for k_tile_unroll in S.range(num_k_tiles // 2):
        k_tile_0 = k_tile_unroll * 2
        k_tile_1 = k_tile_unroll * 2 + 1

        k_base_0 = k_tile_0 * TILE_K
        k_base_1 = k_tile_1 * TILE_K

        # === Phase 1: Compute from buffer 0, load to buffer 1 ===

        # MFMA computation from buffer 0
        a_row_lds = warp_m * 32 + (lane % 32)
        a_col_lds = (lane // 32) * 4

        a_bf16_0 = A_shared_0[a_row_lds, a_col_lds + 0]
        a_bf16_1 = A_shared_0[a_row_lds, a_col_lds + 1]
        a_bf16_2 = A_shared_0[a_row_lds, a_col_lds + 2]
        a_bf16_3 = A_shared_0[a_row_lds, a_col_lds + 3]

        a_u16_0 = S.bitcast(a_bf16_0, S.u16)
        a_u16_1 = S.bitcast(a_bf16_1, S.u16)
        a_u16_2 = S.bitcast(a_bf16_2, S.u16)
        a_u16_3 = S.bitcast(a_bf16_3, S.u16)

        a_u32_0 = a_u16_0 | (a_u16_1 << 16)
        a_u32_1 = a_u16_2 | (a_u16_3 << 16)

        A_frag_lds[warp_id, lane, 0] = a_u32_0
        A_frag_lds[warp_id, lane, 1] = a_u32_1

        b_row_lds = (lane // 32) * 4
        b_col_lds = warp_n * 32 + (lane % 32)

        b_bf16_0 = B_shared_0[b_row_lds + 0, b_col_lds]
        b_bf16_1 = B_shared_0[b_row_lds + 1, b_col_lds]
        b_bf16_2 = B_shared_0[b_row_lds + 2, b_col_lds]
        b_bf16_3 = B_shared_0[b_row_lds + 3, b_col_lds]

        b_u16_0 = S.bitcast(b_bf16_0, S.u16)
        b_u16_1 = S.bitcast(b_bf16_1, S.u16)
        b_u16_2 = S.bitcast(b_bf16_2, S.u16)
        b_u16_3 = S.bitcast(b_bf16_3, S.u16)

        b_u32_0 = b_u16_0 | (b_u16_1 << 16)
        b_u32_1 = b_u16_2 | (b_u16_3 << 16)

        B_frag_lds[warp_id, lane, 0] = b_u32_0
        B_frag_lds[warp_id, lane, 1] = b_u32_1

        a_row_tensor = A_frag_tensor[warp_id, lane]
        b_row_tensor = B_frag_tensor[warp_id, lane]

        a_view = S.view(a_row_tensor, S.Tensor((1, 4, 1), S.bf16))
        b_view = S.view(b_row_tensor, S.Tensor((1, 4, 1), S.bf16))

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_view[0], b_view[0], acc)

        # Load next K-tile to buffer 1
        load_idx_a1 = tid % 64
        row_a1 = load_idx_a1
        global_row_a1 = by * TILE_M + row_a1
        byte_offset_a1 = (global_row_a1 * K + k_base_1) * 2
        data_a1 = S.amdgpu.raw_buffer_load_x4(rsrc_A, byte_offset_a1, 0, 0)
        data_a1_bf16 = S.view(data_a1, S.Tensor((8,), S.bf16))
        for i in S.range(8):
            A_shared_1[row_a1, i] = data_a1_bf16[i]

        load_idx_b1 = tid % 64
        row_b1 = load_idx_b1 // 8
        col_start_b1 = (load_idx_b1 % 8) * 8
        global_row_b1 = k_base_1 + row_b1
        byte_offset_b1 = (global_row_b1 * N + bx * TILE_N + col_start_b1) * 2
        data_b1 = S.amdgpu.raw_buffer_load_x4(rsrc_B, byte_offset_b1, 0, 0)
        data_b1_bf16 = S.view(data_b1, S.Tensor((8,), S.bf16))
        for i in S.range(8):
            B_shared_1[row_b1, col_start_b1 + i] = data_b1_bf16[i]

        S.syncthreads()

        # === Phase 2: Compute from buffer 1, load to buffer 0 for next iteration ===

        # MFMA computation from buffer 1
        a_bf16_0_1 = A_shared_1[a_row_lds, a_col_lds + 0]
        a_bf16_1_1 = A_shared_1[a_row_lds, a_col_lds + 1]
        a_bf16_2_1 = A_shared_1[a_row_lds, a_col_lds + 2]
        a_bf16_3_1 = A_shared_1[a_row_lds, a_col_lds + 3]

        a_u16_0_1 = S.bitcast(a_bf16_0_1, S.u16)
        a_u16_1_1 = S.bitcast(a_bf16_1_1, S.u16)
        a_u16_2_1 = S.bitcast(a_bf16_2_1, S.u16)
        a_u16_3_1 = S.bitcast(a_bf16_3_1, S.u16)

        a_u32_0_1 = a_u16_0_1 | (a_u16_1_1 << 16)
        a_u32_1_1 = a_u16_2_1 | (a_u16_3_1 << 16)

        A_frag_lds[warp_id, lane, 0] = a_u32_0_1
        A_frag_lds[warp_id, lane, 1] = a_u32_1_1

        b_bf16_0_1 = B_shared_1[b_row_lds + 0, b_col_lds]
        b_bf16_1_1 = B_shared_1[b_row_lds + 1, b_col_lds]
        b_bf16_2_1 = B_shared_1[b_row_lds + 2, b_col_lds]
        b_bf16_3_1 = B_shared_1[b_row_lds + 3, b_col_lds]

        b_u16_0_1 = S.bitcast(b_bf16_0_1, S.u16)
        b_u16_1_1 = S.bitcast(b_bf16_1_1, S.u16)
        b_u16_2_1 = S.bitcast(b_bf16_2_1, S.u16)
        b_u16_3_1 = S.bitcast(b_bf16_3_1, S.u16)

        b_u32_0_1 = b_u16_0_1 | (b_u16_1_1 << 16)
        b_u32_1_1 = b_u16_2_1 | (b_u16_3_1 << 16)

        B_frag_lds[warp_id, lane, 0] = b_u32_0_1
        B_frag_lds[warp_id, lane, 1] = b_u32_1_1

        a_row_tensor_1 = A_frag_tensor[warp_id, lane]
        b_row_tensor_1 = B_frag_tensor[warp_id, lane]

        a_view_1 = S.view(a_row_tensor_1, S.Tensor((1, 4, 1), S.bf16))
        b_view_1 = S.view(b_row_tensor_1, S.Tensor((1, 4, 1), S.bf16))

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_view_1[0], b_view_1[0], acc)

        # Load next K-tile to buffer 0 for next iteration
        # No explicit OOB guard needed - range parameter in make_rsrc handles OOB:
        # - OOB loads return 0
        # - OOB stores are discarded
        # Removing this branch reduces divergence and improves performance
        k_base_next = (k_tile_unroll + 1) * 2 * TILE_K
        load_idx_a0 = tid % 64
        row_a0 = load_idx_a0
        global_row_a0 = by * TILE_M + row_a0
        byte_offset_a0 = (global_row_a0 * K + k_base_next) * 2
        data_a0 = S.amdgpu.raw_buffer_load_x4(rsrc_A, byte_offset_a0, 0, 0)
        data_a0_bf16 = S.view(data_a0, S.Tensor((8,), S.bf16))
        for i in S.range(8):
            A_shared_0[row_a0, i] = data_a0_bf16[i]

        load_idx_b0 = tid % 64
        row_b0 = load_idx_b0 // 8
        col_start_b0 = (load_idx_b0 % 8) * 8
        global_row_b0 = k_base_next + row_b0
        byte_offset_b0 = (global_row_b0 * N + bx * TILE_N + col_start_b0) * 2
        data_b0 = S.amdgpu.raw_buffer_load_x4(rsrc_B, byte_offset_b0, 0, 0)
        data_b0_bf16 = S.view(data_b0, S.Tensor((8,), S.bf16))
        for i in S.range(8):
            B_shared_0[row_b0, col_start_b0 + i] = data_b0_bf16[i]

        S.syncthreads()

    # Write results - correct MFMA 32x32x8 output layout
    for acc_idx in S.range(16):
        row_offset = 8 * (acc_idx // 4) + 4 * (lane // 32) + (acc_idx % 4)
        col_offset = lane % 32

        global_row = tile_m_base + row_offset
        global_col = tile_n_base + col_offset

        C[global_row, global_col] = S.convert(acc[acc_idx], S.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if tuple(A.shape) != (8192, 2048) or tuple(B.shape) != (8192, 4096):
            return torch.matmul(A.T, B)

        A2 = A.transpose(-2, -1).contiguous()
        B2 = B.contiguous()
        C = torch.empty((2048, 4096), device=A.device, dtype=A.dtype)

        grid, block = kernel_launch_config()
        gemm_mfma_kernel[lambda: (grid, block)](A2, B2, C)

        return C
