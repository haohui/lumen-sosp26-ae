import torch
import torch.nn as nn

import substrate
import substrate.language as S


M = 4096
K = 4096
N = 4096

# Tile sizes for 4-warps (2x2 warp grid)
TILE_M = 64
TILE_N = 64
TILE_K = 16  # 2 MFMA iterations per K tile (K-loop unrolled by 2)
WARP_SIZE = 64
NUM_WARPS = 4
THREADS = WARP_SIZE * NUM_WARPS


def kernel_launch_config():
    grid_m = (M + TILE_M - 1) // TILE_M
    grid_n = (N + TILE_N - 1) // TILE_N
    return (grid_n, grid_m, 1), (THREADS, 1, 1)


@substrate.jit
def gemm_mfma_kernel(
    A: S.Tensor((4096, 4096), S.bf16),
    B: S.Tensor((4096, 4096), S.bf16),
    C: S.Tensor((4096, 4096), S.bf16),
):
    """GEMM kernel using MFMA 32x32x8_bf16_f32 with raw_buffer_load_x4 and range for OOB handling."""
    bx = S.block_id(0)
    by = S.block_id(1)
    tid = S.thread_id(0)
    lane = tid % WARP_SIZE
    warp_id = tid // WARP_SIZE

    warp_m = warp_id // 2
    warp_n = warp_id % 2

    tile_m_base = by * TILE_M + warp_m * 32
    tile_n_base = bx * TILE_N + warp_n * 32

    # Create buffer resources with range (size in bytes)
    # A: M x K bf16 = M * K * 2 bytes
    # B: K x N bf16 = K * N * 2 bytes
    rsrc_A = S.amdgpu.make_rsrc(A, M * K * 2)
    rsrc_B = S.amdgpu.make_rsrc(B, K * N * 2)

    # Double-buffered LDS
    A_shared_0 = S.make_shared((TILE_M, TILE_K), S.bf16)
    A_shared_1 = S.make_shared((TILE_M, TILE_K), S.bf16)
    B_shared_0 = S.make_shared((TILE_K, TILE_N), S.bf16)
    B_shared_1 = S.make_shared((TILE_K, TILE_N), S.bf16)

    A_frag_lds = S.make_shared((NUM_WARPS, WARP_SIZE, 2), S.u32)
    B_frag_lds = S.make_shared((NUM_WARPS, WARP_SIZE, 2), S.u32)

    A_frag_tensor = S.view(A_frag_lds, S.Tensor((NUM_WARPS, WARP_SIZE, 2), S.u32))
    B_frag_tensor = S.view(B_frag_lds, S.Tensor((NUM_WARPS, WARP_SIZE, 2), S.u32))

    acc = S.full((16,), 0.0, S.f32)

    num_k_tiles = K // TILE_K  # 256

    # Process all K-tiles using double buffering
    # num_k_tiles = 256 is even, so we can process exactly 128 pairs
    num_pairs = num_k_tiles // 2  # 128

    for pair_idx in S.range(num_pairs):
        k_even = pair_idx * 2
        k_odd = pair_idx * 2 + 1

        k_base_even = k_even * TILE_K
        k_base_odd = k_odd * TILE_K

        # Load even tile into buffer 0 using raw_buffer_load_x4
        # A_shared: (64, 16) = 1024 elements, need 128 loads of 8 elements each
        # B_shared: (16, 64) = 1024 elements, need 128 loads of 8 elements each
        #
        # Strategy: All threads participate in loading
        # Use tid % 128 as load index to ensure valid LDS indices
        # Thread participation: tid < 128 for A, tid >= 128 for B (warp-aligned)

        # A tile loading (threads 0-127 actively write)
        # load_idx wraps around for tid >= 128 using modulo
        load_idx_a = tid % 128
        row_a = load_idx_a // 2
        col_start_a = (load_idx_a % 2) * 8

        # Byte offset for A: (row * K + col) * 2
        # For tid 0-127: valid global row
        # For tid 128-255: OOB global row (returns 0 with range)
        global_row_a = by * TILE_M + (tid // 2)
        byte_offset_a_even = (global_row_a * K + k_base_even + col_start_a) * 2
        data_a_even = S.amdgpu.raw_buffer_load_x4(rsrc_A, byte_offset_a_even, 0, 0)
        data_a_even_bf16 = S.view(data_a_even, S.Tensor((8,), S.bf16))

        # Store to shared memory only for threads 0-127
        for i in S.range(8):
            if tid < 128:
                A_shared_0[row_a, col_start_a + i] = data_a_even_bf16[i]

        # B tile loading (threads 128-255 actively write)
        # Use tid % 128 as load index
        load_idx_b = tid % 128
        row_b = load_idx_b // 8
        col_start_b = (load_idx_b % 8) * 8

        # Byte offset for B: (row * N + col) * 2
        # For tid 0-127: row_b is valid, but global_row_b might be OOB
        # For tid 128-255: row_b is valid, global_row_b is valid
        global_row_b = k_base_even + row_b
        byte_offset_b_even = (global_row_b * N + bx * TILE_N + col_start_b) * 2
        data_b_even = S.amdgpu.raw_buffer_load_x4(rsrc_B, byte_offset_b_even, 0, 0)
        data_b_even_bf16 = S.view(data_b_even, S.Tensor((8,), S.bf16))

        # Store to shared memory only for threads 128-255
        for i in S.range(8):
            if tid >= 128:
                B_shared_0[row_b, col_start_b + i] = data_b_even_bf16[i]

        S.syncthreads()

        # Load odd tile into buffer 1 (pipelined)
        byte_offset_a_odd = (global_row_a * K + k_base_odd + col_start_a) * 2
        data_a_odd = S.amdgpu.raw_buffer_load_x4(rsrc_A, byte_offset_a_odd, 0, 0)
        data_a_odd_bf16 = S.view(data_a_odd, S.Tensor((8,), S.bf16))

        for i in S.range(8):
            if tid < 128:
                A_shared_1[row_a, col_start_a + i] = data_a_odd_bf16[i]

        byte_offset_b_odd = ((k_base_odd + row_b) * N + bx * TILE_N + col_start_b) * 2
        data_b_odd = S.amdgpu.raw_buffer_load_x4(rsrc_B, byte_offset_b_odd, 0, 0)
        data_b_odd_bf16 = S.view(data_b_odd, S.Tensor((8,), S.bf16))

        for i in S.range(8):
            if tid >= 128:
                B_shared_1[row_b, col_start_b + i] = data_b_odd_bf16[i]

        # Compute on buffer 0 (overlaps with odd tile load)
        for k_sub in S.range(2):
            k_offset = k_sub * 8

            a_row_lds = warp_m * 32 + (lane % 32)
            a_col_lds = k_offset + (lane // 32) * 4

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

            b_row_lds = k_offset + (lane // 32) * 4
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

            S.syncthreads()

            a_row_tensor = A_frag_tensor[warp_id, lane]
            b_row_tensor = B_frag_tensor[warp_id, lane]

            a_view = S.view(a_row_tensor, S.Tensor((1, 4, 1), S.bf16))
            b_view = S.view(b_row_tensor, S.Tensor((1, 4, 1), S.bf16))

            acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_view[0], b_view[0], acc)

        S.syncthreads()

        # Compute on buffer 1 (odd tile)
        for k_sub in S.range(2):
            k_offset = k_sub * 8

            a_row_lds = warp_m * 32 + (lane % 32)
            a_col_lds = k_offset + (lane // 32) * 4

            a_bf16_0 = A_shared_1[a_row_lds, a_col_lds + 0]
            a_bf16_1 = A_shared_1[a_row_lds, a_col_lds + 1]
            a_bf16_2 = A_shared_1[a_row_lds, a_col_lds + 2]
            a_bf16_3 = A_shared_1[a_row_lds, a_col_lds + 3]

            a_u16_0 = S.bitcast(a_bf16_0, S.u16)
            a_u16_1 = S.bitcast(a_bf16_1, S.u16)
            a_u16_2 = S.bitcast(a_bf16_2, S.u16)
            a_u16_3 = S.bitcast(a_bf16_3, S.u16)

            a_u32_0 = a_u16_0 | (a_u16_1 << 16)
            a_u32_1 = a_u16_2 | (a_u16_3 << 16)

            A_frag_lds[warp_id, lane, 0] = a_u32_0
            A_frag_lds[warp_id, lane, 1] = a_u32_1

            b_row_lds = k_offset + (lane // 32) * 4
            b_col_lds = warp_n * 32 + (lane % 32)

            b_bf16_0 = B_shared_1[b_row_lds + 0, b_col_lds]
            b_bf16_1 = B_shared_1[b_row_lds + 1, b_col_lds]
            b_bf16_2 = B_shared_1[b_row_lds + 2, b_col_lds]
            b_bf16_3 = B_shared_1[b_row_lds + 3, b_col_lds]

            b_u16_0 = S.bitcast(b_bf16_0, S.u16)
            b_u16_1 = S.bitcast(b_bf16_1, S.u16)
            b_u16_2 = S.bitcast(b_bf16_2, S.u16)
            b_u16_3 = S.bitcast(b_bf16_3, S.u16)

            b_u32_0 = b_u16_0 | (b_u16_1 << 16)
            b_u32_1 = b_u16_2 | (b_u16_3 << 16)

            B_frag_lds[warp_id, lane, 0] = b_u32_0
            B_frag_lds[warp_id, lane, 1] = b_u32_1

            S.syncthreads()

            a_row_tensor = A_frag_tensor[warp_id, lane]
            b_row_tensor = B_frag_tensor[warp_id, lane]

            a_view = S.view(a_row_tensor, S.Tensor((1, 4, 1), S.bf16))
            b_view = S.view(b_row_tensor, S.Tensor((1, 4, 1), S.bf16))

            acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_view[0], b_view[0], acc)

        S.syncthreads()

    # Write results using direct tensor assignment
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
        if tuple(A.shape) != (4096, 4096) or tuple(B.shape) != (4096, 4096):
            return torch.matmul(A, B)

        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((4096, 4096), device=A.device, dtype=A.dtype)

        grid, block = kernel_launch_config()
        gemm_mfma_kernel[lambda: (grid, block)](A, B, C)

        return C
