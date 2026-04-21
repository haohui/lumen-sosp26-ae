import torch
import torch.nn as nn

import substrate
import substrate.language as S


M = 8205
K = 2949
N = 5921

# Tile sizes for 4-warps (2x2 warp grid)
TILE_M = 64
TILE_N = 64
TILE_K = 16  # 2 MFMA iterations per K tile
WARP_SIZE = 64
NUM_WARPS = 4
THREADS = WARP_SIZE * NUM_WARPS


@substrate.jit
def gemm_mfma_kernel(
    A: S.Tensor((8205, 2949), S.bf16),
    B: S.Tensor((2949, 5921), S.bf16),
    C: S.Tensor((8205, 5921), S.bf16),
):
    """GEMM kernel using MFMA 32x32x8_bf16_f32."""
    bx = S.block_id(0)
    by = S.block_id(1)
    tid = S.thread_id(0)
    lane = tid % WARP_SIZE
    warp_id = tid // WARP_SIZE

    warp_m = warp_id // 2
    warp_n = warp_id % 2

    tile_m_base = by * TILE_M + warp_m * 32
    tile_n_base = bx * TILE_N + warp_n * 32

    # Buffer descriptors
    rsrc_A = S.amdgpu.make_rsrc(A, M * K * 2)
    rsrc_B = S.amdgpu.make_rsrc(B, K * N * 2)
    C_RANGE_BYTES = M * N * 2

    # LDS for staging
    A_shared = S.make_shared((TILE_M, TILE_K), S.bf16)
    B_shared = S.make_shared((TILE_K, TILE_N), S.bf16)

    # Fragment LDS for MFMA input
    A_frag_lds = S.make_shared((NUM_WARPS, WARP_SIZE, 2), S.u32)
    B_frag_lds = S.make_shared((NUM_WARPS, WARP_SIZE, 2), S.u32)

    acc = S.full((16,), 0.0, S.f32)

    num_k_tiles = (K + TILE_K - 1) // TILE_K

    for k_tile in S.range(num_k_tiles):
        k_base = k_tile * TILE_K

        # Load A into LDS using range to handle OOB
        # 64 rows x 16 cols = 1024 bf16 = 2048 bytes
        # 128 threads x 8 bf16 = 1024 bf16
        if tid < 128:
            row_a = tid // 2
            col_start_a = (tid % 2) * 8

            global_row_a = by * TILE_M + row_a
            global_col_a = k_base + col_start_a

            byte_offset_a = (global_row_a * K + global_col_a) * 2
            data_a = S.amdgpu.raw_buffer_load_x4(rsrc_A, byte_offset_a, 0, 0, range=M * K * 2)
            data_a_bf16 = S.view(data_a, S.Tensor((8,), S.bf16))
            for i in S.range(8):
                A_shared[row_a, col_start_a + i] = data_a_bf16[i]

        # Load B into LDS using range to handle OOB
        # 16 rows x 64 cols = 1024 bf16 = 2048 bytes
        # 128 threads x 8 bf16 = 1024 bf16
        if tid < 128:
            row_b = tid // 8
            col_start_b = (tid % 8) * 8

            global_row_b = k_base + row_b
            global_col_b = bx * TILE_N + col_start_b

            byte_offset_b = (global_row_b * N + global_col_b) * 2
            data_b = S.amdgpu.raw_buffer_load_x4(rsrc_B, byte_offset_b, 0, 0, range=K * N * 2)
            data_b_bf16 = S.view(data_b, S.Tensor((8,), S.bf16))
            for i in S.range(8):
                B_shared[row_b, col_start_b + i] = data_b_bf16[i]

        S.syncthreads()

        # Compute MFMA for this K tile (2 MFMA calls)
        for k_sub in S.range(2):
            k_offset = k_sub * 8

            # Load A fragment from LDS
            a_row_lds = warp_m * 32 + (lane % 32)
            a_col_lds = k_offset + (lane // 32) * 4

            a_bf16_0 = A_shared[a_row_lds, a_col_lds + 0]
            a_bf16_1 = A_shared[a_row_lds, a_col_lds + 1]
            a_bf16_2 = A_shared[a_row_lds, a_col_lds + 2]
            a_bf16_3 = A_shared[a_row_lds, a_col_lds + 3]

            a_u16_0 = S.bitcast(a_bf16_0, S.u16)
            a_u16_1 = S.bitcast(a_bf16_1, S.u16)
            a_u16_2 = S.bitcast(a_bf16_2, S.u16)
            a_u16_3 = S.bitcast(a_bf16_3, S.u16)

            a_u32_0 = a_u16_0 | (a_u16_1 << 16)
            a_u32_1 = a_u16_2 | (a_u16_3 << 16)

            A_frag_lds[warp_id, lane, 0] = a_u32_0
            A_frag_lds[warp_id, lane, 1] = a_u32_1

            # Load B fragment from LDS
            b_row_lds = k_offset + (lane // 32) * 4
            b_col_lds = warp_n * 32 + (lane % 32)

            b_bf16_0 = B_shared[b_row_lds + 0, b_col_lds]
            b_bf16_1 = B_shared[b_row_lds + 1, b_col_lds]
            b_bf16_2 = B_shared[b_row_lds + 2, b_col_lds]
            b_bf16_3 = B_shared[b_row_lds + 3, b_col_lds]

            b_u16_0 = S.bitcast(b_bf16_0, S.u16)
            b_u16_1 = S.bitcast(b_bf16_1, S.u16)
            b_u16_2 = S.bitcast(b_bf16_2, S.u16)
            b_u16_3 = S.bitcast(b_bf16_3, S.u16)

            b_u32_0 = b_u16_0 | (b_u16_1 << 16)
            b_u32_1 = b_u16_2 | (b_u16_3 << 16)

            B_frag_lds[warp_id, lane, 0] = b_u32_0
            B_frag_lds[warp_id, lane, 1] = b_u32_1

            S.syncthreads()

            # View for MFMA
            A_frag_tensor = S.view(A_frag_lds, S.Tensor((NUM_WARPS, WARP_SIZE, 2), S.u32))
            B_frag_tensor = S.view(B_frag_lds, S.Tensor((NUM_WARPS, WARP_SIZE, 2), S.u32))

            a_row_tensor = A_frag_tensor[warp_id, lane]
            b_row_tensor = B_frag_tensor[warp_id, lane]

            a_view = S.view(a_row_tensor, S.Tensor((1, 4, 1), S.bf16))
            b_view = S.view(b_row_tensor, S.Tensor((1, 4, 1), S.bf16))

            acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_view[0], b_view[0], acc)

        S.syncthreads()

    # Write output using raw_buffer_store_x1 with row-specific range to handle OOB
    for acc_idx in S.range(16):
        row_offset = 8 * (acc_idx // 4) + 4 * (lane // 32) + (acc_idx % 4)
        col_offset = lane % 32

        global_row = tile_m_base + row_offset
        global_col = tile_n_base + col_offset

        c_row_range = S.min((global_row * N + N) * 2, C_RANGE_BYTES)
        c_row_rsrc = S.amdgpu.make_rsrc(C, c_row_range)
        c_val = S.bitcast(S.convert(acc[acc_idx], S.bf16), S.u16)
        S.amdgpu.raw_buffer_store_x1(c_val, c_row_rsrc, (global_row * N + global_col) * 2, 0, 0)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if tuple(A.shape) != (8205, 2949) or tuple(B.shape) != (2949, 5921):
            return torch.matmul(A, B)

        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((8205, 5921), device=A.device, dtype=A.dtype)

        grid_m = (M + TILE_M - 1) // TILE_M
        grid_n = (N + TILE_N - 1) // TILE_N

        gemm_mfma_kernel[lambda: ((grid_n, grid_m, 1), (THREADS, 1, 1))](A, B, C)

        return C
