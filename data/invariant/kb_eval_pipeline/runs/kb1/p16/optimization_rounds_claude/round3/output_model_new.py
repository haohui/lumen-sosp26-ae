import torch
import torch.nn as nn

import substrate
import substrate.language as S

M = 2048
K = 8192
N = 4096

# Tile sizes
TILE_M = 64
TILE_N = 64
TILE_K = 16


@substrate.jit
def gemm_kernel(
    A: S.Tensor((2048, 8192), S.bf16),
    B: S.Tensor((8192, 4096), S.bf16),
    C: S.Tensor((2048, 4096), S.bf16),
):
    # Block and thread indices
    block_m = S.block_id(0)
    block_n = S.block_id(1)

    # Thread and warp indices
    tid = S.thread_id(0)
    lane = tid % 64
    warp_id = tid // 64

    # Create resource descriptors for vectorized loads with range (in bytes)
    # The range enables OOB handling: loads return 0 for OOB
    rsrc_A = S.amdgpu.make_rsrc(A, M * K * 2)
    rsrc_B = S.amdgpu.make_rsrc(B, K * N * 2)

    # Double buffered shared memory for A and B tiles
    lds_A_0 = S.make_shared((TILE_M, TILE_K), S.bf16)
    lds_A_1 = S.make_shared((TILE_M, TILE_K), S.bf16)
    lds_B_0 = S.make_shared((TILE_K, TILE_N), S.bf16)
    lds_B_1 = S.make_shared((TILE_K, TILE_N), S.bf16)

    # Local memory for accumulators (4 warps, 16 elements each)
    acc = S.make_local((4, 16), S.f32)

    # Local memory for A and B fragments
    a_frag = S.make_local((4,), S.bf16)
    b_frag = S.make_local((4,), S.bf16)

    # Initialize accumulators to zero
    for acc_i in S.range(16):
        acc[warp_id, acc_i] = S.convert(0.0, S.f32)

    # Base offsets for this block
    base_m = block_m * TILE_M
    base_n = block_n * TILE_N

    # Warp grid: 2x2
    warp_m = warp_id // 2
    warp_n = warp_id % 2
    warp_base_m = warp_m * 32
    warp_base_n = warp_n * 32

    # Fragment load indices
    row_A = lane % 32
    col_B = lane % 32
    row_B_base = (lane // 32) * 4

    # Total number of K-tiles
    NUM_K_TILES = K // TILE_K  # 512

    # Prologue: Load first tile
    base_k = 0
    for load_idx in S.range(2):
        load_id = tid + load_idx * 128
        if load_id < 128:
            row = load_id // 2
            col_start = (load_id % 2) * 8

            global_row = base_m + row
            global_col = base_k + col_start
            byte_offset = global_row * K * 2 + global_col * 2

            data = S.amdgpu.raw_buffer_load_x4(rsrc_A, byte_offset, 0, 0)

            for elem_i in S.range(4):
                u32_val = data[elem_i]
                bf16_view = S.view(u32_val, S.Tensor((2,), S.bf16))
                lds_A_0[row, col_start + elem_i * 2] = bf16_view[0]
                lds_A_0[row, col_start + elem_i * 2 + 1] = bf16_view[1]

    for load_idx in S.range(2):
        load_id = tid + load_idx * 128
        if load_id < 128:
            row = load_id // 8
            col_start = (load_id % 8) * 8

            global_row = base_k + row
            global_col = base_n + col_start
            byte_offset = global_row * N * 2 + global_col * 2

            data = S.amdgpu.raw_buffer_load_x4(rsrc_B, byte_offset, 0, 0)

            for elem_i in S.range(4):
                u32_val = data[elem_i]
                bf16_view = S.view(u32_val, S.Tensor((2,), S.bf16))
                lds_B_0[row, col_start + elem_i * 2] = bf16_view[0]
                lds_B_0[row, col_start + elem_i * 2 + 1] = bf16_view[1]

    S.syncthreads()

    # Main loop: unrolled by 2
    for k_outer in S.range(NUM_K_TILES // 2):
        # === First tile (use buffer 0) ===
        for k_step in S.range(2):
            k_offset = k_step * 8
            col_A_base = k_offset + (lane // 32) * 4
            row_B_local = k_offset + (lane // 32) * 4

            a_frag[0] = lds_A_0[warp_base_m + row_A, col_A_base + 0]
            a_frag[1] = lds_A_0[warp_base_m + row_A, col_A_base + 1]
            a_frag[2] = lds_A_0[warp_base_m + row_A, col_A_base + 2]
            a_frag[3] = lds_A_0[warp_base_m + row_A, col_A_base + 3]

            b_frag[0] = lds_B_0[row_B_local + 0, warp_base_n + col_B]
            b_frag[1] = lds_B_0[row_B_local + 1, warp_base_n + col_B]
            b_frag[2] = lds_B_0[row_B_local + 2, warp_base_n + col_B]
            b_frag[3] = lds_B_0[row_B_local + 3, warp_base_n + col_B]

            a_vec = S.view(a_frag, S.Tensor((4,), S.bf16))
            b_vec = S.view(b_frag, S.Tensor((4,), S.bf16))
            acc_vec = S.view(acc[warp_id], S.Tensor((16,), S.f32))
            acc_vec = S.amdgpu.mfma_32x32x8_bf16_f32(a_vec, b_vec, acc_vec)
            for acc_i in S.range(16):
                acc[warp_id, acc_i] = acc_vec[acc_i]

        # Load second tile into buffer 1
        k_tile_1 = k_outer * 2 + 1
        base_k_1 = k_tile_1 * TILE_K

        for load_idx in S.range(2):
            load_id = tid + load_idx * 128
            if load_id < 128:
                row = load_id // 2
                col_start = (load_id % 2) * 8

                global_row = base_m + row
                global_col = base_k_1 + col_start
                byte_offset = global_row * K * 2 + global_col * 2

                data = S.amdgpu.raw_buffer_load_x4(rsrc_A, byte_offset, 0, 0)

                for elem_i in S.range(4):
                    u32_val = data[elem_i]
                    bf16_view = S.view(u32_val, S.Tensor((2,), S.bf16))
                    lds_A_1[row, col_start + elem_i * 2] = bf16_view[0]
                    lds_A_1[row, col_start + elem_i * 2 + 1] = bf16_view[1]

        for load_idx in S.range(2):
            load_id = tid + load_idx * 128
            if load_id < 128:
                row = load_id // 8
                col_start = (load_id % 8) * 8

                global_row = base_k_1 + row
                global_col = base_n + col_start
                byte_offset = global_row * N * 2 + global_col * 2

                data = S.amdgpu.raw_buffer_load_x4(rsrc_B, byte_offset, 0, 0)

                for elem_i in S.range(4):
                    u32_val = data[elem_i]
                    bf16_view = S.view(u32_val, S.Tensor((2,), S.bf16))
                    lds_B_1[row, col_start + elem_i * 2] = bf16_view[0]
                    lds_B_1[row, col_start + elem_i * 2 + 1] = bf16_view[1]

        S.syncthreads()

        # === Second tile (use buffer 1) ===
        for k_step in S.range(2):
            k_offset = k_step * 8
            col_A_base = k_offset + (lane // 32) * 4
            row_B_local = k_offset + (lane // 32) * 4

            a_frag[0] = lds_A_1[warp_base_m + row_A, col_A_base + 0]
            a_frag[1] = lds_A_1[warp_base_m + row_A, col_A_base + 1]
            a_frag[2] = lds_A_1[warp_base_m + row_A, col_A_base + 2]
            a_frag[3] = lds_A_1[warp_base_m + row_A, col_A_base + 3]

            b_frag[0] = lds_B_1[row_B_local + 0, warp_base_n + col_B]
            b_frag[1] = lds_B_1[row_B_local + 1, warp_base_n + col_B]
            b_frag[2] = lds_B_1[row_B_local + 2, warp_base_n + col_B]
            b_frag[3] = lds_B_1[row_B_local + 3, warp_base_n + col_B]

            a_vec = S.view(a_frag, S.Tensor((4,), S.bf16))
            b_vec = S.view(b_frag, S.Tensor((4,), S.bf16))
            acc_vec = S.view(acc[warp_id], S.Tensor((16,), S.f32))
            acc_vec = S.amdgpu.mfma_32x32x8_bf16_f32(a_vec, b_vec, acc_vec)
            for acc_i in S.range(16):
                acc[warp_id, acc_i] = acc_vec[acc_i]

        # Prefetch next tile into buffer 0
        # Removed k_tile_next < NUM_K_TILES check - OOB global memory access handled by range
        k_tile_next = k_outer * 2 + 2
        base_k_next = k_tile_next * TILE_K

        for load_idx in S.range(2):
            load_id = tid + load_idx * 128
            if load_id < 128:
                row = load_id // 2
                col_start = (load_id % 2) * 8

                global_row = base_m + row
                global_col = base_k_next + col_start
                byte_offset = global_row * K * 2 + global_col * 2

                data = S.amdgpu.raw_buffer_load_x4(rsrc_A, byte_offset, 0, 0)

                for elem_i in S.range(4):
                    u32_val = data[elem_i]
                    bf16_view = S.view(u32_val, S.Tensor((2,), S.bf16))
                    lds_A_0[row, col_start + elem_i * 2] = bf16_view[0]
                    lds_A_0[row, col_start + elem_i * 2 + 1] = bf16_view[1]

        for load_idx in S.range(2):
            load_id = tid + load_idx * 128
            if load_id < 128:
                row = load_id // 8
                col_start = (load_id % 8) * 8

                global_row = base_k_next + row
                global_col = base_n + col_start
                byte_offset = global_row * N * 2 + global_col * 2

                data = S.amdgpu.raw_buffer_load_x4(rsrc_B, byte_offset, 0, 0)

                for elem_i in S.range(4):
                    u32_val = data[elem_i]
                    bf16_view = S.view(u32_val, S.Tensor((2,), S.bf16))
                    lds_B_0[row, col_start + elem_i * 2] = bf16_view[0]
                    lds_B_0[row, col_start + elem_i * 2 + 1] = bf16_view[1]

        S.syncthreads()

    # Write results to global memory
    tile_row_base = warp_m * 32
    tile_col_base = warp_n * 32

    for acc_idx in S.range(16):
        col = tile_col_base + (lane % 32)
        row = tile_row_base + 8 * (acc_idx // 4) + 4 * (lane // 32) + (acc_idx % 4)

        global_row = base_m + row
        global_col = base_n + col

        c_val = acc[warp_id, acc_idx]
        c_bf16 = S.convert(c_val, S.bf16)
        C[global_row, global_col] = c_bf16


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if tuple(A.shape) != (8192, 2048) or tuple(B.shape) != (8192, 4096):
            return torch.matmul(A.T, B)
        A2 = A.transpose(-2, -1).contiguous()
        B2 = B.contiguous()
        C = torch.empty((2048, 4096), device=A.device, dtype=A.dtype)

        grid = (M // TILE_M, N // TILE_N, 1)
        block = (256, 1, 1)
        gemm_kernel[lambda: (grid, block)](A2, B2, C)
        return C
