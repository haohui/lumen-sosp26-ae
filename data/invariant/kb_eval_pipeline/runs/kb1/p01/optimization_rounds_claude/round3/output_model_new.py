import torch
import torch.nn as nn

import substrate
import substrate.language as S


N = 4096

# Tile sizes for MFMA-based GEMM
# Each MFMA computes 32x32x8 (M x N x K)
# We tile the output: 64x64 per block, with 32x32 per warp
# K is processed in tiles of 16 (2 MFMA steps per K tile)

BLOCK_M = 64  # 2 warps in M dimension
BLOCK_N = 64  # 2 warps in N dimension
BLOCK_K = 16  # K tile size per buffer (2 x 8 = 16)

# Raw buffer range is expressed in bytes
A_BYTE_RANGE = N * N * 2  # bf16 = 2 bytes
B_BYTE_RANGE = N * N * 2

# Number of chunks for raw_buffer_load_x4 (each loads 16 bytes = 8 bf16)
A_NUM_CHUNKS = BLOCK_M * BLOCK_K // 8  # 128
B_NUM_CHUNKS = BLOCK_K * BLOCK_N // 8  # 128


@substrate.jit
def gemm_mfma_kernel(
    A: S.Tensor((N, N), S.bf16),
    B: S.Tensor((N, N), S.bf16),
    C: S.Tensor((N, N), S.bf16),
):
    # Thread indices
    lane_id = S.thread_id(0)  # 0-63, lane within wave
    wave_id = S.thread_id(1)  # 0-3, wave within block

    # Block indices
    block_m = S.block_id(0)
    block_n = S.block_id(1)

    # 2x2 warp grid within each block
    warp_row = wave_id // 2
    warp_col = wave_id % 2

    # Warp output tile offset
    warp_m_base = warp_row * 32
    warp_n_base = warp_col * 32

    # Global base row/col for this warp's output tile
    global_m_base = block_m * BLOCK_M + warp_m_base
    global_n_base = block_n * BLOCK_N + warp_n_base

    # Accumulator for 32x32 output tile (16 f32 per lane)
    acc = S.full((16,), 0.0, S.f32)

    # Create buffer resource descriptors with range for OOB handling
    a_rsrc = S.amdgpu.make_rsrc(A, A_BYTE_RANGE)
    b_rsrc = S.amdgpu.make_rsrc(B, B_BYTE_RANGE)

    # Double buffering: 2 LDS buffers for A and B
    # Use stage dimension to match the reference implementation's pattern
    a_stage = S.make_shared((2, BLOCK_M, BLOCK_K), S.bf16)
    b_stage = S.make_shared((2, BLOCK_K, BLOCK_N), S.bf16)

    # View as u32 for raw_buffer_load_x4 (loads 4 x i32 = 8 bf16)
    a_stage_u32 = S.view(a_stage, S.Tensor((2, BLOCK_M, BLOCK_K // 8, 4), S.u32))
    b_stage_u32 = S.view(b_stage, S.Tensor((2, BLOCK_K, BLOCK_N // 8, 4), S.u32))

    num_k_tiles = N // BLOCK_K
    thread_id_in_block = wave_id * 64 + lane_id

    # =================================================================
    # Prologue: Load first K tile into stage 0
    # =================================================================
    if thread_id_in_block < A_NUM_CHUNKS:
        row = thread_id_in_block // 2
        k_chunk = thread_id_in_block % 2
        global_row = block_m * BLOCK_M + row
        global_col = k_chunk * 8
        byte_offset = S.convert((global_row * N + global_col) * 2, S.i32)
        a_stage_u32[0, row, k_chunk] = S.amdgpu.raw_buffer_load_x4(
            a_rsrc,
            byte_offset,
            S.convert(0, S.i32),
            S.convert(0, S.i32),
        )

    if thread_id_in_block < B_NUM_CHUNKS:
        k_row = thread_id_in_block // 8
        col_chunk = thread_id_in_block % 8
        global_row = k_row
        global_col = block_n * BLOCK_N + col_chunk * 8
        byte_offset = S.convert((global_row * N + global_col) * 2, S.i32)
        b_stage_u32[0, k_row, col_chunk] = S.amdgpu.raw_buffer_load_x4(
            b_rsrc,
            byte_offset,
            S.convert(0, S.i32),
            S.convert(0, S.i32),
        )

    S.syncthreads()

    # =================================================================
    # Main loop: Software pipelined with double buffering
    # Unrolled by 2 K tiles per iteration
    # =================================================================
    num_tile_pairs = num_k_tiles // 2

    for pair_idx in S.range(num_tile_pairs):
        k_tile_0 = pair_idx * 2
        k_tile_1 = pair_idx * 2 + 1
        k_tile_next = pair_idx * 2 + 2

        # -------------------------------------------------------------
        # Phase 1: Load tile 1 into stage 1, compute on stage 0
        # -------------------------------------------------------------
        k_base_1 = k_tile_1 * BLOCK_K

        if thread_id_in_block < A_NUM_CHUNKS:
            row = thread_id_in_block // 2
            k_chunk = thread_id_in_block % 2
            global_row = block_m * BLOCK_M + row
            global_col = k_base_1 + k_chunk * 8
            byte_offset = S.convert((global_row * N + global_col) * 2, S.i32)
            a_stage_u32[1, row, k_chunk] = S.amdgpu.raw_buffer_load_x4(
                a_rsrc,
                byte_offset,
                S.convert(0, S.i32),
                S.convert(0, S.i32),
            )

        if thread_id_in_block < B_NUM_CHUNKS:
            k_row = thread_id_in_block // 8
            col_chunk = thread_id_in_block % 8
            global_row = k_base_1 + k_row
            global_col = block_n * BLOCK_N + col_chunk * 8
            byte_offset = S.convert((global_row * N + global_col) * 2, S.i32)
            b_stage_u32[1, k_row, col_chunk] = S.amdgpu.raw_buffer_load_x4(
                b_rsrc,
                byte_offset,
                S.convert(0, S.i32),
                S.convert(0, S.i32),
            )

        # Compute MFMA on stage 0
        for k_step in S.range(2):
            k_offset = k_step * 8
            a_row = lane_id % 32
            k_group = lane_id // 32

            a_frag = S.make_local((4,), S.bf16)
            for j in S.range(4):
                a_frag[j] = a_stage[0, warp_m_base + a_row, k_offset + k_group * 4 + j]

            b_col = lane_id % 32
            b_k_group = lane_id // 32

            b_frag = S.make_local((4,), S.bf16)
            for k in S.range(4):
                b_frag[k] = b_stage[0, k_offset + b_k_group * 4 + k, warp_n_base + b_col]

            a_vec = S.view(a_frag, S.Tensor((4,), S.bf16))
            b_vec = S.view(b_frag, S.Tensor((4,), S.bf16))
            acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_vec, b_vec, acc)

        S.syncthreads()

        # -------------------------------------------------------------
        # Phase 2: Load next tile into stage 0, compute on stage 1
        # -------------------------------------------------------------
        k_base_next = k_tile_next * BLOCK_K

        if thread_id_in_block < A_NUM_CHUNKS:
            row = thread_id_in_block // 2
            k_chunk = thread_id_in_block % 2
            global_row = block_m * BLOCK_M + row
            global_col = k_base_next + k_chunk * 8
            byte_offset = S.convert((global_row * N + global_col) * 2, S.i32)
            a_stage_u32[0, row, k_chunk] = S.amdgpu.raw_buffer_load_x4(
                a_rsrc,
                byte_offset,
                S.convert(0, S.i32),
                S.convert(0, S.i32),
            )

        if thread_id_in_block < B_NUM_CHUNKS:
            k_row = thread_id_in_block // 8
            col_chunk = thread_id_in_block % 8
            global_row = k_base_next + k_row
            global_col = block_n * BLOCK_N + col_chunk * 8
            byte_offset = S.convert((global_row * N + global_col) * 2, S.i32)
            b_stage_u32[0, k_row, col_chunk] = S.amdgpu.raw_buffer_load_x4(
                b_rsrc,
                byte_offset,
                S.convert(0, S.i32),
                S.convert(0, S.i32),
            )

        # Compute MFMA on stage 1
        for k_step in S.range(2):
            k_offset = k_step * 8
            a_row = lane_id % 32
            k_group = lane_id // 32

            a_frag = S.make_local((4,), S.bf16)
            for j in S.range(4):
                a_frag[j] = a_stage[1, warp_m_base + a_row, k_offset + k_group * 4 + j]

            b_col = lane_id % 32
            b_k_group = lane_id // 32

            b_frag = S.make_local((4,), S.bf16)
            for k in S.range(4):
                b_frag[k] = b_stage[1, k_offset + b_k_group * 4 + k, warp_n_base + b_col]

            a_vec = S.view(a_frag, S.Tensor((4,), S.bf16))
            b_vec = S.view(b_frag, S.Tensor((4,), S.bf16))
            acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_vec, b_vec, acc)

        S.syncthreads()

    # Write accumulator to global memory
    for acc_idx in S.range(16):
        col = lane_id % 32
        row = 8 * (acc_idx // 4) + 4 * (lane_id // 32) + (acc_idx % 4)

        global_row = global_m_base + row
        global_col = global_n_base + col

        C[global_row, global_col] = S.convert(acc[acc_idx], S.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        if (
            tuple(A.shape) != (N, N)
            or tuple(B.shape) != (N, N)
            or A.dtype != torch.bfloat16
            or B.dtype != torch.bfloat16
            or A.device != B.device
        ):
            return torch.matmul(A, B)

        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((N, N), device=A.device, dtype=A.dtype)

        grid = lambda: ((N // BLOCK_M, N // BLOCK_N, 1), (64, 4, 1))
        gemm_mfma_kernel[grid](A, B, C)

        return C
