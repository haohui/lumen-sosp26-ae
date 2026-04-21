import torch
import torch.nn as nn

import substrate
import substrate.language as S

M = 32768
K = 64
N = 32768

# MFMA tile dimensions
MFMA_M = 32
MFMA_N = 32
MFMA_K = 8

# Block dimensions: 64x64 output tile with 4 waves
BLOCK_M = 64
BLOCK_N = 64


@substrate.jit
def gemm_kernel(
    A: S.Tensor((32768, 64), S.bf16),
    B: S.Tensor((64, 32768), S.bf16),
    C: S.Tensor((32768, 32768), S.bf16),
):
    # Block indices
    block_m = S.block_id(0)
    block_n = S.block_id(1)

    # Thread/wave indices
    lane = S.thread_id(0)  # 0-255 for 4 waves
    wave_id = lane // 64   # 0, 1, 2, 3
    lane_in_wave = lane % 64  # 0-63 within wave

    # Wave position in 2x2 grid
    wave_row = wave_id // 2  # 0 or 1
    wave_col = wave_id % 2   # 0 or 1

    # Output tile position for this wave
    tile_row_base = block_m * BLOCK_M + wave_row * MFMA_M
    tile_col_base = block_n * BLOCK_N + wave_col * MFMA_N

    # Initialize accumulator (16 f32 values per lane)
    acc = S.full((16,), 0.0, S.f32)

    # Global base addresses for this block
    block_row_base = block_m * BLOCK_M
    block_col_base = block_n * BLOCK_N

    # Create buffer resources with range for OOB handling
    # Range is in bytes - OOB loads return 0, OOB stores are discarded
    # This allows removing the branch guarding OOB access
    A_range = M * K * 2  # 32768 * 64 * 2 = 4194304 bytes
    B_range = K * N * 2  # 64 * 32768 * 2 = 4194304 bytes
    rsrc_A = S.amdgpu.make_rsrc(A, A_range)
    rsrc_B = S.amdgpu.make_rsrc(B, B_range)

    # Double-buffered shared memory
    # A: 64 rows x 16 cols (for K=16 per iteration, 2 MFMA pairs)
    A_shared = S.make_shared((2, 64, 16), S.bf16)
    # B: 16 rows x 64 cols
    B_shared = S.make_shared((2, 16, 64), S.bf16)

    # Thread to element mapping for cooperative loads
    # 256 threads load 64*16=1024 elements for A
    # Each thread loads 4 elements
    elem0 = lane
    row0_a = elem0 // 16
    col0_a = elem0 % 16
    elem1 = lane + 256
    row1_a = elem1 // 16
    col1_a = elem1 % 16
    elem2 = lane + 512
    row2_a = elem2 // 16
    col2_a = elem2 % 16
    elem3 = lane + 768
    row3_a = elem3 // 16
    col3_a = elem3 % 16

    # For B: 16*64=1024 elements
    row0_b = elem0 // 64
    col0_b = elem0 % 64
    row1_b = elem1 // 64
    col1_b = elem1 % 64
    row2_b = elem2 // 64
    col2_b = elem2 % 64
    row3_b = elem3 // 64
    col3_b = elem3 % 64

    # MFMA swizzle indices
    a_row = lane_in_wave % 32
    a_col_base = (lane_in_wave // 32) * 4
    b_col = lane_in_wave % 32
    b_row_base = (lane_in_wave // 32) * 4

    # ========== Software Pipelining ==========
    # K=64 total, K=16 per iteration (2 MFMA), 4 iterations
    # Use make_rsrc with range to eliminate OOB branches

    # Prefetch K=0..15 to buffer 0 (use original scalar loads for correctness)
    k_offset = 0
    buf_idx = 0
    A_shared[buf_idx, row0_a, col0_a] = A[block_row_base + row0_a, k_offset + col0_a]
    A_shared[buf_idx, row1_a, col1_a] = A[block_row_base + row1_a, k_offset + col1_a]
    A_shared[buf_idx, row2_a, col2_a] = A[block_row_base + row2_a, k_offset + col2_a]
    A_shared[buf_idx, row3_a, col3_a] = A[block_row_base + row3_a, k_offset + col3_a]
    B_shared[buf_idx, row0_b, col0_b] = B[k_offset + row0_b, block_col_base + col0_b]
    B_shared[buf_idx, row1_b, col1_b] = B[k_offset + row1_b, block_col_base + col1_b]
    B_shared[buf_idx, row2_b, col2_b] = B[k_offset + row2_b, block_col_base + col2_b]
    B_shared[buf_idx, row3_b, col3_b] = B[k_offset + row3_b, block_col_base + col3_b]

    S.syncthreads()

    # Main loop: unrolled by 2
    # Each unrolled iteration handles K=32 (4 MFMA operations)
    for k_outer in S.range(2):
        # === First sub-iteration: k = k_outer * 2, K offset = k_outer * 32 ===
        k0 = k_outer * 32
        buf0 = (k_outer * 2) % 2

        # First MFMA pair (K=0..7 within this chunk)
        a_frag0 = S.make_local((4,), S.bf16)
        for e in S.range(4):
            a_frag0[e] = A_shared[buf0, wave_row * 32 + a_row, a_col_base + e]

        b_frag0 = S.make_local((4,), S.bf16)
        for e in S.range(4):
            b_frag0[e] = B_shared[buf0, b_row_base + e, wave_col * 32 + b_col]

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0, b_frag0, acc)

        # Second MFMA pair (K=8..15 within this chunk)
        a_frag1 = S.make_local((4,), S.bf16)
        for e in S.range(4):
            a_frag1[e] = A_shared[buf0, wave_row * 32 + a_row, 8 + a_col_base + e]

        b_frag1 = S.make_local((4,), S.bf16)
        for e in S.range(4):
            b_frag1[e] = B_shared[buf0, 8 + b_row_base + e, wave_col * 32 + b_col]

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1, b_frag1, acc)

        # Load next K-chunk to other buffer
        k1 = k0 + 16
        buf1 = 1 - buf0

        A_shared[buf1, row0_a, col0_a] = A[block_row_base + row0_a, k1 + col0_a]
        A_shared[buf1, row1_a, col1_a] = A[block_row_base + row1_a, k1 + col1_a]
        A_shared[buf1, row2_a, col2_a] = A[block_row_base + row2_a, k1 + col2_a]
        A_shared[buf1, row3_a, col3_a] = A[block_row_base + row3_a, k1 + col3_a]
        B_shared[buf1, row0_b, col0_b] = B[k1 + row0_b, block_col_base + col0_b]
        B_shared[buf1, row1_b, col1_b] = B[k1 + row1_b, block_col_base + col1_b]
        B_shared[buf1, row2_b, col2_b] = B[k1 + row2_b, block_col_base + col2_b]
        B_shared[buf1, row3_b, col3_b] = B[k1 + row3_b, block_col_base + col3_b]

        S.syncthreads()

        # === Second sub-iteration: k = k_outer * 2 + 1 ===
        # Third MFMA pair (K=16..23 within this chunk)
        a_frag2 = S.make_local((4,), S.bf16)
        for e in S.range(4):
            a_frag2[e] = A_shared[buf1, wave_row * 32 + a_row, a_col_base + e]

        b_frag2 = S.make_local((4,), S.bf16)
        for e in S.range(4):
            b_frag2[e] = B_shared[buf1, b_row_base + e, wave_col * 32 + b_col]

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag2, b_frag2, acc)

        # Fourth MFMA pair (K=24..31 within this chunk)
        a_frag3 = S.make_local((4,), S.bf16)
        for e in S.range(4):
            a_frag3[e] = A_shared[buf1, wave_row * 32 + a_row, 8 + a_col_base + e]

        b_frag3 = S.make_local((4,), S.bf16)
        for e in S.range(4):
            b_frag3[e] = B_shared[buf1, 8 + b_row_base + e, wave_col * 32 + b_col]

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag3, b_frag3, acc)

        # Prefetch for next iteration - BRANCH REMOVED!
        # Use raw_buffer_load with range to handle OOB safely
        # When k_next is out of bounds, the load returns 0 which is safe
        k_next = (k_outer + 1) * 32
        buf_next = (k_outer + 1) * 2 % 2

        # Calculate byte offsets for raw buffer loads
        # A is row-major: byte offset = (row * K + col) * 2
        # B is row-major: byte offset = (row * N + col) * 2
        offset0_a = (block_row_base + row0_a) * K * 2 + (k_next + col0_a) * 2
        offset1_a = (block_row_base + row1_a) * K * 2 + (k_next + col1_a) * 2
        offset2_a = (block_row_base + row2_a) * K * 2 + (k_next + col2_a) * 2
        offset3_a = (block_row_base + row3_a) * K * 2 + (k_next + col3_a) * 2

        offset0_b = (k_next + row0_b) * N * 2 + (block_col_base + col0_b) * 2
        offset1_b = (k_next + row1_b) * N * 2 + (block_col_base + col1_b) * 2
        offset2_b = (k_next + row2_b) * N * 2 + (block_col_base + col2_b) * 2
        offset3_b = (k_next + row3_b) * N * 2 + (block_col_base + col3_b) * 2

        # Use raw_buffer_load_x1 to load 4 bytes (2 bf16)
        # The range in rsrc ensures OOB returns 0
        val0_a = S.amdgpu.raw_buffer_load_x1(rsrc_A, offset0_a, 0, 0)
        val1_a = S.amdgpu.raw_buffer_load_x1(rsrc_A, offset1_a, 0, 0)
        val2_a = S.amdgpu.raw_buffer_load_x1(rsrc_A, offset2_a, 0, 0)
        val3_a = S.amdgpu.raw_buffer_load_x1(rsrc_A, offset3_a, 0, 0)

        val0_b = S.amdgpu.raw_buffer_load_x1(rsrc_B, offset0_b, 0, 0)
        val1_b = S.amdgpu.raw_buffer_load_x1(rsrc_B, offset1_b, 0, 0)
        val2_b = S.amdgpu.raw_buffer_load_x1(rsrc_B, offset2_b, 0, 0)
        val3_b = S.amdgpu.raw_buffer_load_x1(rsrc_B, offset3_b, 0, 0)

        # View i32 as 2 bf16 and extract the first element
        A_shared[buf_next, row0_a, col0_a] = S.view(val0_a, S.Tensor((2,), S.bf16))[0]
        A_shared[buf_next, row1_a, col1_a] = S.view(val1_a, S.Tensor((2,), S.bf16))[0]
        A_shared[buf_next, row2_a, col2_a] = S.view(val2_a, S.Tensor((2,), S.bf16))[0]
        A_shared[buf_next, row3_a, col3_a] = S.view(val3_a, S.Tensor((2,), S.bf16))[0]

        B_shared[buf_next, row0_b, col0_b] = S.view(val0_b, S.Tensor((2,), S.bf16))[0]
        B_shared[buf_next, row1_b, col1_b] = S.view(val1_b, S.Tensor((2,), S.bf16))[0]
        B_shared[buf_next, row2_b, col2_b] = S.view(val2_b, S.Tensor((2,), S.bf16))[0]
        B_shared[buf_next, row3_b, col3_b] = S.view(val3_b, S.Tensor((2,), S.bf16))[0]

        S.syncthreads()

    # Write results to C using accumulator invariant
    for acc_idx in S.range(16):
        col = tile_col_base + (lane_in_wave % 32)
        row = tile_row_base + 8 * (acc_idx // 4) + 4 * (lane_in_wave // 32) + (acc_idx % 4)
        C[row, col] = S.convert(acc[acc_idx], S.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if tuple(A.shape) != (32768, 64) or tuple(B.shape) != (64, 32768):
            raise ValueError("Input shapes must be (32768, 64) and (64, 32768)")
        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((32768, 32768), device=A.device, dtype=A.dtype)

        grid = (M // BLOCK_M, N // BLOCK_N, 1)
        block = (256, 1, 1)
        gemm_kernel[lambda: (grid, block)](A, B, C)
        return C
