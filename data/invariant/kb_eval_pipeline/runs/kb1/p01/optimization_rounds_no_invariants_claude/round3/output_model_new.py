import torch
import torch.nn as nn

import substrate
import substrate.language as S


N = 4096
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 16
WARP_SIZE = 64
NUM_WARPS = 4
THREADS = NUM_WARPS * WARP_SIZE

# Range in bytes for the full matrix
MATRIX_RANGE = N * N * 2


@substrate.jit
def gemm_mfma_kernel(
    A: S.Tensor((N, N), S.bf16),
    B: S.Tensor((N, N), S.bf16),
    C: S.Tensor((N, N), S.bf16),
):
    tid = S.thread_id(0)
    bx = S.block_id(0)
    by = S.block_id(1)

    warp_id = tid // WARP_SIZE
    lane = tid % WARP_SIZE
    warp_m = warp_id // 2
    warp_n = warp_id % 2

    tile_row_base = by * BLOCK_M + warp_m * 32
    tile_col_base = bx * BLOCK_N + warp_n * 32

    acc = S.full((16,), 0.0, S.f32)

    # Create resource descriptors with range for OOB handling
    # When range is set, raw_buffer_load_x2 returns 0 for OOB elements
    rsrc_A = S.amdgpu.make_rsrc(A, MATRIX_RANGE)
    rsrc_B = S.amdgpu.make_rsrc(B, MATRIX_RANGE)

    # Double-buffered LDS for A and B tiles
    # A: 64x16 bf16, B: 16x64 bf16
    A_lds = S.make_shared((2, BLOCK_M, BLOCK_K), S.bf16)
    B_lds = S.make_shared((2, BLOCK_K, BLOCK_N), S.bf16)

    num_k_tiles = N // BLOCK_K

    # Thread-to-LDS mapping: each thread loads 4 bf16 elements
    # Using the same mapping as the working compare_kernels.py
    a_offset = tid * 4
    b_offset = tid * 4
    a_elem_row = a_offset // 16  # row in A tile (0-63)
    a_elem_col = a_offset % 16   # starting col in A tile (0-15)
    b_elem_row = b_offset // 64  # row in B tile (0-15)
    b_elem_col = b_offset % 64   # starting col in B tile (0-63)

    # Global indices for loading
    a_global_row = by * BLOCK_M + a_elem_row
    b_global_col = bx * BLOCK_N + b_elem_col

    # MFMA fragment indices
    a_row = lane % 32
    a_k_group = lane // 32
    a_lds_row = warp_m * 32 + a_row

    b_col = lane % 32
    b_k_group = lane // 32
    b_lds_col = warp_n * 32 + b_col

    # Prologue: load first K tile into buffer 0 using raw_buffer_load_x2
    # A: byte offset = (row * N + col) * 2
    a_byte_offset = (a_global_row * N + a_elem_col) * 2
    a_data = S.amdgpu.raw_buffer_load_x2(rsrc_A, a_byte_offset, 0, 0)
    a_data_bf16 = S.view(a_data, S.Tensor((4,), S.bf16))
    for i in S.range(4):
        A_lds[0, a_elem_row, a_elem_col + i] = a_data_bf16[i]

    # B: byte offset = (row * N + col) * 2
    b_byte_offset = (b_elem_row * N + b_global_col) * 2
    b_data = S.amdgpu.raw_buffer_load_x2(rsrc_B, b_byte_offset, 0, 0)
    b_data_bf16 = S.view(b_data, S.Tensor((4,), S.bf16))
    for i in S.range(4):
        B_lds[0, b_elem_row, b_elem_col + i] = b_data_bf16[i]

    S.syncthreads()

    # Main K-loop: unrolled by 2 with double buffering
    for k_outer in S.range(num_k_tiles // 2):
        k0 = k_outer * 2
        k1 = k_outer * 2 + 1
        k2 = k_outer * 2 + 2

        buf0 = k0 % 2
        buf1 = k1 % 2
        buf2 = k2 % 2

        # === Compute on buf0 (K tile k0) ===
        # Load A fragment from LDS - split into 2 halves for fine-grained overlap
        a_frag0 = S.make_local((4,), S.bf16)
        for elem in S.range(4):
            k_idx = a_k_group * 4 + elem
            a_frag0[elem] = A_lds[buf0, a_lds_row, k_idx]

        a_frag1 = S.make_local((4,), S.bf16)
        for elem in S.range(4):
            k_idx = 8 + a_k_group * 4 + elem
            a_frag1[elem] = A_lds[buf0, a_lds_row, k_idx]

        # Load B fragment from LDS
        b_frag0 = S.make_local((4,), S.bf16)
        for elem in S.range(4):
            k_idx = b_k_group * 4 + elem
            b_frag0[elem] = B_lds[buf0, k_idx, b_lds_col]

        b_frag1 = S.make_local((4,), S.bf16)
        for elem in S.range(4):
            k_idx = 8 + b_k_group * 4 + elem
            b_frag1[elem] = B_lds[buf0, k_idx, b_lds_col]

        # Issue 2 MFMA instructions for K=16
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0, b_frag0, acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1, b_frag1, acc)

        # === Load buf1 (K tile k1) - overlapped with compute ===
        k_base1 = k1 * BLOCK_K
        a_byte_offset_k1 = (a_global_row * N + k_base1 + a_elem_col) * 2
        a_data_k1 = S.amdgpu.raw_buffer_load_x2(rsrc_A, a_byte_offset_k1, 0, 0)
        a_data_k1_bf16 = S.view(a_data_k1, S.Tensor((4,), S.bf16))
        for i in S.range(4):
            A_lds[buf1, a_elem_row, a_elem_col + i] = a_data_k1_bf16[i]

        b_byte_offset_k1 = ((k_base1 + b_elem_row) * N + b_global_col) * 2
        b_data_k1 = S.amdgpu.raw_buffer_load_x2(rsrc_B, b_byte_offset_k1, 0, 0)
        b_data_k1_bf16 = S.view(b_data_k1, S.Tensor((4,), S.bf16))
        for i in S.range(4):
            B_lds[buf1, b_elem_row, b_elem_col + i] = b_data_k1_bf16[i]

        S.syncthreads()

        # === Compute on buf1 (K tile k1) ===
        a_frag2 = S.make_local((4,), S.bf16)
        for elem in S.range(4):
            k_idx = a_k_group * 4 + elem
            a_frag2[elem] = A_lds[buf1, a_lds_row, k_idx]

        a_frag3 = S.make_local((4,), S.bf16)
        for elem in S.range(4):
            k_idx = 8 + a_k_group * 4 + elem
            a_frag3[elem] = A_lds[buf1, a_lds_row, k_idx]

        b_frag2 = S.make_local((4,), S.bf16)
        for elem in S.range(4):
            k_idx = b_k_group * 4 + elem
            b_frag2[elem] = B_lds[buf1, k_idx, b_lds_col]

        b_frag3 = S.make_local((4,), S.bf16)
        for elem in S.range(4):
            k_idx = 8 + b_k_group * 4 + elem
            b_frag3[elem] = B_lds[buf1, k_idx, b_lds_col]

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag2, b_frag2, acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag3, b_frag3, acc)

        # === Prefetch buf2 (K tile k2) - overlapped with compute ===
        # Range in rsrc handles OOB: returns 0 for OOB loads
        k_base2 = k2 * BLOCK_K
        a_byte_offset_k2 = (a_global_row * N + k_base2 + a_elem_col) * 2
        a_data_k2 = S.amdgpu.raw_buffer_load_x2(rsrc_A, a_byte_offset_k2, 0, 0)
        a_data_k2_bf16 = S.view(a_data_k2, S.Tensor((4,), S.bf16))
        for i in S.range(4):
            A_lds[buf2, a_elem_row, a_elem_col + i] = a_data_k2_bf16[i]

        b_byte_offset_k2 = ((k_base2 + b_elem_row) * N + b_global_col) * 2
        b_data_k2 = S.amdgpu.raw_buffer_load_x2(rsrc_B, b_byte_offset_k2, 0, 0)
        b_data_k2_bf16 = S.view(b_data_k2, S.Tensor((4,), S.bf16))
        for i in S.range(4):
            B_lds[buf2, b_elem_row, b_elem_col + i] = b_data_k2_bf16[i]

        S.syncthreads()

    # Write output with swizzle pattern
    for acc_idx in S.range(16):
        col = tile_col_base + (lane % 32)
        row = tile_row_base + 8 * (acc_idx // 4) + 4 * (lane // 32) + (acc_idx % 4)
        C[row, col] = S.convert(acc[acc_idx], S.bf16)


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

        grid_x = N // BLOCK_N
        grid_y = N // BLOCK_M

        gemm_mfma_kernel[lambda: ((grid_x, grid_y, 1), (THREADS, 1, 1))](A, B, C)
        return C
