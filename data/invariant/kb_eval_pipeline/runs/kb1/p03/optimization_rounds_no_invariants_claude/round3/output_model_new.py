import torch
import torch.nn as nn

import substrate
import substrate.language as S

BATCH = 128
M = 512
K = 1024
N = 2048

# Tile sizes for single wave kernel
TILE_M = 32
TILE_N = 32
TILE_K = 8
WARP_SIZE = 64


@substrate.jit
def bmm_kernel(
    A: S.Tensor((BATCH, M, K), S.bf16),
    B: S.Tensor((BATCH, K, N), S.bf16),
    C: S.Tensor((BATCH, M, N), S.bf16),
):
    batch = S.block_id(0)
    block_m = S.block_id(1)
    block_n = S.block_id(2)

    lane = S.thread_id(0)  # 0-63 for single wave

    # Base offsets for this block's output tile
    m_base = block_m * TILE_M
    n_base = block_n * TILE_N

    # Allocate double-buffered LDS for A and B tiles
    A_shared = S.make_shared((2, TILE_M, TILE_K), S.bf16)
    B_shared = S.make_shared((2, TILE_K, TILE_N), S.bf16)

    # Fragment LDS for MFMA input (2 u32 per lane = 4 bf16 values)
    A_frag_lds = S.make_shared((WARP_SIZE, 2), S.u32)
    B_frag_lds = S.make_shared((WARP_SIZE, 2), S.u32)

    A_frag_tensor = S.view(A_frag_lds, S.Tensor((WARP_SIZE, 2), S.u32))
    B_frag_tensor = S.view(B_frag_lds, S.Tensor((WARP_SIZE, 2), S.u32))

    # Accumulator for this wave's 32x32 output tile (16 f32 values per lane)
    c_acc = S.full((16,), 0.0, S.f32)

    # Number of K iterations
    num_k_tiles = K // TILE_K

    # Prologue: Load first tile to buffer 0
    k_base = 0
    for load_offset in S.range(4):
        row_in_tile = (lane % 32)
        src_row_a = m_base + row_in_tile
        src_col_a = k_base + (lane // 32) * 4 + load_offset
        val_a = A[batch, src_row_a, src_col_a]
        A_shared[0, row_in_tile, (lane // 32) * 4 + load_offset] = val_a

    for load_offset in S.range(4):
        src_row_b = k_base + (lane // 32) * 4 + load_offset
        src_col_b = n_base + (lane % 32)
        val_b = B[batch, src_row_b, src_col_b]
        B_shared[0, (lane // 32) * 4 + load_offset, lane % 32] = val_b

    S.syncthreads()

    # Main loop - unroll by 2 to minimize branching
    # Process pairs of K tiles: even tile uses buffer 0, odd tile uses buffer 1
    num_pairs = num_k_tiles // 2

    for pair_idx in S.range(num_pairs):
        k_even = pair_idx * 2
        k_odd = pair_idx * 2 + 1

        k_base_even = k_even * TILE_K
        k_base_odd = k_odd * TILE_K
        k_base_next_even = (pair_idx + 1) * 2 * TILE_K

        # ===== Process even tile (buffer 0) =====

        # Load A fragment from LDS and pack into u32 for MFMA
        a_bf16_0 = A_shared[0, lane % 32, (lane // 32) * 4 + 0]
        a_bf16_1 = A_shared[0, lane % 32, (lane // 32) * 4 + 1]
        a_bf16_2 = A_shared[0, lane % 32, (lane // 32) * 4 + 2]
        a_bf16_3 = A_shared[0, lane % 32, (lane // 32) * 4 + 3]

        a_u16_0 = S.bitcast(a_bf16_0, S.u16)
        a_u16_1 = S.bitcast(a_bf16_1, S.u16)
        a_u16_2 = S.bitcast(a_bf16_2, S.u16)
        a_u16_3 = S.bitcast(a_bf16_3, S.u16)

        a_u32_0 = a_u16_0 | (a_u16_1 << 16)
        a_u32_1 = a_u16_2 | (a_u16_3 << 16)

        A_frag_lds[lane, 0] = a_u32_0
        A_frag_lds[lane, 1] = a_u32_1

        # Load B fragment from LDS and pack into u32 for MFMA
        b_bf16_0 = B_shared[0, (lane // 32) * 4 + 0, lane % 32]
        b_bf16_1 = B_shared[0, (lane // 32) * 4 + 1, lane % 32]
        b_bf16_2 = B_shared[0, (lane // 32) * 4 + 2, lane % 32]
        b_bf16_3 = B_shared[0, (lane // 32) * 4 + 3, lane % 32]

        b_u16_0 = S.bitcast(b_bf16_0, S.u16)
        b_u16_1 = S.bitcast(b_bf16_1, S.u16)
        b_u16_2 = S.bitcast(b_bf16_2, S.u16)
        b_u16_3 = S.bitcast(b_bf16_3, S.u16)

        b_u32_0 = b_u16_0 | (b_u16_1 << 16)
        b_u32_1 = b_u16_2 | (b_u16_3 << 16)

        B_frag_lds[lane, 0] = b_u32_0
        B_frag_lds[lane, 1] = b_u32_1

        S.syncthreads()

        # View the packed data as bf16 for MFMA
        a_row_tensor = A_frag_tensor[lane]
        b_row_tensor = B_frag_tensor[lane]

        a_view = S.view(a_row_tensor, S.Tensor((1, 4, 1), S.bf16))
        b_view = S.view(b_row_tensor, S.Tensor((1, 4, 1), S.bf16))

        # Perform MFMA for even tile
        c_acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_view[0], b_view[0], c_acc)

        # Prefetch: load odd tile to buffer 1 (overlaps with even tile MFMA)
        for load_offset in S.range(4):
            row_in_tile = (lane % 32)
            src_row_a = m_base + row_in_tile
            src_col_a = k_base_odd + (lane // 32) * 4 + load_offset
            val_a = A[batch, src_row_a, src_col_a]
            A_shared[1, row_in_tile, (lane // 32) * 4 + load_offset] = val_a

        for load_offset in S.range(4):
            src_row_b = k_base_odd + (lane // 32) * 4 + load_offset
            src_col_b = n_base + (lane % 32)
            val_b = B[batch, src_row_b, src_col_b]
            B_shared[1, (lane // 32) * 4 + load_offset, lane % 32] = val_b

        S.syncthreads()

        # ===== Process odd tile (buffer 1) =====

        # Load A fragment from LDS and pack into u32 for MFMA
        a_bf16_0 = A_shared[1, lane % 32, (lane // 32) * 4 + 0]
        a_bf16_1 = A_shared[1, lane % 32, (lane // 32) * 4 + 1]
        a_bf16_2 = A_shared[1, lane % 32, (lane // 32) * 4 + 2]
        a_bf16_3 = A_shared[1, lane % 32, (lane // 32) * 4 + 3]

        a_u16_0 = S.bitcast(a_bf16_0, S.u16)
        a_u16_1 = S.bitcast(a_bf16_1, S.u16)
        a_u16_2 = S.bitcast(a_bf16_2, S.u16)
        a_u16_3 = S.bitcast(a_bf16_3, S.u16)

        a_u32_0 = a_u16_0 | (a_u16_1 << 16)
        a_u32_1 = a_u16_2 | (a_u16_3 << 16)

        A_frag_lds[lane, 0] = a_u32_0
        A_frag_lds[lane, 1] = a_u32_1

        # Load B fragment from LDS and pack into u32 for MFMA
        b_bf16_0 = B_shared[1, (lane // 32) * 4 + 0, lane % 32]
        b_bf16_1 = B_shared[1, (lane // 32) * 4 + 1, lane % 32]
        b_bf16_2 = B_shared[1, (lane // 32) * 4 + 2, lane % 32]
        b_bf16_3 = B_shared[1, (lane // 32) * 4 + 3, lane % 32]

        b_u16_0 = S.bitcast(b_bf16_0, S.u16)
        b_u16_1 = S.bitcast(b_bf16_1, S.u16)
        b_u16_2 = S.bitcast(b_bf16_2, S.u16)
        b_u16_3 = S.bitcast(b_bf16_3, S.u16)

        b_u32_0 = b_u16_0 | (b_u16_1 << 16)
        b_u32_1 = b_u16_2 | (b_u16_3 << 16)

        B_frag_lds[lane, 0] = b_u32_0
        B_frag_lds[lane, 1] = b_u32_1

        S.syncthreads()

        # View the packed data as bf16 for MFMA
        a_row_tensor = A_frag_tensor[lane]
        b_row_tensor = B_frag_tensor[lane]

        a_view = S.view(a_row_tensor, S.Tensor((1, 4, 1), S.bf16))
        b_view = S.view(b_row_tensor, S.Tensor((1, 4, 1), S.bf16))

        # Perform MFMA for odd tile
        c_acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_view[0], b_view[0], c_acc)

        # Prefetch: load next even tile to buffer 0 (overlaps with odd tile MFMA)
        if pair_idx < num_pairs - 1:
            for load_offset in S.range(4):
                row_in_tile = (lane % 32)
                src_row_a = m_base + row_in_tile
                src_col_a = k_base_next_even + (lane // 32) * 4 + load_offset
                val_a = A[batch, src_row_a, src_col_a]
                A_shared[0, row_in_tile, (lane // 32) * 4 + load_offset] = val_a

            for load_offset in S.range(4):
                src_row_b = k_base_next_even + (lane // 32) * 4 + load_offset
                src_col_b = n_base + (lane % 32)
                val_b = B[batch, src_row_b, src_col_b]
                B_shared[0, (lane // 32) * 4 + load_offset, lane % 32] = val_b

        S.syncthreads()

    # Write C to global memory with correct MFMA swizzled output layout
    for acc_idx in S.range(16):
        # MFMA output swizzle pattern:
        # row_offset = 8 * (acc_idx // 4) + 4 * (lane // 32) + (acc_idx % 4)
        row_offset = 8 * (acc_idx // 4) + 4 * (lane // 32) + (acc_idx % 4)
        col_offset = lane % 32

        out_row = m_base + row_offset
        out_col = n_base + col_offset
        if out_row < M and out_col < N:
            C[batch, out_row, out_col] = S.convert(c_acc[acc_idx], S.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if tuple(A.shape) != (128, 512, 1024) or tuple(B.shape) != (128, 1024, 2048):
            return torch.bmm(A, B)
        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((128, 512, 2048), device=A.device, dtype=A.dtype)

        # Grid: (batch, M/32, N/32) = (128, 16, 64)
        # Block: 64 threads (1 wave)
        bmm_kernel[lambda: ((128, 16, 64), (64, 1, 1))](A, B, C)
        return C
