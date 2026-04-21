import torch
import torch.nn as nn

import substrate
import substrate.language as S

# Problem dimensions
BATCH = 16
M = 1024
K = 2048
N = 768

# Tiling parameters for 4-warps (2x2 warp grid)
WARP_SIZE = 64
NUM_WARPS = 4
TILE_M = 32  # per warp
TILE_N = 32  # per warp
TILE_K = 16  # 2 MFMA instructions per K step
BLOCK_M = TILE_M * 2  # 64
BLOCK_N = TILE_N * 2  # 64


@substrate.jit
def matmul3d_mfma_kernel(
    A: S.Tensor((BATCH, M, K), S.bf16),
    B: S.Tensor((K, N), S.bf16),
    C: S.Tensor((BATCH, M, N), S.bf16),
):
    block_idx = S.block_id(0)
    tid = S.thread_id(0)  # 0-255

    # Warp structure: 2x2 grid
    warp_id = tid // WARP_SIZE  # 0-3
    lane_id = tid % WARP_SIZE  # 0-63

    warp_row = warp_id // 2  # 0 or 1
    warp_col = warp_id % 2   # 0 or 1

    # Compute block position
    blocks_per_batch = (M // BLOCK_M) * (N // BLOCK_N)
    batch_idx = block_idx // blocks_per_batch
    residual = block_idx % blocks_per_batch

    blocks_n = N // BLOCK_N
    block_m_idx = residual // blocks_n
    block_n_idx = residual % blocks_n

    tile_m_base = block_m_idx * BLOCK_M
    tile_n_base = block_n_idx * BLOCK_N

    # Create resource descriptors with range for OOB handling
    # Range is in bytes; bf16 = 2 bytes
    rsrc_A = S.amdgpu.make_rsrc(A, BATCH * M * K * 2)
    rsrc_B = S.amdgpu.make_rsrc(B, K * N * 2)

    # Double buffering LDS
    A_lds = S.make_shared((2, BLOCK_M, TILE_K), S.bf16)  # 2 x 64 x 16
    B_lds = S.make_shared((2, TILE_K, BLOCK_N), S.bf16)  # 2 x 16 x 64

    # Accumulator (16 f32 values per lane for 32x32 output)
    acc = S.full((16,), 0.0, S.f32)

    # Thread offsets for cooperative loading
    a_offset = tid * 4
    b_offset = tid * 4

    # Compute element positions for loading
    a_elem_row = a_offset // TILE_K
    a_elem_col = a_offset % TILE_K
    a_global_row = tile_m_base + a_elem_row

    b_elem_row = b_offset // BLOCK_N
    b_elem_col = b_offset % BLOCK_N
    b_global_col = tile_n_base + b_elem_col

    # Lane swizzle for MFMA
    a_row = lane_id % 32
    a_k_group = lane_id // 32
    a_lds_row = warp_row * TILE_M + a_row

    b_col = lane_id % 32
    b_k_group = lane_id // 32
    b_lds_col = warp_col * TILE_N + b_col

    num_k_tiles = K // TILE_K

    # Prologue: Load first K tile into buffer 0 using raw_buffer_load
    a_byte_offset = (batch_idx * M * K + a_global_row * K + a_elem_col) * 2
    a_data = S.amdgpu.raw_buffer_load_x2(rsrc_A, a_byte_offset, 0, 0)
    a_data_bf16 = S.view(a_data, S.Tensor((4,), S.bf16))
    for i in S.range(4):
        A_lds[0, a_elem_row, a_elem_col + i] = a_data_bf16[i]

    b_byte_offset = (b_elem_row * N + b_global_col) * 2
    b_data = S.amdgpu.raw_buffer_load_x2(rsrc_B, b_byte_offset, 0, 0)
    b_data_bf16 = S.view(b_data, S.Tensor((4,), S.bf16))
    for i in S.range(4):
        B_lds[0, b_elem_row, b_elem_col + i] = b_data_bf16[i]

    S.syncthreads()

    # Main K loop unrolled by 2 with double buffering
    for kt in S.range(num_k_tiles // 2):
        k0 = kt * 2
        k1 = kt * 2 + 1
        buf0 = k0 % 2
        buf1 = k1 % 2

        # Compute from buffer 0 (2 MFMA ops for K=16)
        a_frag0 = S.make_local((4,), S.bf16)
        for elem in S.range(4):
            k_idx = a_k_group * 4 + elem
            a_frag0[elem] = A_lds[buf0, a_lds_row, k_idx]

        a_frag1 = S.make_local((4,), S.bf16)
        for elem in S.range(4):
            k_idx = 8 + a_k_group * 4 + elem
            a_frag1[elem] = A_lds[buf0, a_lds_row, k_idx]

        b_frag0 = S.make_local((4,), S.bf16)
        for elem in S.range(4):
            k_idx = b_k_group * 4 + elem
            b_frag0[elem] = B_lds[buf0, k_idx, b_lds_col]

        b_frag1 = S.make_local((4,), S.bf16)
        for elem in S.range(4):
            k_idx = 8 + b_k_group * 4 + elem
            b_frag1[elem] = B_lds[buf0, k_idx, b_lds_col]

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0, b_frag0, acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1, b_frag1, acc)

        # Load next tile into buffer 1 using raw_buffer_load
        k_base1 = k1 * TILE_K
        a_byte_offset_k1 = (batch_idx * M * K + a_global_row * K + k_base1 + a_elem_col) * 2
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

        # Compute from buffer 1 (2 MFMA ops for K=16)
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

        # Prefetch next tile - no branch needed, OOB returns 0
        k_next = (kt + 1) * 2
        buf_next = k_next % 2
        k_base_next = k_next * TILE_K
        a_byte_offset_next = (batch_idx * M * K + a_global_row * K + k_base_next + a_elem_col) * 2
        a_data_next = S.amdgpu.raw_buffer_load_x2(rsrc_A, a_byte_offset_next, 0, 0)
        a_data_next_bf16 = S.view(a_data_next, S.Tensor((4,), S.bf16))
        for i in S.range(4):
            A_lds[buf_next, a_elem_row, a_elem_col + i] = a_data_next_bf16[i]

        b_byte_offset_next = ((k_base_next + b_elem_row) * N + b_global_col) * 2
        b_data_next = S.amdgpu.raw_buffer_load_x2(rsrc_B, b_byte_offset_next, 0, 0)
        b_data_next_bf16 = S.view(b_data_next, S.Tensor((4,), S.bf16))
        for i in S.range(4):
            B_lds[buf_next, b_elem_row, b_elem_col + i] = b_data_next_bf16[i]

        S.syncthreads()

    # Write C output
    for acc_idx in S.range(16):
        out_col = tile_n_base + warp_col * TILE_N + (lane_id % 32)
        out_row = tile_m_base + warp_row * TILE_M + 8 * (acc_idx // 4) + 4 * (lane_id // 32) + (acc_idx % 4)
        C[batch_idx, out_row, out_col] = S.convert(acc[acc_idx], S.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if tuple(A.shape) != (BATCH, M, K) or tuple(B.shape) != (K, N):
            return torch.matmul(A, B)
        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((BATCH, M, N), device=A.device, dtype=A.dtype)

        blocks_m = M // BLOCK_M
        blocks_n = N // BLOCK_N
        num_blocks = blocks_m * blocks_n * BATCH

        matmul3d_mfma_kernel[lambda: ((num_blocks, 1, 1), (NUM_WARPS * WARP_SIZE, 1, 1))](A, B, C)
        return C
