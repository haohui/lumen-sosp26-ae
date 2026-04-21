import torch
import torch.nn as nn

import substrate
import substrate.language as S


BATCH = 128
M = 512
K = 1024
N = 2048

# Tile sizes for 4-wave (2x2 warp grid)
TILE_M = 64  # 2 warps * 32
TILE_N = 64  # 2 warps * 32
TILE_K = 16  # Two MFMA instructions per K iteration (K=8 each)

WAVE_M = 32
WAVE_N = 32

WAVE_SIZE = 64
NUM_WARPS = 4
BLOCK_SIZE = 256


@substrate.jit
def bmm_mfma_kernel(
    A: S.Tensor((BATCH, M, K), S.bf16),
    B: S.Tensor((BATCH, K, N), S.bf16),
    C: S.Tensor((BATCH, M, N), S.bf16),
):
    # Block indices
    batch_idx = S.block_id(0)
    m_tile = S.block_id(1)
    n_tile = S.block_id(2)

    # Thread and warp info
    tid = S.thread_id(0)
    warp_id = tid // WAVE_SIZE
    lane_id = tid % WAVE_SIZE

    # Warp grid (2x2)
    warp_row = warp_id // 2
    warp_col = warp_id % 2

    # Output tile base for this warp
    tile_m_base = m_tile * TILE_M + warp_row * WAVE_M
    tile_n_base = n_tile * TILE_N + warp_col * WAVE_N

    # Create resource descriptors with range (in bytes)
    # Range is set to the full tensor size - OOB accesses return 0 for loads
    rsrc_A = S.amdgpu.make_rsrc(A, BATCH * M * K * 2)
    rsrc_B = S.amdgpu.make_rsrc(B, BATCH * K * N * 2)

    # Double-buffered LDS allocation
    A_lds = S.make_shared((2, 64, 16), S.bf16)
    B_lds = S.make_shared((2, 16, 64), S.bf16)

    # Accumulator - 16 f32 values per lane
    acc = S.full((16,), 0.0, S.f32)

    num_k_tiles = K // TILE_K

    # Thread offsets for loading (element index base)
    a_offset = tid * 4
    b_offset = tid * 4

    # Pre-compute row/col indices for A and B loads (constant across all K tiles)
    # A: elements at indices tid*4 to tid*4+3, all in same row, consecutive cols
    a_elem_row = a_offset // 16  # row in tile (0-63)
    a_elem_col = a_offset % 16   # starting col in tile (0-15)
    a_global_row = m_tile * TILE_M + a_elem_row

    # B: elements at indices tid*4 to tid*4+3, all in same row, consecutive cols
    b_elem_row = b_offset // 64  # row in tile (0-15)
    b_elem_col = b_offset % 64   # starting col in tile (0-63)
    b_global_col = n_tile * TILE_N + b_elem_col

    # Warp-local variables for MFMA
    a_row = lane_id % 32
    a_k_group = lane_id // 32
    a_lds_row = warp_row * WAVE_M + a_row

    b_col = lane_id % 32
    b_k_group = lane_id // 32
    b_lds_col = warp_col * WAVE_N + b_col

    # Prologue: Load first K tile into buffer 0 using raw_buffer_load
    # Each thread loads 4 consecutive bf16 elements (8 bytes = 2 x i32)
    # A: byte offset = (batch_idx * M * K + a_global_row * K + col) * 2
    a_byte_offset = (batch_idx * M * K + a_global_row * K + a_elem_col) * 2
    a_data = S.amdgpu.raw_buffer_load_x2(rsrc_A, a_byte_offset, 0, 0)
    a_data_bf16 = S.view(a_data, S.Tensor((4,), S.bf16))
    for i in S.range(4):
        A_lds[0, a_elem_row, a_elem_col + i] = a_data_bf16[i]

    # B: byte offset = (batch_idx * K * N + b_elem_row * N + b_global_col) * 2
    b_byte_offset = (batch_idx * K * N + b_elem_row * N + b_global_col) * 2
    b_data = S.amdgpu.raw_buffer_load_x2(rsrc_B, b_byte_offset, 0, 0)
    b_data_bf16 = S.view(b_data, S.Tensor((4,), S.bf16))
    for i in S.range(4):
        B_lds[0, b_elem_row, b_elem_col + i] = b_data_bf16[i]

    S.syncthreads()

    # Main K-loop with unrolling by 2
    # Each iteration processes 2 K tiles
    for kt in S.range(num_k_tiles // 2):
        # First K tile index
        k0 = kt * 2
        # Second K tile index
        k1 = kt * 2 + 1

        buf0 = k0 % 2
        buf1 = k1 % 2

        # === Compute on first K tile ===
        # Load fragments from LDS
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

        # Issue MFMA for first K tile
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0, b_frag0, acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1, b_frag1, acc)

        # === Load second K tile into alternate buffer using raw_buffer_load ===
        k_base1 = k1 * TILE_K

        # Load A elements for k1
        a_byte_offset_k1 = (batch_idx * M * K + a_global_row * K + k_base1 + a_elem_col) * 2
        a_data_k1 = S.amdgpu.raw_buffer_load_x2(rsrc_A, a_byte_offset_k1, 0, 0)
        a_data_k1_bf16 = S.view(a_data_k1, S.Tensor((4,), S.bf16))
        for i in S.range(4):
            A_lds[buf1, a_elem_row, a_elem_col + i] = a_data_k1_bf16[i]

        # Load B elements for k1
        b_byte_offset_k1 = (batch_idx * K * N + (k_base1 + b_elem_row) * N + b_global_col) * 2
        b_data_k1 = S.amdgpu.raw_buffer_load_x2(rsrc_B, b_byte_offset_k1, 0, 0)
        b_data_k1_bf16 = S.view(b_data_k1, S.Tensor((4,), S.bf16))
        for i in S.range(4):
            B_lds[buf1, b_elem_row, b_elem_col + i] = b_data_k1_bf16[i]

        S.syncthreads()

        # === Compute on second K tile ===
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

        # Issue MFMA for second K tile
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag2, b_frag2, acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag3, b_frag3, acc)

        # === Prefetch next iteration's first K tile using raw_buffer_load ===
        k_next = (kt + 1) * 2
        buf_next = k_next % 2
        k_base_next = k_next * TILE_K

        # Load A elements for k_next
        a_byte_offset_next = (batch_idx * M * K + a_global_row * K + k_base_next + a_elem_col) * 2
        a_data_next = S.amdgpu.raw_buffer_load_x2(rsrc_A, a_byte_offset_next, 0, 0)
        a_data_next_bf16 = S.view(a_data_next, S.Tensor((4,), S.bf16))
        for i in S.range(4):
            A_lds[buf_next, a_elem_row, a_elem_col + i] = a_data_next_bf16[i]

        # Load B elements for k_next
        b_byte_offset_next = (batch_idx * K * N + (k_base_next + b_elem_row) * N + b_global_col) * 2
        b_data_next = S.amdgpu.raw_buffer_load_x2(rsrc_B, b_byte_offset_next, 0, 0)
        b_data_next_bf16 = S.view(b_data_next, S.Tensor((4,), S.bf16))
        for i in S.range(4):
            B_lds[buf_next, b_elem_row, b_elem_col + i] = b_data_next_bf16[i]

        S.syncthreads()

    # Write accumulator to global memory
    for acc_idx in S.range(16):
        out_col = tile_n_base + (lane_id % 32)
        out_row = tile_m_base + 8 * (acc_idx // 4) + 4 * (lane_id // 32) + (acc_idx % 4)
        C[batch_idx, out_row, out_col] = S.convert(acc[acc_idx], S.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if tuple(A.shape) != (128, 512, 1024) or tuple(B.shape) != (128, 1024, 2048):
            return torch.bmm(A, B)

        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((128, 512, 2048), device=A.device, dtype=A.dtype)

        bmm_mfma_kernel[lambda: ((BATCH, M // TILE_M, N // TILE_N), (BLOCK_SIZE, 1, 1))](A, B, C)

        return C
