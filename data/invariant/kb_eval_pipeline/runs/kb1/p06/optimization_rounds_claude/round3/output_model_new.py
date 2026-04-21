import torch
import torch.nn as nn

import substrate
import substrate.language as S


M = 256
K = 524288
N = 256

# MFMA 32x32x8 parameters
WARP_SIZE = 64
NUM_WARPS = 4
BLOCK_SIZE = WARP_SIZE * NUM_WARPS  # 256 threads

# Tile sizes
TILE_M = 32  # MFMA output tile M
TILE_N = 32  # MFMA output tile N
TILE_K = 16  # K step for double MFMA (32x32x16)

# Block output tile: 64x64 (2x2 warps)
BLOCK_M = TILE_M * 2  # 64
BLOCK_N = TILE_N * 2  # 64


def get_launch_config():
    """Compute grid and block dimensions for the kernel launch."""
    grid_m = (M + BLOCK_M - 1) // BLOCK_M  # 4
    grid_n = (N + BLOCK_N - 1) // BLOCK_N  # 4
    grid = (grid_m * grid_n, 1, 1)
    block = (BLOCK_SIZE, 1, 1)
    return grid, block


@substrate.jit
def gemm_mfma_kernel(
    A: S.Tensor((256, 524288), S.bf16),
    B: S.Tensor((524288, 256), S.bf16),
    C: S.Tensor((256, 256), S.bf16),
):
    """GEMM kernel using MFMA 32x32x8 with software pipelining and double buffering.

    Uses range in raw buffer operations to remove OOB branch.
    """
    tid = S.thread_id(0)
    bid = S.block_id(0)

    # Warp and lane identification
    warp_id = tid // WARP_SIZE  # 0-3
    lane_id = tid % WARP_SIZE   # 0-63

    # Block tile position (grid is 4x4 blocks, each 64x64 output)
    block_m = (bid // 4) * BLOCK_M
    block_n = (bid % 4) * BLOCK_N

    # Warp position within block (2x2 warp grid)
    warp_row = warp_id // 2
    warp_col = warp_id % 2

    # Warp's global output position
    warp_m = block_m + warp_row * TILE_M
    warp_n = block_n + warp_col * TILE_N

    # Double buffered LDS: each buffer is 64x16 for A, 16x64 for B
    lds_a_0 = S.make_shared((64, 16), S.bf16)
    lds_a_1 = S.make_shared((64, 16), S.bf16)
    lds_b_0 = S.make_shared((16, 64), S.bf16)
    lds_b_1 = S.make_shared((16, 64), S.bf16)

    # Local storage for A and B fragments
    a_frag = S.make_local((4,), S.bf16)
    b_frag = S.make_local((4,), S.bf16)

    # Accumulator for 32x32 output (16 f32 per lane)
    acc = S.full((16,), 0.0, S.f32)

    # Number of K tiles (each tile is 16 columns of A and 16 rows of B)
    num_k_tiles = K // TILE_K  # 524288 / 16 = 32768

    # Total sizes in bytes for range parameter
    A_SIZE_BYTES = M * K * 2  # 268435456 bytes
    B_SIZE_BYTES = K * N * 2  # 268435456 bytes

    # Create buffer resource descriptors with range for OOB handling
    # When range is set, raw_buffer_load_x4 returns 0 for OOB elements
    rsrc_a = S.amdgpu.make_rsrc(A, A_SIZE_BYTES)
    rsrc_b = S.amdgpu.make_rsrc(B, B_SIZE_BYTES)

    # Prologue: Load first tile into buffer 0
    k_base = 0
    for load_iter in S.range(4):
        load_idx = tid * 4 + load_iter
        row_a = load_idx // 16
        col_a = load_idx % 16
        global_row = block_m + row_a
        global_col = k_base + col_a
        lds_a_0[row_a, col_a] = A[global_row, global_col]

    for load_iter in S.range(4):
        load_idx = tid * 4 + load_iter
        row_b = load_idx // 64
        col_b = load_idx % 64
        global_row_b = k_base + row_b
        global_col_b = block_n + col_b
        lds_b_0[row_b, col_b] = B[global_row_b, global_col_b]

    S.syncthreads()

    # Main loop with K-loop unrolled by 2
    # Each iteration processes 2 consecutive K-tiles
    # Use double buffering: while computing from buffer 0, load into buffer 1
    num_k_pairs = num_k_tiles // 2  # 16384 pairs

    for k_pair in S.range(num_k_pairs):
        k_tile_0 = k_pair * 2
        k_tile_1 = k_pair * 2 + 1

        k_base_0 = k_tile_0 * TILE_K
        k_base_1 = k_tile_1 * TILE_K

        # === First tile of the pair ===
        # Compute from buffer 0 (already loaded)
        for mfma_iter in S.range(2):
            k_offset = mfma_iter * 8

            row_a = warp_row * 32 + (lane_id % 32)
            col_a = k_offset + (lane_id // 32) * 4
            for i in S.range(4):
                a_frag[i] = lds_a_0[row_a, col_a + i]

            row_b = k_offset + (lane_id % 8)
            col_b = warp_col * 32 + (lane_id // 8) * 4
            for i in S.range(4):
                b_frag[i] = lds_b_0[row_b, col_b + i]

            acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag, b_frag, acc)

        # Load second tile into buffer 1 (software pipelining)
        for load_iter in S.range(4):
            load_idx = tid * 4 + load_iter
            row_a = load_idx // 16
            col_a = load_idx % 16
            global_row = block_m + row_a
            global_col = k_base_1 + col_a
            lds_a_1[row_a, col_a] = A[global_row, global_col]

        for load_iter in S.range(4):
            load_idx = tid * 4 + load_iter
            row_b = load_idx // 64
            col_b = load_idx % 64
            global_row_b = k_base_1 + row_b
            global_col_b = block_n + col_b
            lds_b_1[row_b, col_b] = B[global_row_b, global_col_b]

        S.syncthreads()

        # === Second tile of the pair ===
        # Compute from buffer 1
        for mfma_iter in S.range(2):
            k_offset = mfma_iter * 8

            row_a = warp_row * 32 + (lane_id % 32)
            col_a = k_offset + (lane_id // 32) * 4
            for i in S.range(4):
                a_frag[i] = lds_a_1[row_a, col_a + i]

            row_b = k_offset + (lane_id % 8)
            col_b = warp_col * 32 + (lane_id // 8) * 4
            for i in S.range(4):
                b_frag[i] = lds_b_1[row_b, col_b + i]

            acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag, b_frag, acc)

        # Prefetch next pair's first tile into buffer 0 using raw buffer loads with range
        # OOB accesses will return 0, which is safe since they won't be used
        # Branch removed - all iterations execute the prefetch
        next_k_tile = (k_pair + 1) * 2
        next_k_base = next_k_tile * TILE_K

        # Load A using raw buffer operations
        # Each thread loads 4 elements, compute the starting position
        load_idx_start = tid * 4
        row_a_start = load_idx_start // 16
        col_a_start = load_idx_start % 16
        global_row_a = block_m + row_a_start
        global_col_a_start = next_k_base + col_a_start

        # Load 8 bf16 values starting at (global_row_a, global_col_a_start)
        # These cover col_a_start to col_a_start+7, which includes all 4 elements we need
        byte_offset_a = global_row_a * K * 2 + global_col_a_start * 2
        data_a = S.amdgpu.raw_buffer_load_x4(rsrc_a, byte_offset_a, 0, 0)
        bf16_view_a = S.view(data_a, S.Tensor((8,), S.bf16))

        # Store the 4 elements to LDS
        for i in S.range(4):
            lds_a_0[row_a_start, col_a_start + i] = bf16_view_a[i]

        # Load B using raw buffer operations
        row_b_start = load_idx_start // 64
        col_b_start = load_idx_start % 64
        global_row_b_start = next_k_base + row_b_start
        global_col_b = block_n + col_b_start

        # Load 8 bf16 values starting at (global_row_b_start, global_col_b)
        byte_offset_b = global_row_b_start * N * 2 + global_col_b * 2
        data_b = S.amdgpu.raw_buffer_load_x4(rsrc_b, byte_offset_b, 0, 0)
        bf16_view_b = S.view(data_b, S.Tensor((8,), S.bf16))

        # Store the 4 elements to LDS
        for i in S.range(4):
            lds_b_0[row_b_start, col_b_start + i] = bf16_view_b[i]

        S.syncthreads()

    # Write results to global memory using the specified accumulator invariant
    for acc_idx in S.range(16):
        col_offset = lane_id % 32
        row_offset = 8 * (acc_idx // 4) + 4 * (lane_id // 32) + (acc_idx % 4)

        global_row = warp_m + row_offset
        global_col = warp_n + col_offset

        C[global_row, global_col] = S.convert(acc[acc_idx], S.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.grid, self.block = get_launch_config()

    def forward(self, A, B):
        if tuple(A.shape) != (256, 524288) or tuple(B.shape) != (524288, 256):
            raise ValueError(f"Expected shapes (256, 524288) and (524288, 256), got {A.shape} and {B.shape}")

        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((256, 256), device=A.device, dtype=A.dtype)

        grid, block = self.grid, self.block
        gemm_mfma_kernel[lambda: (grid, block)](A, B, C)

        return C
