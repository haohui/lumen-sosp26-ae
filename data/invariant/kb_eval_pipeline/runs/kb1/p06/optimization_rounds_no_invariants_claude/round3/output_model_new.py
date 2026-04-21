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
    """GEMM kernel using MFMA 32x32x8 instructions with LDS staging and software pipelining."""
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

    # Create resource descriptors with range for OOB handling
    # Range is in bytes: total elements * 2 bytes per bf16
    rsrc_A = S.amdgpu.make_rsrc(A, M * K * 2)
    rsrc_B = S.amdgpu.make_rsrc(B, K * N * 2)

    # Double buffering: two LDS buffers for A and B
    lds_a0 = S.make_shared((64, 16), S.bf16)
    lds_b0 = S.make_shared((16, 64), S.bf16)
    lds_a1 = S.make_shared((64, 16), S.bf16)
    lds_b1 = S.make_shared((16, 64), S.bf16)

    # Local storage for A and B fragments
    a_frag = S.make_local((4,), S.bf16)
    b_frag = S.make_local((4,), S.bf16)

    # Accumulator for 32x32 output (16 f32 per lane)
    acc = S.full((16,), 0.0, S.f32)

    # Number of K tiles (each tile is 16 columns of A and 16 rows of B)
    num_k_tiles = K // TILE_K  # 524288 / 16 = 32768

    # Prefetch first tile into buffer 0 using raw_buffer_load_x2
    # Each thread loads 4 contiguous bf16 elements (8 bytes)
    for load_iter in S.range(4):
        load_idx = tid * 4 + load_iter
        row_a = load_idx // 16
        col_a = load_idx % 16
        global_row = block_m + row_a
        global_col = col_a
        # Byte offset for 4 contiguous bf16 elements starting at (global_row, global_col)
        byte_offset = (global_row * K + global_col) * 2
        data = S.amdgpu.raw_buffer_load_x2(rsrc_A, byte_offset, 0, 0)
        data_bf16 = S.view(data, S.Tensor((4,), S.bf16))
        lds_a0[row_a, col_a] = data_bf16[load_iter]

    for load_iter in S.range(4):
        load_idx = tid * 4 + load_iter
        row_b = load_idx // 64
        col_b = load_idx % 64
        global_row_b = row_b
        global_col_b = block_n + col_b
        # Byte offset for 4 contiguous bf16 elements
        byte_offset = (global_row_b * N + global_col_b) * 2
        data = S.amdgpu.raw_buffer_load_x2(rsrc_B, byte_offset, 0, 0)
        data_bf16 = S.view(data, S.Tensor((4,), S.bf16))
        lds_b0[row_b, col_b] = data_bf16[load_iter]

    S.syncthreads()

    # Main loop with K-loop unroll by 2 and software pipelining
    # Removed OOB guards - raw_buffer_load with range handles OOB by returning 0
    for k_tile in S.range(0, num_k_tiles, 2):
        # === First tile: k_tile ===
        # Buffer 0 has tile k_tile (prefetched or from previous iteration)

        # Load tile k_tile + 1 into buffer 1 (overlapped with compute)
        # No OOB check needed - range in rsrc handles it
        next_k_base = (k_tile + 1) * TILE_K
        for load_iter in S.range(4):
            load_idx = tid * 4 + load_iter
            row_a = load_idx // 16
            col_a = load_idx % 16
            global_row = block_m + row_a
            global_col = next_k_base + col_a
            byte_offset = (global_row * K + global_col) * 2
            data = S.amdgpu.raw_buffer_load_x2(rsrc_A, byte_offset, 0, 0)
            data_bf16 = S.view(data, S.Tensor((4,), S.bf16))
            lds_a1[row_a, col_a] = data_bf16[load_iter]

        for load_iter in S.range(4):
            load_idx = tid * 4 + load_iter
            row_b = load_idx // 64
            col_b = load_idx % 64
            global_row_b = next_k_base + row_b
            global_col_b = block_n + col_b
            byte_offset = (global_row_b * N + global_col_b) * 2
            data = S.amdgpu.raw_buffer_load_x2(rsrc_B, byte_offset, 0, 0)
            data_bf16 = S.view(data, S.Tensor((4,), S.bf16))
            lds_b1[row_b, col_b] = data_bf16[load_iter]

        # Compute on buffer 0 (tile k_tile)
        for mfma_iter in S.range(2):
            k_offset = mfma_iter * 8

            row_a = warp_row * 32 + (lane_id % 32)
            col_a = k_offset + (lane_id // 32) * 4
            for i in S.range(4):
                a_frag[i] = lds_a0[row_a, col_a + i]

            row_b = k_offset + (lane_id % 8)
            col_b = warp_col * 32 + (lane_id // 8) * 4
            for i in S.range(4):
                b_frag[i] = lds_b0[row_b, col_b + i]

            acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag, b_frag, acc)

        S.syncthreads()

        # === Second tile: k_tile + 1 ===
        # Buffer 1 has tile k_tile + 1 (just loaded)

        # Load tile k_tile + 2 into buffer 0 (for next iteration)
        # No OOB check needed - range in rsrc handles it
        next_k_base = (k_tile + 2) * TILE_K
        for load_iter in S.range(4):
            load_idx = tid * 4 + load_iter
            row_a = load_idx // 16
            col_a = load_idx % 16
            global_row = block_m + row_a
            global_col = next_k_base + col_a
            byte_offset = (global_row * K + global_col) * 2
            data = S.amdgpu.raw_buffer_load_x2(rsrc_A, byte_offset, 0, 0)
            data_bf16 = S.view(data, S.Tensor((4,), S.bf16))
            lds_a0[row_a, col_a] = data_bf16[load_iter]

        for load_iter in S.range(4):
            load_idx = tid * 4 + load_iter
            row_b = load_idx // 64
            col_b = load_idx % 64
            global_row_b = next_k_base + row_b
            global_col_b = block_n + col_b
            byte_offset = (global_row_b * N + global_col_b) * 2
            data = S.amdgpu.raw_buffer_load_x2(rsrc_B, byte_offset, 0, 0)
            data_bf16 = S.view(data, S.Tensor((4,), S.bf16))
            lds_b0[row_b, col_b] = data_bf16[load_iter]

        # Compute on buffer 1 (tile k_tile + 1)
        # No OOB check needed - extra computation with 0 doesn't affect result
        for mfma_iter in S.range(2):
            k_offset = mfma_iter * 8

            row_a = warp_row * 32 + (lane_id % 32)
            col_a = k_offset + (lane_id // 32) * 4
            for i in S.range(4):
                a_frag[i] = lds_a1[row_a, col_a + i]

            row_b = k_offset + (lane_id % 8)
            col_b = warp_col * 32 + (lane_id // 8) * 4
            for i in S.range(4):
                b_frag[i] = lds_b1[row_b, col_b + i]

            acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag, b_frag, acc)

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
        self._cached_inputs = None
        self._cached_output = None
        self.grid, self.block = get_launch_config()

    def forward(self, A, B):
        if tuple(A.shape) != (256, 524288) or tuple(B.shape) != (524288, 256):
            raise ValueError(f"Expected shapes (256, 524288) and (524288, 256), got {A.shape} and {B.shape}")

        A = A.contiguous()
        B = B.contiguous()

        cache_key = (A.data_ptr(), B.data_ptr())
        if self._cached_inputs == cache_key and self._cached_output is not None:
            C = self._cached_output
        else:
            C = torch.empty((256, 256), device=A.device, dtype=A.dtype)
            self._cached_output = C
            self._cached_inputs = cache_key

        grid, block = self.grid, self.block
        gemm_mfma_kernel[lambda: (grid, block)](A, B, C)

        return C
