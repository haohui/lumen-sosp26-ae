import torch
import torch.nn as nn

import substrate
import substrate.language as S


M = 4096
N = 4096
K = 4096
TILE_M = 64
TILE_N = 64
BLOCK_K = 16


@substrate.jit
def diag_left_mfma_kernel(
    A: S.Tensor((4096,), S.bf16),
    B: S.Tensor((4096, 4096), S.bf16),
    C: S.Tensor((4096, 4096), S.bf16),
):
    """
    Computes C = diag(A) @ B using element-wise computation with software pipelining.

    Uses double buffering for LDS.
    Each workgroup handles a 64x64 tile with 256 threads (4 waves).
    Each wave handles a 32x32 sub-tile in a 2x2 grid.
    """
    tid = S.thread_id(0)
    lane = tid % 64
    wave_id = tid // 64
    wg_id = S.block_id(0)

    # Workgroup position in the grid
    num_tiles_n = N // TILE_N
    wg_row = wg_id // num_tiles_n
    wg_col = wg_id % num_tiles_n

    # 4 waves in 2x2 grid within workgroup
    warp_row = wave_id // 2
    warp_col = wave_id % 2

    # Each warp handles a 32x32 tile
    tile_row = wg_row * TILE_M + warp_row * 32
    tile_col = wg_col * TILE_N + warp_col * 32

    # Position within warp (64 lanes for 32x32 tile)
    row_in_warp = lane % 32
    col_group = lane // 32

    global_row = tile_row + row_in_warp

    # Double-buffered LDS for A diagonal values (shared across all warps)
    lds_A_0 = S.make_shared((TILE_M,), S.bf16)
    lds_A_1 = S.make_shared((TILE_M,), S.bf16)

    # Double-buffered LDS for B - each warp has its own 32x16 buffer
    # Total: 4 warps * 2 buffers * 32 rows * 16 cols = 4096 bf16 = 8KB
    lds_B_0 = S.make_shared((4, 32, 16), S.bf16)
    lds_B_1 = S.make_shared((4, 32, 16), S.bf16)

    num_k_iters = 2

    # ========== Prologue ==========
    # Load A diagonal - all 256 threads cooperatively load 64 values
    if tid < TILE_M:
        lds_A_0[tid] = A[wg_row * TILE_M + tid]

    # Load first B block for each warp
    # Each warp loads 32x16 = 512 elements
    # 64 threads per warp, each loads 8 elements
    for i in S.range(8):
        elem_idx = lane * 8 + i  # lane in 0..63, elem_idx in 0..511
        b_row = elem_idx // 16  # 0..31
        b_col = elem_idx % 16   # 0..15
        lds_B_0[wave_id, b_row, b_col] = B[wg_row * TILE_M + warp_row * 32 + b_row,
                                            wg_col * TILE_N + warp_col * 32 + b_col]

    S.syncthreads()

    # ========== Main Loop ==========
    for k_iter in S.range(num_k_iters):
        buf_cur = k_iter % 2
        buf_next = 1 - buf_cur
        k_col_offset = k_iter * 16

        # Load next buffers
        if k_iter < num_k_iters - 1:
            # Load next A diagonal
            if tid < TILE_M:
                if buf_next == 0:
                    lds_A_0[tid] = A[wg_row * TILE_M + tid]
                else:
                    lds_A_1[tid] = A[wg_row * TILE_M + tid]

            # Load next B block
            next_col_offset = (k_iter + 1) * 16
            for i in S.range(8):
                elem_idx = lane * 8 + i
                b_row = elem_idx // 16
                b_col = elem_idx % 16
                if buf_next == 0:
                    lds_B_0[wave_id, b_row, b_col] = B[wg_row * TILE_M + warp_row * 32 + b_row,
                                                        wg_col * TILE_N + warp_col * 32 + next_col_offset + b_col]
                else:
                    lds_B_1[wave_id, b_row, b_col] = B[wg_row * TILE_M + warp_row * 32 + b_row,
                                                        wg_col * TILE_N + warp_col * 32 + next_col_offset + b_col]

        # ===== Compute =====
        lds_A_cur = lds_A_0 if buf_cur == 0 else lds_A_1
        lds_B_cur = lds_B_0 if buf_cur == 0 else lds_B_1

        diag_val = lds_A_cur[warp_row * 32 + row_in_warp]
        diag_f32 = S.convert(diag_val, S.f32)

        # ===== Store results =====
        # OOB branch removed - exact tiling guarantees in-bounds access
        for col_offset in S.range(8):
            col_in_block = col_group * 8 + col_offset
            b_val = lds_B_cur[wave_id, row_in_warp, col_in_block]
            result_f32 = diag_f32 * S.convert(b_val, S.f32)
            col = tile_col + k_col_offset + col_in_block
            C[global_row, col] = S.convert(result_f32, S.bf16)

        if k_iter < num_k_iters - 1:
            S.syncthreads()


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if tuple(A.shape) != (4096,) or tuple(B.shape) != (4096, 4096):
            return torch.diag(A) @ B
        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((4096, 4096), device=B.device, dtype=B.dtype)

        num_wg = (M // TILE_M) * (N // TILE_N)
        diag_left_mfma_kernel[lambda: ((num_wg, 1, 1), (256, 1, 1))](A, B, C)
        return C
