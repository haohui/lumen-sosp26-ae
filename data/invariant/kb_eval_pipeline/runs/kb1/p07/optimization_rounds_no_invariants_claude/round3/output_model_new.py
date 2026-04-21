import torch
import torch.nn as nn

import substrate
import substrate.language as S


M = 32768
K = 64
N = 32768

# Tile sizes for 4 waves (2x2 wave grid)
WG_M = 64
WG_N = 64
WG_K = 16  # Two MFMA (32x32x8 each) for K=16

WAVE_M = 32
WAVE_N = 32
WAVE_SIZE = 64

NUM_WAVES = 4
BLOCK_SIZE = NUM_WAVES * WAVE_SIZE

BYTE_SIZE_BF16 = 2
TOTAL_A_BYTES = M * K * BYTE_SIZE_BF16
TOTAL_B_BYTES = K * N * BYTE_SIZE_BF16


@substrate.jit
def gemm_mfma_kernel(
    A: S.Tensor((M, K), S.bf16),
    B: S.Tensor((K, N), S.bf16),
    C: S.Tensor((M, N), S.bf16),
):
    """
    MFMA-based GEMM kernel using software pipelining and double buffering.
    OOB access is handled by range in raw buffer loads - returns 0 for OOB.
    """
    tid = S.thread_id(0)
    lane = tid % WAVE_SIZE
    warp = tid // WAVE_SIZE
    warp_row = warp // 2  # Row in 2x2 wave grid
    warp_col = warp % 2   # Col in 2x2 wave grid

    # Note: block_id(1) is row, block_id(0) is col (matching grid order)
    block_row = S.block_id(1) * WG_M
    block_col = S.block_id(0) * WG_N

    # Double buffered shared memory
    a_tile_0 = S.make_shared((WG_M, WG_K), S.bf16)
    a_tile_1 = S.make_shared((WG_M, WG_K), S.bf16)
    b_tile_0 = S.make_shared((WG_K, WG_N), S.bf16)
    b_tile_1 = S.make_shared((WG_K, WG_N), S.bf16)

    # Resource descriptors for raw buffer loads with range set for OOB handling
    a_rsrc = S.amdgpu.make_rsrc(A, TOTAL_A_BYTES)
    b_rsrc = S.amdgpu.make_rsrc(B, TOTAL_B_BYTES)

    # Accumulator
    acc = S.full((16,), 0.0, S.f32)

    # Local fragments for MFMA inputs
    a_frag = S.make_local((2, 4), S.bf16)
    b_frag = S.make_local((2, 4), S.bf16)

    # Precompute indices for loading from shared memory
    a_row = warp_row * 32 + (lane % 32)
    a_col_group = lane // 32

    b_col = warp_col * 32 + (lane % 32)
    b_k_group = lane // 32

    # ============================================
    # Prologue: Load first K-tile using raw buffer loads
    # ============================================
    k_iter = 0
    if tid < 128:
        load_id = tid
        row = load_id // (WG_K // 8)
        chunk = load_id % (WG_K // 8)
        byte_offset = ((block_row + row) * K + k_iter + chunk * 8) * BYTE_SIZE_BF16
        vec = S.amdgpu.raw_buffer_load_x4(a_rsrc, 0, byte_offset, 0)
        vals = S.view(vec, S.Tensor((8,), S.bf16))
        for t in S.range(8):
            a_tile_0[row, chunk * 8 + t] = vals[t]
    else:
        load_id = tid - 128
        row = load_id // (WG_N // 8)
        chunk = load_id % (WG_N // 8)
        byte_offset = ((k_iter + row) * N + block_col + chunk * 8) * BYTE_SIZE_BF16
        vec = S.amdgpu.raw_buffer_load_x4(b_rsrc, 0, byte_offset, 0)
        vals = S.view(vec, S.Tensor((8,), S.bf16))
        for t in S.range(8):
            b_tile_0[row, chunk * 8 + t] = vals[t]

    S.syncthreads()

    # ============================================
    # Main loop - unrolled by 2 with double buffering
    # OOB branches removed - range in rsrc handles OOB access
    # ============================================
    for k_base in S.range(0, K, WG_K * 2):
        # First K-tile of the pair
        k_iter_0 = k_base
        k_iter_1 = k_base + WG_K

        # Compute on first K-tile (from buffer 0)
        for t in S.range(4):
            a_frag[0, t] = a_tile_0[a_row, 4 * a_col_group + t]
            a_frag[1, t] = a_tile_0[a_row, 8 + 4 * a_col_group + t]

            b_frag[0, t] = b_tile_0[4 * b_k_group + t, b_col]
            b_frag[1, t] = b_tile_0[8 + 4 * b_k_group + t, b_col]

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[1], b_frag[1], acc)

        # Prefetch second K-tile into buffer 1 (overlapping)
        if tid < 128:
            load_id = tid
            row = load_id // (WG_K // 8)
            chunk = load_id % (WG_K // 8)
            byte_offset = ((block_row + row) * K + k_iter_1 + chunk * 8) * BYTE_SIZE_BF16
            vec = S.amdgpu.raw_buffer_load_x4(a_rsrc, 0, byte_offset, 0)
            vals = S.view(vec, S.Tensor((8,), S.bf16))
            for t in S.range(8):
                a_tile_1[row, chunk * 8 + t] = vals[t]
        else:
            load_id = tid - 128
            row = load_id // (WG_N // 8)
            chunk = load_id % (WG_N // 8)
            byte_offset = ((k_iter_1 + row) * N + block_col + chunk * 8) * BYTE_SIZE_BF16
            vec = S.amdgpu.raw_buffer_load_x4(b_rsrc, 0, byte_offset, 0)
            vals = S.view(vec, S.Tensor((8,), S.bf16))
            for t in S.range(8):
                b_tile_1[row, chunk * 8 + t] = vals[t]

        S.syncthreads()

        # Compute on second K-tile (from buffer 1)
        for t in S.range(4):
            a_frag[0, t] = a_tile_1[a_row, 4 * a_col_group + t]
            a_frag[1, t] = a_tile_1[a_row, 8 + 4 * a_col_group + t]

            b_frag[0, t] = b_tile_1[4 * b_k_group + t, b_col]
            b_frag[1, t] = b_tile_1[8 + 4 * b_k_group + t, b_col]

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[1], b_frag[1], acc)

        # Prefetch next pair's first K-tile into buffer 0 (overlapping)
        # No OOB branch needed - range in rsrc returns 0 for OOB access
        next_k_iter = k_base + 2 * WG_K
        if tid < 128:
            load_id = tid
            row = load_id // (WG_K // 8)
            chunk = load_id % (WG_K // 8)
            byte_offset = ((block_row + row) * K + next_k_iter + chunk * 8) * BYTE_SIZE_BF16
            vec = S.amdgpu.raw_buffer_load_x4(a_rsrc, 0, byte_offset, 0)
            vals = S.view(vec, S.Tensor((8,), S.bf16))
            for t in S.range(8):
                a_tile_0[row, chunk * 8 + t] = vals[t]
        else:
            load_id = tid - 128
            row = load_id // (WG_N // 8)
            chunk = load_id % (WG_N // 8)
            byte_offset = ((next_k_iter + row) * N + block_col + chunk * 8) * BYTE_SIZE_BF16
            vec = S.amdgpu.raw_buffer_load_x4(b_rsrc, 0, byte_offset, 0)
            vals = S.view(vec, S.Tensor((8,), S.bf16))
            for t in S.range(8):
                b_tile_0[row, chunk * 8 + t] = vals[t]

        S.syncthreads()

    # Store results to global memory
    tile_row_base = block_row + warp_row * 32
    tile_col_base = block_col + warp_col * 32
    col = tile_col_base + (lane % 32)
    row_group = 4 * (lane // 32)
    for acc_idx in S.range(16):
        row = tile_row_base + 8 * (acc_idx // 4) + row_group + (acc_idx % 4)
        C[row, col] = S.convert(acc[acc_idx], S.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._cached_shapes = None

    def forward(self, A, B):
        if tuple(A.shape) != (M, K) or tuple(B.shape) != (K, N):
            return torch.matmul(A, B)

        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)

        # Note: grid is (N // WG_N, M // WG_M, 1) - col major order
        grid = (N // WG_N, M // WG_M, 1)
        block = (BLOCK_SIZE, 1, 1)
        gemm_mfma_kernel[lambda: (grid, block)](A, B, C, num_warps=4)
        return C
