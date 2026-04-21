import torch
import torch.nn as nn

import substrate
import substrate.language as S


M = 32768
K = 32
N = 32768

BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 16
NUM_K_ITERS = K // BLOCK_K  # 2
WAVE_SIZE = 64
WAVES_M = 2
WAVES_N = 2
THREADS = WAVE_SIZE * WAVES_M * WAVES_N

BYTE_SIZE_BF16 = 2
TOTAL_A_BYTES = M * K * BYTE_SIZE_BF16
TOTAL_B_BYTES = K * N * BYTE_SIZE_BF16


@substrate.jit
def gemm_kernel(
    A: S.Tensor((M, K), S.bf16),
    B: S.Tensor((K, N), S.bf16),
    C: S.Tensor((M, N), S.bf16),
):
    tid = S.thread_id(0)
    lane = tid % WAVE_SIZE
    warp = tid // WAVE_SIZE
    warp_row = warp // WAVES_N
    warp_col = warp % WAVES_N

    block_row = S.block_id(1) * BLOCK_M
    block_col = S.block_id(0) * BLOCK_N

    # Double buffering: two shared memory tiles
    a_tile = S.make_shared((2, BLOCK_M, BLOCK_K), S.bf16)
    b_tile = S.make_shared((2, BLOCK_K, BLOCK_N), S.bf16)

    acc = S.full((16,), 0.0, S.f32)

    # Create resource descriptors with range for OOB handling
    # When range is set, raw_buffer_load returns 0 for OOB elements
    # and raw_buffer_store discards OOB writes
    a_rsrc = S.amdgpu.make_rsrc(A, TOTAL_A_BYTES)
    b_rsrc = S.amdgpu.make_rsrc(B, TOTAL_B_BYTES)

    # Fragment storage for MFMA - split to enable overlap
    a_frag = S.make_local((2, 4), S.bf16)
    b_frag = S.make_local((2, 4), S.bf16)

    a_row = warp_row * 32 + (lane % 32)
    a_col_group = lane // 32

    b_col = warp_col * 32 + (lane % 32)
    b_k_group = lane // 32

    # Prologue: Load first tile (k=0) into buffer 0
    if tid < 128:
        load_id = tid
        row = load_id // (BLOCK_K // 8)
        chunk = load_id % (BLOCK_K // 8)
        byte_offset = ((block_row + row) * K + chunk * 8) * BYTE_SIZE_BF16
        vec = S.amdgpu.raw_buffer_load_x4(a_rsrc, 0, byte_offset, 0)
        vals = S.view(vec, S.Tensor((8,), S.bf16))
        for t in S.range(8):
            a_tile[0, row, chunk * 8 + t] = vals[t]
    else:
        load_id = tid - 128
        row = load_id // (BLOCK_N // 8)
        chunk = load_id % (BLOCK_N // 8)
        byte_offset = ((row) * N + block_col + chunk * 8) * BYTE_SIZE_BF16
        vec = S.amdgpu.raw_buffer_load_x4(b_rsrc, 0, byte_offset, 0)
        vals = S.view(vec, S.Tensor((8,), S.bf16))
        for t in S.range(8):
            b_tile[0, row, chunk * 8 + t] = vals[t]

    S.syncthreads()

    # Software pipelined main loop with double buffering
    # Unroll by 2 (NUM_K_ITERS = 2) to minimize branching
    # Removed the k_iter < NUM_K_ITERS - 1 branch guard by using range
    # in make_rsrc - OOB accesses return 0 which is harmless
    for k_iter in S.range(NUM_K_ITERS):
        buf_idx = k_iter % 2
        next_buf_idx = 1 - buf_idx

        # Prefetch next tile from global memory (double buffering)
        # This overlaps with MFMA computation
        # No branch guard needed - range in rsrc handles OOB gracefully
        next_k_offset = (k_iter + 1) * BLOCK_K
        if tid < 128:
            load_id = tid
            row = load_id // (BLOCK_K // 8)
            chunk = load_id % (BLOCK_K // 8)
            byte_offset = ((block_row + row) * K + next_k_offset + chunk * 8) * BYTE_SIZE_BF16
            vec = S.amdgpu.raw_buffer_load_x4(a_rsrc, 0, byte_offset, 0)
            vals = S.view(vec, S.Tensor((8,), S.bf16))
            for t in S.range(8):
                a_tile[next_buf_idx, row, chunk * 8 + t] = vals[t]
        else:
            load_id = tid - 128
            row = load_id // (BLOCK_N // 8)
            chunk = load_id % (BLOCK_N // 8)
            byte_offset = ((next_k_offset + row) * N + block_col + chunk * 8) * BYTE_SIZE_BF16
            vec = S.amdgpu.raw_buffer_load_x4(b_rsrc, 0, byte_offset, 0)
            vals = S.view(vec, S.Tensor((8,), S.bf16))
            for t in S.range(8):
                b_tile[next_buf_idx, row, chunk * 8 + t] = vals[t]

        # Fine-grained split: Load first half of fragments from LDS
        for t in S.range(4):
            a_frag[0, t] = a_tile[buf_idx, a_row, 4 * a_col_group + t]
            b_frag[0, t] = b_tile[buf_idx, 4 * b_k_group + t, b_col]

        # Issue first MFMA instruction
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], acc)

        # Load second half of fragments while first MFMA executes
        # This overlaps LDS access with MFMA computation
        for t in S.range(4):
            a_frag[1, t] = a_tile[buf_idx, a_row, 8 + 4 * a_col_group + t]
            b_frag[1, t] = b_tile[buf_idx, 8 + 4 * b_k_group + t, b_col]

        # Issue second MFMA instruction
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[1], b_frag[1], acc)

        # Synchronize to ensure prefetch is complete for next iteration
        S.syncthreads()

    # Write output
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

    def forward(self, A, B):
        if tuple(A.shape) != (32768, 32) or tuple(B.shape) != (32, 32768):
            return torch.matmul(A, B)
        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((32768, 32768), device=A.device, dtype=A.dtype)
        grid = (N // BLOCK_N, M // BLOCK_M, 1)
        block = (THREADS, 1, 1)
        gemm_kernel[lambda: (grid, block)](A, B, C, num_warps=4)
        return C
