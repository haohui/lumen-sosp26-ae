import torch
import torch.nn as nn

import substrate
import substrate.language as S


# Problem dimensions: C = A @ B where
# A is (2048, 8192), B is (8192, 4096), C is (2048, 4096)
# Note: forward() receives A as (8192, 2048) and B as (4096, 8192)
# and computes C = A.T @ B.T
M = 2048
K = 8192
N = 4096

BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 16
WAVE_SIZE = 64
WAVES_M = 2
WAVES_N = 2
THREADS = WAVE_SIZE * WAVES_M * WAVES_N

BYTE_SIZE_BF16 = 2
TOTAL_A_BYTES = M * K * BYTE_SIZE_BF16
TOTAL_B_BYTES = K * N * BYTE_SIZE_BF16

NUM_K_TILES = K // BLOCK_K  # 512 total K tiles


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

    block_row = S.block_id(0) * BLOCK_M
    block_col = S.block_id(1) * BLOCK_N

    # Double buffering: two tiles each for A and B
    a_tile = S.make_shared((2, BLOCK_M, BLOCK_K), S.bf16)
    b_tile = S.make_shared((2, BLOCK_K, BLOCK_N), S.bf16)

    acc = S.full((16,), 0.0, S.f32)

    # Create buffer descriptors for raw buffer loads
    a_rsrc = S.amdgpu.make_rsrc(A, TOTAL_A_BYTES)
    b_rsrc = S.amdgpu.make_rsrc(B, TOTAL_B_BYTES)

    a_frag = S.make_local((2, 4), S.bf16)
    b_frag = S.make_local((2, 4), S.bf16)

    a_row = warp_row * 32 + (lane % 32)
    a_col_group = lane // 32

    b_col = warp_col * 32 + (lane % 32)
    b_k_group = lane // 32

    # Prologue: load first tile into buffer 0
    k_iter = 0
    if tid < 128:
        load_id = tid
        row = load_id // (BLOCK_K // 8)
        chunk = load_id % (BLOCK_K // 8)
        byte_offset = (
            ((block_row + row) * K + k_iter + chunk * 8) * BYTE_SIZE_BF16
        )
        vec = S.amdgpu.raw_buffer_load_x4(a_rsrc, 0, byte_offset, 0)
        vals = S.view(vec, S.Tensor((8,), S.bf16))
        for t in S.range(8):
            a_tile[0, row, chunk * 8 + t] = vals[t]
    else:
        load_id = tid - 128
        row = load_id // (BLOCK_N // 8)
        chunk = load_id % (BLOCK_N // 8)
        byte_offset = (
            ((k_iter + row) * N + block_col + chunk * 8) * BYTE_SIZE_BF16
        )
        vec = S.amdgpu.raw_buffer_load_x4(b_rsrc, 0, byte_offset, 0)
        vals = S.view(vec, S.Tensor((8,), S.bf16))
        for t in S.range(8):
            b_tile[0, row, chunk * 8 + t] = vals[t]

    S.syncthreads()

    # Main loop with software pipelining and double buffering
    # Each iteration processes one K tile with two MFMA instructions
    # Branch for OOB access removed - range in make_rsrc handles it:
    # OOB loads return 0, OOB stores are discarded
    for k_iter_idx in S.range(NUM_K_TILES):
        current_buf = k_iter_idx % 2
        next_buf = 1 - current_buf
        k_iter = k_iter_idx * BLOCK_K

        # Load next tile (prefetch) - overlaps with computation
        # Range in make_rsrc ensures OOB loads return 0, OOB stores discarded
        next_k_iter = (k_iter_idx + 1) * BLOCK_K

        # Prefetch A tile: first 128 threads
        if tid < 128:
            load_id = tid
            row = load_id // (BLOCK_K // 8)
            chunk = load_id % (BLOCK_K // 8)
            byte_offset = (
                ((block_row + row) * K + next_k_iter + chunk * 8) * BYTE_SIZE_BF16
            )
            vec = S.amdgpu.raw_buffer_load_x4(a_rsrc, 0, byte_offset, 0)
            vals = S.view(vec, S.Tensor((8,), S.bf16))
            for t in S.range(8):
                a_tile[next_buf, row, chunk * 8 + t] = vals[t]
        else:
            # Prefetch B tile: next 128 threads
            load_id = tid - 128
            row = load_id // (BLOCK_N // 8)
            chunk = load_id % (BLOCK_N // 8)
            byte_offset = (
                ((next_k_iter + row) * N + block_col + chunk * 8) * BYTE_SIZE_BF16
            )
            vec = S.amdgpu.raw_buffer_load_x4(b_rsrc, 0, byte_offset, 0)
            vals = S.view(vec, S.Tensor((8,), S.bf16))
            for t in S.range(8):
                b_tile[next_buf, row, chunk * 8 + t] = vals[t]

        # Compute on current buffer - two MFMA instructions
        # Fine-grained overlap: split LDS reads and MFMA
        # Load first half of fragments
        for t in S.range(4):
            a_frag[0, t] = a_tile[current_buf, a_row, 4 * a_col_group + t]
            b_frag[0, t] = b_tile[current_buf, 4 * b_k_group + t, b_col]

        # Issue first MFMA instruction
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], acc)

        # Load second half of fragments while first MFMA executes
        # This overlaps LDS access with MFMA computation
        for t in S.range(4):
            a_frag[1, t] = a_tile[current_buf, a_row, 8 + 4 * a_col_group + t]
            b_frag[1, t] = b_tile[current_buf, 8 + 4 * b_k_group + t, b_col]

        # Issue second MFMA instruction
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[1], b_frag[1], acc)

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
        if tuple(A.shape) != (8192, 2048) or tuple(B.shape) != (4096, 8192):
            return torch.matmul(A.T, B.T)

        A2 = A.transpose(-2, -1).contiguous()
        B2 = B.transpose(-2, -1).contiguous()
        C = torch.empty((2048, 4096), device=A.device, dtype=A.dtype)

        # Grid: (M / 64, N / 64, 1) = (32, 64, 1)
        # Block: (256, 1, 1) for 4 warps
        grid = (M // BLOCK_M, N // BLOCK_N, 1)
        block = (THREADS, 1, 1)
        gemm_kernel[lambda: (grid, block)](A2, B2, C, num_warps=4)
        return C
