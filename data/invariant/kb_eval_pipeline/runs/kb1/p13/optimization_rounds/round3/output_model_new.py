import torch
import torch.nn as nn

import substrate
import substrate.language as S


M = 4096
K = 4096
N = 4096

BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 16
K_UNROLL = 2
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

    a_tile = S.make_shared((BLOCK_M, BLOCK_K), S.bf16)
    b_tile = S.make_shared((BLOCK_K, BLOCK_N), S.bf16)

    acc = S.full((16,), 0.0, S.f32)

    # The resource range is specified in bytes. Raw buffer loads use it to
    # zero-fill OOB accesses instead of requiring explicit bounds branches.
    a_rsrc = S.amdgpu.make_rsrc(A, TOTAL_A_BYTES)
    b_rsrc = S.amdgpu.make_rsrc(B, TOTAL_B_BYTES)

    a_frag = S.make_local((2, 4), S.bf16)
    b_frag = S.make_local((2, 4), S.bf16)

    a_row = warp_row * 32 + (lane % 32)
    a_col_group = lane // 32

    b_col = warp_col * 32 + (lane % 32)
    b_k_group = lane // 32

    for k_base in S.range(0, K, BLOCK_K * K_UNROLL):
        for k_step in S.range(K_UNROLL):
            k_iter = k_base + k_step * BLOCK_K

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
                    a_tile[row, chunk * 8 + t] = vals[t]
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
                    b_tile[row, chunk * 8 + t] = vals[t]

            S.syncthreads()

            for t in S.range(4):
                a_frag[0, t] = a_tile[a_row, 4 * a_col_group + t]
                a_frag[1, t] = a_tile[a_row, 8 + 4 * a_col_group + t]

                b_frag[0, t] = b_tile[4 * b_k_group + t, b_col]
                b_frag[1, t] = b_tile[8 + 4 * b_k_group + t, b_col]

            acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], acc)
            acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[1], b_frag[1], acc)

            S.syncthreads()

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
        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)
        grid = (N // BLOCK_N, M // BLOCK_M, 1)
        block = (THREADS, 1, 1)
        gemm_kernel[lambda: (grid, block)](A, B, C, num_warps=4)
        return C
