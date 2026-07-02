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
WARP_M = 32
WARP_N = 32
WAVE_SIZE = 64
NUM_WARPS = 4
THREADS = WAVE_SIZE * NUM_WARPS
RANGE_BYTES = M * K * 2


@substrate.jit
def tri_gemm_kernel(
    A: S.Tensor((M, K), S.bf16),
    B: S.Tensor((K, N), S.bf16),
    C: S.Tensor((M, N), S.bf16),
):
    block_row = S.block_id(1) * BLOCK_M
    block_col = S.block_id(0) * BLOCK_N

    tid = S.thread_id(0)
    warp_id = tid // WAVE_SIZE
    lane = tid % WAVE_SIZE
    warp_row = warp_id // 2
    warp_col = warp_id % 2

    tile_row_base = block_row + warp_row * WARP_M
    tile_col_base = block_col + warp_col * WARP_N

    acc = S.full((16,), 0.0, S.f32)

    if block_col + BLOCK_N <= block_row:
        for acc_idx in S.range(16):
            out_col = tile_col_base + (lane % 32)
            out_row = tile_row_base + 8 * (acc_idx // 4) + 4 * (lane // 32) + (acc_idx % 4)
            C[out_row, out_col] = S.convert(0.0, S.bf16)
        return

    a_words = S.make_shared((NUM_WARPS, WAVE_SIZE, 4), S.u32)
    b_words = S.make_shared((NUM_WARPS, WAVE_SIZE, 4), S.u32)
    lane_layout = S.make_layout((NUM_WARPS, WAVE_SIZE, 2, 4, 1), (WAVE_SIZE * 8, 8, 4, 1, 1))
    a_shared = S.view(a_words, S.bf16, lane_layout)
    b_shared = S.view(b_words, S.bf16, lane_layout)

    a_rsrc = S.amdgpu.make_rsrc(A, S.convert(RANGE_BYTES, S.i32))
    b_rsrc = S.amdgpu.make_rsrc(B, S.convert(RANGE_BYTES, S.i32))
    zero = S.convert(0, S.i32)

    for k_tile in S.range(K // BLOCK_K):
        k_base = k_tile * BLOCK_K

        a_row = tile_row_base + (lane % 32)
        a_seg = lane // 32
        a_col = k_base + a_seg * 8
        a_offset = S.convert((a_row * K + a_col) * 2, S.i32)
        a_pack = S.amdgpu.raw_buffer_load_x4(a_rsrc, zero, a_offset, 0)
        a_frag = S.view(a_pack, S.Tensor((2, 4, 1), S.bf16))

        for e in S.range(4):
            if a_seg == 0:
                a_shared[warp_id, lane % 32, 0, e, 0] = a_frag[0, e, 0]
                a_shared[warp_id, lane % 32 + 32, 0, e, 0] = a_frag[1, e, 0]
            else:
                a_shared[warp_id, lane % 32, 1, e, 0] = a_frag[0, e, 0]
                a_shared[warp_id, lane % 32 + 32, 1, e, 0] = a_frag[1, e, 0]

        b_k = k_base + (lane % 16)
        b_chunk = lane // 16
        b_col = tile_col_base + b_chunk * 8
        b_offset = S.convert((b_k * N + b_col) * 2, S.i32)
        b_pack = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero, b_offset, 0)
        b_frag = S.view(b_pack, S.Tensor((2, 4, 1), S.bf16))

        for e in S.range(4):
            col0 = b_chunk * 8 + e
            col1 = b_chunk * 8 + 4 + e
            if lane % 16 < 4:
                b_shared[warp_id, col0, 0, lane % 16, 0] = b_frag[0, e, 0]
                b_shared[warp_id, col1, 0, lane % 16, 0] = b_frag[1, e, 0]
            elif lane % 16 < 8:
                kk = lane % 16 - 4
                b_shared[warp_id, col0 + 32, 0, kk, 0] = b_frag[0, e, 0]
                b_shared[warp_id, col1 + 32, 0, kk, 0] = b_frag[1, e, 0]
            elif lane % 16 < 12:
                kk = lane % 16 - 8
                b_shared[warp_id, col0, 1, kk, 0] = b_frag[0, e, 0]
                b_shared[warp_id, col1, 1, kk, 0] = b_frag[1, e, 0]
            else:
                kk = lane % 16 - 12
                b_shared[warp_id, col0 + 32, 1, kk, 0] = b_frag[0, e, 0]
                b_shared[warp_id, col1 + 32, 1, kk, 0] = b_frag[1, e, 0]

        S.syncthreads()

        a_mfma = S.view(a_words[warp_id, lane], S.Tensor((2, 4, 1), S.bf16))
        b_mfma = S.view(b_words[warp_id, lane], S.Tensor((2, 4, 1), S.bf16))

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_mfma[0], b_mfma[0], acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_mfma[1], b_mfma[1], acc)

        S.syncthreads()

    for acc_idx in S.range(16):
        out_col = tile_col_base + (lane % 32)
        out_row = tile_row_base + 8 * (acc_idx // 4) + 4 * (lane // 32) + (acc_idx % 4)
        if out_col >= out_row:
            C[out_row, out_col] = S.convert(acc[acc_idx], S.bf16)
        else:
            C[out_row, out_col] = S.convert(0.0, S.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if tuple(A.shape) != (M, K) or tuple(B.shape) != (K, N):
            raise ValueError("ModelNew only supports 4096x4096 inputs.")
        if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
            raise ValueError("ModelNew requires bfloat16 inputs.")

        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)
        tri_gemm_kernel[lambda: ((N // BLOCK_N, M // BLOCK_M, 1), (THREADS, 1, 1))](A, B, C)
        return C
