import torch
import torch.nn as nn

import substrate
import substrate.language as S


M = 2048
K = 8192
N = 4096
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 16
WAVE_M = 32
WAVE_N = 32
WARP_SIZE = 64
NUM_WARPS = 4
THREADS = WARP_SIZE * NUM_WARPS


@substrate.jit
def gemm_kernel(
    A: S.Tensor((2048, 8192), S.bf16),
    B: S.Tensor((8192, 4096), S.bf16),
    C: S.Tensor((2048, 4096), S.bf16),
):
    tid = S.thread_id(0)
    lane = tid % WARP_SIZE
    warp = tid // WARP_SIZE
    warp_row = warp % 2
    warp_col = warp // 2
    block_row = S.block_id(1) * BLOCK_M
    block_col = S.block_id(0) * BLOCK_N

    a_words = S.make_shared((2 * 2 * WARP_SIZE * 4,), S.u32)
    b_words = S.make_shared((2 * 2 * WARP_SIZE * 4,), S.u32)
    a_word_layout = S.make_layout((2, 2, WARP_SIZE, 4), (2 * WARP_SIZE * 4, WARP_SIZE * 4, 4, 1))
    b_word_layout = S.make_layout((2, 2, WARP_SIZE, 4), (2 * WARP_SIZE * 4, WARP_SIZE * 4, 4, 1))
    a_word_view = S.view(a_words, S.u32, a_word_layout)
    b_word_view = S.view(b_words, S.u32, b_word_layout)

    zero = S.convert(0, S.i32)
    a_range = S.convert(M * K * 2, S.i32)
    b_range = S.convert(K * N * 2, S.i32)
    a_rsrc = S.amdgpu.make_rsrc(A, a_range)
    b_rsrc = S.amdgpu.make_rsrc(B, b_range)

    c_lane = S.full((16,), 0.0, S.f32)

    if tid < 128:
        a_owner = tid // WARP_SIZE
        a_lane = tid % WARP_SIZE
        a_row = block_row + a_owner * WAVE_M + (a_lane % 32)
        a_k_base = (a_lane // 32) * 8
        a_offset_el = a_row * K + a_k_base
        a_packed = S.amdgpu.raw_buffer_load_x4(
            a_rsrc, zero, S.convert(a_offset_el * 2, S.i32), 0
        )
        if a_lane < 32:
            a_word_view[0, a_owner, a_lane, 0] = a_packed[0]
            a_word_view[0, a_owner, a_lane, 1] = a_packed[1]
            a_word_view[0, a_owner, a_lane + 32, 0] = a_packed[2]
            a_word_view[0, a_owner, a_lane + 32, 1] = a_packed[3]
        else:
            a_dst_lane = a_lane - 32
            a_word_view[0, a_owner, a_dst_lane, 2] = a_packed[0]
            a_word_view[0, a_owner, a_dst_lane, 3] = a_packed[1]
            a_word_view[0, a_owner, a_dst_lane + 32, 2] = a_packed[2]
            a_word_view[0, a_owner, a_dst_lane + 32, 3] = a_packed[3]
    else:
        b_tid = tid - 128
        b_owner = b_tid // WARP_SIZE
        b_lane = b_tid % WARP_SIZE
        b_k = (b_lane % 8) + (b_lane // 32) * 8
        b_col_base = b_owner * WAVE_N + ((b_lane % 32) // 8) * 8
        b_offset_el = b_k * N + (block_col + b_col_base)
        b_packed = S.amdgpu.raw_buffer_load_x4(
            b_rsrc, zero, S.convert(b_offset_el * 2, S.i32), 0
        )
        if b_lane < 32:
            b_word_view[0, b_owner, b_lane, 0] = b_packed[0]
            b_word_view[0, b_owner, b_lane, 1] = b_packed[1]
            b_word_view[0, b_owner, b_lane + 32, 0] = b_packed[2]
            b_word_view[0, b_owner, b_lane + 32, 1] = b_packed[3]
        else:
            b_dst_lane = b_lane - 32
            b_word_view[0, b_owner, b_dst_lane, 2] = b_packed[0]
            b_word_view[0, b_owner, b_dst_lane, 3] = b_packed[1]
            b_word_view[0, b_owner, b_dst_lane + 32, 2] = b_packed[2]
            b_word_view[0, b_owner, b_dst_lane + 32, 3] = b_packed[3]

    S.syncthreads()

    for k0 in S.range(0, K, 2 * BLOCK_K):
        a_mfma_words = a_word_view[0, warp_row, lane]
        b_mfma_words = b_word_view[0, warp_col, lane]
        mfma_a = S.view(a_mfma_words, S.Tensor((2, 4, 1), S.bf16))
        mfma_b = S.view(b_mfma_words, S.Tensor((2, 4, 1), S.bf16))
        c_lane = S.amdgpu.mfma_32x32x8_bf16_f32(mfma_a[0], mfma_b[0], c_lane)

        if k0 + BLOCK_K < K:
            if tid < 128:
                a_owner = tid // WARP_SIZE
                a_lane = tid % WARP_SIZE
                a_row = block_row + a_owner * WAVE_M + (a_lane % 32)
                a_k_base = (a_lane // 32) * 8
                a_offset_el = a_row * K + (k0 + BLOCK_K + a_k_base)
                a_packed = S.amdgpu.raw_buffer_load_x4(
                    a_rsrc, zero, S.convert(a_offset_el * 2, S.i32), 0
                )
                if a_lane < 32:
                    a_word_view[1, a_owner, a_lane, 0] = a_packed[0]
                    a_word_view[1, a_owner, a_lane, 1] = a_packed[1]
                    a_word_view[1, a_owner, a_lane + 32, 0] = a_packed[2]
                    a_word_view[1, a_owner, a_lane + 32, 1] = a_packed[3]
                else:
                    a_dst_lane = a_lane - 32
                    a_word_view[1, a_owner, a_dst_lane, 2] = a_packed[0]
                    a_word_view[1, a_owner, a_dst_lane, 3] = a_packed[1]
                    a_word_view[1, a_owner, a_dst_lane + 32, 2] = a_packed[2]
                    a_word_view[1, a_owner, a_dst_lane + 32, 3] = a_packed[3]
            else:
                b_tid = tid - 128
                b_owner = b_tid // WARP_SIZE
                b_lane = b_tid % WARP_SIZE
                b_k = (b_lane % 8) + (b_lane // 32) * 8
                b_col_base = b_owner * WAVE_N + ((b_lane % 32) // 8) * 8
                b_offset_el = (k0 + BLOCK_K + b_k) * N + (block_col + b_col_base)
                b_packed = S.amdgpu.raw_buffer_load_x4(
                    b_rsrc, zero, S.convert(b_offset_el * 2, S.i32), 0
                )
                if b_lane < 32:
                    b_word_view[1, b_owner, b_lane, 0] = b_packed[0]
                    b_word_view[1, b_owner, b_lane, 1] = b_packed[1]
                    b_word_view[1, b_owner, b_lane + 32, 0] = b_packed[2]
                    b_word_view[1, b_owner, b_lane + 32, 1] = b_packed[3]
                else:
                    b_dst_lane = b_lane - 32
                    b_word_view[1, b_owner, b_dst_lane, 2] = b_packed[0]
                    b_word_view[1, b_owner, b_dst_lane, 3] = b_packed[1]
                    b_word_view[1, b_owner, b_dst_lane + 32, 2] = b_packed[2]
                    b_word_view[1, b_owner, b_dst_lane + 32, 3] = b_packed[3]

        a_mfma_words = a_word_view[0, warp_row, lane]
        b_mfma_words = b_word_view[0, warp_col, lane]
        mfma_a = S.view(a_mfma_words, S.Tensor((2, 4, 1), S.bf16))
        mfma_b = S.view(b_mfma_words, S.Tensor((2, 4, 1), S.bf16))
        c_lane = S.amdgpu.mfma_32x32x8_bf16_f32(mfma_a[1], mfma_b[1], c_lane)

        if k0 + BLOCK_K < K:
            S.syncthreads()

            a_mfma_words = a_word_view[1, warp_row, lane]
            b_mfma_words = b_word_view[1, warp_col, lane]
            mfma_a = S.view(a_mfma_words, S.Tensor((2, 4, 1), S.bf16))
            mfma_b = S.view(b_mfma_words, S.Tensor((2, 4, 1), S.bf16))
            c_lane = S.amdgpu.mfma_32x32x8_bf16_f32(mfma_a[0], mfma_b[0], c_lane)

            if k0 + 2 * BLOCK_K < K:
                if tid < 128:
                    a_owner = tid // WARP_SIZE
                    a_lane = tid % WARP_SIZE
                    a_row = block_row + a_owner * WAVE_M + (a_lane % 32)
                    a_k_base = (a_lane // 32) * 8
                    a_offset_el = a_row * K + (k0 + 2 * BLOCK_K + a_k_base)
                    a_packed = S.amdgpu.raw_buffer_load_x4(
                        a_rsrc, zero, S.convert(a_offset_el * 2, S.i32), 0
                    )
                    if a_lane < 32:
                        a_word_view[0, a_owner, a_lane, 0] = a_packed[0]
                        a_word_view[0, a_owner, a_lane, 1] = a_packed[1]
                        a_word_view[0, a_owner, a_lane + 32, 0] = a_packed[2]
                        a_word_view[0, a_owner, a_lane + 32, 1] = a_packed[3]
                    else:
                        a_dst_lane = a_lane - 32
                        a_word_view[0, a_owner, a_dst_lane, 2] = a_packed[0]
                        a_word_view[0, a_owner, a_dst_lane, 3] = a_packed[1]
                        a_word_view[0, a_owner, a_dst_lane + 32, 2] = a_packed[2]
                        a_word_view[0, a_owner, a_dst_lane + 32, 3] = a_packed[3]
                else:
                    b_tid = tid - 128
                    b_owner = b_tid // WARP_SIZE
                    b_lane = b_tid % WARP_SIZE
                    b_k = (b_lane % 8) + (b_lane // 32) * 8
                    b_col_base = b_owner * WAVE_N + ((b_lane % 32) // 8) * 8
                    b_offset_el = (k0 + 2 * BLOCK_K + b_k) * N + (block_col + b_col_base)
                    b_packed = S.amdgpu.raw_buffer_load_x4(
                        b_rsrc, zero, S.convert(b_offset_el * 2, S.i32), 0
                    )
                    if b_lane < 32:
                        b_word_view[0, b_owner, b_lane, 0] = b_packed[0]
                        b_word_view[0, b_owner, b_lane, 1] = b_packed[1]
                        b_word_view[0, b_owner, b_lane + 32, 0] = b_packed[2]
                        b_word_view[0, b_owner, b_lane + 32, 1] = b_packed[3]
                    else:
                        b_dst_lane = b_lane - 32
                        b_word_view[0, b_owner, b_dst_lane, 2] = b_packed[0]
                        b_word_view[0, b_owner, b_dst_lane, 3] = b_packed[1]
                        b_word_view[0, b_owner, b_dst_lane + 32, 2] = b_packed[2]
                        b_word_view[0, b_owner, b_dst_lane + 32, 3] = b_packed[3]

            a_mfma_words = a_word_view[1, warp_row, lane]
            b_mfma_words = b_word_view[1, warp_col, lane]
            mfma_a = S.view(a_mfma_words, S.Tensor((2, 4, 1), S.bf16))
            mfma_b = S.view(b_mfma_words, S.Tensor((2, 4, 1), S.bf16))
            c_lane = S.amdgpu.mfma_32x32x8_bf16_f32(mfma_a[1], mfma_b[1], c_lane)

            if k0 + 2 * BLOCK_K < K:
                S.syncthreads()

    tile_row_base = block_row + warp_row * WAVE_M
    tile_col_base = block_col + warp_col * WAVE_N
    for acc_idx in S.range(16):
        row = (
            tile_row_base
            + 8 * (acc_idx // 4)
            + 4 * (lane // 32)
            + (acc_idx % 4)
        )
        col = tile_col_base + (lane % 32)
        C[row, col] = S.convert(c_lane[acc_idx], S.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if tuple(A.shape) != (M, K) or tuple(B.shape) != (K, N):
            raise ValueError(
                f"ModelNew only supports A={(M, K)} and B={(K, N)}, "
                f"got {tuple(A.shape)} and {tuple(B.shape)}"
            )
        if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
            raise ValueError(
                f"ModelNew requires bfloat16 inputs, got {A.dtype} and {B.dtype}"
            )

        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)
        gemm_kernel[lambda: ((N // BLOCK_N, M // BLOCK_M, 1), (THREADS, 1, 1))](A, B, C)
        return C
