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
WAVE_SIZE = 64
WAVES_PER_BLOCK = 4
THREADS_PER_BLOCK = WAVE_SIZE * WAVES_PER_BLOCK
K_UNROLL = 2
PIPE_STAGES = 2
RANGE_BYTES = M * K * 2


@substrate.jit
def tri_gemm_mfma_kernel(
    A: S.Tensor((M, K), S.bf16),
    B: S.Tensor((K, N), S.bf16),
    C: S.Tensor((M, N), S.bf16),
    range_bytes: S.i32,
):
    tid = S.thread_id(0)
    wave_id = tid // WAVE_SIZE
    lane = tid % WAVE_SIZE
    warp_row = wave_id // 2
    warp_col = wave_id % 2

    block_row = S.block_id(1) * BLOCK_M
    block_col = S.block_id(0) * BLOCK_N
    tile_row_base = block_row + warp_row * 32
    tile_col_base = block_col + warp_col * 32

    a_rsrc = S.amdgpu.make_rsrc(A, range_bytes)
    b_rsrc = S.amdgpu.make_rsrc(B, range_bytes)

    a_lds_words = S.make_shared((PIPE_STAGES, WAVES_PER_BLOCK, WAVE_SIZE, 4), S.u32)
    b_lds_words = S.make_shared((PIPE_STAGES, WAVES_PER_BLOCK, WAVE_SIZE, 4), S.u32)

    zero_i32 = S.convert(0, S.i32)
    c_lane = S.full((16,), 0.0, S.f32)

    a_row = tile_row_base + (lane % 32)
    a_load_row = tile_row_base + (lane // 2)
    a_chunk = lane % 2

    b_load_k_row = (lane // 32) * 8 + (lane % 8)
    b_load_col = tile_col_base + ((lane % 32) // 8) * 8

    a_offset_0 = S.convert((a_load_row * K + a_chunk * 8) * 2, S.i32)
    b_offset_0 = S.convert((b_load_k_row * N + b_load_col) * 2, S.i32)
    a_pack_0 = S.amdgpu.raw_buffer_load_x4(a_rsrc, zero_i32, a_offset_0, 0)
    b_pack_0 = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero_i32, b_offset_0, 0)
    if a_chunk == 0:
        a_lds_words[0, wave_id, lane // 2, 0] = a_pack_0[0]
        a_lds_words[0, wave_id, lane // 2, 1] = a_pack_0[1]
        a_lds_words[0, wave_id, lane // 2 + 32, 0] = a_pack_0[2]
        a_lds_words[0, wave_id, lane // 2 + 32, 1] = a_pack_0[3]
    else:
        a_lds_words[0, wave_id, lane // 2, 2] = a_pack_0[0]
        a_lds_words[0, wave_id, lane // 2, 3] = a_pack_0[1]
        a_lds_words[0, wave_id, lane // 2 + 32, 2] = a_pack_0[2]
        a_lds_words[0, wave_id, lane // 2 + 32, 3] = a_pack_0[3]
    for word_idx in S.range(4):
        b_lds_words[0, wave_id, lane, word_idx] = b_pack_0[word_idx]

    k1_base = BLOCK_K
    a_offset_1 = S.convert((a_load_row * K + k1_base + a_chunk * 8) * 2, S.i32)
    b_offset_1 = S.convert(((k1_base + b_load_k_row) * N + b_load_col) * 2, S.i32)
    a_pack_1 = S.amdgpu.raw_buffer_load_x4(a_rsrc, zero_i32, a_offset_1, 0)
    b_pack_1 = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero_i32, b_offset_1, 0)
    if a_chunk == 0:
        a_lds_words[1, wave_id, lane // 2, 0] = a_pack_1[0]
        a_lds_words[1, wave_id, lane // 2, 1] = a_pack_1[1]
        a_lds_words[1, wave_id, lane // 2 + 32, 0] = a_pack_1[2]
        a_lds_words[1, wave_id, lane // 2 + 32, 1] = a_pack_1[3]
    else:
        a_lds_words[1, wave_id, lane // 2, 2] = a_pack_1[0]
        a_lds_words[1, wave_id, lane // 2, 3] = a_pack_1[1]
        a_lds_words[1, wave_id, lane // 2 + 32, 2] = a_pack_1[2]
        a_lds_words[1, wave_id, lane // 2 + 32, 3] = a_pack_1[3]
    for word_idx in S.range(4):
        b_lds_words[1, wave_id, lane, word_idx] = b_pack_1[word_idx]

    S.syncthreads()

    for k_pair_base in S.range(0, K, BLOCK_K * K_UNROLL):
        a_words_0 = a_lds_words[0, wave_id, lane]
        b_words_0 = b_lds_words[0, wave_id, lane]
        a_frag_0 = S.view(a_words_0, S.Tensor((2, 4, 1), S.bf16))
        b_frag_0 = S.view(b_words_0, S.Tensor((2, 4, 1), S.bf16))
        c_lane = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag_0[0], b_frag_0[0], c_lane)

        next_pair_base = k_pair_base + BLOCK_K * K_UNROLL

        c_lane = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag_0[1], b_frag_0[1], c_lane)

        if next_pair_base < K:
            next_stage0_a_offset = S.convert((a_load_row * K + next_pair_base + a_chunk * 8) * 2, S.i32)
            next_stage0_b_offset = S.convert(((next_pair_base + b_load_k_row) * N + b_load_col) * 2, S.i32)
            next_a_pack_0 = S.amdgpu.raw_buffer_load_x4(a_rsrc, zero_i32, next_stage0_a_offset, 0)
            next_b_pack_0 = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero_i32, next_stage0_b_offset, 0)
            if a_chunk == 0:
                a_lds_words[0, wave_id, lane // 2, 0] = next_a_pack_0[0]
                a_lds_words[0, wave_id, lane // 2, 1] = next_a_pack_0[1]
                a_lds_words[0, wave_id, lane // 2 + 32, 0] = next_a_pack_0[2]
                a_lds_words[0, wave_id, lane // 2 + 32, 1] = next_a_pack_0[3]
            else:
                a_lds_words[0, wave_id, lane // 2, 2] = next_a_pack_0[0]
                a_lds_words[0, wave_id, lane // 2, 3] = next_a_pack_0[1]
                a_lds_words[0, wave_id, lane // 2 + 32, 2] = next_a_pack_0[2]
                a_lds_words[0, wave_id, lane // 2 + 32, 3] = next_a_pack_0[3]
            for word_idx in S.range(4):
                b_lds_words[0, wave_id, lane, word_idx] = next_b_pack_0[word_idx]

        a_words_1 = a_lds_words[1, wave_id, lane]
        b_words_1 = b_lds_words[1, wave_id, lane]
        a_frag_1 = S.view(a_words_1, S.Tensor((2, 4, 1), S.bf16))
        b_frag_1 = S.view(b_words_1, S.Tensor((2, 4, 1), S.bf16))
        c_lane = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag_1[0], b_frag_1[0], c_lane)

        c_lane = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag_1[1], b_frag_1[1], c_lane)

        if next_pair_base < K:
            next_stage1_base = next_pair_base + BLOCK_K
            next_stage1_a_offset = S.convert((a_load_row * K + next_stage1_base + a_chunk * 8) * 2, S.i32)
            next_stage1_b_offset = S.convert(((next_stage1_base + b_load_k_row) * N + b_load_col) * 2, S.i32)
            next_a_pack_1 = S.amdgpu.raw_buffer_load_x4(a_rsrc, zero_i32, next_stage1_a_offset, 0)
            next_b_pack_1 = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero_i32, next_stage1_b_offset, 0)
            if a_chunk == 0:
                a_lds_words[1, wave_id, lane // 2, 0] = next_a_pack_1[0]
                a_lds_words[1, wave_id, lane // 2, 1] = next_a_pack_1[1]
                a_lds_words[1, wave_id, lane // 2 + 32, 0] = next_a_pack_1[2]
                a_lds_words[1, wave_id, lane // 2 + 32, 1] = next_a_pack_1[3]
            else:
                a_lds_words[1, wave_id, lane // 2, 2] = next_a_pack_1[0]
                a_lds_words[1, wave_id, lane // 2, 3] = next_a_pack_1[1]
                a_lds_words[1, wave_id, lane // 2 + 32, 2] = next_a_pack_1[2]
                a_lds_words[1, wave_id, lane // 2 + 32, 3] = next_a_pack_1[3]
            for word_idx in S.range(4):
                b_lds_words[1, wave_id, lane, word_idx] = next_b_pack_1[word_idx]

        S.syncthreads()

    lane_col = tile_col_base + (lane % 32)
    lane_row_quad = tile_row_base + 4 * (lane // 32)
    for acc_idx in S.range(16):
        row = lane_row_quad + 8 * (acc_idx // 4) + (acc_idx % 4)
        if lane_col <= row:
            C[row, lane_col] = S.convert(c_lane[acc_idx], S.bf16)
        else:
            C[row, lane_col] = S.convert(0.0, S.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._range_bytes = RANGE_BYTES

    def forward(self, A, B):
        if tuple(A.shape) != (M, K) or tuple(B.shape) != (K, N):
            raise ValueError("ModelNew only supports 4096x4096 bf16 inputs")
        if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
            raise ValueError("ModelNew expects bf16 inputs")

        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)
        tri_gemm_mfma_kernel[lambda: ((N // BLOCK_N, M // BLOCK_M, 1), (THREADS_PER_BLOCK, 1, 1))](
            A, B, C, self._range_bytes
        )
        return C
