import torch
import torch.nn as nn

import substrate
import substrate.language as S


M = 2048
K = 1048576
N = 1

BLOCK_ROWS = 64
BLOCK_THREADS = 256
WAVE_SIZE = 64
WAVES_PER_BLOCK = 4
PIPE_STAGES = 2

A_RANGE_BYTES = M * K * 2
B_RANGE_BYTES = K * N * 2
C_RANGE_BYTES = M * N * 2


def _launch_config():
    return ((M + BLOCK_ROWS - 1) // BLOCK_ROWS, 1, 1), (BLOCK_THREADS, 1, 1)


@substrate.jit
def gemv_mfma_kernel(
    A: S.Tensor((M, K), S.bf16),
    B: S.Tensor((K, N), S.bf16),
    C: S.Tensor((M, N), S.bf16),
):
    tid = S.thread_id(0)
    lane = tid % WAVE_SIZE
    wave = tid // WAVE_SIZE
    warp_row = wave // 2
    warp_col = wave % 2
    lane_lo = lane % 32
    lane_hi = lane // 32

    block_row = S.block_id(0) * BLOCK_ROWS
    wave_row_base = block_row + warp_row * 32

    a_rsrc = S.amdgpu.make_rsrc(A, A_RANGE_BYTES)
    b_rsrc = S.amdgpu.make_rsrc(B, B_RANGE_BYTES)
    c_rsrc = S.amdgpu.make_rsrc(C, C_RANGE_BYTES)

    a_stage = S.make_shared((PIPE_STAGES, WAVES_PER_BLOCK, WAVE_SIZE, 4), S.u32)
    b_stage = S.make_shared((PIPE_STAGES, WAVES_PER_BLOCK, WAVE_SIZE, 4), S.u32)
    a_pack_stage = S.make_shared((PIPE_STAGES, WAVES_PER_BLOCK, WAVE_SIZE, 4), S.u32)
    b_pack_stage = S.make_shared((PIPE_STAGES, WAVES_PER_BLOCK, WAVE_SIZE, 4), S.u32)

    a_row = wave_row_base + lane // 2
    a_chunk = lane % 2
    b_source_chunk = lane % 2

    a_elem_offset = (a_row * K) + (a_chunk * 8)
    a_vec = S.amdgpu.raw_buffer_load_x4(a_rsrc, a_elem_offset * 2, 0, 0)
    for i in S.range(4):
        a_stage[0, wave, lane, i] = a_vec[i]

    b_vec = S.amdgpu.raw_buffer_load_x4(b_rsrc, (b_source_chunk * 8) * 2, 0, 0)
    for i in S.range(4):
        b_stage[0, wave, lane, i] = b_vec[i]

    a_elem_offset = (a_row * K) + 16 + (a_chunk * 8)
    a_vec = S.amdgpu.raw_buffer_load_x4(a_rsrc, a_elem_offset * 2, 0, 0)
    for i in S.range(4):
        a_stage[1, wave, lane, i] = a_vec[i]

    b_vec = S.amdgpu.raw_buffer_load_x4(b_rsrc, (16 + b_source_chunk * 8) * 2, 0, 0)
    for i in S.range(4):
        b_stage[1, wave, lane, i] = b_vec[i]

    S.syncthreads()

    acc = S.full((16,), 0.0, S.f32)

    for k_base in S.range(0, K - 32, 32):
        a_src0_0 = a_stage[0, wave, 2 * lane_lo + 0]
        a_src0_1 = a_stage[0, wave, 2 * lane_lo + 1]
        if lane_hi == 0:
            a_pack_stage[0, wave, lane, 0] = a_src0_0[0]
            a_pack_stage[0, wave, lane, 1] = a_src0_0[1]
            a_pack_stage[0, wave, lane, 2] = a_src0_1[0]
            a_pack_stage[0, wave, lane, 3] = a_src0_1[1]
        else:
            a_pack_stage[0, wave, lane, 0] = a_src0_0[2]
            a_pack_stage[0, wave, lane, 1] = a_src0_0[3]
            a_pack_stage[0, wave, lane, 2] = a_src0_1[2]
            a_pack_stage[0, wave, lane, 3] = a_src0_1[3]
        for i in S.range(4):
            b_pack_stage[0, wave, lane, i] = 0
        if warp_col == 0 and lane < 8:
            b_src0_lo = b_stage[0, wave, 0]
            b_src0_hi = b_stage[0, wave, 1]
            b_word0_lo = b_src0_lo[lane // 2]
            b_word0_hi = b_src0_hi[lane // 2]
            if lane % 2 == 0:
                b_pack_stage[0, wave, lane, 0] = b_word0_lo & 0xFFFF
                b_pack_stage[0, wave, lane, 2] = b_word0_hi & 0xFFFF
            else:
                b_pack_stage[0, wave, lane, 0] = (b_word0_lo >> 16) & 0xFFFF
                b_pack_stage[0, wave, lane, 2] = (b_word0_hi >> 16) & 0xFFFF
        a_frag0 = S.view(a_pack_stage[0, wave, lane], S.Tensor((2, 4, 1), S.bf16))
        b_frag0 = S.view(b_pack_stage[0, wave, lane], S.Tensor((2, 4, 1), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[0], b_frag0[0], acc)

        a_elem_offset = (a_row * K) + k_base + 32 + (a_chunk * 8)
        a_vec = S.amdgpu.raw_buffer_load_x4(a_rsrc, a_elem_offset * 2, 0, 0)
        for i in S.range(4):
            a_stage[0, wave, lane, i] = a_vec[i]
        b_vec = S.amdgpu.raw_buffer_load_x4(
            b_rsrc, (k_base + 32 + b_source_chunk * 8) * 2, 0, 0
        )
        for i in S.range(4):
            b_stage[0, wave, lane, i] = b_vec[i]

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[1], b_frag0[1], acc)

        a_src1_0 = a_stage[1, wave, 2 * lane_lo + 0]
        a_src1_1 = a_stage[1, wave, 2 * lane_lo + 1]
        if lane_hi == 0:
            a_pack_stage[1, wave, lane, 0] = a_src1_0[0]
            a_pack_stage[1, wave, lane, 1] = a_src1_0[1]
            a_pack_stage[1, wave, lane, 2] = a_src1_1[0]
            a_pack_stage[1, wave, lane, 3] = a_src1_1[1]
        else:
            a_pack_stage[1, wave, lane, 0] = a_src1_0[2]
            a_pack_stage[1, wave, lane, 1] = a_src1_0[3]
            a_pack_stage[1, wave, lane, 2] = a_src1_1[2]
            a_pack_stage[1, wave, lane, 3] = a_src1_1[3]
        for i in S.range(4):
            b_pack_stage[1, wave, lane, i] = 0
        if warp_col == 0 and lane < 8:
            b_src1_lo = b_stage[1, wave, 0]
            b_src1_hi = b_stage[1, wave, 1]
            b_word1_lo = b_src1_lo[lane // 2]
            b_word1_hi = b_src1_hi[lane // 2]
            if lane % 2 == 0:
                b_pack_stage[1, wave, lane, 0] = b_word1_lo & 0xFFFF
                b_pack_stage[1, wave, lane, 2] = b_word1_hi & 0xFFFF
            else:
                b_pack_stage[1, wave, lane, 0] = (b_word1_lo >> 16) & 0xFFFF
                b_pack_stage[1, wave, lane, 2] = (b_word1_hi >> 16) & 0xFFFF
        a_frag1 = S.view(a_pack_stage[1, wave, lane], S.Tensor((2, 4, 1), S.bf16))
        b_frag1 = S.view(b_pack_stage[1, wave, lane], S.Tensor((2, 4, 1), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[0], b_frag1[0], acc)

        a_elem_offset = (a_row * K) + k_base + 48 + (a_chunk * 8)
        a_vec = S.amdgpu.raw_buffer_load_x4(a_rsrc, a_elem_offset * 2, 0, 0)
        for i in S.range(4):
            a_stage[1, wave, lane, i] = a_vec[i]
        b_vec = S.amdgpu.raw_buffer_load_x4(
            b_rsrc, (k_base + 48 + b_source_chunk * 8) * 2, 0, 0
        )
        for i in S.range(4):
            b_stage[1, wave, lane, i] = b_vec[i]

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[1], b_frag1[1], acc)

        S.syncthreads()

    a_src_tail0_0 = a_stage[0, wave, 2 * lane_lo + 0]
    a_src_tail0_1 = a_stage[0, wave, 2 * lane_lo + 1]
    if lane_hi == 0:
        a_pack_stage[0, wave, lane, 0] = a_src_tail0_0[0]
        a_pack_stage[0, wave, lane, 1] = a_src_tail0_0[1]
        a_pack_stage[0, wave, lane, 2] = a_src_tail0_1[0]
        a_pack_stage[0, wave, lane, 3] = a_src_tail0_1[1]
    else:
        a_pack_stage[0, wave, lane, 0] = a_src_tail0_0[2]
        a_pack_stage[0, wave, lane, 1] = a_src_tail0_0[3]
        a_pack_stage[0, wave, lane, 2] = a_src_tail0_1[2]
        a_pack_stage[0, wave, lane, 3] = a_src_tail0_1[3]
    for i in S.range(4):
        b_pack_stage[0, wave, lane, i] = 0
    if warp_col == 0 and lane < 8:
        b_src_tail0_lo = b_stage[0, wave, 0]
        b_src_tail0_hi = b_stage[0, wave, 1]
        b_word_tail0_lo = b_src_tail0_lo[lane // 2]
        b_word_tail0_hi = b_src_tail0_hi[lane // 2]
        if lane % 2 == 0:
            b_pack_stage[0, wave, lane, 0] = b_word_tail0_lo & 0xFFFF
            b_pack_stage[0, wave, lane, 2] = b_word_tail0_hi & 0xFFFF
        else:
            b_pack_stage[0, wave, lane, 0] = (b_word_tail0_lo >> 16) & 0xFFFF
            b_pack_stage[0, wave, lane, 2] = (b_word_tail0_hi >> 16) & 0xFFFF
    a_frag_tail0 = S.view(a_pack_stage[0, wave, lane], S.Tensor((2, 4, 1), S.bf16))
    b_frag_tail0 = S.view(b_pack_stage[0, wave, lane], S.Tensor((2, 4, 1), S.bf16))
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag_tail0[0], b_frag_tail0[0], acc)
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag_tail0[1], b_frag_tail0[1], acc)

    a_src_tail1_0 = a_stage[1, wave, 2 * lane_lo + 0]
    a_src_tail1_1 = a_stage[1, wave, 2 * lane_lo + 1]
    if lane_hi == 0:
        a_pack_stage[1, wave, lane, 0] = a_src_tail1_0[0]
        a_pack_stage[1, wave, lane, 1] = a_src_tail1_0[1]
        a_pack_stage[1, wave, lane, 2] = a_src_tail1_1[0]
        a_pack_stage[1, wave, lane, 3] = a_src_tail1_1[1]
    else:
        a_pack_stage[1, wave, lane, 0] = a_src_tail1_0[2]
        a_pack_stage[1, wave, lane, 1] = a_src_tail1_0[3]
        a_pack_stage[1, wave, lane, 2] = a_src_tail1_1[2]
        a_pack_stage[1, wave, lane, 3] = a_src_tail1_1[3]
    for i in S.range(4):
        b_pack_stage[1, wave, lane, i] = 0
    if warp_col == 0 and lane < 8:
        b_src_tail1_lo = b_stage[1, wave, 0]
        b_src_tail1_hi = b_stage[1, wave, 1]
        b_word_tail1_lo = b_src_tail1_lo[lane // 2]
        b_word_tail1_hi = b_src_tail1_hi[lane // 2]
        if lane % 2 == 0:
            b_pack_stage[1, wave, lane, 0] = b_word_tail1_lo & 0xFFFF
            b_pack_stage[1, wave, lane, 2] = b_word_tail1_hi & 0xFFFF
        else:
            b_pack_stage[1, wave, lane, 0] = (b_word_tail1_lo >> 16) & 0xFFFF
            b_pack_stage[1, wave, lane, 2] = (b_word_tail1_hi >> 16) & 0xFFFF
    a_frag_tail1 = S.view(a_pack_stage[1, wave, lane], S.Tensor((2, 4, 1), S.bf16))
    b_frag_tail1 = S.view(b_pack_stage[1, wave, lane], S.Tensor((2, 4, 1), S.bf16))
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag_tail1[0], b_frag_tail1[0], acc)
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag_tail1[1], b_frag_tail1[1], acc)

    if warp_col == 0 and lane_lo == 0:
        for acc_idx in S.range(0, 16, 2):
            row = wave_row_base + 8 * (acc_idx // 4) + 4 * lane_hi + (acc_idx % 4)
            lo = S.bitcast(S.convert(acc[acc_idx], S.bf16), S.u16)
            hi = S.bitcast(S.convert(acc[acc_idx + 1], S.bf16), S.u16)
            packed = S.convert(lo, S.u32) | (S.convert(hi, S.u32) << 16)
            S.amdgpu.raw_buffer_store_x1(packed, c_rsrc, row * 2, 0, 0)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        A = A.contiguous()
        B = B.contiguous()
        return torch.matmul(A, B)
