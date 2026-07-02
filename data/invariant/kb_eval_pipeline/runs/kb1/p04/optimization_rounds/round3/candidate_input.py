import torch
import torch.nn as nn

import substrate
import substrate.language as S


M = 2048
K = 1048576
N = 1
BLOCK_M = 64
WAVES_PER_BLOCK = 4
LANES_PER_WAVE = 64
BLOCK_THREADS = WAVES_PER_BLOCK * LANES_PER_WAVE
K_STEP = 16


@substrate.jit
def gemm_kernel(
    A: S.Tensor((M, K // 2), S.u32),
    B: S.Tensor((K // 2,), S.u32),
    C: S.Tensor((M, N), S.bf16),
):
    tid = S.thread_id(0)
    lane = tid % LANES_PER_WAVE
    warp = tid // LANES_PER_WAVE
    warp_row = warp // 2
    warp_col = warp % 2
    block_row = S.block_id(1) * BLOCK_M
    tile_row_base = block_row + warp_row * 32
    tile_col_base = warp_col * 32

    shared_a = S.make_shared((2, WAVES_PER_BLOCK, LANES_PER_WAVE, 4), S.u32)
    shared_b = S.make_shared((2, WAVES_PER_BLOCK, LANES_PER_WAVE, 4), S.u32)

    acc = S.full((16,), 0.0, S.f32)
    zero = S.convert(0, S.i32)
    b_rsrc = S.amdgpu.make_rsrc(B, (K // 2) * 4)
    row_in_wave = lane % 32
    half = lane // 32
    a_row = tile_row_base + row_in_wave

    panel_word_base0 = 0
    lane_word_base0 = panel_word_base0 + half * 2
    b_offset0 = S.convert(lane_word_base0 * 4, S.i32)
    b_words0 = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero, b_offset0, 0)
    shared_a[0, warp, lane, 0] = A[a_row, lane_word_base0 + 0]
    shared_a[0, warp, lane, 1] = A[a_row, lane_word_base0 + 1]
    shared_a[0, warp, lane, 2] = A[a_row, lane_word_base0 + 2]
    shared_a[0, warp, lane, 3] = A[a_row, lane_word_base0 + 3]
    for i in S.range(4):
        shared_b[0, warp, lane, i] = b_words0[i]
    S.syncthreads()

    for pair_idx in S.range(0, K // (2 * K_STEP) - 1):
        a_frag0 = S.view(shared_a[0, warp, lane], S.Tensor((2, 4, 1), S.bf16))
        b_frag0 = S.view(shared_b[0, warp, lane], S.Tensor((2, 4, 1), S.bf16))

        panel1_word_base = (2 * pair_idx + 1) * (K_STEP // 2)
        lane_word_base1 = panel1_word_base + half * 2
        b_offset1 = S.convert(lane_word_base1 * 4, S.i32)
        b_words1 = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero, b_offset1, 0)

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[0], b_frag0[0], acc)
        shared_a[1, warp, lane, 0] = A[a_row, lane_word_base1 + 0]
        shared_a[1, warp, lane, 1] = A[a_row, lane_word_base1 + 1]
        shared_a[1, warp, lane, 2] = A[a_row, lane_word_base1 + 2]
        shared_a[1, warp, lane, 3] = A[a_row, lane_word_base1 + 3]
        for i in S.range(4):
            shared_b[1, warp, lane, i] = b_words1[i]
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[1], b_frag0[1], acc)
        S.syncthreads()

        a_frag1 = S.view(shared_a[1, warp, lane], S.Tensor((2, 4, 1), S.bf16))
        b_frag1 = S.view(shared_b[1, warp, lane], S.Tensor((2, 4, 1), S.bf16))

        panel2_word_base = (2 * pair_idx + 2) * (K_STEP // 2)
        lane_word_base2 = panel2_word_base + half * 2
        b_offset2 = S.convert(lane_word_base2 * 4, S.i32)
        b_words2 = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero, b_offset2, 0)

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[0], b_frag1[0], acc)
        shared_a[0, warp, lane, 0] = A[a_row, lane_word_base2 + 0]
        shared_a[0, warp, lane, 1] = A[a_row, lane_word_base2 + 1]
        shared_a[0, warp, lane, 2] = A[a_row, lane_word_base2 + 2]
        shared_a[0, warp, lane, 3] = A[a_row, lane_word_base2 + 3]
        for i in S.range(4):
            shared_b[0, warp, lane, i] = b_words2[i]
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[1], b_frag1[1], acc)
        S.syncthreads()

    a_frag0 = S.view(shared_a[0, warp, lane], S.Tensor((2, 4, 1), S.bf16))
    b_frag0 = S.view(shared_b[0, warp, lane], S.Tensor((2, 4, 1), S.bf16))

    final_panel1_word_base = (K // K_STEP - 1) * (K_STEP // 2)
    final_lane_word_base1 = final_panel1_word_base + half * 2
    final_b_offset1 = S.convert(final_lane_word_base1 * 4, S.i32)
    final_b_words1 = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero, final_b_offset1, 0)

    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[0], b_frag0[0], acc)
    shared_a[1, warp, lane, 0] = A[a_row, final_lane_word_base1 + 0]
    shared_a[1, warp, lane, 1] = A[a_row, final_lane_word_base1 + 1]
    shared_a[1, warp, lane, 2] = A[a_row, final_lane_word_base1 + 2]
    shared_a[1, warp, lane, 3] = A[a_row, final_lane_word_base1 + 3]
    for i in S.range(4):
        shared_b[1, warp, lane, i] = final_b_words1[i]
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[1], b_frag0[1], acc)
    S.syncthreads()

    a_frag1 = S.view(shared_a[1, warp, lane], S.Tensor((2, 4, 1), S.bf16))
    b_frag1 = S.view(shared_b[1, warp, lane], S.Tensor((2, 4, 1), S.bf16))
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[0], b_frag1[0], acc)
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[1], b_frag1[1], acc)

    if tile_col_base == 0 and (lane % 32) == 0:
        lane_half = lane // 32
        for acc_idx in S.range(16):
            row = tile_row_base + 8 * (acc_idx // 4) + 4 * lane_half + (acc_idx % 4)
            C[row, 0] = S.convert(acc[acc_idx], S.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if tuple(A.shape) != (M, K) or tuple(B.shape) != (K, N):
            raise ValueError(f"Expected A={(M, K)} and B={(K, N)}, got {tuple(A.shape)} and {tuple(B.shape)}")
        if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
            raise TypeError("ModelNew expects bfloat16 inputs")

        A = A.contiguous()
        B = B.contiguous()
        A_u32 = A.view(torch.int32).reshape(M, K // 2)
        B_u32 = B.reshape(K).view(torch.int32).reshape(K // 2)
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)
        gemm_kernel[lambda: ((1, M // BLOCK_M, 1), (BLOCK_THREADS, 1, 1))](A_u32, B_u32, C)
        return C
