import torch
import torch.nn as nn

import substrate
import substrate.language as S


M = 32768
K = 32
N = 32768
BLOCK_M = 64
BLOCK_N = 64
WAVE_SIZE = 64
WAVES_PER_BLOCK = 4
THREADS_PER_BLOCK = WAVE_SIZE * WAVES_PER_BLOCK


@substrate.jit
def gemm_kernel(
    A: S.Tensor((M, K), S.bf16),
    B: S.Tensor((K, N), S.bf16),
    C: S.Tensor((M, N), S.bf16),
    a_range_bytes: S.i32,
    b_range_bytes: S.i32,
):
    tid = S.thread_id(0)
    lane = tid & 63
    wave = tid >> 6

    block_row = S.block_id(1) * BLOCK_M
    block_col = S.block_id(0) * BLOCK_N
    wave_row = wave >> 1
    wave_col = wave & 1
    tile_row_base = block_row + wave_row * 32
    tile_col_base = block_col + wave_col * 32

    a_rsrc = S.amdgpu.make_rsrc(A, a_range_bytes)
    b_rsrc = S.amdgpu.make_rsrc(B, b_range_bytes)

    a_stage = S.make_shared((2 * WAVES_PER_BLOCK, WAVE_SIZE, 8), S.bf16)
    b_stage = S.make_shared((2 * WAVES_PER_BLOCK, WAVE_SIZE, 8), S.bf16)

    acc = S.full((16,), 0.0, S.f32)
    zero = S.convert(0, S.i32)

    a_row = lane >> 1
    a_part = lane & 1
    b_row = lane >> 2
    b_part = lane & 3
    b_lane_base = (b_part << 3) + ((b_row & 4) << 3)
    b_slot_base = 0
    if b_row >= 8:
        b_slot_base = 4

    stage0_wave = wave
    stage1_wave = wave + WAVES_PER_BLOCK

    a_offset = ((tile_row_base + a_row) * K + a_part * 8) * 2
    a_words = S.amdgpu.raw_buffer_load_x4(a_rsrc, zero, a_offset, 0)
    a_vals = S.view(a_words, S.Tensor((2, 4, 1), S.bf16))
    if a_part == 0:
        for kk in S.range(4):
            a_stage[stage0_wave, a_row, kk] = a_vals[0, kk, 0]
            a_stage[stage0_wave, a_row + 32, kk] = a_vals[1, kk, 0]
    else:
        for kk in S.range(4):
            a_stage[stage0_wave, a_row, kk + 4] = a_vals[0, kk, 0]
            a_stage[stage0_wave, a_row + 32, kk + 4] = a_vals[1, kk, 0]

    b_offset = (b_row * N + tile_col_base + b_part * 8) * 2
    b_words = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero, b_offset, 0)
    b_vals = S.view(b_words, S.Tensor((2, 4, 1), S.bf16))
    for kk in S.range(4):
        b_stage[stage0_wave, b_lane_base + kk, b_slot_base + (b_row & 3)] = b_vals[0, kk, 0]
        b_stage[stage0_wave, b_lane_base + 4 + kk, b_slot_base + (b_row & 3)] = b_vals[1, kk, 0]

    S.syncthreads()

    next_k_base = 16
    next_a_offset = ((tile_row_base + a_row) * K + next_k_base + a_part * 8) * 2
    next_a_words = S.amdgpu.raw_buffer_load_x4(a_rsrc, zero, next_a_offset, 0)
    next_a_vals = S.view(next_a_words, S.Tensor((2, 4, 1), S.bf16))

    next_b_offset = ((next_k_base + b_row) * N + tile_col_base + b_part * 8) * 2
    next_b_words = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero, next_b_offset, 0)
    next_b_vals = S.view(next_b_words, S.Tensor((2, 4, 1), S.bf16))

    curr_a_frag = S.view(a_stage[stage0_wave, lane], S.Tensor((2, 4, 1), S.bf16))
    curr_b_frag = S.view(b_stage[stage0_wave, lane], S.Tensor((2, 4, 1), S.bf16))

    acc = S.amdgpu.mfma_32x32x8_bf16_f32(curr_a_frag[0], curr_b_frag[0], acc)

    if a_part == 0:
        for kk in S.range(4):
            a_stage[stage1_wave, a_row, kk] = next_a_vals[0, kk, 0]
            a_stage[stage1_wave, a_row + 32, kk] = next_a_vals[1, kk, 0]
    else:
        for kk in S.range(4):
            a_stage[stage1_wave, a_row, kk + 4] = next_a_vals[0, kk, 0]
            a_stage[stage1_wave, a_row + 32, kk + 4] = next_a_vals[1, kk, 0]

    for kk in S.range(4):
        b_stage[stage1_wave, b_lane_base + kk, b_slot_base + (b_row & 3)] = next_b_vals[0, kk, 0]
        b_stage[stage1_wave, b_lane_base + 4 + kk, b_slot_base + (b_row & 3)] = next_b_vals[1, kk, 0]

    acc = S.amdgpu.mfma_32x32x8_bf16_f32(curr_a_frag[1], curr_b_frag[1], acc)

    S.syncthreads()

    next_a_frag = S.view(a_stage[stage1_wave, lane], S.Tensor((2, 4, 1), S.bf16))
    next_b_frag = S.view(b_stage[stage1_wave, lane], S.Tensor((2, 4, 1), S.bf16))

    acc = S.amdgpu.mfma_32x32x8_bf16_f32(next_a_frag[0], next_b_frag[0], acc)
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(next_a_frag[1], next_b_frag[1], acc)

    lane_col = lane % 32
    lane_row_group = lane >> 5
    for acc_idx in S.range(16):
        out_row = tile_row_base + 8 * (acc_idx // 4) + 4 * lane_row_group + (acc_idx % 4)
        out_col = tile_col_base + lane_col
        C[out_row, out_col] = S.convert(acc[acc_idx], S.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._a_range_bytes = M * K * 2
        self._b_range_bytes = K * N * 2

    def forward(self, A, B):
        if tuple(A.shape) != (M, K) or tuple(B.shape) != (K, N):
            raise ValueError(f"Expected A=({M}, {K}) and B=({K}, {N}), got {tuple(A.shape)} and {tuple(B.shape)}")
        if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
            raise TypeError(f"Expected bf16 inputs, got {A.dtype} and {B.dtype}")

        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((M, N), device=A.device, dtype=torch.bfloat16)
        gemm_kernel[lambda: ((N // BLOCK_N, M // BLOCK_M, 1), (THREADS_PER_BLOCK, 1, 1))](
            A,
            B,
            C,
            self._a_range_bytes,
            self._b_range_bytes,
        )
        return C
