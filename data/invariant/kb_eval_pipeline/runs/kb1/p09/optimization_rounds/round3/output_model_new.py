import torch
import torch.nn as nn

import substrate
import substrate.language as S


M = 32768
K = 32
N = 32768

BLOCK_M = 64
BLOCK_N = 64
WAVE_TILE_M = 32
WAVE_TILE_N = 32
WAVES_PER_BLOCK = 4
THREADS_PER_WAVE = 64
THREADS_PER_BLOCK = WAVES_PER_BLOCK * THREADS_PER_WAVE
K_STAGE = 16
A_RANGE_BYTES = M * K * 2
B_RANGE_BYTES = K * N * 2
C_RANGE_BYTES = M * N * 2


@substrate.jit
def gemm_kernel(
    A: S.Tensor((M, K), S.bf16),
    B: S.Tensor((K, N), S.bf16),
    C: S.Tensor((M, N), S.bf16),
):
    tid = S.thread_id(0)
    lane = tid % THREADS_PER_WAVE
    wave = tid // THREADS_PER_WAVE

    block_row = S.block_id(1) * BLOCK_M
    block_col = S.block_id(0) * BLOCK_N

    wave_row = wave // 2
    wave_col = wave % 2

    row_in_wave = lane % WAVE_TILE_M
    lane_half = lane // WAVE_TILE_M
    b_load_row = lane % K_STAGE
    b_load_col_group = lane // K_STAGE

    wave_row_base = block_row + wave_row * WAVE_TILE_M
    wave_col_base = block_col + wave_col * WAVE_TILE_N

    a_rsrc = S.amdgpu.make_rsrc(A, A_RANGE_BYTES)
    b_rsrc = S.amdgpu.make_rsrc(B, B_RANGE_BYTES)
    c_rsrc = S.amdgpu.make_rsrc(C, C_RANGE_BYTES)

    # Layout: [buffer, wave, half, row/col, chunk, elem4]. Viewing one buffer as
    # [wave, lane, 4] u32 preserves the fixed MFMA swizzle invariants.
    a_stage = S.make_shared((2, WAVES_PER_BLOCK, 2, WAVE_TILE_M, 2, 4), S.bf16)
    b_stage = S.make_shared((2, WAVES_PER_BLOCK, 2, WAVE_TILE_N, 2, 4), S.bf16)

    acc = S.full((16,), 0.0, S.f32)

    a_words = S.view(a_stage, S.Tensor((2, WAVES_PER_BLOCK, THREADS_PER_WAVE, 4), S.u32))
    b_words = S.view(b_stage, S.Tensor((2, WAVES_PER_BLOCK, THREADS_PER_WAVE, 4), S.u32))

    b_chunk = b_load_row // 8
    b_half = (b_load_row % 8) // 4
    b_elem_k = b_load_row % 4

    a_byte_offset0 = ((wave_row_base + row_in_wave) * K + lane_half * 8) * 2
    b_byte_offset0 = (b_load_row * N + wave_col_base + b_load_col_group * 8) * 2
    a_vec0 = S.amdgpu.raw_buffer_load_x4(a_rsrc, a_byte_offset0, 0, 0)
    b_vec0 = S.amdgpu.raw_buffer_load_x4(b_rsrc, b_byte_offset0, 0, 0)

    a_vals0 = S.view(a_vec0, S.Tensor((2, 4, 1), S.bf16))
    b_vals0 = S.view(b_vec0, S.Tensor((2, 4, 1), S.bf16))

    for e in S.range(4):
        a_stage[0, wave, 0, row_in_wave, lane_half, e] = a_vals0[0, e, 0]
        a_stage[0, wave, 1, row_in_wave, lane_half, e] = a_vals0[1, e, 0]
        b_stage[0, wave, b_half, b_load_col_group * 8 + e, b_chunk, b_elem_k] = b_vals0[0, e, 0]
        b_stage[0, wave, b_half, b_load_col_group * 8 + 4 + e, b_chunk, b_elem_k] = b_vals0[1, e, 0]

    S.syncthreads()

    a_byte_offset1 = ((wave_row_base + row_in_wave) * K + K_STAGE + lane_half * 8) * 2
    b_byte_offset1 = ((K_STAGE + b_load_row) * N + wave_col_base + b_load_col_group * 8) * 2
    a_vec1 = S.amdgpu.raw_buffer_load_x4(a_rsrc, a_byte_offset1, 0, 0)
    b_vec1 = S.amdgpu.raw_buffer_load_x4(b_rsrc, b_byte_offset1, 0, 0)
    a_vals1 = S.view(a_vec1, S.Tensor((2, 4, 1), S.bf16))
    b_vals1 = S.view(b_vec1, S.Tensor((2, 4, 1), S.bf16))

    a_frag0 = S.view(a_words[0, wave, lane], S.Tensor((2, 4, 1), S.bf16))
    b_frag0 = S.view(b_words[0, wave, lane], S.Tensor((2, 4, 1), S.bf16))

    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[0], b_frag0[0], acc)
    for e in S.range(4):
        a_stage[1, wave, 0, row_in_wave, lane_half, e] = a_vals1[0, e, 0]
        a_stage[1, wave, 1, row_in_wave, lane_half, e] = a_vals1[1, e, 0]
        b_stage[1, wave, b_half, b_load_col_group * 8 + e, b_chunk, b_elem_k] = b_vals1[0, e, 0]
        b_stage[1, wave, b_half, b_load_col_group * 8 + 4 + e, b_chunk, b_elem_k] = b_vals1[1, e, 0]
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[1], b_frag0[1], acc)
    S.syncthreads()

    a_frag1 = S.view(a_words[1, wave, lane], S.Tensor((2, 4, 1), S.bf16))
    b_frag1 = S.view(b_words[1, wave, lane], S.Tensor((2, 4, 1), S.bf16))

    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[0], b_frag1[0], acc)
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[1], b_frag1[1], acc)

    for acc_idx in S.range(16):
        col = wave_col_base + (lane % 32)
        row = (
            wave_row_base
            + 8 * (acc_idx // 4)
            + 4 * (lane // 32)
            + (acc_idx % 4)
        )
        c_byte_offset = (row * N + col) * 2
        c_val = S.bitcast(S.convert(acc[acc_idx], S.bf16), S.i16)
        S.amdgpu.raw_buffer_store_x1(c_val, c_rsrc, c_byte_offset, 0, 0)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._grid = (N // BLOCK_N, M // BLOCK_M, 1)
        self._block = (THREADS_PER_BLOCK, 1, 1)

    def forward(self, A, B):
        if tuple(A.shape) != (M, K) or tuple(B.shape) != (K, N):
            raise ValueError(f"Expected A {(M, K)} and B {(K, N)}, got {tuple(A.shape)} and {tuple(B.shape)}")

        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)
        gemm_kernel[lambda: (self._grid, self._block)](A, B, C)
        return C
