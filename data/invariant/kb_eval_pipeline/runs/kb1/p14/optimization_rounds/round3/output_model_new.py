import torch
import torch.nn as nn

import substrate
import substrate.language as S


M = 4096
N = 4096
K = 4096

WAVE_SIZE = 64
NUM_WAVES = 4
THREADS = WAVE_SIZE * NUM_WAVES

BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 16
PIPE_K = 2 * BLOCK_K

A_ROW_BYTES = K * 2
B_ROW_BYTES = N * 2
NUM_PIPE_TILES = K // PIPE_K


@substrate.jit
def tri_gemm_kernel(
    A: S.Tensor((M, K), S.bf16),
    B: S.Tensor((K, N), S.bf16),
    C: S.Tensor((M, N), S.bf16),
):
    tid = S.thread_id(0)
    lane = tid % WAVE_SIZE
    warp = tid // WAVE_SIZE
    warp_row = warp // 2
    warp_col = warp % 2

    block_row = S.block_id(1)
    block_col = S.block_id(0)
    tile_row_base = block_row * BLOCK_M + warp_row * 32
    tile_col_base = block_col * BLOCK_N + warp_col * 32

    acc = S.full((16,), 0.0, S.f32)

    if block_col * BLOCK_N + (BLOCK_N - 1) < block_row * BLOCK_M:
        for acc_idx in S.range(16):
            out_col = tile_col_base + (lane % 32)
            out_row = tile_row_base + 8 * (acc_idx // 4) + 4 * (lane // 32) + (acc_idx % 4)
            C[out_row, out_col] = S.convert(0.0, S.bf16)
        return

    a_stage = S.make_shared((2, NUM_WAVES, WAVE_SIZE, 8), S.bf16)
    b_stage = S.make_shared((2, NUM_WAVES, WAVE_SIZE, 8), S.bf16)

    a_row = tile_row_base + (lane % 32)
    a_k_quad = (lane // 32) * 4
    b_col = tile_col_base + (lane % 32)
    b_k_quad = (lane // 32) * 4
    b_col_pair = (b_col // 2) * 4
    b_col_is_odd = b_col % 2

    a_rsrc = S.amdgpu.make_rsrc(A[a_row], A_ROW_BYTES)

    a_words0 = S.amdgpu.raw_buffer_load_x4(a_rsrc, a_k_quad * 2, 0, 0)
    a_vals0 = S.view(a_words0, S.Tensor((8,), S.bf16))
    a_words1 = S.amdgpu.raw_buffer_load_x4(a_rsrc, (8 + a_k_quad) * 2, 0, 0)
    a_vals1 = S.view(a_words1, S.Tensor((8,), S.bf16))
    for e in S.range(4):
        a_stage[0, warp, lane, e] = a_vals0[e]
        a_stage[0, warp, lane, 4 + e] = a_vals1[e]

    for e in S.range(4):
        b_row0 = b_k_quad + e
        b_pack0 = S.amdgpu.raw_buffer_load_x1(S.amdgpu.make_rsrc(B[b_row0], B_ROW_BYTES), b_col_pair, 0, 0)
        b_pack0_u32 = S.bitcast(b_pack0, S.u32)
        if b_col_is_odd == 0:
            b_bits0 = S.convert(b_pack0_u32 & 0xFFFF, S.u16)
        else:
            b_bits0 = S.convert((b_pack0_u32 >> 16) & 0xFFFF, S.u16)
        b_stage[0, warp, lane, e] = S.bitcast(b_bits0, S.bf16)

        b_row1 = 8 + b_k_quad + e
        b_pack1 = S.amdgpu.raw_buffer_load_x1(S.amdgpu.make_rsrc(B[b_row1], B_ROW_BYTES), b_col_pair, 0, 0)
        b_pack1_u32 = S.bitcast(b_pack1, S.u32)
        if b_col_is_odd == 0:
            b_bits1 = S.convert(b_pack1_u32 & 0xFFFF, S.u16)
        else:
            b_bits1 = S.convert((b_pack1_u32 >> 16) & 0xFFFF, S.u16)
        b_stage[0, warp, lane, 4 + e] = S.bitcast(b_bits1, S.bf16)

    S.syncthreads()

    for tile_idx in S.range(NUM_PIPE_TILES - 1):
        k_base = tile_idx * PIPE_K
        next_k0 = k_base + BLOCK_K

        a_words0 = S.amdgpu.raw_buffer_load_x4(a_rsrc, (next_k0 + a_k_quad) * 2, 0, 0)
        a_vals0 = S.view(a_words0, S.Tensor((8,), S.bf16))
        a_words1 = S.amdgpu.raw_buffer_load_x4(a_rsrc, (next_k0 + 8 + a_k_quad) * 2, 0, 0)
        a_vals1 = S.view(a_words1, S.Tensor((8,), S.bf16))
        for e in S.range(4):
            a_stage[1, warp, lane, e] = a_vals0[e]
            a_stage[1, warp, lane, 4 + e] = a_vals1[e]

            b_row0 = next_k0 + b_k_quad + e
            b_pack0 = S.amdgpu.raw_buffer_load_x1(S.amdgpu.make_rsrc(B[b_row0], B_ROW_BYTES), b_col_pair, 0, 0)
            b_pack0_u32 = S.bitcast(b_pack0, S.u32)
            if b_col_is_odd == 0:
                b_bits0 = S.convert(b_pack0_u32 & 0xFFFF, S.u16)
            else:
                b_bits0 = S.convert((b_pack0_u32 >> 16) & 0xFFFF, S.u16)
            b_stage[1, warp, lane, e] = S.bitcast(b_bits0, S.bf16)

            b_row1 = next_k0 + 8 + b_k_quad + e
            b_pack1 = S.amdgpu.raw_buffer_load_x1(S.amdgpu.make_rsrc(B[b_row1], B_ROW_BYTES), b_col_pair, 0, 0)
            b_pack1_u32 = S.bitcast(b_pack1, S.u32)
            if b_col_is_odd == 0:
                b_bits1 = S.convert(b_pack1_u32 & 0xFFFF, S.u16)
            else:
                b_bits1 = S.convert((b_pack1_u32 >> 16) & 0xFFFF, S.u16)
            b_stage[1, warp, lane, 4 + e] = S.bitcast(b_bits1, S.bf16)

        a_frag0 = S.view(a_stage[0, warp, lane], S.Tensor((2, 4, 1), S.bf16))
        b_frag0 = S.view(b_stage[0, warp, lane], S.Tensor((2, 4, 1), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[0], b_frag0[0], acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[1], b_frag0[1], acc)

        S.syncthreads()

        next_k1 = k_base + PIPE_K
        a_words0 = S.amdgpu.raw_buffer_load_x4(a_rsrc, (next_k1 + a_k_quad) * 2, 0, 0)
        a_vals0 = S.view(a_words0, S.Tensor((8,), S.bf16))
        a_words1 = S.amdgpu.raw_buffer_load_x4(a_rsrc, (next_k1 + 8 + a_k_quad) * 2, 0, 0)
        a_vals1 = S.view(a_words1, S.Tensor((8,), S.bf16))
        for e in S.range(4):
            a_stage[0, warp, lane, e] = a_vals0[e]
            a_stage[0, warp, lane, 4 + e] = a_vals1[e]

            b_row0 = next_k1 + b_k_quad + e
            b_pack0 = S.amdgpu.raw_buffer_load_x1(S.amdgpu.make_rsrc(B[b_row0], B_ROW_BYTES), b_col_pair, 0, 0)
            b_pack0_u32 = S.bitcast(b_pack0, S.u32)
            if b_col_is_odd == 0:
                b_bits0 = S.convert(b_pack0_u32 & 0xFFFF, S.u16)
            else:
                b_bits0 = S.convert((b_pack0_u32 >> 16) & 0xFFFF, S.u16)
            b_stage[0, warp, lane, e] = S.bitcast(b_bits0, S.bf16)

            b_row1 = next_k1 + 8 + b_k_quad + e
            b_pack1 = S.amdgpu.raw_buffer_load_x1(S.amdgpu.make_rsrc(B[b_row1], B_ROW_BYTES), b_col_pair, 0, 0)
            b_pack1_u32 = S.bitcast(b_pack1, S.u32)
            if b_col_is_odd == 0:
                b_bits1 = S.convert(b_pack1_u32 & 0xFFFF, S.u16)
            else:
                b_bits1 = S.convert((b_pack1_u32 >> 16) & 0xFFFF, S.u16)
            b_stage[0, warp, lane, 4 + e] = S.bitcast(b_bits1, S.bf16)

        a_frag1 = S.view(a_stage[1, warp, lane], S.Tensor((2, 4, 1), S.bf16))
        b_frag1 = S.view(b_stage[1, warp, lane], S.Tensor((2, 4, 1), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[0], b_frag1[0], acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[1], b_frag1[1], acc)

        S.syncthreads()

    a_frag_last = S.view(a_stage[0, warp, lane], S.Tensor((2, 4, 1), S.bf16))
    b_frag_last = S.view(b_stage[0, warp, lane], S.Tensor((2, 4, 1), S.bf16))
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag_last[0], b_frag_last[0], acc)
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag_last[1], b_frag_last[1], acc)

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
            raise ValueError("ModelNew expects two contiguous 4096x4096 tensors")
        if A.device.type != "cuda" or B.device.type != "cuda":
            raise ValueError("ModelNew expects CUDA tensors")
        if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
            raise ValueError("ModelNew expects bfloat16 tensors")

        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)
        tri_gemm_kernel[
            lambda: (((N + BLOCK_N - 1) // BLOCK_N, (M + BLOCK_M - 1) // BLOCK_M, 1), (THREADS, 1, 1))
        ](A, B, C)
        return C
