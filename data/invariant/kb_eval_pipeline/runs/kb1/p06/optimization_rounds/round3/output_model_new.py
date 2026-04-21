import torch
import torch.nn as nn

import substrate
import substrate.language as S


M = 256
K = 524288
N = 256

BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 16
PIPE_STAGES = 2
K_UNROLL = 2
PAIR_K = BLOCK_K * K_UNROLL
NUM_K_PAIRS = K // PAIR_K
WAVE_SIZE = 64
WAVES_PER_BLOCK = 4
THREADS_PER_BLOCK = WAVE_SIZE * WAVES_PER_BLOCK

A_BYTES = M * K * 2
B_BYTES = K * N * 2
C_BYTES = M * N * 2


@substrate.jit
def gemm_kernel(
    A: S.Tensor((256, 524288), S.bf16),
    B: S.Tensor((524288, 256), S.bf16),
    C: S.Tensor((256, 256), S.bf16),
):
    tid = S.thread_id(0)
    wave = tid // WAVE_SIZE
    lane = tid % WAVE_SIZE

    warp_row = wave // 2
    warp_col = wave % 2

    block_row_base = S.block_id(1) * BLOCK_M
    block_col_base = S.block_id(0) * BLOCK_N
    tile_row_base = block_row_base + warp_row * 32
    tile_col_base = block_col_base + warp_col * 32

    a_rsrc = S.amdgpu.make_rsrc(A, A_BYTES)
    b_rsrc = S.amdgpu.make_rsrc(B, B_BYTES)
    c_rsrc = S.amdgpu.make_rsrc(C, C_BYTES)

    a_shared = S.make_shared((PIPE_STAGES, THREADS_PER_BLOCK, 4), S.u32)
    b_shared = S.make_shared((PIPE_STAGES, THREADS_PER_BLOCK, 4), S.u32)

    acc = S.full((16,), 0.0, S.f32)

    a_row = tile_row_base + (lane % 32)
    a_half = lane // 32
    b_group = lane % 8
    b_half = b_group % 2
    b_col8 = tile_col_base + (b_group // 2) * 8

    next_a0 = S.make_local((4,), S.u32)
    next_b0 = S.make_local((4,), S.u32)
    next_a1 = S.make_local((4,), S.u32)
    next_b1 = S.make_local((4,), S.u32)

    a_off0 = S.convert((a_row * K + 0) * 2, S.i32)
    a_off1 = S.convert((a_row * K + 8) * 2, S.i32)
    a_vec0 = S.amdgpu.raw_buffer_load_x4(a_rsrc, 0, a_off0, 0)
    a_vec1 = S.amdgpu.raw_buffer_load_x4(a_rsrc, 0, a_off1, 0)
    a_shared[0, tid, 0] = a_vec0[a_half * 2 + 0]
    a_shared[0, tid, 1] = a_vec0[a_half * 2 + 1]
    a_shared[0, tid, 2] = a_vec1[a_half * 2 + 0]
    a_shared[0, tid, 3] = a_vec1[a_half * 2 + 1]

    b_k0 = lane // 8
    b_off0 = S.convert((b_k0 * N + b_col8) * 2, S.i32)
    b_off1 = S.convert(((b_k0 + 8) * N + b_col8) * 2, S.i32)
    b_vec0 = S.amdgpu.raw_buffer_load_x4(b_rsrc, 0, b_off0, 0)
    b_vec1 = S.amdgpu.raw_buffer_load_x4(b_rsrc, 0, b_off1, 0)
    b_shared[0, tid, 0] = b_vec0[b_half * 2 + 0]
    b_shared[0, tid, 1] = b_vec0[b_half * 2 + 1]
    b_shared[0, tid, 2] = b_vec1[b_half * 2 + 0]
    b_shared[0, tid, 3] = b_vec1[b_half * 2 + 1]

    a_off0 = S.convert((a_row * K + BLOCK_K) * 2, S.i32)
    a_off1 = S.convert((a_row * K + BLOCK_K + 8) * 2, S.i32)
    a_vec0 = S.amdgpu.raw_buffer_load_x4(a_rsrc, 0, a_off0, 0)
    a_vec1 = S.amdgpu.raw_buffer_load_x4(a_rsrc, 0, a_off1, 0)
    a_shared[1, tid, 0] = a_vec0[a_half * 2 + 0]
    a_shared[1, tid, 1] = a_vec0[a_half * 2 + 1]
    a_shared[1, tid, 2] = a_vec1[a_half * 2 + 0]
    a_shared[1, tid, 3] = a_vec1[a_half * 2 + 1]

    b_k1 = BLOCK_K + (lane // 8)
    b_off0 = S.convert((b_k1 * N + b_col8) * 2, S.i32)
    b_off1 = S.convert(((b_k1 + 8) * N + b_col8) * 2, S.i32)
    b_vec0 = S.amdgpu.raw_buffer_load_x4(b_rsrc, 0, b_off0, 0)
    b_vec1 = S.amdgpu.raw_buffer_load_x4(b_rsrc, 0, b_off1, 0)
    b_shared[1, tid, 0] = b_vec0[b_half * 2 + 0]
    b_shared[1, tid, 1] = b_vec0[b_half * 2 + 1]
    b_shared[1, tid, 2] = b_vec1[b_half * 2 + 0]
    b_shared[1, tid, 3] = b_vec1[b_half * 2 + 1]

    S.syncthreads()

    for pair_idx in S.range(NUM_K_PAIRS - 1):
        k_pair = pair_idx * PAIR_K

        a_frag = S.view(a_shared[0, tid], S.Tensor((2, 4, 1), S.bf16))
        b_frag = S.view(b_shared[0, tid], S.Tensor((2, 4, 1), S.bf16))

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], acc)

        a_off0 = S.convert((a_row * K + k_pair + PAIR_K) * 2, S.i32)
        a_off1 = S.convert((a_row * K + k_pair + PAIR_K + 8) * 2, S.i32)
        a_vec0 = S.amdgpu.raw_buffer_load_x4(a_rsrc, 0, a_off0, 0)
        a_vec1 = S.amdgpu.raw_buffer_load_x4(a_rsrc, 0, a_off1, 0)
        next_a0[0] = a_vec0[a_half * 2 + 0]
        next_a0[1] = a_vec0[a_half * 2 + 1]
        next_a0[2] = a_vec1[a_half * 2 + 0]
        next_a0[3] = a_vec1[a_half * 2 + 1]

        b_k0 = k_pair + PAIR_K + (lane // 8)
        b_off0 = S.convert((b_k0 * N + b_col8) * 2, S.i32)
        b_off1 = S.convert(((b_k0 + 8) * N + b_col8) * 2, S.i32)
        b_vec0 = S.amdgpu.raw_buffer_load_x4(b_rsrc, 0, b_off0, 0)
        b_vec1 = S.amdgpu.raw_buffer_load_x4(b_rsrc, 0, b_off1, 0)
        next_b0[0] = b_vec0[b_half * 2 + 0]
        next_b0[1] = b_vec0[b_half * 2 + 1]
        next_b0[2] = b_vec1[b_half * 2 + 0]
        next_b0[3] = b_vec1[b_half * 2 + 1]

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[1], b_frag[1], acc)

        a_shared[0, tid] = next_a0
        b_shared[0, tid] = next_b0

        a_frag = S.view(a_shared[1, tid], S.Tensor((2, 4, 1), S.bf16))
        b_frag = S.view(b_shared[1, tid], S.Tensor((2, 4, 1), S.bf16))

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], acc)

        a_off0 = S.convert((a_row * K + k_pair + PAIR_K + BLOCK_K) * 2, S.i32)
        a_off1 = S.convert((a_row * K + k_pair + PAIR_K + BLOCK_K + 8) * 2, S.i32)
        a_vec0 = S.amdgpu.raw_buffer_load_x4(a_rsrc, 0, a_off0, 0)
        a_vec1 = S.amdgpu.raw_buffer_load_x4(a_rsrc, 0, a_off1, 0)
        next_a1[0] = a_vec0[a_half * 2 + 0]
        next_a1[1] = a_vec0[a_half * 2 + 1]
        next_a1[2] = a_vec1[a_half * 2 + 0]
        next_a1[3] = a_vec1[a_half * 2 + 1]

        b_k1 = k_pair + PAIR_K + BLOCK_K + (lane // 8)
        b_off0 = S.convert((b_k1 * N + b_col8) * 2, S.i32)
        b_off1 = S.convert(((b_k1 + 8) * N + b_col8) * 2, S.i32)
        b_vec0 = S.amdgpu.raw_buffer_load_x4(b_rsrc, 0, b_off0, 0)
        b_vec1 = S.amdgpu.raw_buffer_load_x4(b_rsrc, 0, b_off1, 0)
        next_b1[0] = b_vec0[b_half * 2 + 0]
        next_b1[1] = b_vec0[b_half * 2 + 1]
        next_b1[2] = b_vec1[b_half * 2 + 0]
        next_b1[3] = b_vec1[b_half * 2 + 1]

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[1], b_frag[1], acc)

        a_shared[1, tid] = next_a1
        b_shared[1, tid] = next_b1

        S.syncthreads()

    a_frag = S.view(a_shared[0, tid], S.Tensor((2, 4, 1), S.bf16))
    b_frag = S.view(b_shared[0, tid], S.Tensor((2, 4, 1), S.bf16))
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], acc)
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[1], b_frag[1], acc)

    a_frag = S.view(a_shared[1, tid], S.Tensor((2, 4, 1), S.bf16))
    b_frag = S.view(b_shared[1, tid], S.Tensor((2, 4, 1), S.bf16))
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], acc)
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[1], b_frag[1], acc)

    if (lane % 2) == 0:
        for acc_idx in S.range(16):
            row = tile_row_base + 8 * (acc_idx // 4) + 4 * (lane // 32) + (acc_idx % 4)
            col = tile_col_base + (lane % 32)
            lo = S.convert(acc[acc_idx], S.bf16)
            hi = S.shuffle_down(lo, 1, 32)
            lo_bits = S.convert(S.bitcast(lo, S.u16), S.u32)
            hi_bits = S.convert(S.bitcast(hi, S.u16), S.u32)
            packed = lo_bits | (hi_bits << 16)
            c_off = S.convert((row * N + col) * 2, S.i32)
            S.amdgpu.raw_buffer_store_x1(packed, c_rsrc, 0, c_off, 0)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if tuple(A.shape) != (M, K) or tuple(B.shape) != (K, N):
            raise RuntimeError("ModelNew only supports A=(256, 524288), B=(524288, 256)")
        if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
            raise RuntimeError("ModelNew requires bfloat16 inputs")
        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((M, N), device=A.device, dtype=torch.bfloat16)
        gemm_kernel[lambda: ((N // BLOCK_N, M // BLOCK_M, 1), (THREADS_PER_BLOCK, 1, 1))](A, B, C)
        return C
