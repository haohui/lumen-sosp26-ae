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
WAVE_SIZE = 64
WAVES_PER_BLOCK = 4
THREADS_PER_BLOCK = WAVE_SIZE * WAVES_PER_BLOCK
NUM_K_TILES = K // BLOCK_K


@substrate.jit
def gemm_kernel_mfma_pipelined(
    A: S.Tensor((M * K,), S.bf16),
    B: S.Tensor((N * K,), S.bf16),
    C: S.Tensor((M, N), S.bf16),
):
    tid = S.thread_id(0)
    lane = tid % WAVE_SIZE
    wave = tid // WAVE_SIZE
    warp_row = wave // 2
    warp_col = wave % 2

    block_row = S.block_id(1) * BLOCK_M
    block_col = S.block_id(0) * BLOCK_N

    a_shared0 = S.make_shared((128, 4), S.u32)
    a_shared1 = S.make_shared((128, 4), S.u32)
    b_shared0 = S.make_shared((128, 4), S.u32)
    b_shared1 = S.make_shared((128, 4), S.u32)

    a_rsrc = S.amdgpu.make_rsrc(A, S.convert(M * K * 2, S.i32))
    b_rsrc = S.amdgpu.make_rsrc(B, S.convert(K * N * 2, S.i32))
    zero = S.convert(0, S.i32)

    c_lane = S.full((16,), 0.0, S.f32)

    if tid < 128:
        row_in_block = tid % BLOCK_M
        k8 = tid // BLOCK_M
        row = block_row + row_in_block
        elem_offset = row * K + k8 * 8
        packed = S.amdgpu.raw_buffer_load_x4(
            a_rsrc,
            zero,
            S.convert(elem_offset * 2, S.i32),
            0,
        )

        row_group = row_in_block // 32
        row_lane = row_in_block % 32
        lane0 = row_group * 64 + row_lane
        lane1 = row_group * 64 + 32 + row_lane
        word_base = k8 * 2

        a_shared0[lane0, word_base + 0] = packed[0]
        a_shared0[lane0, word_base + 1] = packed[1]
        a_shared0[lane1, word_base + 0] = packed[2]
        a_shared0[lane1, word_base + 1] = packed[3]
    else:
        load_idx = tid - 128
        col_in_block = load_idx % BLOCK_N
        k8 = load_idx // BLOCK_N
        col = block_col + col_in_block
        elem_offset = col * K + k8 * 8
        packed = S.amdgpu.raw_buffer_load_x4(
            b_rsrc,
            zero,
            S.convert(elem_offset * 2, S.i32),
            0,
        )

        wave_col = col_in_block // 32
        col_lane = col_in_block % 32
        lane0 = wave_col * 64 + col_lane
        lane1 = wave_col * 64 + 32 + col_lane
        word_base = k8 * 2

        b_shared0[lane0, word_base + 0] = packed[0]
        b_shared0[lane0, word_base + 1] = packed[1]
        b_shared0[lane1, word_base + 0] = packed[2]
        b_shared0[lane1, word_base + 1] = packed[3]

    S.syncthreads()

    if NUM_K_TILES > 1:
        if tid < 128:
            row_in_block = tid % BLOCK_M
            k8 = tid // BLOCK_M
            row = block_row + row_in_block
            elem_offset = row * K + BLOCK_K + k8 * 8
            packed = S.amdgpu.raw_buffer_load_x4(
                a_rsrc,
                zero,
                S.convert(elem_offset * 2, S.i32),
                0,
            )

            row_group = row_in_block // 32
            row_lane = row_in_block % 32
            lane0 = row_group * 64 + row_lane
            lane1 = row_group * 64 + 32 + row_lane
            word_base = k8 * 2

            a_shared1[lane0, word_base + 0] = packed[0]
            a_shared1[lane0, word_base + 1] = packed[1]
            a_shared1[lane1, word_base + 0] = packed[2]
            a_shared1[lane1, word_base + 1] = packed[3]
        else:
            load_idx = tid - 128
            col_in_block = load_idx % BLOCK_N
            k8 = load_idx // BLOCK_N
            col = block_col + col_in_block
            elem_offset = col * K + BLOCK_K + k8 * 8
            packed = S.amdgpu.raw_buffer_load_x4(
                b_rsrc,
                zero,
                S.convert(elem_offset * 2, S.i32),
                0,
            )

            wave_col = col_in_block // 32
            col_lane = col_in_block % 32
            lane0 = wave_col * 64 + col_lane
            lane1 = wave_col * 64 + 32 + col_lane
            word_base = k8 * 2

            b_shared1[lane0, word_base + 0] = packed[0]
            b_shared1[lane0, word_base + 1] = packed[1]
            b_shared1[lane1, word_base + 0] = packed[2]
            b_shared1[lane1, word_base + 1] = packed[3]
        S.syncthreads()

    num_pairs = NUM_K_TILES // 2
    for pair in S.range(num_pairs - 1):
        a_frag_0 = S.view(a_shared0[warp_row * 64 + lane], S.Tensor((2, 4, 1), S.bf16))
        b_frag_0 = S.view(b_shared0[warp_col * 64 + lane], S.Tensor((2, 4, 1), S.bf16))
        c_lane = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag_0[0], b_frag_0[0], c_lane)
        c_lane = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag_0[1], b_frag_0[1], c_lane)
        S.syncthreads()
        next_even = pair * 2 + 2
        if tid < 128:
            row_in_block = tid % BLOCK_M
            k8 = tid // BLOCK_M
            row = block_row + row_in_block
            elem_offset = row * K + next_even * BLOCK_K + k8 * 8
            packed = S.amdgpu.raw_buffer_load_x4(
                a_rsrc,
                zero,
                S.convert(elem_offset * 2, S.i32),
                0,
            )

            row_group = row_in_block // 32
            row_lane = row_in_block % 32
            lane0 = row_group * 64 + row_lane
            lane1 = row_group * 64 + 32 + row_lane
            word_base = k8 * 2

            a_shared0[lane0, word_base + 0] = packed[0]
            a_shared0[lane0, word_base + 1] = packed[1]
            a_shared0[lane1, word_base + 0] = packed[2]
            a_shared0[lane1, word_base + 1] = packed[3]
        else:
            load_idx = tid - 128
            col_in_block = load_idx % BLOCK_N
            k8 = load_idx // BLOCK_N
            col = block_col + col_in_block
            elem_offset = col * K + next_even * BLOCK_K + k8 * 8
            packed = S.amdgpu.raw_buffer_load_x4(
                b_rsrc,
                zero,
                S.convert(elem_offset * 2, S.i32),
                0,
            )

            wave_col = col_in_block // 32
            col_lane = col_in_block % 32
            lane0 = wave_col * 64 + col_lane
            lane1 = wave_col * 64 + 32 + col_lane
            word_base = k8 * 2

            b_shared0[lane0, word_base + 0] = packed[0]
            b_shared0[lane0, word_base + 1] = packed[1]
            b_shared0[lane1, word_base + 0] = packed[2]
            b_shared0[lane1, word_base + 1] = packed[3]
        S.syncthreads()

        a_frag_1 = S.view(a_shared1[warp_row * 64 + lane], S.Tensor((2, 4, 1), S.bf16))
        b_frag_1 = S.view(b_shared1[warp_col * 64 + lane], S.Tensor((2, 4, 1), S.bf16))
        c_lane = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag_1[0], b_frag_1[0], c_lane)
        c_lane = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag_1[1], b_frag_1[1], c_lane)
        S.syncthreads()
        next_odd = pair * 2 + 3
        if tid < 128:
            row_in_block = tid % BLOCK_M
            k8 = tid // BLOCK_M
            row = block_row + row_in_block
            elem_offset = row * K + next_odd * BLOCK_K + k8 * 8
            packed = S.amdgpu.raw_buffer_load_x4(
                a_rsrc,
                zero,
                S.convert(elem_offset * 2, S.i32),
                0,
            )

            row_group = row_in_block // 32
            row_lane = row_in_block % 32
            lane0 = row_group * 64 + row_lane
            lane1 = row_group * 64 + 32 + row_lane
            word_base = k8 * 2

            a_shared1[lane0, word_base + 0] = packed[0]
            a_shared1[lane0, word_base + 1] = packed[1]
            a_shared1[lane1, word_base + 0] = packed[2]
            a_shared1[lane1, word_base + 1] = packed[3]
        else:
            load_idx = tid - 128
            col_in_block = load_idx % BLOCK_N
            k8 = load_idx // BLOCK_N
            col = block_col + col_in_block
            elem_offset = col * K + next_odd * BLOCK_K + k8 * 8
            packed = S.amdgpu.raw_buffer_load_x4(
                b_rsrc,
                zero,
                S.convert(elem_offset * 2, S.i32),
                0,
            )

            wave_col = col_in_block // 32
            col_lane = col_in_block % 32
            lane0 = wave_col * 64 + col_lane
            lane1 = wave_col * 64 + 32 + col_lane
            word_base = k8 * 2

            b_shared1[lane0, word_base + 0] = packed[0]
            b_shared1[lane0, word_base + 1] = packed[1]
            b_shared1[lane1, word_base + 0] = packed[2]
            b_shared1[lane1, word_base + 1] = packed[3]
        S.syncthreads()

    a_frag_last0 = S.view(a_shared0[warp_row * 64 + lane], S.Tensor((2, 4, 1), S.bf16))
    b_frag_last0 = S.view(b_shared0[warp_col * 64 + lane], S.Tensor((2, 4, 1), S.bf16))
    c_lane = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag_last0[0], b_frag_last0[0], c_lane)
    c_lane = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag_last0[1], b_frag_last0[1], c_lane)

    a_frag_last1 = S.view(a_shared1[warp_row * 64 + lane], S.Tensor((2, 4, 1), S.bf16))
    b_frag_last1 = S.view(b_shared1[warp_col * 64 + lane], S.Tensor((2, 4, 1), S.bf16))
    c_lane = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag_last1[0], b_frag_last1[0], c_lane)
    c_lane = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag_last1[1], b_frag_last1[1], c_lane)

    tile_row_base = block_row + warp_row * 32
    tile_col_base = block_col + warp_col * 32
    out_col = tile_col_base + (lane % 32)
    lane_quad = lane // 32
    for acc_idx in S.range(16):
        out_row = (
            tile_row_base
            + 8 * (acc_idx // 4)
            + 4 * lane_quad
            + (acc_idx % 4)
        )
        C[out_row, out_col] = S.convert(c_lane[acc_idx], S.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        A2 = A.transpose(-2, -1).contiguous()
        B2 = B.transpose(-2, -1).contiguous()
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)
        gemm_kernel_mfma_pipelined[lambda: ((N // BLOCK_N, M // BLOCK_M, 1), (THREADS_PER_BLOCK, 1, 1))](
            A2.view(-1), B2.view(-1), C
        )
        return C
