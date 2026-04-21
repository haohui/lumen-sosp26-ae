import torch
import torch.nn as nn

import substrate
import substrate.language as S


BATCH = 16
M = 1024
K = 2048
N = 768
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 16
WAVE_SIZE = 64
WAVES_M = 2
WAVES_N = 2
THREADS = WAVE_SIZE * WAVES_M * WAVES_N
BLOCKS_M = M // BLOCK_M
BLOCKS_N = N // BLOCK_N
A_RSRC_RANGE = BATCH * M * K * 2
B_RSRC_RANGE = K * N * 2
C_RSRC_RANGE = BATCH * M * N * 2
PAIR_K = BLOCK_K * 2


@substrate.jit
def matmul3d_mfma_kernel(
    A: S.Tensor((16, 1024, 2048), S.bf16),
    B: S.Tensor((2048, 768), S.bf16),
    C: S.Tensor((16, 1024, 768), S.bf16),
):
    tid = S.thread_id(0)
    wave_id = tid // WAVE_SIZE
    lane = tid % WAVE_SIZE
    wave_row = wave_id // WAVES_N
    wave_col = wave_id % WAVES_N

    batch = S.block_id(2)
    block_row = S.block_id(1)
    block_col = S.block_id(0)

    tile_row_base = block_row * BLOCK_M + wave_row * 32
    tile_col_base = block_col * BLOCK_N + wave_col * 32

    a_rsrc = S.amdgpu.make_rsrc(A, A_RSRC_RANGE)
    b_rsrc = S.amdgpu.make_rsrc(B, B_RSRC_RANGE)
    c_rsrc = S.amdgpu.make_rsrc(C, C_RSRC_RANGE)

    a_raw0 = S.make_shared((64, 2, 4), S.u32)
    a_raw1 = S.make_shared((64, 2, 4), S.u32)
    b_raw0 = S.make_shared((16, 8, 4), S.u32)
    b_raw1 = S.make_shared((16, 8, 4), S.u32)
    a_lane_packed0 = S.make_shared((128, 4), S.u32)
    a_lane_packed1 = S.make_shared((128, 4), S.u32)
    b_lane_packed0 = S.make_shared((128, 4), S.u32)
    b_lane_packed1 = S.make_shared((128, 4), S.u32)

    acc = S.full((16,), 0.0, S.f32)
    mask = S.convert(65535, S.u32)
    zero = S.convert(0, S.i32)

    if tid < 128:
        a_row = tid // 2
        a_chunk = tid % 2
        global_row = block_row * BLOCK_M + a_row
        a_offset = ((batch * M + global_row) * K + a_chunk * 8) * 2
        a_vec = S.amdgpu.raw_buffer_load_x4(
            a_rsrc,
            zero,
            S.convert(a_offset, S.i32),
            zero,
        )
        a_raw0[a_row, a_chunk] = a_vec
    else:
        b_tid = tid - 128
        b_row = b_tid // 8
        b_chunk = b_tid % 8
        global_col = block_col * BLOCK_N + b_chunk * 8
        b_offset = (b_row * N + global_col) * 2
        b_vec = S.amdgpu.raw_buffer_load_x4(
            b_rsrc,
            zero,
            S.convert(b_offset, S.i32),
            zero,
        )
        b_raw0[b_row, b_chunk] = b_vec

    S.syncthreads()

    for kk in S.range(0, K, PAIR_K):
        kk1 = kk + BLOCK_K

        if tid < 128:
            a_row = tid // 2
            a_chunk = tid % 2
            global_row = block_row * BLOCK_M + a_row
            a_offset = ((batch * M + global_row) * K + kk1 + a_chunk * 8) * 2
            a_vec = S.amdgpu.raw_buffer_load_x4(
                a_rsrc,
                zero,
                S.convert(a_offset, S.i32),
                zero,
            )
            a_raw1[a_row, a_chunk] = a_vec
        else:
            b_tid = tid - 128
            b_row = b_tid // 8
            b_chunk = b_tid % 8
            global_col = block_col * BLOCK_N + b_chunk * 8
            b_offset = ((kk1 + b_row) * N + global_col) * 2
            b_vec = S.amdgpu.raw_buffer_load_x4(
                b_rsrc,
                zero,
                S.convert(b_offset, S.i32),
                zero,
            )
            b_raw1[b_row, b_chunk] = b_vec

        if tid < 128:
            pack_idx = tid
            pack_wave_row = pack_idx // 64
            pack_lane = pack_idx % 64
            pack_row = pack_wave_row * 32 + (pack_lane % 32)
            pack_half = pack_lane // 32
            src0 = a_raw0[pack_row, 0]
            src1 = a_raw0[pack_row, 1]
            dst = S.full((4,), 0, S.u32)
            dst[0] = src0[pack_half * 2 + 0]
            dst[1] = src0[pack_half * 2 + 1]
            dst[2] = src1[pack_half * 2 + 0]
            dst[3] = src1[pack_half * 2 + 1]
            a_lane_packed0[pack_idx] = dst
        else:
            pack_idx = tid - 128
            pack_wave_col = pack_idx // 64
            pack_lane = pack_idx % 64
            pack_kgroup = pack_lane // 32
            pack_col = pack_lane % 32
            src_chunk = pack_wave_col * 4 + pack_col // 8
            src_word = (pack_col % 8) // 2
            src_half = pack_col % 2
            row0_word0 = b_raw0[pack_kgroup * 4 + 0, src_chunk][src_word]
            row0_word1 = b_raw0[pack_kgroup * 4 + 1, src_chunk][src_word]
            row0_word2 = b_raw0[pack_kgroup * 4 + 2, src_chunk][src_word]
            row0_word3 = b_raw0[pack_kgroup * 4 + 3, src_chunk][src_word]
            row1_word0 = b_raw0[pack_kgroup * 4 + 8, src_chunk][src_word]
            row1_word1 = b_raw0[pack_kgroup * 4 + 9, src_chunk][src_word]
            row1_word2 = b_raw0[pack_kgroup * 4 + 10, src_chunk][src_word]
            row1_word3 = b_raw0[pack_kgroup * 4 + 11, src_chunk][src_word]
            shift = src_half * 16
            dst = S.full((4,), 0, S.u32)
            dst[0] = ((row0_word0 >> shift) & mask) | (((row0_word1 >> shift) & mask) << 16)
            dst[1] = ((row0_word2 >> shift) & mask) | (((row0_word3 >> shift) & mask) << 16)
            dst[2] = ((row1_word0 >> shift) & mask) | (((row1_word1 >> shift) & mask) << 16)
            dst[3] = ((row1_word2 >> shift) & mask) | (((row1_word3 >> shift) & mask) << 16)
            b_lane_packed0[pack_idx] = dst

        S.syncthreads()

        a_frag_u32 = a_lane_packed0[wave_row * 64 + lane]
        b_frag_u32 = b_lane_packed0[wave_col * 64 + lane]
        a_frag = S.view(a_frag_u32, S.Tensor((2, 4, 1), S.bf16))
        b_frag = S.view(b_frag_u32, S.Tensor((2, 4, 1), S.bf16))

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[1], b_frag[1], acc)

        S.syncthreads()

        if tid < 128:
            pack_idx = tid
            pack_wave_row = pack_idx // 64
            pack_lane = pack_idx % 64
            pack_row = pack_wave_row * 32 + (pack_lane % 32)
            pack_half = pack_lane // 32
            src0 = a_raw1[pack_row, 0]
            src1 = a_raw1[pack_row, 1]
            dst = S.full((4,), 0, S.u32)
            dst[0] = src0[pack_half * 2 + 0]
            dst[1] = src0[pack_half * 2 + 1]
            dst[2] = src1[pack_half * 2 + 0]
            dst[3] = src1[pack_half * 2 + 1]
            a_lane_packed1[pack_idx] = dst
        else:
            pack_idx = tid - 128
            pack_wave_col = pack_idx // 64
            pack_lane = pack_idx % 64
            pack_kgroup = pack_lane // 32
            pack_col = pack_lane % 32
            src_chunk = pack_wave_col * 4 + pack_col // 8
            src_word = (pack_col % 8) // 2
            src_half = pack_col % 2
            row0_word0 = b_raw1[pack_kgroup * 4 + 0, src_chunk][src_word]
            row0_word1 = b_raw1[pack_kgroup * 4 + 1, src_chunk][src_word]
            row0_word2 = b_raw1[pack_kgroup * 4 + 2, src_chunk][src_word]
            row0_word3 = b_raw1[pack_kgroup * 4 + 3, src_chunk][src_word]
            row1_word0 = b_raw1[pack_kgroup * 4 + 8, src_chunk][src_word]
            row1_word1 = b_raw1[pack_kgroup * 4 + 9, src_chunk][src_word]
            row1_word2 = b_raw1[pack_kgroup * 4 + 10, src_chunk][src_word]
            row1_word3 = b_raw1[pack_kgroup * 4 + 11, src_chunk][src_word]
            shift = src_half * 16
            dst = S.full((4,), 0, S.u32)
            dst[0] = ((row0_word0 >> shift) & mask) | (((row0_word1 >> shift) & mask) << 16)
            dst[1] = ((row0_word2 >> shift) & mask) | (((row0_word3 >> shift) & mask) << 16)
            dst[2] = ((row1_word0 >> shift) & mask) | (((row1_word1 >> shift) & mask) << 16)
            dst[3] = ((row1_word2 >> shift) & mask) | (((row1_word3 >> shift) & mask) << 16)
            b_lane_packed1[pack_idx] = dst

        S.syncthreads()

        next_kk = kk + PAIR_K
        if next_kk < K:
            if tid < 128:
                a_row = tid // 2
                a_chunk = tid % 2
                global_row = block_row * BLOCK_M + a_row
                a_offset = ((batch * M + global_row) * K + next_kk + a_chunk * 8) * 2
                a_vec = S.amdgpu.raw_buffer_load_x4(
                    a_rsrc,
                    zero,
                    S.convert(a_offset, S.i32),
                    zero,
                )
                a_raw0[a_row, a_chunk] = a_vec
            else:
                b_tid = tid - 128
                b_row = b_tid // 8
                b_chunk = b_tid % 8
                global_col = block_col * BLOCK_N + b_chunk * 8
                b_offset = ((next_kk + b_row) * N + global_col) * 2
                b_vec = S.amdgpu.raw_buffer_load_x4(
                    b_rsrc,
                    zero,
                    S.convert(b_offset, S.i32),
                    zero,
                )
                b_raw0[b_row, b_chunk] = b_vec

        a_frag_u32 = a_lane_packed1[wave_row * 64 + lane]
        b_frag_u32 = b_lane_packed1[wave_col * 64 + lane]
        a_frag = S.view(a_frag_u32, S.Tensor((2, 4, 1), S.bf16))
        b_frag = S.view(b_frag_u32, S.Tensor((2, 4, 1), S.bf16))

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[1], b_frag[1], acc)

        if next_kk < K:
            S.syncthreads()

    for acc_idx in S.range(16):
        col = tile_col_base + (lane % 32)
        row = tile_row_base + 8 * (acc_idx // 4) + 4 * (lane // 32) + (acc_idx % 4)
        c_offset = ((batch * M + row) * N + col) * 2
        c_value = S.bitcast(S.convert(acc[acc_idx], S.bf16), S.u16)
        S.amdgpu.raw_buffer_store_x1(
            S.convert(c_value, S.u32),
            c_rsrc,
            zero,
            S.convert(c_offset, S.i32),
            zero,
        )


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if tuple(A.shape) != (BATCH, M, K) or tuple(B.shape) != (K, N):
            raise ValueError("ModelNew only supports the benchmark input shapes")
        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((16, 1024, 768), device=A.device, dtype=A.dtype)
        matmul3d_mfma_kernel[lambda: ((BLOCKS_N, BLOCKS_M, BATCH), (THREADS, 1, 1))](A, B, C)
        return C
