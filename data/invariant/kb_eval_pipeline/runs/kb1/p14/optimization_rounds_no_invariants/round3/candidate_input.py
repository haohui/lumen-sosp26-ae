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
NUM_WAVES = 4
THREADS = WAVE_SIZE * NUM_WAVES
A_RANGE_BYTES = M * K * 2
B_RANGE_BYTES = K * N * 2


@substrate.jit
def tri_gemm_kernel(
    A: S.Tensor((4096, 4096), S.bf16),
    B: S.Tensor((4096, 4096), S.bf16),
    C: S.Tensor((4096, 4096), S.bf16),
):
    tid = S.thread_id(0)
    lane = tid % WAVE_SIZE
    warp = tid // WAVE_SIZE
    warp_m = warp // 2
    warp_n = warp % 2

    block_m = S.block_id(1) * BLOCK_M
    block_n = S.block_id(0) * BLOCK_N

    a_rsrc = S.amdgpu.make_rsrc(A, A_RANGE_BYTES)
    b_rsrc = S.amdgpu.make_rsrc(B, B_RANGE_BYTES)

    zero = S.convert(0, S.i32)
    a_shared = S.make_shared((BLOCK_M, 8), S.u32)
    b_shared = S.make_shared((BLOCK_K, 32), S.u32)

    acc = S.full((16,), 0.0, S.f32)

    a_lane_row = lane % 32
    a_lane_col_group = lane // 32
    b_lane_row = lane // 8
    b_lane_col_group = lane % 8

    for k0 in S.range(K // BLOCK_K):
        k_base = k0 * BLOCK_K

        a_loader = tid % 128
        a_row = a_loader // 2
        a_word = (a_loader % 2) * 4
        a_offset = ((block_m + a_row) * K + k_base + a_word * 2) * 2
        a_pack = S.amdgpu.raw_buffer_load_x4(
            a_rsrc,
            zero,
            S.convert(a_offset, S.i32),
            0,
        )
        a_words = S.view(a_pack, S.Tensor((4,), S.u32))
        for i in S.range(4):
            a_shared[a_row, a_word + i] = a_words[i]

        b_loader = tid % 128
        b_row = b_loader // 8
        b_word = (b_loader % 8) * 4
        b_offset = ((k_base + b_row) * N + block_n + b_word * 2) * 2
        b_pack = S.amdgpu.raw_buffer_load_x4(
            b_rsrc,
            zero,
            S.convert(b_offset, S.i32),
            0,
        )
        b_words = S.view(b_pack, S.Tensor((4,), S.u32))
        for i in S.range(4):
            b_shared[b_row, b_word + i] = b_words[i]

        S.syncthreads()

        a_base = a_lane_col_group * 4
        a_lane_words_0 = S.full((2,), S.convert(0, S.u32), S.u32)
        a_lane_words_1 = S.full((2,), S.convert(0, S.u32), S.u32)
        a_lane_words_0[0] = a_shared[warp_m * 32 + a_lane_row, a_base]
        a_lane_words_0[1] = a_shared[warp_m * 32 + a_lane_row, a_base + 1]
        a_lane_words_1[0] = a_shared[warp_m * 32 + a_lane_row, a_base + 2]
        a_lane_words_1[1] = a_shared[warp_m * 32 + a_lane_row, a_base + 3]
        a_frag_0 = S.view(a_lane_words_0, S.Tensor((1, 4, 1), S.bf16))
        a_frag_1 = S.view(a_lane_words_1, S.Tensor((1, 4, 1), S.bf16))

        b_base = warp_n * 16 + b_lane_col_group * 2
        b_lane_words_0 = S.full((2,), S.convert(0, S.u32), S.u32)
        b_lane_words_1 = S.full((2,), S.convert(0, S.u32), S.u32)
        b_lane_words_0[0] = b_shared[b_lane_row, b_base]
        b_lane_words_0[1] = b_shared[b_lane_row, b_base + 1]
        b_lane_words_1[0] = b_shared[8 + b_lane_row, b_base]
        b_lane_words_1[1] = b_shared[8 + b_lane_row, b_base + 1]
        b_frag_0 = S.view(b_lane_words_0, S.Tensor((1, 4, 1), S.bf16))
        b_frag_1 = S.view(b_lane_words_1, S.Tensor((1, 4, 1), S.bf16))

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag_0[0], b_frag_0[0], acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag_1[0], b_frag_1[0], acc)

        S.syncthreads()

    lane_row_group = lane % 8
    lane_col_group = lane // 8
    tile_row = block_m + warp_m * 32 + lane_row_group * 4
    tile_col = block_n + warp_n * 32 + lane_col_group * 4

    for i in S.range(16):
        out_row = tile_row + i // 4
        out_col = tile_col + i % 4
        if out_col >= out_row:
            C[out_row, out_col] = S.convert(acc[i], S.bf16)
        else:
            C[out_row, out_col] = S.convert(0.0, S.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if tuple(A.shape) != (4096, 4096) or tuple(B.shape) != (4096, 4096):
            raise ValueError("ModelNew only supports 4096x4096 bf16 inputs")
        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((4096, 4096), device=A.device, dtype=A.dtype)
        tri_gemm_kernel[lambda: ((N // BLOCK_N, M // BLOCK_M, 1), (THREADS, 1, 1))](A, B, C)
        return C
