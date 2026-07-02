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
THREADS = 256


@substrate.jit
def gemm_kernel_mfma(
    A: S.Tensor((M, K), S.bf16),
    B: S.Tensor((N, K), S.bf16),
    C: S.Tensor((M, N), S.bf16),
    a_range_bytes: S.i32,
    b_range_bytes: S.i32,
):
    tid = S.thread_id(0)
    lane = tid % 64
    warp = tid // 64
    warp_row = warp // 2
    warp_col = warp % 2

    block_col = S.block_id(0) * BLOCK_N
    block_row = S.block_id(1) * BLOCK_M

    a_rsrc = S.amdgpu.make_rsrc(A, a_range_bytes)
    b_rsrc = S.amdgpu.make_rsrc(B, b_range_bytes)
    zero = S.convert(0, S.i32)

    a_words = S.make_shared((2, 64, 4), S.u32)
    b_words = S.make_shared((2, 64, 4), S.u32)

    acc = S.full((16,), 0.0, S.f32)

    for k0 in S.range(0, K, BLOCK_K):
        if tid < 128:
            a_group = tid // 64
            a_chunk = tid % 64
            a_row = block_row + a_group * 32 + (a_chunk % 32)
            a_k_chunk = a_chunk // 32
            a_col = k0 + a_k_chunk * 8
            a_offset = S.convert((a_row * K + a_col) * 2, S.i32)
            a_pack = S.amdgpu.raw_buffer_load_x4(a_rsrc, zero, a_offset, 0)
            a_lane0 = a_chunk % 32
            a_lane1 = a_lane0 + 32
            a_dst_word = a_k_chunk * 2

            a_words[a_group, a_lane0, a_dst_word + 0] = a_pack[0]
            a_words[a_group, a_lane0, a_dst_word + 1] = a_pack[1]
            a_words[a_group, a_lane1, a_dst_word + 0] = a_pack[2]
            a_words[a_group, a_lane1, a_dst_word + 1] = a_pack[3]

            b_group = tid // 64
            b_chunk = tid % 64
            b_col_local = b_chunk % 32
            b_k_chunk = b_chunk // 32
            b_col = block_col + b_group * 32 + b_col_local
            b_k = k0 + b_k_chunk * 8
            b_offset = S.convert((b_col * K + b_k) * 2, S.i32)
            b_pack = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero, b_offset, 0)
            b_lane0 = b_col_local
            b_lane1 = b_col_local + 32
            b_dst_word = b_k_chunk * 2

            b_words[b_group, b_lane0, b_dst_word + 0] = b_pack[0]
            b_words[b_group, b_lane0, b_dst_word + 1] = b_pack[1]
            b_words[b_group, b_lane1, b_dst_word + 0] = b_pack[2]
            b_words[b_group, b_lane1, b_dst_word + 1] = b_pack[3]

        S.syncthreads()

        a_lane_words = a_words[warp_row, lane]
        b_lane_words = b_words[warp_col, lane]
        a_mfma = S.view(a_lane_words, S.Tensor((2, 4, 1), S.bf16))
        b_mfma = S.view(b_lane_words, S.Tensor((2, 4, 1), S.bf16))

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_mfma[0], b_mfma[0], acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_mfma[1], b_mfma[1], acc)

        S.syncthreads()

    tile_row_base = block_row + warp_row * 32
    tile_col_base = block_col + warp_col * 32
    lane_col = lane % 32
    lane_row_quad = lane // 32

    for acc_idx in S.range(16):
        out_col = tile_col_base + lane_col
        out_row = tile_row_base + 8 * (acc_idx // 4) + 4 * lane_row_quad + (acc_idx % 4)
        C[out_row, out_col] = S.convert(acc[acc_idx], S.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._a_range_bytes = M * K * 2
        self._b_range_bytes = K * N * 2

    def forward(self, A, B):
        if tuple(A.shape) != (8192, 2048) or tuple(B.shape) != (4096, 8192):
            raise RuntimeError("ModelNew only supports A=(8192, 2048), B=(4096, 8192)")

        A2 = A.transpose(-2, -1).contiguous()
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)
        gemm_kernel_mfma[lambda: ((N // BLOCK_N, M // BLOCK_M, 1), (THREADS, 1, 1))](
            A2, B.contiguous(), C, self._a_range_bytes, self._b_range_bytes
        )
        return C
