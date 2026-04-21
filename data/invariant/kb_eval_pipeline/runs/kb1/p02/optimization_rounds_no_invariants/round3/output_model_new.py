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


@substrate.jit
def gemm_kernel(
    A: S.Tensor((2048, 8192), S.bf16),
    B: S.Tensor((8192, 4096), S.bf16),
    C: S.Tensor((2048, 4096), S.bf16),
):
    tid = S.thread_id(0)
    lane = tid % WAVE_SIZE
    wave = tid // WAVE_SIZE
    wave_row = wave // 2
    wave_col = wave % 2

    block_n = S.block_id(0)
    block_m = S.block_id(1)

    a_rsrc = S.amdgpu.make_rsrc(A, M * K * 2)
    b_rsrc = S.amdgpu.make_rsrc(B, K * N * 2)

    a_shared = S.make_shared((BLOCK_M, 2, 4), S.u32)
    b_shared = S.make_shared((BLOCK_K, BLOCK_N // 8, 4), S.u32)
    a_lane_words = S.make_shared((THREADS_PER_BLOCK, 4), S.u32)
    b_lane_words = S.make_shared((THREADS_PER_BLOCK, 4), S.u32)

    acc = S.full((16,), 0.0, S.f32)

    for k_tile in S.range(K // BLOCK_K):
        if tid < (BLOCK_M * 2):
            a_row = tid // 2
            a_chunk = tid % 2
            a_global_row = block_m * BLOCK_M + a_row
            a_global_k = k_tile * BLOCK_K + a_chunk * 8
            a_byte_offset = (a_global_row * K + a_global_k) * 2
            a_shared[a_row, a_chunk] = S.amdgpu.raw_buffer_load_x4(
                a_rsrc, a_byte_offset, 0, 0
            )
        else:
            b_linear = tid - BLOCK_M * 2
            b_k = b_linear // (BLOCK_N // 8)
            b_chunk = b_linear % (BLOCK_N // 8)
            b_global_k = k_tile * BLOCK_K + b_k
            b_global_n = block_n * BLOCK_N + b_chunk * 8
            b_byte_offset = (b_global_k * N + b_global_n) * 2
            b_shared[b_k, b_chunk] = S.amdgpu.raw_buffer_load_x4(
                b_rsrc, b_byte_offset, 0, 0
            )

        S.syncthreads()

        a_row = wave_row * 32 + (lane % 32)
        k_group = lane // 32

        a_lane_words[tid, 0] = a_shared[a_row, 0, k_group * 2 + 0]
        a_lane_words[tid, 1] = a_shared[a_row, 0, k_group * 2 + 1]
        a_lane_words[tid, 2] = a_shared[a_row, 1, k_group * 2 + 0]
        a_lane_words[tid, 3] = a_shared[a_row, 1, k_group * 2 + 1]
        a_frag = S.view(a_lane_words[tid], S.Tensor((2, 4, 1), S.bf16))

        col_in_wave = lane % 32
        b_chunk = wave_col * 4 + col_in_wave // 8
        b_word = (col_in_wave % 8) // 2
        b_shift = S.convert((col_in_wave % 2) * 16, S.u32)
        b_mask = S.convert(0xFFFF, S.u32)

        b_src0 = (b_shared[k_group * 4 + 0, b_chunk, b_word] >> b_shift) & b_mask
        b_src1 = (b_shared[k_group * 4 + 1, b_chunk, b_word] >> b_shift) & b_mask
        b_src2 = (b_shared[k_group * 4 + 2, b_chunk, b_word] >> b_shift) & b_mask
        b_src3 = (b_shared[k_group * 4 + 3, b_chunk, b_word] >> b_shift) & b_mask
        b_src4 = (b_shared[8 + k_group * 4 + 0, b_chunk, b_word] >> b_shift) & b_mask
        b_src5 = (b_shared[8 + k_group * 4 + 1, b_chunk, b_word] >> b_shift) & b_mask
        b_src6 = (b_shared[8 + k_group * 4 + 2, b_chunk, b_word] >> b_shift) & b_mask
        b_src7 = (b_shared[8 + k_group * 4 + 3, b_chunk, b_word] >> b_shift) & b_mask

        b_lane_words[tid, 0] = b_src0 | (b_src1 << S.convert(16, S.u32))
        b_lane_words[tid, 1] = b_src2 | (b_src3 << S.convert(16, S.u32))
        b_lane_words[tid, 2] = b_src4 | (b_src5 << S.convert(16, S.u32))
        b_lane_words[tid, 3] = b_src6 | (b_src7 << S.convert(16, S.u32))
        b_frag = S.view(b_lane_words[tid], S.Tensor((2, 4, 1), S.bf16))

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[1], b_frag[1], acc)

        S.syncthreads()

    out_col = block_n * BLOCK_N + wave_col * 32 + (lane % 32)
    for acc_idx in S.range(16):
        out_row = (
            block_m * BLOCK_M
            + wave_row * 32
            + (acc_idx % 4)
            + 8 * (acc_idx // 4)
            + 4 * (lane // 32)
        )
        C[out_row, out_col] = S.convert(acc[acc_idx], S.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if tuple(A.shape) != (M, K) or tuple(B.shape) != (K, N):
            return torch.matmul(A, B)
        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)
        gemm_kernel[lambda: ((N // BLOCK_N, M // BLOCK_M, 1), (THREADS_PER_BLOCK, 1, 1))](
            A, B, C
        )
        return C
