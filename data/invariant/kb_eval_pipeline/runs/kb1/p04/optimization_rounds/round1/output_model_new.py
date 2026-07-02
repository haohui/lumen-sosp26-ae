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
    B: S.Tensor((K // 2, N), S.u32),
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

    shared_a = S.make_shared((WAVES_PER_BLOCK, LANES_PER_WAVE, 4), S.u32)
    shared_b = S.make_shared((WAVES_PER_BLOCK, LANES_PER_WAVE, 4), S.u32)

    acc = S.full((16,), 0.0, S.f32)

    for k_base in S.range(0, K, K_STEP):
        row_in_wave = lane % 32
        half = lane // 32

        a_row = tile_row_base + row_in_wave

        word_base = k_base // 2
        if half == 0:
            shared_a[warp, lane, 0] = A[a_row, word_base + 0]
            shared_a[warp, lane, 1] = A[a_row, word_base + 1]
            shared_a[warp, lane, 2] = A[a_row, word_base + 4]
            shared_a[warp, lane, 3] = A[a_row, word_base + 5]

            shared_b[warp, lane, 0] = B[word_base + 0, 0]
            shared_b[warp, lane, 1] = B[word_base + 1, 0]
            shared_b[warp, lane, 2] = B[word_base + 4, 0]
            shared_b[warp, lane, 3] = B[word_base + 5, 0]
        else:
            shared_a[warp, lane, 0] = A[a_row, word_base + 2]
            shared_a[warp, lane, 1] = A[a_row, word_base + 3]
            shared_a[warp, lane, 2] = A[a_row, word_base + 6]
            shared_a[warp, lane, 3] = A[a_row, word_base + 7]

            shared_b[warp, lane, 0] = B[word_base + 2, 0]
            shared_b[warp, lane, 1] = B[word_base + 3, 0]
            shared_b[warp, lane, 2] = B[word_base + 6, 0]
            shared_b[warp, lane, 3] = B[word_base + 7, 0]

        S.syncthreads()

        a_frag = S.view(shared_a[warp, lane], S.Tensor((2, 4, 1), S.bf16))
        b_frag = S.view(shared_b[warp, lane], S.Tensor((2, 4, 1), S.bf16))

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[1], b_frag[1], acc)

        S.syncthreads()

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
        B_u32 = B.reshape(K).view(torch.int32).reshape(K // 2, N)
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)
        gemm_kernel[lambda: ((1, M // BLOCK_M, 1), (BLOCK_THREADS, 1, 1))](A_u32, B_u32, C)
        return C
