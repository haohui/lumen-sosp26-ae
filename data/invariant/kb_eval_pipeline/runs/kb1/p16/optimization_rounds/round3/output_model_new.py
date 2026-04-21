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
THREADS = WAVE_SIZE * WAVES_PER_BLOCK


@substrate.jit
def gemm_kernel(
    A: S.Tensor((M, K), S.bf16),
    B: S.Tensor((K, N), S.bf16),
    C: S.Tensor((M, N), S.bf16),
):
    tid = S.thread_id(0)
    warp = tid // WAVE_SIZE
    lane = tid % WAVE_SIZE

    wave_row = warp // 2
    wave_col = warp % 2

    block_col = S.block_id(0) * BLOCK_N
    block_row = S.block_id(1) * BLOCK_M

    row = block_row + wave_row * 32 + (lane % 32)
    col = block_col + wave_col * 32 + (lane % 32)
    lane_k_group = (lane // 32) * 4

    a_shared = S.make_shared((2, THREADS, 4), S.u32)
    b_shared = S.make_shared((2, THREADS, 4), S.u32)
    a_local = S.make_local((8,), S.bf16)
    b_local = S.make_local((8,), S.bf16)

    acc = S.full((16,), 0.0, S.f32)

    for elem in S.range(4):
        a_local[elem] = A[row, lane_k_group + elem]
        a_local[4 + elem] = A[row, 8 + lane_k_group + elem]
        b_local[elem] = B[lane_k_group + elem, col]
        b_local[4 + elem] = B[8 + lane_k_group + elem, col]
    a_pack = S.view(a_local, S.Tensor((4,), S.u32))
    b_pack = S.view(b_local, S.Tensor((4,), S.u32))
    for word in S.range(4):
        a_shared[0, tid, word] = a_pack[word]
        b_shared[0, tid, word] = b_pack[word]

    for elem in S.range(4):
        a_local[elem] = A[row, BLOCK_K + lane_k_group + elem]
        a_local[4 + elem] = A[row, BLOCK_K + 8 + lane_k_group + elem]
        b_local[elem] = B[BLOCK_K + lane_k_group + elem, col]
        b_local[4 + elem] = B[BLOCK_K + 8 + lane_k_group + elem, col]
    a_pack = S.view(a_local, S.Tensor((4,), S.u32))
    b_pack = S.view(b_local, S.Tensor((4,), S.u32))
    for word in S.range(4):
        a_shared[1, tid, word] = a_pack[word]
        b_shared[1, tid, word] = b_pack[word]

    S.syncthreads()

    for ko in S.range(0, K - 2 * BLOCK_K, 2 * BLOCK_K):
        a_mfma = S.view(a_shared[0, tid], S.Tensor((2, 4, 1), S.bf16))
        b_mfma = S.view(b_shared[0, tid], S.Tensor((2, 4, 1), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_mfma[0], b_mfma[0], acc)

        next_ko = ko + 2 * BLOCK_K
        for elem in S.range(4):
            a_local[elem] = A[row, next_ko + lane_k_group + elem]
            a_local[4 + elem] = A[row, next_ko + 8 + lane_k_group + elem]
            b_local[elem] = B[next_ko + lane_k_group + elem, col]
            b_local[4 + elem] = B[next_ko + 8 + lane_k_group + elem, col]

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_mfma[1], b_mfma[1], acc)
        a_pack = S.view(a_local, S.Tensor((4,), S.u32))
        b_pack = S.view(b_local, S.Tensor((4,), S.u32))
        for word in S.range(4):
            a_shared[0, tid, word] = a_pack[word]
            b_shared[0, tid, word] = b_pack[word]

        a_mfma = S.view(a_shared[1, tid], S.Tensor((2, 4, 1), S.bf16))
        b_mfma = S.view(b_shared[1, tid], S.Tensor((2, 4, 1), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_mfma[0], b_mfma[0], acc)

        next_ko = ko + 3 * BLOCK_K
        for elem in S.range(4):
            a_local[elem] = A[row, next_ko + lane_k_group + elem]
            a_local[4 + elem] = A[row, next_ko + 8 + lane_k_group + elem]
            b_local[elem] = B[next_ko + lane_k_group + elem, col]
            b_local[4 + elem] = B[next_ko + 8 + lane_k_group + elem, col]

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_mfma[1], b_mfma[1], acc)
        a_pack = S.view(a_local, S.Tensor((4,), S.u32))
        b_pack = S.view(b_local, S.Tensor((4,), S.u32))
        for word in S.range(4):
            a_shared[1, tid, word] = a_pack[word]
            b_shared[1, tid, word] = b_pack[word]

        S.syncthreads()

    a_mfma = S.view(a_shared[0, tid], S.Tensor((2, 4, 1), S.bf16))
    b_mfma = S.view(b_shared[0, tid], S.Tensor((2, 4, 1), S.bf16))
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_mfma[0], b_mfma[0], acc)
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_mfma[1], b_mfma[1], acc)

    a_mfma = S.view(a_shared[1, tid], S.Tensor((2, 4, 1), S.bf16))
    b_mfma = S.view(b_shared[1, tid], S.Tensor((2, 4, 1), S.bf16))
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_mfma[0], b_mfma[0], acc)
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_mfma[1], b_mfma[1], acc)

    out_col = block_col + wave_col * 32 + (lane % 32)
    lane_row_group = lane // 32
    for acc_idx in S.range(16):
        out_row = (
            block_row
            + wave_row * 32
            + 8 * (acc_idx // 4)
            + 4 * lane_row_group
            + (acc_idx % 4)
        )
        C[out_row, out_col] = S.convert(acc[acc_idx], S.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if tuple(A.shape) != (K, M) or tuple(B.shape) != (K, N):
            raise RuntimeError(
                f"ModelNew expects A.shape == ({K}, {M}) and B.shape == ({K}, {N})"
            )
        if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
            raise RuntimeError("ModelNew expects bf16 inputs")
        if A.device != B.device:
            raise RuntimeError("A and B must be on the same device")

        a_t = A.transpose(-2, -1).contiguous()
        b_c = B.contiguous()
        c = torch.empty((M, N), device=A.device, dtype=torch.bfloat16)
        gemm_kernel[lambda: ((N // BLOCK_N, M // BLOCK_M, 1), (THREADS, 1, 1))](
            a_t, b_c, c
        )
        return c
