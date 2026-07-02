import torch
import torch.nn as nn

import substrate
import substrate.language as S


M = 4096
N = 4096
K = 4096

BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 16
WARP_M = 32
WARP_N = 32
WAVES_PER_BLOCK = 4
THREADS_PER_BLOCK = WAVES_PER_BLOCK * 32
BF16_BYTES = 2
U32_BYTES = 4


@substrate.jit
def gemm_kernel(
    A: S.Tensor((M, K), S.bf16),
    B: S.Tensor((K, N), S.bf16),
    C: S.Tensor((M, N), S.bf16),
):
    tid = S.thread_id(0)
    lane = tid % 32
    warp = tid // 32

    warp_row = warp // 2
    warp_col = warp % 2

    block_row = S.block_id(1) * BLOCK_M
    block_col = S.block_id(0) * BLOCK_N

    a_rsrc = S.amdgpu.make_rsrc(A, M * K * BF16_BYTES)
    b_rsrc = S.amdgpu.make_rsrc(B, K * N * BF16_BYTES)

    zero = S.convert(0, S.i32)

    a_shared = S.make_shared((WAVES_PER_BLOCK, 32, 4), S.u32)
    b_shared = S.make_shared((WAVES_PER_BLOCK, 32, 4), S.u32)
    a_tile = S.make_shared((BLOCK_M, BLOCK_K), S.bf16)
    b_tile = S.make_shared((BLOCK_K, BLOCK_N), S.bf16)

    mfma_acc = S.full((16,), 0.0, S.f32)
    acc = S.full((32,), 0.0, S.f32)
    thread_row_group = tid // 16
    thread_col_group = tid % 16

    for k0 in S.range(K // BLOCK_K):
        a_block_row = tid // 2
        a_block_col_u32 = (tid % 2) * 4
        a_block_offset = ((block_row + a_block_row) * K + k0 * BLOCK_K) * BF16_BYTES + a_block_col_u32 * U32_BYTES
        a_block_packed = S.amdgpu.raw_buffer_load_x4(a_rsrc, zero, a_block_offset, 0)
        a_block_vals = S.view(a_block_packed, S.Tensor((2, 4, 1), S.bf16))
        for h in S.range(2):
            for j in S.range(4):
                a_tile[a_block_row, (tid % 2) * 8 + h * 4 + j] = a_block_vals[h, j, 0]

        a_row = block_row + warp_row * WARP_M + lane // 2
        a_col_u32 = (lane % 2) * 4
        a_offset = (a_row * K + k0 * BLOCK_K) * BF16_BYTES + a_col_u32 * U32_BYTES
        a_packed = S.amdgpu.raw_buffer_load_x4(a_rsrc, zero, a_offset, 0)
        for i in S.range(4):
            a_shared[warp, lane, i] = a_packed[i]

        b_block_row = tid // 8
        b_block_col_u32 = (tid % 8) * 4
        b_block_offset = ((k0 * BLOCK_K + b_block_row) * N + block_col) * BF16_BYTES + b_block_col_u32 * U32_BYTES
        b_block_packed = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero, b_block_offset, 0)
        b_block_vals = S.view(b_block_packed, S.Tensor((2, 4, 1), S.bf16))
        for h in S.range(2):
            for j in S.range(4):
                b_tile[b_block_row, (tid % 8) * 8 + h * 4 + j] = b_block_vals[h, j, 0]

        b_row = k0 * BLOCK_K + lane // 8
        b_col_u32 = warp_col * (WARP_N // 2) + (lane % 8) * 4
        b_offset = (b_row * N + block_col) * BF16_BYTES + b_col_u32 * U32_BYTES
        b_packed = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero, b_offset, 0)
        for i in S.range(4):
            b_shared[warp, lane, i] = b_packed[i]

        S.syncthreads()

        a_frag = S.view(a_shared[warp, lane], S.Tensor((2, 4, 1), S.bf16))
        b_frag = S.view(b_shared[warp, lane], S.Tensor((2, 4, 1), S.bf16))

        mfma_acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], mfma_acc)
        mfma_acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[1], b_frag[1], mfma_acc)

        for kk in S.range(BLOCK_K):
            for rr in S.range(8):
                a_val = S.convert(a_tile[thread_row_group + rr * 8, kk], S.f32)
                for cc in S.range(4):
                    idx = rr * 4 + cc
                    b_val = S.convert(b_tile[kk, thread_col_group * 4 + cc], S.f32)
                    acc[idx] += a_val * b_val

        S.syncthreads()
    for rr in S.range(8):
        row = block_row + thread_row_group + rr * 8
        for cc in S.range(4):
            col = block_col + thread_col_group * 4 + cc
            C[row, col] = S.convert(acc[rr * 4 + cc], S.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._launch = lambda: ((N // BLOCK_N, M // BLOCK_M, 1), (THREADS_PER_BLOCK, 1, 1))

    def forward(self, A, B):
        if tuple(A.shape) != (M, K) or tuple(B.shape) != (K, N):
            raise ValueError("ModelNew expects A and B with shape (4096, 4096)")
        if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
            raise ValueError("ModelNew expects bfloat16 inputs")
        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((M, N), device=A.device, dtype=torch.bfloat16)
        gemm_kernel[self._launch](A, B, C)
        return C
