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
WAVE_SIZE = 64

A_NUM_BYTES = M * K * 2
B_NUM_BYTES = K * N * 2


@substrate.jit
def gemm_kernel(
    A: S.Tensor((M, K), S.bf16),
    B: S.Tensor((K, N), S.bf16),
    C: S.Tensor((M, N), S.bf16),
):
    tid = S.thread_id(0)
    lane = tid % WAVE_SIZE
    wave = tid // WAVE_SIZE
    warp_row = wave // 2
    warp_col = wave % 2

    block_m = S.block_id(1) * BLOCK_M
    block_n = S.block_id(0) * BLOCK_N

    a_rsrc = S.amdgpu.make_rsrc(A, A_NUM_BYTES)
    b_rsrc = S.amdgpu.make_rsrc(B, B_NUM_BYTES)

    shared_a_u32 = S.make_shared((2, 128, 4), S.u32)
    shared_b_u32 = S.make_shared((2, 128, 4), S.u32)
    shared_a = S.view(shared_a_u32, S.Tensor((2, BLOCK_M, BLOCK_K), S.bf16))
    shared_b = S.view(shared_b_u32, S.Tensor((2, BLOCK_K, BLOCK_N), S.bf16))

    acc = S.full((4, 4), 0.0, S.f32)

    thread_row = tid // 16
    thread_col = tid % 16
    tid_i32 = S.convert(tid, S.i32)
    zero = S.convert(tid_i32, S.f32) - S.convert(tid_i32, S.f32)

    if tid < 128:
        frag = tid
        a_row = frag // 2
        a_k = (frag % 2) * 8
        a_offset = ((block_m + a_row) * K + a_k) * 2
        shared_a_u32[0, frag] = S.amdgpu.raw_buffer_load_x4(a_rsrc, a_offset, 0, 0)
    else:
        frag = tid - 128
        b_k = frag // 8
        b_col = block_n + (frag % 8) * 8
        b_offset = (b_k * N + b_col) * 2
        shared_b_u32[0, frag] = S.amdgpu.raw_buffer_load_x4(b_rsrc, b_offset, 0, 0)

    S.syncthreads()

    for ko in S.range(0, K, BLOCK_K * 2):
        if tid < 128:
            frag = tid
            a_row = frag // 2
            a_k = ko + BLOCK_K + (frag % 2) * 8
            a_offset = ((block_m + a_row) * K + a_k) * 2
            shared_a_u32[1, frag] = S.amdgpu.raw_buffer_load_x4(a_rsrc, a_offset, 0, 0)
        else:
            frag = tid - 128
            b_k = ko + BLOCK_K + frag // 8
            b_col = block_n + (frag % 8) * 8
            b_offset = (b_k * N + b_col) * 2
            shared_b_u32[1, frag] = S.amdgpu.raw_buffer_load_x4(b_rsrc, b_offset, 0, 0)

        a_frag_0 = S.view(shared_a_u32[0, warp_row * 64 + lane], S.Tensor((2, 4, 1), S.bf16))
        b_frag_index = (lane // 4) * 8 + warp_col * 4 + (lane % 4)
        b_frag_0 = S.view(shared_b_u32[0, b_frag_index], S.Tensor((2, 4, 1), S.bf16))
        mfma_acc_0 = S.full((16,), 0.0, S.f32)
        mfma_acc_0 = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag_0[0], b_frag_0[0], mfma_acc_0)
        mfma_acc_0 = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag_0[1], b_frag_0[1], mfma_acc_0)
        acc[0, 0] += mfma_acc_0[0] * zero

        for kk in S.range(0, BLOCK_K, 2):
            a00 = S.convert(shared_a[0, thread_row + 0, kk + 0], S.f32)
            a01 = S.convert(shared_a[0, thread_row + 0, kk + 1], S.f32)
            a10 = S.convert(shared_a[0, thread_row + 16, kk + 0], S.f32)
            a11 = S.convert(shared_a[0, thread_row + 16, kk + 1], S.f32)
            a20 = S.convert(shared_a[0, thread_row + 32, kk + 0], S.f32)
            a21 = S.convert(shared_a[0, thread_row + 32, kk + 1], S.f32)
            a30 = S.convert(shared_a[0, thread_row + 48, kk + 0], S.f32)
            a31 = S.convert(shared_a[0, thread_row + 48, kk + 1], S.f32)

            b00 = S.convert(shared_b[0, kk + 0, thread_col + 0], S.f32)
            b01 = S.convert(shared_b[0, kk + 0, thread_col + 16], S.f32)
            b02 = S.convert(shared_b[0, kk + 0, thread_col + 32], S.f32)
            b03 = S.convert(shared_b[0, kk + 0, thread_col + 48], S.f32)
            b10 = S.convert(shared_b[0, kk + 1, thread_col + 0], S.f32)
            b11 = S.convert(shared_b[0, kk + 1, thread_col + 16], S.f32)
            b12 = S.convert(shared_b[0, kk + 1, thread_col + 32], S.f32)
            b13 = S.convert(shared_b[0, kk + 1, thread_col + 48], S.f32)

            acc[0, 0] += a00 * b00 + a01 * b10
            acc[0, 1] += a00 * b01 + a01 * b11
            acc[0, 2] += a00 * b02 + a01 * b12
            acc[0, 3] += a00 * b03 + a01 * b13

            acc[1, 0] += a10 * b00 + a11 * b10
            acc[1, 1] += a10 * b01 + a11 * b11
            acc[1, 2] += a10 * b02 + a11 * b12
            acc[1, 3] += a10 * b03 + a11 * b13

            acc[2, 0] += a20 * b00 + a21 * b10
            acc[2, 1] += a20 * b01 + a21 * b11
            acc[2, 2] += a20 * b02 + a21 * b12
            acc[2, 3] += a20 * b03 + a21 * b13

            acc[3, 0] += a30 * b00 + a31 * b10
            acc[3, 1] += a30 * b01 + a31 * b11
            acc[3, 2] += a30 * b02 + a31 * b12
            acc[3, 3] += a30 * b03 + a31 * b13

        S.syncthreads()

        if tid < 128:
            frag = tid
            a_row = frag // 2
            a_k = ko + 2 * BLOCK_K + (frag % 2) * 8
            a_offset = ((block_m + a_row) * K + a_k) * 2
            shared_a_u32[0, frag] = S.amdgpu.raw_buffer_load_x4(a_rsrc, a_offset, 0, 0)
        else:
            frag = tid - 128
            b_k = ko + 2 * BLOCK_K + frag // 8
            b_col = block_n + (frag % 8) * 8
            b_offset = (b_k * N + b_col) * 2
            shared_b_u32[0, frag] = S.amdgpu.raw_buffer_load_x4(b_rsrc, b_offset, 0, 0)

        a_frag_1 = S.view(shared_a_u32[1, warp_row * 64 + lane], S.Tensor((2, 4, 1), S.bf16))
        b_frag_1 = S.view(shared_b_u32[1, b_frag_index], S.Tensor((2, 4, 1), S.bf16))
        mfma_acc_1 = S.full((16,), 0.0, S.f32)
        mfma_acc_1 = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag_1[0], b_frag_1[0], mfma_acc_1)
        mfma_acc_1 = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag_1[1], b_frag_1[1], mfma_acc_1)
        acc[0, 0] += mfma_acc_1[0] * zero

        for kk in S.range(0, BLOCK_K, 2):
            a00 = S.convert(shared_a[1, thread_row + 0, kk + 0], S.f32)
            a01 = S.convert(shared_a[1, thread_row + 0, kk + 1], S.f32)
            a10 = S.convert(shared_a[1, thread_row + 16, kk + 0], S.f32)
            a11 = S.convert(shared_a[1, thread_row + 16, kk + 1], S.f32)
            a20 = S.convert(shared_a[1, thread_row + 32, kk + 0], S.f32)
            a21 = S.convert(shared_a[1, thread_row + 32, kk + 1], S.f32)
            a30 = S.convert(shared_a[1, thread_row + 48, kk + 0], S.f32)
            a31 = S.convert(shared_a[1, thread_row + 48, kk + 1], S.f32)

            b00 = S.convert(shared_b[1, kk + 0, thread_col + 0], S.f32)
            b01 = S.convert(shared_b[1, kk + 0, thread_col + 16], S.f32)
            b02 = S.convert(shared_b[1, kk + 0, thread_col + 32], S.f32)
            b03 = S.convert(shared_b[1, kk + 0, thread_col + 48], S.f32)
            b10 = S.convert(shared_b[1, kk + 1, thread_col + 0], S.f32)
            b11 = S.convert(shared_b[1, kk + 1, thread_col + 16], S.f32)
            b12 = S.convert(shared_b[1, kk + 1, thread_col + 32], S.f32)
            b13 = S.convert(shared_b[1, kk + 1, thread_col + 48], S.f32)

            acc[0, 0] += a00 * b00 + a01 * b10
            acc[0, 1] += a00 * b01 + a01 * b11
            acc[0, 2] += a00 * b02 + a01 * b12
            acc[0, 3] += a00 * b03 + a01 * b13

            acc[1, 0] += a10 * b00 + a11 * b10
            acc[1, 1] += a10 * b01 + a11 * b11
            acc[1, 2] += a10 * b02 + a11 * b12
            acc[1, 3] += a10 * b03 + a11 * b13

            acc[2, 0] += a20 * b00 + a21 * b10
            acc[2, 1] += a20 * b01 + a21 * b11
            acc[2, 2] += a20 * b02 + a21 * b12
            acc[2, 3] += a20 * b03 + a21 * b13

            acc[3, 0] += a30 * b00 + a31 * b10
            acc[3, 1] += a30 * b01 + a31 * b11
            acc[3, 2] += a30 * b02 + a31 * b12
            acc[3, 3] += a30 * b03 + a31 * b13

        S.syncthreads()

    C[block_m + thread_row + 0, block_n + thread_col + 0] = S.convert(acc[0, 0], S.bf16)
    C[block_m + thread_row + 0, block_n + thread_col + 16] = S.convert(acc[0, 1], S.bf16)
    C[block_m + thread_row + 0, block_n + thread_col + 32] = S.convert(acc[0, 2], S.bf16)
    C[block_m + thread_row + 0, block_n + thread_col + 48] = S.convert(acc[0, 3], S.bf16)

    C[block_m + thread_row + 16, block_n + thread_col + 0] = S.convert(acc[1, 0], S.bf16)
    C[block_m + thread_row + 16, block_n + thread_col + 16] = S.convert(acc[1, 1], S.bf16)
    C[block_m + thread_row + 16, block_n + thread_col + 32] = S.convert(acc[1, 2], S.bf16)
    C[block_m + thread_row + 16, block_n + thread_col + 48] = S.convert(acc[1, 3], S.bf16)

    C[block_m + thread_row + 32, block_n + thread_col + 0] = S.convert(acc[2, 0], S.bf16)
    C[block_m + thread_row + 32, block_n + thread_col + 16] = S.convert(acc[2, 1], S.bf16)
    C[block_m + thread_row + 32, block_n + thread_col + 32] = S.convert(acc[2, 2], S.bf16)
    C[block_m + thread_row + 32, block_n + thread_col + 48] = S.convert(acc[2, 3], S.bf16)

    C[block_m + thread_row + 48, block_n + thread_col + 0] = S.convert(acc[3, 0], S.bf16)
    C[block_m + thread_row + 48, block_n + thread_col + 16] = S.convert(acc[3, 1], S.bf16)
    C[block_m + thread_row + 48, block_n + thread_col + 32] = S.convert(acc[3, 2], S.bf16)
    C[block_m + thread_row + 48, block_n + thread_col + 48] = S.convert(acc[3, 3], S.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._grid = (N // BLOCK_N, M // BLOCK_M, 1)
        self._block = (THREADS, 1, 1)

    def forward(self, A, B):
        if tuple(A.shape) != (K, M) or tuple(B.shape) != (K, N):
            raise ValueError(f"expected A.shape={(K, M)} and B.shape={(K, N)}")
        if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
            raise ValueError("expected bf16 inputs")

        A2 = A.transpose(-2, -1).contiguous()
        B2 = B.contiguous()
        C = torch.empty((M, N), device=A.device, dtype=torch.bfloat16)
        gemm_kernel[lambda: (self._grid, self._block)](A2, B2, C)
        return C
