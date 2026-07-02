import torch
import torch.nn as nn

import substrate
import substrate.language as S


M = 32768
K = 64
N = 32768

BLOCK_M = 64
BLOCK_N = 64
WAVE_M = 32
WAVE_N = 32
WAVES_PER_BLOCK = 4
THREADS_PER_WAVE = 64
THREADS_PER_BLOCK = WAVES_PER_BLOCK * THREADS_PER_WAVE
K_CHUNK = 16

A_RANGE_BYTES = M * K * 2
B_RANGE_BYTES = K * N * 2


@substrate.jit
def gemm_kernel(
    A: S.Tensor((32768, 64), S.bf16),
    B: S.Tensor((64, 32768), S.bf16),
    C: S.Tensor((32768, 32768), S.bf16),
):
    tid = S.thread_id(0)
    lane = tid % THREADS_PER_WAVE
    wave = tid // THREADS_PER_WAVE
    wave_row = wave // 2
    wave_col = wave % 2

    block_m = S.block_id(1) * BLOCK_M
    block_n = S.block_id(0) * BLOCK_N

    row_group = lane // 8
    col_group = lane % 8
    row_base = block_m + wave_row * WAVE_M + row_group * 4
    col_base = block_n + wave_col * WAVE_N + col_group * 4

    acc00 = S.convert(0.0, S.f32)
    acc01 = S.convert(0.0, S.f32)
    acc02 = S.convert(0.0, S.f32)
    acc03 = S.convert(0.0, S.f32)
    acc10 = S.convert(0.0, S.f32)
    acc11 = S.convert(0.0, S.f32)
    acc12 = S.convert(0.0, S.f32)
    acc13 = S.convert(0.0, S.f32)
    acc20 = S.convert(0.0, S.f32)
    acc21 = S.convert(0.0, S.f32)
    acc22 = S.convert(0.0, S.f32)
    acc23 = S.convert(0.0, S.f32)
    acc30 = S.convert(0.0, S.f32)
    acc31 = S.convert(0.0, S.f32)
    acc32 = S.convert(0.0, S.f32)
    acc33 = S.convert(0.0, S.f32)
    mfma_acc = S.full((16,), 0.0, S.f32)

    a_words0 = S.make_shared((512,), S.u32)
    a_words1 = S.make_shared((512,), S.u32)
    b_words0 = S.make_shared((512,), S.u32)
    b_words1 = S.make_shared((512,), S.u32)
    a_frags0 = S.view(a_words0, S.u32, S.make_layout((128, 4), (4, 1)))
    a_frags1 = S.view(a_words1, S.u32, S.make_layout((128, 4), (4, 1)))
    b_frags0 = S.view(b_words0, S.u32, S.make_layout((128, 4), (4, 1)))
    b_frags1 = S.view(b_words1, S.u32, S.make_layout((128, 4), (4, 1)))

    a_rsrc = S.amdgpu.make_rsrc(A, S.convert(A_RANGE_BYTES, S.i32))
    b_rsrc = S.amdgpu.make_rsrc(B, S.convert(B_RANGE_BYTES, S.i32))
    zero = S.convert(0, S.i32)

    if tid < 128:
        a_row = tid // 2
        a_part = tid % 2
        a_byte_offset = S.convert(((block_m + a_row) * K + a_part * 8) * 2, S.i32)
        a_packed = S.amdgpu.raw_buffer_load_x4(a_rsrc, a_byte_offset, zero, 0)
        for j in S.range(4):
            a_frags0[tid, j] = a_packed[j]
    else:
        b_frag = tid - 128
        b_k = b_frag // 8
        b_part = b_frag % 8
        b_byte_offset = S.convert((b_k * N + block_n + b_part * 8) * 2, S.i32)
        b_packed = S.amdgpu.raw_buffer_load_x4(b_rsrc, b_byte_offset, zero, 0)
        for j in S.range(4):
            b_frags0[b_frag, j] = b_packed[j]

    if tid < 128:
        a_row = tid // 2
        a_part = tid % 2
        a_byte_offset = S.convert(((block_m + a_row) * K + K_CHUNK + a_part * 8) * 2, S.i32)
        a_packed = S.amdgpu.raw_buffer_load_x4(a_rsrc, a_byte_offset, zero, 0)
        for j in S.range(4):
            a_frags1[tid, j] = a_packed[j]
    else:
        b_frag = tid - 128
        b_k = b_frag // 8
        b_part = b_frag % 8
        b_byte_offset = S.convert(((K_CHUNK + b_k) * N + block_n + b_part * 8) * 2, S.i32)
        b_packed = S.amdgpu.raw_buffer_load_x4(b_rsrc, b_byte_offset, zero, 0)
        for j in S.range(4):
            b_frags1[b_frag, j] = b_packed[j]

    S.syncthreads()

    a_mfma = S.view(a_frags0[wave_row * 64 + lane], S.Tensor((2, 4, 1), S.bf16))
    b_mfma = S.view(
        b_frags0[(lane // 4) * 8 + wave_col * 4 + (lane % 4)],
        S.Tensor((2, 4, 1), S.bf16),
    )
    mfma_acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_mfma[0], b_mfma[0], mfma_acc)
    mfma_acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_mfma[1], b_mfma[1], mfma_acc)
    for kk in S.range(K_CHUNK):
        global_k = kk
        a0 = S.convert(A[row_base + 0, global_k], S.f32)
        a1 = S.convert(A[row_base + 1, global_k], S.f32)
        a2 = S.convert(A[row_base + 2, global_k], S.f32)
        a3 = S.convert(A[row_base + 3, global_k], S.f32)
        b0 = S.convert(B[global_k, col_base + 0], S.f32)
        b1 = S.convert(B[global_k, col_base + 1], S.f32)
        b2 = S.convert(B[global_k, col_base + 2], S.f32)
        b3 = S.convert(B[global_k, col_base + 3], S.f32)
        acc00 += a0 * b0
        acc01 += a0 * b1
        acc02 += a0 * b2
        acc03 += a0 * b3
        acc10 += a1 * b0
        acc11 += a1 * b1
        acc12 += a1 * b2
        acc13 += a1 * b3
        acc20 += a2 * b0
        acc21 += a2 * b1
        acc22 += a2 * b2
        acc23 += a2 * b3
        acc30 += a3 * b0
        acc31 += a3 * b1
        acc32 += a3 * b2
        acc33 += a3 * b3

    if tid < 128:
        a_row = tid // 2
        a_part = tid % 2
        a_byte_offset = S.convert(((block_m + a_row) * K + 2 * K_CHUNK + a_part * 8) * 2, S.i32)
        a_packed = S.amdgpu.raw_buffer_load_x4(a_rsrc, a_byte_offset, zero, 0)
        for j in S.range(4):
            a_frags0[tid, j] = a_packed[j]
    else:
        b_frag = tid - 128
        b_k = b_frag // 8
        b_part = b_frag % 8
        b_byte_offset = S.convert(((2 * K_CHUNK + b_k) * N + block_n + b_part * 8) * 2, S.i32)
        b_packed = S.amdgpu.raw_buffer_load_x4(b_rsrc, b_byte_offset, zero, 0)
        for j in S.range(4):
            b_frags0[b_frag, j] = b_packed[j]

    a_mfma = S.view(a_frags1[wave_row * 64 + lane], S.Tensor((2, 4, 1), S.bf16))
    b_mfma = S.view(
        b_frags1[(lane // 4) * 8 + wave_col * 4 + (lane % 4)],
        S.Tensor((2, 4, 1), S.bf16),
    )
    mfma_acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_mfma[0], b_mfma[0], mfma_acc)
    mfma_acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_mfma[1], b_mfma[1], mfma_acc)
    for kk in S.range(K_CHUNK):
        global_k = K_CHUNK + kk
        a0 = S.convert(A[row_base + 0, global_k], S.f32)
        a1 = S.convert(A[row_base + 1, global_k], S.f32)
        a2 = S.convert(A[row_base + 2, global_k], S.f32)
        a3 = S.convert(A[row_base + 3, global_k], S.f32)
        b0 = S.convert(B[global_k, col_base + 0], S.f32)
        b1 = S.convert(B[global_k, col_base + 1], S.f32)
        b2 = S.convert(B[global_k, col_base + 2], S.f32)
        b3 = S.convert(B[global_k, col_base + 3], S.f32)
        acc00 += a0 * b0
        acc01 += a0 * b1
        acc02 += a0 * b2
        acc03 += a0 * b3
        acc10 += a1 * b0
        acc11 += a1 * b1
        acc12 += a1 * b2
        acc13 += a1 * b3
        acc20 += a2 * b0
        acc21 += a2 * b1
        acc22 += a2 * b2
        acc23 += a2 * b3
        acc30 += a3 * b0
        acc31 += a3 * b1
        acc32 += a3 * b2
        acc33 += a3 * b3

    if tid < 128:
        a_row = tid // 2
        a_part = tid % 2
        a_byte_offset = S.convert(((block_m + a_row) * K + 3 * K_CHUNK + a_part * 8) * 2, S.i32)
        a_packed = S.amdgpu.raw_buffer_load_x4(a_rsrc, a_byte_offset, zero, 0)
        for j in S.range(4):
            a_frags1[tid, j] = a_packed[j]
    else:
        b_frag = tid - 128
        b_k = b_frag // 8
        b_part = b_frag % 8
        b_byte_offset = S.convert(((3 * K_CHUNK + b_k) * N + block_n + b_part * 8) * 2, S.i32)
        b_packed = S.amdgpu.raw_buffer_load_x4(b_rsrc, b_byte_offset, zero, 0)
        for j in S.range(4):
            b_frags1[b_frag, j] = b_packed[j]

    S.syncthreads()

    a_mfma = S.view(a_frags0[wave_row * 64 + lane], S.Tensor((2, 4, 1), S.bf16))
    b_mfma = S.view(
        b_frags0[(lane // 4) * 8 + wave_col * 4 + (lane % 4)],
        S.Tensor((2, 4, 1), S.bf16),
    )
    mfma_acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_mfma[0], b_mfma[0], mfma_acc)
    mfma_acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_mfma[1], b_mfma[1], mfma_acc)
    for kk in S.range(K_CHUNK):
        global_k = 2 * K_CHUNK + kk
        a0 = S.convert(A[row_base + 0, global_k], S.f32)
        a1 = S.convert(A[row_base + 1, global_k], S.f32)
        a2 = S.convert(A[row_base + 2, global_k], S.f32)
        a3 = S.convert(A[row_base + 3, global_k], S.f32)
        b0 = S.convert(B[global_k, col_base + 0], S.f32)
        b1 = S.convert(B[global_k, col_base + 1], S.f32)
        b2 = S.convert(B[global_k, col_base + 2], S.f32)
        b3 = S.convert(B[global_k, col_base + 3], S.f32)
        acc00 += a0 * b0
        acc01 += a0 * b1
        acc02 += a0 * b2
        acc03 += a0 * b3
        acc10 += a1 * b0
        acc11 += a1 * b1
        acc12 += a1 * b2
        acc13 += a1 * b3
        acc20 += a2 * b0
        acc21 += a2 * b1
        acc22 += a2 * b2
        acc23 += a2 * b3
        acc30 += a3 * b0
        acc31 += a3 * b1
        acc32 += a3 * b2
        acc33 += a3 * b3

    a_mfma = S.view(a_frags1[wave_row * 64 + lane], S.Tensor((2, 4, 1), S.bf16))
    b_mfma = S.view(
        b_frags1[(lane // 4) * 8 + wave_col * 4 + (lane % 4)],
        S.Tensor((2, 4, 1), S.bf16),
    )
    mfma_acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_mfma[0], b_mfma[0], mfma_acc)
    mfma_acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_mfma[1], b_mfma[1], mfma_acc)
    for kk in S.range(K_CHUNK):
        global_k = 3 * K_CHUNK + kk
        a0 = S.convert(A[row_base + 0, global_k], S.f32)
        a1 = S.convert(A[row_base + 1, global_k], S.f32)
        a2 = S.convert(A[row_base + 2, global_k], S.f32)
        a3 = S.convert(A[row_base + 3, global_k], S.f32)
        b0 = S.convert(B[global_k, col_base + 0], S.f32)
        b1 = S.convert(B[global_k, col_base + 1], S.f32)
        b2 = S.convert(B[global_k, col_base + 2], S.f32)
        b3 = S.convert(B[global_k, col_base + 3], S.f32)
        acc00 += a0 * b0
        acc01 += a0 * b1
        acc02 += a0 * b2
        acc03 += a0 * b3
        acc10 += a1 * b0
        acc11 += a1 * b1
        acc12 += a1 * b2
        acc13 += a1 * b3
        acc20 += a2 * b0
        acc21 += a2 * b1
        acc22 += a2 * b2
        acc23 += a2 * b3
        acc30 += a3 * b0
        acc31 += a3 * b1
        acc32 += a3 * b2
        acc33 += a3 * b3

    C[row_base + 0, col_base + 0] = S.convert(acc00, S.bf16)
    C[row_base + 0, col_base + 1] = S.convert(acc01, S.bf16)
    C[row_base + 0, col_base + 2] = S.convert(acc02, S.bf16)
    C[row_base + 0, col_base + 3] = S.convert(acc03, S.bf16)
    C[row_base + 1, col_base + 0] = S.convert(acc10, S.bf16)
    C[row_base + 1, col_base + 1] = S.convert(acc11, S.bf16)
    C[row_base + 1, col_base + 2] = S.convert(acc12, S.bf16)
    C[row_base + 1, col_base + 3] = S.convert(acc13, S.bf16)
    C[row_base + 2, col_base + 0] = S.convert(acc20, S.bf16)
    C[row_base + 2, col_base + 1] = S.convert(acc21, S.bf16)
    C[row_base + 2, col_base + 2] = S.convert(acc22, S.bf16)
    C[row_base + 2, col_base + 3] = S.convert(acc23, S.bf16)
    C[row_base + 3, col_base + 0] = S.convert(acc30, S.bf16)
    C[row_base + 3, col_base + 1] = S.convert(acc31, S.bf16)
    C[row_base + 3, col_base + 2] = S.convert(acc32, S.bf16)
    C[row_base + 3, col_base + 3] = S.convert(acc33, S.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if tuple(A.shape) != (M, K) or tuple(B.shape) != (K, N):
            raise ValueError(f"Expected A {(M, K)} and B {(K, N)}, got {tuple(A.shape)} and {tuple(B.shape)}")
        if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
            raise ValueError(f"Expected bf16 inputs, got {A.dtype} and {B.dtype}")
        if A.device != B.device:
            raise ValueError("A and B must be on the same device")

        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((M, N), device=A.device, dtype=torch.bfloat16)
        gemm_kernel[lambda: ((N // BLOCK_N, M // BLOCK_M, 1), (THREADS_PER_BLOCK, 1, 1))](A, B, C)
        return C
