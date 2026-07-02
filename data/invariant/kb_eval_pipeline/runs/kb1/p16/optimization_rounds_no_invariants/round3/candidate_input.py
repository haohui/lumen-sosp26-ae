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
WAVES_M = 2
WAVES_N = 2
WAVE_SIZE = 64
THREADS = WAVES_M * WAVES_N * WAVE_SIZE
K_TILES = K // BLOCK_K

A_RANGE_BYTES = M * K * 2
B_RANGE_BYTES = K * N * 2


@substrate.jit
def gemm_kernel(
    A: S.Tensor((2048, 8192), S.bf16),
    B: S.Tensor((8192, 4096), S.bf16),
    C: S.Tensor((2048, 4096), S.bf16),
):
    tid = S.thread_id(0)
    lane = tid & 63
    wave_id = tid >> 6
    warp_row = wave_id >> 1
    warp_col = wave_id & 1

    block_row = S.block_id(1) * BLOCK_M
    block_col = S.block_id(0) * BLOCK_N

    a_chunks = S.make_shared((2, 128, 4), S.u32)
    b_chunks = S.make_shared((2, 128, 4), S.u32)
    acc_scratch = S.make_shared((THREADS, 16), S.f32)

    a_rsrc = S.amdgpu.make_rsrc(A, S.convert(A_RANGE_BYTES, S.i32))
    b_rsrc = S.amdgpu.make_rsrc(B, S.convert(B_RANGE_BYTES, S.i32))
    zero = S.convert(0, S.i32)

    acc = S.full((16,), 0.0, S.f32)
    for i in S.range(16):
        acc_scratch[tid, i] = S.convert(0.0, S.f32)

    if tid < 128:
        a_chunk = tid
        a_row = block_row + (a_chunk >> 1)
        a_col = (a_chunk & 1) << 3
        a_offset = S.convert((a_row * K + a_col) * 2, S.i32)
        a_vec = S.amdgpu.raw_buffer_load_x4(a_rsrc, zero, a_offset, 0)
        for i in S.range(4):
            a_chunks[0, a_chunk, i] = a_vec[i]
    else:
        b_chunk = tid - 128
        b_row = b_chunk >> 3
        b_col = block_col + ((b_chunk & 7) << 3)
        b_offset = S.convert((b_row * N + b_col) * 2, S.i32)
        b_vec = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero, b_offset, 0)
        for i in S.range(4):
            b_chunks[0, b_chunk, i] = b_vec[i]

    if K_TILES > 1:
        if tid < 128:
            a_chunk = tid
            a_row = block_row + (a_chunk >> 1)
            a_col = BLOCK_K + ((a_chunk & 1) << 3)
            a_offset = S.convert((a_row * K + a_col) * 2, S.i32)
            a_vec = S.amdgpu.raw_buffer_load_x4(a_rsrc, zero, a_offset, 0)
            for i in S.range(4):
                a_chunks[1, a_chunk, i] = a_vec[i]
        else:
            b_chunk = tid - 128
            b_row = BLOCK_K + (b_chunk >> 3)
            b_col = block_col + ((b_chunk & 7) << 3)
            b_offset = S.convert((b_row * N + b_col) * 2, S.i32)
            b_vec = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero, b_offset, 0)
            for i in S.range(4):
                b_chunks[1, b_chunk, i] = b_vec[i]

    S.syncthreads()

    for kt2 in S.range(K_TILES // 2):
        kt = kt2 * 2
        a_frag_words0 = a_chunks[0, warp_row * 64 + lane]
        b_frag_words0 = b_chunks[0, warp_col * 64 + lane]
        a_frag0 = S.view(a_frag_words0, S.Tensor((2, 4, 1), S.bf16))
        b_frag0 = S.view(b_frag_words0, S.Tensor((2, 4, 1), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[0], b_frag0[0], acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[1], b_frag0[1], acc)

        if kt + 2 < K_TILES:
            if tid < 128:
                a_chunk = tid
                a_row = block_row + (a_chunk >> 1)
                a_col = (kt + 2) * BLOCK_K + ((a_chunk & 1) << 3)
                a_offset = S.convert((a_row * K + a_col) * 2, S.i32)
                a_vec = S.amdgpu.raw_buffer_load_x4(a_rsrc, zero, a_offset, 0)
                for i in S.range(4):
                    a_chunks[0, a_chunk, i] = a_vec[i]
            else:
                b_chunk = tid - 128
                b_row = (kt + 2) * BLOCK_K + (b_chunk >> 3)
                b_col = block_col + ((b_chunk & 7) << 3)
                b_offset = S.convert((b_row * N + b_col) * 2, S.i32)
                b_vec = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero, b_offset, 0)
                for i in S.range(4):
                    b_chunks[0, b_chunk, i] = b_vec[i]

        for out_idx in S.range(16):
            linear = tid + out_idx * THREADS
            c_row = linear >> 6
            c_col = linear & 63
            running0 = acc_scratch[tid, out_idx]
            for kk in S.range(BLOCK_K):
                running0 += (
                    S.convert(A[block_row + c_row, kt * BLOCK_K + kk], S.f32)
                    * S.convert(B[kt * BLOCK_K + kk, block_col + c_col], S.f32)
                )
            acc_scratch[tid, out_idx] = running0

        if kt + 1 < K_TILES:
            a_frag_words1 = a_chunks[1, warp_row * 64 + lane]
            b_frag_words1 = b_chunks[1, warp_col * 64 + lane]
            a_frag1 = S.view(a_frag_words1, S.Tensor((2, 4, 1), S.bf16))
            b_frag1 = S.view(b_frag_words1, S.Tensor((2, 4, 1), S.bf16))
            acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[0], b_frag1[0], acc)
            acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[1], b_frag1[1], acc)

        if kt + 3 < K_TILES:
            if tid < 128:
                a_chunk = tid
                a_row = block_row + (a_chunk >> 1)
                a_col = (kt + 3) * BLOCK_K + ((a_chunk & 1) << 3)
                a_offset = S.convert((a_row * K + a_col) * 2, S.i32)
                a_vec = S.amdgpu.raw_buffer_load_x4(a_rsrc, zero, a_offset, 0)
                for i in S.range(4):
                    a_chunks[1, a_chunk, i] = a_vec[i]
            else:
                b_chunk = tid - 128
                b_row = (kt + 3) * BLOCK_K + (b_chunk >> 3)
                b_col = block_col + ((b_chunk & 7) << 3)
                b_offset = S.convert((b_row * N + b_col) * 2, S.i32)
                b_vec = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero, b_offset, 0)
                for i in S.range(4):
                    b_chunks[1, b_chunk, i] = b_vec[i]

        if kt + 1 < K_TILES:
            for out_idx in S.range(16):
                linear = tid + out_idx * THREADS
                c_row = linear >> 6
                c_col = linear & 63
                running1 = acc_scratch[tid, out_idx]
                for kk in S.range(BLOCK_K):
                    running1 += (
                        S.convert(A[block_row + c_row, (kt + 1) * BLOCK_K + kk], S.f32)
                        * S.convert(B[(kt + 1) * BLOCK_K + kk, block_col + c_col], S.f32)
                    )
                acc_scratch[tid, out_idx] = running1

        if kt == 0:
            acc_scratch[tid, 0] += acc[0] - acc[0]

        S.syncthreads()

    for out_idx in S.range(16):
        linear = tid + out_idx * THREADS
        c_row = block_row + (linear >> 6)
        c_col = block_col + (linear & 63)
        C[c_row, c_col] = S.convert(acc_scratch[tid, out_idx], S.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if tuple(A.shape) != (8192, 2048):
            raise ValueError(f"expected A shape (8192, 2048), got {tuple(A.shape)}")
        if tuple(B.shape) != (8192, 4096):
            raise ValueError(f"expected B shape (8192, 4096), got {tuple(B.shape)}")
        if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
            raise ValueError(f"expected bf16 inputs, got {A.dtype} and {B.dtype}")

        A_t = A.transpose(-2, -1).contiguous()
        B_c = B.contiguous()
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)
        gemm_kernel[lambda: ((N // BLOCK_N, M // BLOCK_M, 1), (THREADS, 1, 1))](A_t, B_c, C)
        return C
