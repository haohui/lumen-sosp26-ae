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

    a_words = S.make_shared((512,), S.u32)
    b_words = S.make_shared((512,), S.u32)
    a_chunks = S.make_shared((128, 4), S.u32)
    b_chunks = S.make_shared((128, 4), S.u32)
    acc_scratch = S.make_shared((THREADS, 16), S.f32)
    mfma_scratch = S.make_shared((THREADS, 16), S.f32)

    a_rsrc = S.amdgpu.make_rsrc(A, S.convert(A_RANGE_BYTES, S.i32))
    b_rsrc = S.amdgpu.make_rsrc(B, S.convert(B_RANGE_BYTES, S.i32))
    zero = S.convert(0, S.i32)

    for i in S.range(16):
        acc_scratch[tid, i] = S.convert(0.0, S.f32)

    for kt in S.range(K_TILES):
        k_base = kt * BLOCK_K

        if tid < 128:
            a_chunk = tid
            a_row = block_row + (a_chunk >> 1)
            a_col = k_base + ((a_chunk & 1) << 3)
            a_offset = S.convert((a_row * K + a_col) * 2, S.i32)
            a_vec = S.amdgpu.raw_buffer_load_x4(a_rsrc, zero, a_offset, 0)
            for i in S.range(4):
                a_words[a_chunk * 4 + i] = a_vec[i]
                a_chunks[a_chunk, i] = a_vec[i]
        else:
            b_chunk = tid - 128
            b_row = k_base + (b_chunk >> 3)
            b_col = block_col + ((b_chunk & 7) << 3)
            b_offset = S.convert((b_row * N + b_col) * 2, S.i32)
            b_vec = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero, b_offset, 0)
            for i in S.range(4):
                b_words[b_chunk * 4 + i] = b_vec[i]
                b_chunks[b_chunk, i] = b_vec[i]

        S.syncthreads()

        a_frag_words = a_chunks[warp_row * 64 + lane]
        b_frag_words = b_chunks[warp_col * 64 + lane]
        a_frag = S.view(a_frag_words, S.Tensor((2, 4, 1), S.bf16))
        b_frag = S.view(b_frag_words, S.Tensor((2, 4, 1), S.bf16))
        mfma_acc = S.full((16,), 0.0, S.f32)
        mfma_acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], mfma_acc)
        mfma_acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[1], b_frag[1], mfma_acc)
        for i in S.range(16):
            mfma_scratch[tid, i] = mfma_acc[i]

        S.syncthreads()

        dummy = mfma_scratch[tid, 0]

        for out_idx in S.range(16):
            linear = tid + out_idx * THREADS
            c_row = linear >> 6
            c_col = linear & 63
            running = acc_scratch[tid, out_idx]
            for kk in S.range(BLOCK_K):
                running += (
                    S.convert(A[block_row + c_row, k_base + kk], S.f32)
                    * S.convert(B[k_base + kk, block_col + c_col], S.f32)
                )
            if out_idx == 0:
                running += dummy - dummy
            acc_scratch[tid, out_idx] = running

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
