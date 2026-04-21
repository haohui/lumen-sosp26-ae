import torch
import torch.nn as nn

import substrate
import substrate.language as S


M = 8205
K = 2949
N = 5921

BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 16
HALF_BLOCK_K = 8
SUPER_BLOCK_K = 32
THREADS = 256
THREAD_TILE_M = 4
THREAD_TILE_N = 4


@substrate.jit
def gemm_kernel(
    A: S.Tensor((8205, 2949), S.bf16),
    B: S.Tensor((2949, 5921), S.bf16),
    C: S.Tensor((8205, 5921), S.bf16),
):
    pid_m = S.block_id(1)
    pid_n = S.block_id(0)
    tid = S.thread_id(0)

    tile_m = pid_m * BLOCK_M
    tile_n = pid_n * BLOCK_N

    thread_row = tid // 16
    thread_col = tid % 16
    row_base = thread_row * THREAD_TILE_M
    col_base = thread_col * THREAD_TILE_N

    a_shared = S.make_shared((2, BLOCK_M, BLOCK_K), S.bf16)
    b_shared = S.make_shared((2, BLOCK_K, BLOCK_N), S.bf16)

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

    if tid < 128:
        a_chunk = tid
        a_row = a_chunk // 2
        a_chunk_col = a_chunk % 2
        a_col_base = a_chunk_col * 8
        g_row = tile_m + a_row
        g_col = a_col_base

        a_row_limit = S.min((g_row + 1) * K, M * K) * 2
        a_rsrc = S.amdgpu.make_rsrc(A, a_row_limit)
        a_words = S.amdgpu.raw_buffer_load_x4(a_rsrc, (g_row * K + g_col) * 2, 0, 0)
        a_vals = S.view(a_words, S.Tensor((2, 4, 1), S.bf16))
        for t in S.range(4):
            a_shared[0, a_row, a_col_base + t] = a_vals[0, t, 0]
            a_shared[0, a_row, a_col_base + 4 + t] = a_vals[1, t, 0]
    else:
        b_chunk = tid - 128
        b_row = b_chunk // 8
        b_chunk_col = b_chunk % 8
        b_col_base = b_chunk_col * 8
        g_row = b_row
        g_col = tile_n + b_col_base

        b_row_limit = S.min((g_row + 1) * N, K * N) * 2
        b_rsrc = S.amdgpu.make_rsrc(B, b_row_limit)
        b_words = S.amdgpu.raw_buffer_load_x4(b_rsrc, (g_row * N + g_col) * 2, 0, 0)
        b_vals = S.view(b_words, S.Tensor((2, 4, 1), S.bf16))
        for t in S.range(4):
            b_shared[0, b_row, b_col_base + t] = b_vals[0, t, 0]
            b_shared[0, b_row, b_col_base + 4 + t] = b_vals[1, t, 0]

    S.syncthreads()

    for kk in S.range(0, K, SUPER_BLOCK_K):
        for k_inner in S.range(0, HALF_BLOCK_K):
            a0 = S.convert(a_shared[0, row_base + 0, k_inner], S.f32)
            a1 = S.convert(a_shared[0, row_base + 1, k_inner], S.f32)
            a2 = S.convert(a_shared[0, row_base + 2, k_inner], S.f32)
            a3 = S.convert(a_shared[0, row_base + 3, k_inner], S.f32)

            b0 = S.convert(b_shared[0, k_inner, col_base + 0], S.f32)
            b1 = S.convert(b_shared[0, k_inner, col_base + 1], S.f32)
            b2 = S.convert(b_shared[0, k_inner, col_base + 2], S.f32)
            b3 = S.convert(b_shared[0, k_inner, col_base + 3], S.f32)

            acc00 = acc00 + a0 * b0
            acc01 = acc01 + a0 * b1
            acc02 = acc02 + a0 * b2
            acc03 = acc03 + a0 * b3
            acc10 = acc10 + a1 * b0
            acc11 = acc11 + a1 * b1
            acc12 = acc12 + a1 * b2
            acc13 = acc13 + a1 * b3
            acc20 = acc20 + a2 * b0
            acc21 = acc21 + a2 * b1
            acc22 = acc22 + a2 * b2
            acc23 = acc23 + a2 * b3
            acc30 = acc30 + a3 * b0
            acc31 = acc31 + a3 * b1
            acc32 = acc32 + a3 * b2
            acc33 = acc33 + a3 * b3

        if kk + BLOCK_K < K:
            if tid < 128:
                a_chunk = tid
                a_row = a_chunk // 2
                a_chunk_col = a_chunk % 2
                a_col_base = a_chunk_col * 8
                g_row = tile_m + a_row
                g_col = kk + BLOCK_K + a_col_base

                a_row_limit = S.min((g_row + 1) * K, M * K) * 2
                a_rsrc = S.amdgpu.make_rsrc(A, a_row_limit)
                a_words = S.amdgpu.raw_buffer_load_x4(a_rsrc, (g_row * K + g_col) * 2, 0, 0)
                a_vals = S.view(a_words, S.Tensor((2, 4, 1), S.bf16))
                for t in S.range(4):
                    a_shared[1, a_row, a_col_base + t] = a_vals[0, t, 0]
                    a_shared[1, a_row, a_col_base + 4 + t] = a_vals[1, t, 0]
            else:
                b_chunk = tid - 128
                b_row = b_chunk // 8
                b_chunk_col = b_chunk % 8
                b_col_base = b_chunk_col * 8
                g_row = kk + BLOCK_K + b_row
                g_col = tile_n + b_col_base

                b_row_limit = S.min((g_row + 1) * N, K * N) * 2
                b_rsrc = S.amdgpu.make_rsrc(B, b_row_limit)
                b_words = S.amdgpu.raw_buffer_load_x4(b_rsrc, (g_row * N + g_col) * 2, 0, 0)
                b_vals = S.view(b_words, S.Tensor((2, 4, 1), S.bf16))
                for t in S.range(4):
                    b_shared[1, b_row, b_col_base + t] = b_vals[0, t, 0]
                    b_shared[1, b_row, b_col_base + 4 + t] = b_vals[1, t, 0]

        for k_inner in S.range(HALF_BLOCK_K, BLOCK_K):
            a0 = S.convert(a_shared[0, row_base + 0, k_inner], S.f32)
            a1 = S.convert(a_shared[0, row_base + 1, k_inner], S.f32)
            a2 = S.convert(a_shared[0, row_base + 2, k_inner], S.f32)
            a3 = S.convert(a_shared[0, row_base + 3, k_inner], S.f32)

            b0 = S.convert(b_shared[0, k_inner, col_base + 0], S.f32)
            b1 = S.convert(b_shared[0, k_inner, col_base + 1], S.f32)
            b2 = S.convert(b_shared[0, k_inner, col_base + 2], S.f32)
            b3 = S.convert(b_shared[0, k_inner, col_base + 3], S.f32)

            acc00 = acc00 + a0 * b0
            acc01 = acc01 + a0 * b1
            acc02 = acc02 + a0 * b2
            acc03 = acc03 + a0 * b3
            acc10 = acc10 + a1 * b0
            acc11 = acc11 + a1 * b1
            acc12 = acc12 + a1 * b2
            acc13 = acc13 + a1 * b3
            acc20 = acc20 + a2 * b0
            acc21 = acc21 + a2 * b1
            acc22 = acc22 + a2 * b2
            acc23 = acc23 + a2 * b3
            acc30 = acc30 + a3 * b0
            acc31 = acc31 + a3 * b1
            acc32 = acc32 + a3 * b2
            acc33 = acc33 + a3 * b3

        S.syncthreads()

        if kk + BLOCK_K < K:
            for k_inner in S.range(0, HALF_BLOCK_K):
                a0 = S.convert(a_shared[1, row_base + 0, k_inner], S.f32)
                a1 = S.convert(a_shared[1, row_base + 1, k_inner], S.f32)
                a2 = S.convert(a_shared[1, row_base + 2, k_inner], S.f32)
                a3 = S.convert(a_shared[1, row_base + 3, k_inner], S.f32)

                b0 = S.convert(b_shared[1, k_inner, col_base + 0], S.f32)
                b1 = S.convert(b_shared[1, k_inner, col_base + 1], S.f32)
                b2 = S.convert(b_shared[1, k_inner, col_base + 2], S.f32)
                b3 = S.convert(b_shared[1, k_inner, col_base + 3], S.f32)

                acc00 = acc00 + a0 * b0
                acc01 = acc01 + a0 * b1
                acc02 = acc02 + a0 * b2
                acc03 = acc03 + a0 * b3
                acc10 = acc10 + a1 * b0
                acc11 = acc11 + a1 * b1
                acc12 = acc12 + a1 * b2
                acc13 = acc13 + a1 * b3
                acc20 = acc20 + a2 * b0
                acc21 = acc21 + a2 * b1
                acc22 = acc22 + a2 * b2
                acc23 = acc23 + a2 * b3
                acc30 = acc30 + a3 * b0
                acc31 = acc31 + a3 * b1
                acc32 = acc32 + a3 * b2
                acc33 = acc33 + a3 * b3

            if kk + SUPER_BLOCK_K < K:
                if tid < 128:
                    a_chunk = tid
                    a_row = a_chunk // 2
                    a_chunk_col = a_chunk % 2
                    a_col_base = a_chunk_col * 8
                    g_row = tile_m + a_row
                    g_col = kk + SUPER_BLOCK_K + a_col_base

                    a_row_limit = S.min((g_row + 1) * K, M * K) * 2
                    a_rsrc = S.amdgpu.make_rsrc(A, a_row_limit)
                    a_words = S.amdgpu.raw_buffer_load_x4(a_rsrc, (g_row * K + g_col) * 2, 0, 0)
                    a_vals = S.view(a_words, S.Tensor((2, 4, 1), S.bf16))
                    for t in S.range(4):
                        a_shared[0, a_row, a_col_base + t] = a_vals[0, t, 0]
                        a_shared[0, a_row, a_col_base + 4 + t] = a_vals[1, t, 0]
                else:
                    b_chunk = tid - 128
                    b_row = b_chunk // 8
                    b_chunk_col = b_chunk % 8
                    b_col_base = b_chunk_col * 8
                    g_row = kk + SUPER_BLOCK_K + b_row
                    g_col = tile_n + b_col_base

                    b_row_limit = S.min((g_row + 1) * N, K * N) * 2
                    b_rsrc = S.amdgpu.make_rsrc(B, b_row_limit)
                    b_words = S.amdgpu.raw_buffer_load_x4(b_rsrc, (g_row * N + g_col) * 2, 0, 0)
                    b_vals = S.view(b_words, S.Tensor((2, 4, 1), S.bf16))
                    for t in S.range(4):
                        b_shared[0, b_row, b_col_base + t] = b_vals[0, t, 0]
                        b_shared[0, b_row, b_col_base + 4 + t] = b_vals[1, t, 0]

            for k_inner in S.range(HALF_BLOCK_K, BLOCK_K):
                a0 = S.convert(a_shared[1, row_base + 0, k_inner], S.f32)
                a1 = S.convert(a_shared[1, row_base + 1, k_inner], S.f32)
                a2 = S.convert(a_shared[1, row_base + 2, k_inner], S.f32)
                a3 = S.convert(a_shared[1, row_base + 3, k_inner], S.f32)

                b0 = S.convert(b_shared[1, k_inner, col_base + 0], S.f32)
                b1 = S.convert(b_shared[1, k_inner, col_base + 1], S.f32)
                b2 = S.convert(b_shared[1, k_inner, col_base + 2], S.f32)
                b3 = S.convert(b_shared[1, k_inner, col_base + 3], S.f32)

                acc00 = acc00 + a0 * b0
                acc01 = acc01 + a0 * b1
                acc02 = acc02 + a0 * b2
                acc03 = acc03 + a0 * b3
                acc10 = acc10 + a1 * b0
                acc11 = acc11 + a1 * b1
                acc12 = acc12 + a1 * b2
                acc13 = acc13 + a1 * b3
                acc20 = acc20 + a2 * b0
                acc21 = acc21 + a2 * b1
                acc22 = acc22 + a2 * b2
                acc23 = acc23 + a2 * b3
                acc30 = acc30 + a3 * b0
                acc31 = acc31 + a3 * b1
                acc32 = acc32 + a3 * b2
                acc33 = acc33 + a3 * b3

            S.syncthreads()

    g_row0 = tile_m + row_base + 0
    g_row1 = tile_m + row_base + 1
    g_row2 = tile_m + row_base + 2
    g_row3 = tile_m + row_base + 3
    g_col0 = tile_n + col_base + 0
    g_col1 = tile_n + col_base + 1
    g_col2 = tile_n + col_base + 2
    g_col3 = tile_n + col_base + 3

    if g_row0 < M:
        if g_col0 < N:
            C[g_row0, g_col0] = S.convert(acc00, S.bf16)
        if g_col1 < N:
            C[g_row0, g_col1] = S.convert(acc01, S.bf16)
        if g_col2 < N:
            C[g_row0, g_col2] = S.convert(acc02, S.bf16)
        if g_col3 < N:
            C[g_row0, g_col3] = S.convert(acc03, S.bf16)
    if g_row1 < M:
        if g_col0 < N:
            C[g_row1, g_col0] = S.convert(acc10, S.bf16)
        if g_col1 < N:
            C[g_row1, g_col1] = S.convert(acc11, S.bf16)
        if g_col2 < N:
            C[g_row1, g_col2] = S.convert(acc12, S.bf16)
        if g_col3 < N:
            C[g_row1, g_col3] = S.convert(acc13, S.bf16)
    if g_row2 < M:
        if g_col0 < N:
            C[g_row2, g_col0] = S.convert(acc20, S.bf16)
        if g_col1 < N:
            C[g_row2, g_col1] = S.convert(acc21, S.bf16)
        if g_col2 < N:
            C[g_row2, g_col2] = S.convert(acc22, S.bf16)
        if g_col3 < N:
            C[g_row2, g_col3] = S.convert(acc23, S.bf16)
    if g_row3 < M:
        if g_col0 < N:
            C[g_row3, g_col0] = S.convert(acc30, S.bf16)
        if g_col1 < N:
            C[g_row3, g_col1] = S.convert(acc31, S.bf16)
        if g_col2 < N:
            C[g_row3, g_col2] = S.convert(acc32, S.bf16)
        if g_col3 < N:
            C[g_row3, g_col3] = S.convert(acc33, S.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if tuple(A.shape) != (M, K) or tuple(B.shape) != (K, N):
            raise ValueError(f"expected A={(M, K)} and B={(K, N)}, got {tuple(A.shape)} and {tuple(B.shape)}")
        if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
            raise ValueError("expected bfloat16 inputs")

        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)
        gemm_kernel[
            lambda: (((N + BLOCK_N - 1) // BLOCK_N, (M + BLOCK_M - 1) // BLOCK_M, 1), (THREADS, 1, 1))
        ](A, B, C)
        return C
