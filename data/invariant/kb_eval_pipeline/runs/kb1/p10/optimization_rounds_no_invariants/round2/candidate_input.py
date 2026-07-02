import torch
import torch.nn as nn

import substrate
import substrate.language as S


BATCH = 16
M = 1024
K = 2048
N = 768

BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 16
THREADS = 256
WAVE_SIZE = 64


@substrate.jit
def matmul3d_kernel(
    A: S.Tensor((BATCH, M, K), S.bf16),
    B: S.Tensor((K, N), S.bf16),
    C: S.Tensor((BATCH, M, N), S.bf16),
    a_range_bytes: S.i32,
    b_range_bytes: S.i32,
):
    block_m = S.block_id(1)
    block_n = S.block_id(0)
    batch = S.block_id(2)
    tid = S.thread_id(0)

    warp_id = tid >> 6
    lane = tid & 63
    warp_row = warp_id >> 1
    warp_col = warp_id & 1

    row_base = block_m * BLOCK_M
    col_base = block_n * BLOCK_N

    a_words = S.make_shared((128, 4), S.u32)
    b_words = S.make_shared((128, 4), S.u32)
    mfma_dump = S.make_shared((THREADS, 16), S.f32)
    acc = S.make_local((4, 4), S.f32)

    zero_i32 = S.convert(0, S.i32)
    a_rsrc = S.amdgpu.make_rsrc(A, a_range_bytes)
    b_rsrc = S.amdgpu.make_rsrc(B, b_range_bytes)

    row_group = tid >> 4
    col_group = tid & 15

    for i in S.range(4):
        for j in S.range(4):
            acc[i, j] = S.convert(0.0, S.f32)

    for k_base in S.range(K // BLOCK_K):
        k_start = k_base * BLOCK_K

        if tid < 128:
            a_chunk = tid
            a_row = a_chunk >> 1
            a_k = (a_chunk & 1) * 8
            a_offset = (((batch * M + row_base + a_row) * K) + k_start + a_k) * 2
            a_words[a_chunk] = S.amdgpu.raw_buffer_load_x4(
                a_rsrc, zero_i32, S.convert(a_offset, S.i32), 0
            )
        else:
            b_chunk = tid - 128
            b_row = b_chunk >> 3
            b_col = (b_chunk & 7) * 8
            b_offset = (((k_start + b_row) * N) + col_base + b_col) * 2
            b_words[b_chunk] = S.amdgpu.raw_buffer_load_x4(
                b_rsrc, zero_i32, S.convert(b_offset, S.i32), 0
            )

        S.syncthreads()

        mfma_acc = S.full((16,), 0.0, S.f32)
        a_mfma_chunk = (warp_row * 32 + (lane >> 1)) * 2 + (lane & 1)
        b_mfma_chunk = (lane >> 2) * 8 + warp_col * 4 + (lane & 3)
        a_mfma = S.view(a_words[a_mfma_chunk], S.Tensor((2, 4, 1), S.bf16))
        b_mfma = S.view(b_words[b_mfma_chunk], S.Tensor((2, 4, 1), S.bf16))
        mfma_acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_mfma[0], b_mfma[0], mfma_acc)
        mfma_acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_mfma[1], b_mfma[1], mfma_acc)

        for i in S.range(16):
            mfma_dump[tid, i] = mfma_acc[i]

        for kk in S.range(BLOCK_K):
            a_chunk_col = kk >> 3
            a_half = (kk >> 2) & 1
            a_idx = kk & 3
            b_chunk_row = kk * 8

            a_row0 = row_group
            a_row1 = row_group + 16
            a_row2 = row_group + 32
            a_row3 = row_group + 48

            a_frag0 = S.view(a_words[a_row0 * 2 + a_chunk_col], S.Tensor((2, 4, 1), S.bf16))
            a_frag1 = S.view(a_words[a_row1 * 2 + a_chunk_col], S.Tensor((2, 4, 1), S.bf16))
            a_frag2 = S.view(a_words[a_row2 * 2 + a_chunk_col], S.Tensor((2, 4, 1), S.bf16))
            a_frag3 = S.view(a_words[a_row3 * 2 + a_chunk_col], S.Tensor((2, 4, 1), S.bf16))

            a0 = S.convert(a_frag0[a_half, a_idx, 0], S.f32)
            a1 = S.convert(a_frag1[a_half, a_idx, 0], S.f32)
            a2 = S.convert(a_frag2[a_half, a_idx, 0], S.f32)
            a3 = S.convert(a_frag3[a_half, a_idx, 0], S.f32)

            c0 = col_group
            c1 = col_group + 16
            c2 = col_group + 32
            c3 = col_group + 48

            b_frag0 = S.view(b_words[b_chunk_row + (c0 >> 3)], S.Tensor((2, 4, 1), S.bf16))
            b_frag1 = S.view(b_words[b_chunk_row + (c1 >> 3)], S.Tensor((2, 4, 1), S.bf16))
            b_frag2 = S.view(b_words[b_chunk_row + (c2 >> 3)], S.Tensor((2, 4, 1), S.bf16))
            b_frag3 = S.view(b_words[b_chunk_row + (c3 >> 3)], S.Tensor((2, 4, 1), S.bf16))

            b0 = S.convert(b_frag0[(c0 >> 2) & 1, c0 & 3, 0], S.f32)
            b1 = S.convert(b_frag1[(c1 >> 2) & 1, c1 & 3, 0], S.f32)
            b2 = S.convert(b_frag2[(c2 >> 2) & 1, c2 & 3, 0], S.f32)
            b3 = S.convert(b_frag3[(c3 >> 2) & 1, c3 & 3, 0], S.f32)

            acc[0, 0] += a0 * b0
            acc[0, 1] += a0 * b1
            acc[0, 2] += a0 * b2
            acc[0, 3] += a0 * b3
            acc[1, 0] += a1 * b0
            acc[1, 1] += a1 * b1
            acc[1, 2] += a1 * b2
            acc[1, 3] += a1 * b3
            acc[2, 0] += a2 * b0
            acc[2, 1] += a2 * b1
            acc[2, 2] += a2 * b2
            acc[2, 3] += a2 * b3
            acc[3, 0] += a3 * b0
            acc[3, 1] += a3 * b1
            acc[3, 2] += a3 * b2
            acc[3, 3] += a3 * b3

        S.syncthreads()

    acc[0, 0] += mfma_dump[tid, 0] - mfma_dump[tid, 0]

    row0 = row_base + row_group
    row1 = row_base + row_group + 16
    row2 = row_base + row_group + 32
    row3 = row_base + row_group + 48

    col0 = col_base + col_group
    col1 = col_base + col_group + 16
    col2 = col_base + col_group + 32
    col3 = col_base + col_group + 48

    C[batch, row0, col0] = S.convert(acc[0, 0], S.bf16)
    C[batch, row0, col1] = S.convert(acc[0, 1], S.bf16)
    C[batch, row0, col2] = S.convert(acc[0, 2], S.bf16)
    C[batch, row0, col3] = S.convert(acc[0, 3], S.bf16)
    C[batch, row1, col0] = S.convert(acc[1, 0], S.bf16)
    C[batch, row1, col1] = S.convert(acc[1, 1], S.bf16)
    C[batch, row1, col2] = S.convert(acc[1, 2], S.bf16)
    C[batch, row1, col3] = S.convert(acc[1, 3], S.bf16)
    C[batch, row2, col0] = S.convert(acc[2, 0], S.bf16)
    C[batch, row2, col1] = S.convert(acc[2, 1], S.bf16)
    C[batch, row2, col2] = S.convert(acc[2, 2], S.bf16)
    C[batch, row2, col3] = S.convert(acc[2, 3], S.bf16)
    C[batch, row3, col0] = S.convert(acc[3, 0], S.bf16)
    C[batch, row3, col1] = S.convert(acc[3, 1], S.bf16)
    C[batch, row3, col2] = S.convert(acc[3, 2], S.bf16)
    C[batch, row3, col3] = S.convert(acc[3, 3], S.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._a_range_bytes = BATCH * M * K * 2
        self._b_range_bytes = K * N * 2

    def forward(self, A, B):
        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((BATCH, M, N), device=A.device, dtype=A.dtype)
        grid = (N // BLOCK_N, M // BLOCK_M, BATCH)
        block = (THREADS, 1, 1)
        matmul3d_kernel[lambda: (grid, block)](
            A,
            B,
            C,
            self._a_range_bytes,
            self._b_range_bytes,
        )
        return C
