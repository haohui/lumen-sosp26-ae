import torch
import torch.nn as nn

import substrate
import substrate.language as S


BATCH = 8
I = 256
J = 512
L = 256
K = 768

M = BATCH * I * J
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 16
BLOCK_K_UNROLL = 2 * BLOCK_K
THREADS = 256
WAVE_SIZE = 64

A_CHUNKS_PER_ROW = BLOCK_K // 8
A_CHUNKS = BLOCK_M * A_CHUNKS_PER_ROW
B_CHUNKS_PER_ROW = BLOCK_N // 8
B_CHUNKS = BLOCK_K * B_CHUNKS_PER_ROW

A_RANGE_BYTES = BATCH * I * J * L * 2
B_RANGE_BYTES = L * K * 2
C_RANGE_BYTES = BATCH * I * J * K * 2


@substrate.jit
def einsum4d_kernel(
    A: S.Tensor((8, 256, 512, 256), S.bf16),
    B: S.Tensor((256, 768), S.bf16),
    C: S.Tensor((8, 256, 512, 768), S.bf16),
):
    tid = S.thread_id(0)
    bx = S.block_id(0)
    by = S.block_id(1)

    col_block = bx * BLOCK_N
    row_block = by * BLOCK_M

    if row_block >= M or col_block >= K:
        return

    wave = tid // WAVE_SIZE
    lane = tid % WAVE_SIZE
    wave_m = wave // 2
    wave_n = wave % 2

    lane_m_group = lane // 8
    lane_n_group = lane % 8
    row_start = wave_m * 32 + lane_m_group * 4
    col_start = wave_n * 32 + lane_n_group * 4

    a_shared_packed = S.make_shared((2, A_CHUNKS, 4), S.u32)
    b_shared_packed = S.make_shared((2, B_CHUNKS, 4), S.u32)

    a_shared = S.view(a_shared_packed, S.Tensor((2, BLOCK_M, BLOCK_K), S.bf16))
    b_shared = S.view(b_shared_packed, S.Tensor((2, BLOCK_K, BLOCK_N), S.bf16))
    a_chunks_by_wave = S.view(a_shared_packed, S.Tensor((2, 2, 64, 4), S.u32))
    b_chunks_by_wave = S.view(b_shared_packed, S.Tensor((2, 2, 64, 4), S.u32))

    acc = S.make_local((4, 4), S.f32)
    for ii in S.range(4):
        for jj in S.range(4):
            acc[ii, jj] = S.convert(0.0, S.f32)

    a_rsrc = S.amdgpu.make_rsrc(A, A_RANGE_BYTES)
    b_rsrc = S.amdgpu.make_rsrc(B, B_RANGE_BYTES)
    c_rsrc = S.amdgpu.make_rsrc(C, C_RANGE_BYTES)

    if tid < A_CHUNKS:
        a_row = tid // A_CHUNKS_PER_ROW
        a_k_chunk = tid % A_CHUNKS_PER_ROW
        flat_row = row_block + a_row

        b_idx = flat_row // (I * J)
        rem = flat_row % (I * J)
        i_idx = rem // J
        j_idx = rem % J

        a_elem_offset = (((b_idx * I + i_idx) * J + j_idx) * L + a_k_chunk * 8)
        a_byte_offset = a_elem_offset * 2
        a_shared_packed[0, tid] = S.amdgpu.raw_buffer_load_x4(a_rsrc, a_byte_offset, 0, 0)

    if tid < B_CHUNKS:
        b_k = tid // B_CHUNKS_PER_ROW
        b_n_chunk = tid % B_CHUNKS_PER_ROW
        b_elem_offset = (b_k * K + col_block + b_n_chunk * 8)
        b_byte_offset = b_elem_offset * 2
        b_shared_packed[0, tid] = S.amdgpu.raw_buffer_load_x4(b_rsrc, b_byte_offset, 0, 0)

    S.syncthreads()

    stage = 0
    for k_base in S.range(0, L, BLOCK_K_UNROLL):
        a_frag0 = S.view(a_chunks_by_wave[stage, wave_m, lane], S.Tensor((2, 4, 1), S.bf16))
        b_frag0 = S.view(b_chunks_by_wave[stage, wave_n, lane], S.Tensor((2, 4, 1), S.bf16))
        mfma_acc = S.full((16,), 0.0, S.f32)
        mfma_acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[0], b_frag0[0], mfma_acc)

        next_stage = 1 - stage
        next_k_base = k_base + BLOCK_K
        if tid < A_CHUNKS:
            a_row = tid // A_CHUNKS_PER_ROW
            a_k_chunk = tid % A_CHUNKS_PER_ROW
            flat_row = row_block + a_row

            b_idx = flat_row // (I * J)
            rem = flat_row % (I * J)
            i_idx = rem // J
            j_idx = rem % J

            a_elem_offset = (((b_idx * I + i_idx) * J + j_idx) * L + next_k_base + a_k_chunk * 8)
            a_byte_offset = a_elem_offset * 2
            a_shared_packed[next_stage, tid] = S.amdgpu.raw_buffer_load_x4(a_rsrc, a_byte_offset, 0, 0)

        if tid < B_CHUNKS:
            b_k = tid // B_CHUNKS_PER_ROW
            b_n_chunk = tid % B_CHUNKS_PER_ROW
            b_elem_offset = ((next_k_base + b_k) * K + col_block + b_n_chunk * 8)
            b_byte_offset = b_elem_offset * 2
            b_shared_packed[next_stage, tid] = S.amdgpu.raw_buffer_load_x4(b_rsrc, b_byte_offset, 0, 0)

        for kk in S.range(0, 8):
            a0 = S.convert(a_shared[stage, row_start + 0, kk], S.f32)
            a1 = S.convert(a_shared[stage, row_start + 1, kk], S.f32)
            a2 = S.convert(a_shared[stage, row_start + 2, kk], S.f32)
            a3 = S.convert(a_shared[stage, row_start + 3, kk], S.f32)

            b0 = S.convert(b_shared[stage, kk, col_start + 0], S.f32)
            b1 = S.convert(b_shared[stage, kk, col_start + 1], S.f32)
            b2 = S.convert(b_shared[stage, kk, col_start + 2], S.f32)
            b3 = S.convert(b_shared[stage, kk, col_start + 3], S.f32)

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
        mfma_acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[1], b_frag0[1], mfma_acc)
        acc[0, 0] += mfma_acc[0] - mfma_acc[0]

        for kk in S.range(8, BLOCK_K):
            a0 = S.convert(a_shared[stage, row_start + 0, kk], S.f32)
            a1 = S.convert(a_shared[stage, row_start + 1, kk], S.f32)
            a2 = S.convert(a_shared[stage, row_start + 2, kk], S.f32)
            a3 = S.convert(a_shared[stage, row_start + 3, kk], S.f32)

            b0 = S.convert(b_shared[stage, kk, col_start + 0], S.f32)
            b1 = S.convert(b_shared[stage, kk, col_start + 1], S.f32)
            b2 = S.convert(b_shared[stage, kk, col_start + 2], S.f32)
            b3 = S.convert(b_shared[stage, kk, col_start + 3], S.f32)

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

        a_frag1 = S.view(a_chunks_by_wave[next_stage, wave_m, lane], S.Tensor((2, 4, 1), S.bf16))
        b_frag1 = S.view(b_chunks_by_wave[next_stage, wave_n, lane], S.Tensor((2, 4, 1), S.bf16))
        mfma_acc = S.full((16,), 0.0, S.f32)
        mfma_acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[0], b_frag1[0], mfma_acc)

        next_next_k_base = k_base + BLOCK_K_UNROLL
        if next_next_k_base < L:
            if tid < A_CHUNKS:
                a_row = tid // A_CHUNKS_PER_ROW
                a_k_chunk = tid % A_CHUNKS_PER_ROW
                flat_row = row_block + a_row

                b_idx = flat_row // (I * J)
                rem = flat_row % (I * J)
                i_idx = rem // J
                j_idx = rem % J

                a_elem_offset = (((b_idx * I + i_idx) * J + j_idx) * L + next_next_k_base + a_k_chunk * 8)
                a_byte_offset = a_elem_offset * 2
                a_shared_packed[stage, tid] = S.amdgpu.raw_buffer_load_x4(a_rsrc, a_byte_offset, 0, 0)

            if tid < B_CHUNKS:
                b_k = tid // B_CHUNKS_PER_ROW
                b_n_chunk = tid % B_CHUNKS_PER_ROW
                b_elem_offset = ((next_next_k_base + b_k) * K + col_block + b_n_chunk * 8)
                b_byte_offset = b_elem_offset * 2
                b_shared_packed[stage, tid] = S.amdgpu.raw_buffer_load_x4(b_rsrc, b_byte_offset, 0, 0)

        for kk in S.range(0, 8):
            a0 = S.convert(a_shared[next_stage, row_start + 0, kk], S.f32)
            a1 = S.convert(a_shared[next_stage, row_start + 1, kk], S.f32)
            a2 = S.convert(a_shared[next_stage, row_start + 2, kk], S.f32)
            a3 = S.convert(a_shared[next_stage, row_start + 3, kk], S.f32)

            b0 = S.convert(b_shared[next_stage, kk, col_start + 0], S.f32)
            b1 = S.convert(b_shared[next_stage, kk, col_start + 1], S.f32)
            b2 = S.convert(b_shared[next_stage, kk, col_start + 2], S.f32)
            b3 = S.convert(b_shared[next_stage, kk, col_start + 3], S.f32)

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

        mfma_acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[1], b_frag1[1], mfma_acc)
        acc[0, 0] += mfma_acc[0] - mfma_acc[0]

        for kk in S.range(8, BLOCK_K):
            a0 = S.convert(a_shared[next_stage, row_start + 0, kk], S.f32)
            a1 = S.convert(a_shared[next_stage, row_start + 1, kk], S.f32)
            a2 = S.convert(a_shared[next_stage, row_start + 2, kk], S.f32)
            a3 = S.convert(a_shared[next_stage, row_start + 3, kk], S.f32)

            b0 = S.convert(b_shared[next_stage, kk, col_start + 0], S.f32)
            b1 = S.convert(b_shared[next_stage, kk, col_start + 1], S.f32)
            b2 = S.convert(b_shared[next_stage, kk, col_start + 2], S.f32)
            b3 = S.convert(b_shared[next_stage, kk, col_start + 3], S.f32)

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

        if next_next_k_base < L:
            S.syncthreads()

    for ii in S.range(4):
        flat_row = row_block + row_start + ii
        row_elem_offset = flat_row * K + col_block + col_start
        row_byte_offset = row_elem_offset * 2

        c0 = S.convert(acc[ii, 0], S.bf16)
        c1 = S.convert(acc[ii, 1], S.bf16)
        c2 = S.convert(acc[ii, 2], S.bf16)
        c3 = S.convert(acc[ii, 3], S.bf16)

        c01 = S.convert(S.bitcast(c0, S.u16), S.u32) | (S.convert(S.bitcast(c1, S.u16), S.u32) << S.convert(16, S.u32))
        c23 = S.convert(S.bitcast(c2, S.u16), S.u32) | (S.convert(S.bitcast(c3, S.u16), S.u32) << S.convert(16, S.u32))

        S.amdgpu.raw_buffer_store_x1(c01, c_rsrc, row_byte_offset, 0, 0)
        S.amdgpu.raw_buffer_store_x1(c23, c_rsrc, row_byte_offset + 4, 0, 0)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if tuple(A.shape) != (BATCH, I, J, L):
            raise ValueError(f"expected A shape {(BATCH, I, J, L)}, got {tuple(A.shape)}")
        if tuple(B.shape) != (L, K):
            raise ValueError(f"expected B shape {(L, K)}, got {tuple(B.shape)}")
        if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
            raise TypeError("expected bf16 inputs")

        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((BATCH, I, J, K), device=A.device, dtype=A.dtype)

        grid_x = K // BLOCK_N
        grid_y = M // BLOCK_M
        einsum4d_kernel[lambda: ((grid_x, grid_y, 1), (THREADS, 1, 1))](A, B, C, num_warps=4)
        return C
