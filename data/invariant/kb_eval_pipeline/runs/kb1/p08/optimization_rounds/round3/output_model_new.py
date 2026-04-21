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
UNROLL_K = BLOCK_K * 2
WAVE_SIZE = 64
WAVES_PER_BLOCK = 4
THREADS_PER_BLOCK = WAVE_SIZE * WAVES_PER_BLOCK

GRID_X = (N + BLOCK_N - 1) // BLOCK_N
GRID_Y = (M + BLOCK_M - 1) // BLOCK_M

A_RANGE_BYTES = M * K * 2
B_RANGE_BYTES = K * N * 2
C_RANGE_BYTES = M * N * 2


@substrate.jit
def gemm_kernel(
    A: S.Tensor((M, K), S.bf16),
    B: S.Tensor((K, N), S.bf16),
    C: S.Tensor((M, N), S.bf16),
):
    tid = S.thread_id(0)
    lane = tid % WAVE_SIZE
    wave = tid // WAVE_SIZE
    wave_row = wave // 2
    wave_col = wave % 2
    bx = S.block_id(0)
    by = S.block_id(1)

    block_row = by * BLOCK_M
    block_col = bx * BLOCK_N

    a_nat = S.make_shared((2, BLOCK_M, BLOCK_K), S.bf16)
    b_nat = S.make_shared((2, BLOCK_K, BLOCK_N), S.bf16)
    a_pack = S.make_shared((2, 2, WAVE_SIZE, 8), S.bf16)
    b_pack = S.make_shared((2, 2, WAVE_SIZE, 8), S.bf16)

    acc = S.full((16,), 0.0, S.f32)

    if tid < 128:
        a_row = tid // 2
        a_half = tid % 2
        g_row = block_row + a_row
        g_col = a_half * 8
        a_row_range = S.min((g_row * K + K) * 2, A_RANGE_BYTES)
        a_row_rsrc = S.amdgpu.make_rsrc(A, a_row_range)
        a_vec = S.amdgpu.raw_buffer_load_x4(a_row_rsrc, (g_row * K + g_col) * 2, 0, 0)
        a_vals = S.view(a_vec, S.Tensor((8,), S.bf16))
        for e in S.range(8):
            a_nat[0, a_row, a_half * 8 + e] = a_vals[e]
    else:
        b_tid = tid - 128
        b_row = b_tid // 8
        b_chunk = b_tid % 8
        g_row = b_row
        g_col = block_col + b_chunk * 8
        b_row_range = S.min((g_row * N + N) * 2, B_RANGE_BYTES)
        b_row_rsrc = S.amdgpu.make_rsrc(B, b_row_range)
        b_vec = S.amdgpu.raw_buffer_load_x4(b_row_rsrc, (g_row * N + g_col) * 2, 0, 0)
        b_vals = S.view(b_vec, S.Tensor((8,), S.bf16))
        for e in S.range(8):
            b_nat[0, b_row, b_chunk * 8 + e] = b_vals[e]

    S.syncthreads()

    if tid < 128:
        pack_wave_row = tid // WAVE_SIZE
        pack_lane = tid % WAVE_SIZE
        row_in_wave = pack_lane % 32
        k_group = pack_lane // 32
        src_row = pack_wave_row * 32 + row_in_wave

        a_pack[0, pack_wave_row, pack_lane, 0] = a_nat[0, src_row, k_group * 4 + 0]
        a_pack[0, pack_wave_row, pack_lane, 1] = a_nat[0, src_row, k_group * 4 + 1]
        a_pack[0, pack_wave_row, pack_lane, 2] = a_nat[0, src_row, k_group * 4 + 2]
        a_pack[0, pack_wave_row, pack_lane, 3] = a_nat[0, src_row, k_group * 4 + 3]
        a_pack[0, pack_wave_row, pack_lane, 4] = a_nat[0, src_row, 8 + k_group * 4 + 0]
        a_pack[0, pack_wave_row, pack_lane, 5] = a_nat[0, src_row, 8 + k_group * 4 + 1]
        a_pack[0, pack_wave_row, pack_lane, 6] = a_nat[0, src_row, 8 + k_group * 4 + 2]
        a_pack[0, pack_wave_row, pack_lane, 7] = a_nat[0, src_row, 8 + k_group * 4 + 3]
    else:
        pack_tid = tid - 128
        pack_wave_col = pack_tid // WAVE_SIZE
        pack_lane = pack_tid % WAVE_SIZE
        col_in_wave = pack_lane % 32
        k_group = pack_lane // 32
        chunk = pack_wave_col * 4 + (col_in_wave // 8)
        col_in_chunk = col_in_wave % 8
        col = chunk * 8 + col_in_chunk

        b_pack[0, pack_wave_col, pack_lane, 0] = b_nat[0, k_group * 4 + 0, col]
        b_pack[0, pack_wave_col, pack_lane, 1] = b_nat[0, k_group * 4 + 1, col]
        b_pack[0, pack_wave_col, pack_lane, 2] = b_nat[0, k_group * 4 + 2, col]
        b_pack[0, pack_wave_col, pack_lane, 3] = b_nat[0, k_group * 4 + 3, col]
        b_pack[0, pack_wave_col, pack_lane, 4] = b_nat[0, 8 + k_group * 4 + 0, col]
        b_pack[0, pack_wave_col, pack_lane, 5] = b_nat[0, 8 + k_group * 4 + 1, col]
        b_pack[0, pack_wave_col, pack_lane, 6] = b_nat[0, 8 + k_group * 4 + 2, col]
        b_pack[0, pack_wave_col, pack_lane, 7] = b_nat[0, 8 + k_group * 4 + 3, col]

    if BLOCK_K < K:
        S.syncthreads()
        if tid < 128:
            a_row = tid // 2
            a_half = tid % 2
            g_row = block_row + a_row
            g_col = BLOCK_K + a_half * 8
            a_row_range = S.min((g_row * K + K) * 2, A_RANGE_BYTES)
            a_row_rsrc = S.amdgpu.make_rsrc(A, a_row_range)
            a_vec = S.amdgpu.raw_buffer_load_x4(a_row_rsrc, (g_row * K + g_col) * 2, 0, 0)
            a_vals = S.view(a_vec, S.Tensor((8,), S.bf16))
            for e in S.range(8):
                a_nat[1, a_row, a_half * 8 + e] = a_vals[e]
        else:
            b_tid = tid - 128
            b_row = b_tid // 8
            b_chunk = b_tid % 8
            g_row = BLOCK_K + b_row
            g_col = block_col + b_chunk * 8
            b_row_range = S.min((g_row * N + N) * 2, B_RANGE_BYTES)
            b_row_rsrc = S.amdgpu.make_rsrc(B, b_row_range)
            b_vec = S.amdgpu.raw_buffer_load_x4(b_row_rsrc, (g_row * N + g_col) * 2, 0, 0)
            b_vals = S.view(b_vec, S.Tensor((8,), S.bf16))
            for e in S.range(8):
                b_nat[1, b_row, b_chunk * 8 + e] = b_vals[e]

        S.syncthreads()

        if tid < 128:
            pack_wave_row = tid // WAVE_SIZE
            pack_lane = tid % WAVE_SIZE
            row_in_wave = pack_lane % 32
            k_group = pack_lane // 32
            src_row = pack_wave_row * 32 + row_in_wave

            a_pack[1, pack_wave_row, pack_lane, 0] = a_nat[1, src_row, k_group * 4 + 0]
            a_pack[1, pack_wave_row, pack_lane, 1] = a_nat[1, src_row, k_group * 4 + 1]
            a_pack[1, pack_wave_row, pack_lane, 2] = a_nat[1, src_row, k_group * 4 + 2]
            a_pack[1, pack_wave_row, pack_lane, 3] = a_nat[1, src_row, k_group * 4 + 3]
            a_pack[1, pack_wave_row, pack_lane, 4] = a_nat[1, src_row, 8 + k_group * 4 + 0]
            a_pack[1, pack_wave_row, pack_lane, 5] = a_nat[1, src_row, 8 + k_group * 4 + 1]
            a_pack[1, pack_wave_row, pack_lane, 6] = a_nat[1, src_row, 8 + k_group * 4 + 2]
            a_pack[1, pack_wave_row, pack_lane, 7] = a_nat[1, src_row, 8 + k_group * 4 + 3]
        else:
            pack_tid = tid - 128
            pack_wave_col = pack_tid // WAVE_SIZE
            pack_lane = pack_tid % WAVE_SIZE
            col_in_wave = pack_lane % 32
            k_group = pack_lane // 32
            chunk = pack_wave_col * 4 + (col_in_wave // 8)
            col_in_chunk = col_in_wave % 8
            col = chunk * 8 + col_in_chunk

            b_pack[1, pack_wave_col, pack_lane, 0] = b_nat[1, k_group * 4 + 0, col]
            b_pack[1, pack_wave_col, pack_lane, 1] = b_nat[1, k_group * 4 + 1, col]
            b_pack[1, pack_wave_col, pack_lane, 2] = b_nat[1, k_group * 4 + 2, col]
            b_pack[1, pack_wave_col, pack_lane, 3] = b_nat[1, k_group * 4 + 3, col]
            b_pack[1, pack_wave_col, pack_lane, 4] = b_nat[1, 8 + k_group * 4 + 0, col]
            b_pack[1, pack_wave_col, pack_lane, 5] = b_nat[1, 8 + k_group * 4 + 1, col]
            b_pack[1, pack_wave_col, pack_lane, 6] = b_nat[1, 8 + k_group * 4 + 2, col]
            b_pack[1, pack_wave_col, pack_lane, 7] = b_nat[1, 8 + k_group * 4 + 3, col]

    S.syncthreads()

    for k_base in S.range(0, K, UNROLL_K):
        a_frag0 = S.view(a_pack[0, wave_row, lane], S.Tensor((2, 4, 1), S.bf16))
        b_frag0 = S.view(b_pack[0, wave_col, lane], S.Tensor((2, 4, 1), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[0], b_frag0[0], acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[1], b_frag0[1], acc)

        future0_k = k_base + UNROLL_K
        if future0_k < K:
            if tid < 128:
                a_row = tid // 2
                a_half = tid % 2
                g_row = block_row + a_row
                g_col = future0_k + a_half * 8
                a_row_range = S.min((g_row * K + K) * 2, A_RANGE_BYTES)
                a_row_rsrc = S.amdgpu.make_rsrc(A, a_row_range)
                a_vec = S.amdgpu.raw_buffer_load_x4(a_row_rsrc, (g_row * K + g_col) * 2, 0, 0)
                a_vals = S.view(a_vec, S.Tensor((8,), S.bf16))
                for e in S.range(8):
                    a_nat[0, a_row, a_half * 8 + e] = a_vals[e]
            else:
                b_tid = tid - 128
                b_row = b_tid // 8
                b_chunk = b_tid % 8
                g_row = future0_k + b_row
                g_col = block_col + b_chunk * 8
                b_row_range = S.min((g_row * N + N) * 2, B_RANGE_BYTES)
                b_row_rsrc = S.amdgpu.make_rsrc(B, b_row_range)
                b_vec = S.amdgpu.raw_buffer_load_x4(b_row_rsrc, (g_row * N + g_col) * 2, 0, 0)
                b_vals = S.view(b_vec, S.Tensor((8,), S.bf16))
                for e in S.range(8):
                    b_nat[0, b_row, b_chunk * 8 + e] = b_vals[e]

            S.syncthreads()

            if tid < 128:
                pack_wave_row = tid // WAVE_SIZE
                pack_lane = tid % WAVE_SIZE
                row_in_wave = pack_lane % 32
                k_group = pack_lane // 32
                src_row = pack_wave_row * 32 + row_in_wave

                a_pack[0, pack_wave_row, pack_lane, 0] = a_nat[0, src_row, k_group * 4 + 0]
                a_pack[0, pack_wave_row, pack_lane, 1] = a_nat[0, src_row, k_group * 4 + 1]
                a_pack[0, pack_wave_row, pack_lane, 2] = a_nat[0, src_row, k_group * 4 + 2]
                a_pack[0, pack_wave_row, pack_lane, 3] = a_nat[0, src_row, k_group * 4 + 3]
                a_pack[0, pack_wave_row, pack_lane, 4] = a_nat[0, src_row, 8 + k_group * 4 + 0]
                a_pack[0, pack_wave_row, pack_lane, 5] = a_nat[0, src_row, 8 + k_group * 4 + 1]
                a_pack[0, pack_wave_row, pack_lane, 6] = a_nat[0, src_row, 8 + k_group * 4 + 2]
                a_pack[0, pack_wave_row, pack_lane, 7] = a_nat[0, src_row, 8 + k_group * 4 + 3]
            else:
                pack_tid = tid - 128
                pack_wave_col = pack_tid // WAVE_SIZE
                pack_lane = pack_tid % WAVE_SIZE
                col_in_wave = pack_lane % 32
                k_group = pack_lane // 32
                chunk = pack_wave_col * 4 + (col_in_wave // 8)
                col_in_chunk = col_in_wave % 8
                col = chunk * 8 + col_in_chunk

                b_pack[0, pack_wave_col, pack_lane, 0] = b_nat[0, k_group * 4 + 0, col]
                b_pack[0, pack_wave_col, pack_lane, 1] = b_nat[0, k_group * 4 + 1, col]
                b_pack[0, pack_wave_col, pack_lane, 2] = b_nat[0, k_group * 4 + 2, col]
                b_pack[0, pack_wave_col, pack_lane, 3] = b_nat[0, k_group * 4 + 3, col]
                b_pack[0, pack_wave_col, pack_lane, 4] = b_nat[0, 8 + k_group * 4 + 0, col]
                b_pack[0, pack_wave_col, pack_lane, 5] = b_nat[0, 8 + k_group * 4 + 1, col]
                b_pack[0, pack_wave_col, pack_lane, 6] = b_nat[0, 8 + k_group * 4 + 2, col]
                b_pack[0, pack_wave_col, pack_lane, 7] = b_nat[0, 8 + k_group * 4 + 3, col]

            S.syncthreads()

        stage1_k = k_base + BLOCK_K
        if stage1_k < K:
            a_frag1 = S.view(a_pack[1, wave_row, lane], S.Tensor((2, 4, 1), S.bf16))
            b_frag1 = S.view(b_pack[1, wave_col, lane], S.Tensor((2, 4, 1), S.bf16))
            acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[0], b_frag1[0], acc)
            acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[1], b_frag1[1], acc)

            future1_k = stage1_k + UNROLL_K
            if future1_k < K:
                if tid < 128:
                    a_row = tid // 2
                    a_half = tid % 2
                    g_row = block_row + a_row
                    g_col = future1_k + a_half * 8
                    a_row_range = S.min((g_row * K + K) * 2, A_RANGE_BYTES)
                    a_row_rsrc = S.amdgpu.make_rsrc(A, a_row_range)
                    a_vec = S.amdgpu.raw_buffer_load_x4(a_row_rsrc, (g_row * K + g_col) * 2, 0, 0)
                    a_vals = S.view(a_vec, S.Tensor((8,), S.bf16))
                    for e in S.range(8):
                        a_nat[1, a_row, a_half * 8 + e] = a_vals[e]
                else:
                    b_tid = tid - 128
                    b_row = b_tid // 8
                    b_chunk = b_tid % 8
                    g_row = future1_k + b_row
                    g_col = block_col + b_chunk * 8
                    b_row_range = S.min((g_row * N + N) * 2, B_RANGE_BYTES)
                    b_row_rsrc = S.amdgpu.make_rsrc(B, b_row_range)
                    b_vec = S.amdgpu.raw_buffer_load_x4(b_row_rsrc, (g_row * N + g_col) * 2, 0, 0)
                    b_vals = S.view(b_vec, S.Tensor((8,), S.bf16))
                    for e in S.range(8):
                        b_nat[1, b_row, b_chunk * 8 + e] = b_vals[e]

                S.syncthreads()

                if tid < 128:
                    pack_wave_row = tid // WAVE_SIZE
                    pack_lane = tid % WAVE_SIZE
                    row_in_wave = pack_lane % 32
                    k_group = pack_lane // 32
                    src_row = pack_wave_row * 32 + row_in_wave

                    a_pack[1, pack_wave_row, pack_lane, 0] = a_nat[1, src_row, k_group * 4 + 0]
                    a_pack[1, pack_wave_row, pack_lane, 1] = a_nat[1, src_row, k_group * 4 + 1]
                    a_pack[1, pack_wave_row, pack_lane, 2] = a_nat[1, src_row, k_group * 4 + 2]
                    a_pack[1, pack_wave_row, pack_lane, 3] = a_nat[1, src_row, k_group * 4 + 3]
                    a_pack[1, pack_wave_row, pack_lane, 4] = a_nat[1, src_row, 8 + k_group * 4 + 0]
                    a_pack[1, pack_wave_row, pack_lane, 5] = a_nat[1, src_row, 8 + k_group * 4 + 1]
                    a_pack[1, pack_wave_row, pack_lane, 6] = a_nat[1, src_row, 8 + k_group * 4 + 2]
                    a_pack[1, pack_wave_row, pack_lane, 7] = a_nat[1, src_row, 8 + k_group * 4 + 3]
                else:
                    pack_tid = tid - 128
                    pack_wave_col = pack_tid // WAVE_SIZE
                    pack_lane = pack_tid % WAVE_SIZE
                    col_in_wave = pack_lane % 32
                    k_group = pack_lane // 32
                    chunk = pack_wave_col * 4 + (col_in_wave // 8)
                    col_in_chunk = col_in_wave % 8
                    col = chunk * 8 + col_in_chunk

                    b_pack[1, pack_wave_col, pack_lane, 0] = b_nat[1, k_group * 4 + 0, col]
                    b_pack[1, pack_wave_col, pack_lane, 1] = b_nat[1, k_group * 4 + 1, col]
                    b_pack[1, pack_wave_col, pack_lane, 2] = b_nat[1, k_group * 4 + 2, col]
                    b_pack[1, pack_wave_col, pack_lane, 3] = b_nat[1, k_group * 4 + 3, col]
                    b_pack[1, pack_wave_col, pack_lane, 4] = b_nat[1, 8 + k_group * 4 + 0, col]
                    b_pack[1, pack_wave_col, pack_lane, 5] = b_nat[1, 8 + k_group * 4 + 1, col]
                    b_pack[1, pack_wave_col, pack_lane, 6] = b_nat[1, 8 + k_group * 4 + 2, col]
                    b_pack[1, pack_wave_col, pack_lane, 7] = b_nat[1, 8 + k_group * 4 + 3, col]

                S.syncthreads()

    tile_row_base = block_row + wave_row * 32
    tile_col_base = block_col + wave_col * 32
    lane_col = lane % 32
    lane_row_quad = lane // 32

    for acc_idx in S.range(16):
        out_col = tile_col_base + lane_col
        out_row = tile_row_base + 8 * (acc_idx // 4) + 4 * lane_row_quad + (acc_idx % 4)
        c_row_range = S.min((out_row * N + N) * 2, C_RANGE_BYTES)
        c_row_rsrc = S.amdgpu.make_rsrc(C, c_row_range)
        c_value = S.bitcast(S.convert(acc[acc_idx], S.bf16), S.u16)
        S.amdgpu.raw_buffer_store_x1(c_value, c_row_rsrc, (out_row * N + out_col) * 2, 0, 0)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if tuple(A.shape) != (M, K) or tuple(B.shape) != (K, N):
            raise ValueError(f"Expected A.shape={(M, K)} and B.shape={(K, N)}")
        if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
            raise ValueError("Expected bf16 inputs")
        if A.device != B.device:
            raise ValueError("Inputs must be on the same device")

        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((M, N), device=A.device, dtype=torch.bfloat16)
        gemm_kernel[lambda: ((GRID_X, GRID_Y, 1), (THREADS_PER_BLOCK, 1, 1))](A, B, C)
        return C
