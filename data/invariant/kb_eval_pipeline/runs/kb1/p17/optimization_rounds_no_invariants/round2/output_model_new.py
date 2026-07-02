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
BF16_BYTES = 2
A_BYTES = M * K * BF16_BYTES
B_BYTES = K * N * BF16_BYTES
K_TILES = K // BLOCK_K
K_PAIRS = K_TILES // 2


@substrate.jit
def gemm_kernel(
    A: S.Tensor((M * K,), S.bf16),
    B: S.Tensor((K * N,), S.bf16),
    C: S.Tensor((M * N,), S.bf16),
):
    block_m = S.block_id(1)
    block_n = S.block_id(0)
    tid = S.thread_id(0)

    wave = tid >> 6
    lane = tid & 63
    wave_m = wave >> 1
    wave_n = wave & 1

    a_rsrc = S.amdgpu.make_rsrc(A, A_BYTES)
    b_rsrc = S.amdgpu.make_rsrc(B, B_BYTES)
    zero = S.convert(0, S.i32)

    shared_a_words = S.make_shared((2, 2, 64, 4), S.u32)
    shared_b_words = S.make_shared((2, 2, 64, 4), S.u32)
    shared_a_layout = S.make_layout((2, 2, 64, 8), (1024, 512, 8, 1))
    shared_b_layout = S.make_layout((2, 2, 64, 8), (1024, 512, 8, 1))
    shared_a = S.view(shared_a_words, S.bf16, shared_a_layout)
    shared_b = S.view(shared_b_words, S.bf16, shared_b_layout)

    acc = S.full((16,), 0.0, S.f32)

    block_row = block_m * BLOCK_M
    block_col = block_n * BLOCK_N

    if tid < 128:
        row = tid >> 1
        half = tid & 1
        panel = row >> 5
        row_in_panel = row & 31

        a_base0 = (block_row + row) * K + half * 8
        a_offset0 = S.convert(a_base0 * BF16_BYTES, S.i32)
        a_packed0 = S.amdgpu.raw_buffer_load_x4(a_rsrc, zero, a_offset0, 0)
        shared_a_words[0, panel, row_in_panel, half * 2 + 0] = a_packed0[0]
        shared_a_words[0, panel, row_in_panel, half * 2 + 1] = a_packed0[1]
        shared_a_words[0, panel, row_in_panel + 32, half * 2 + 0] = a_packed0[2]
        shared_a_words[0, panel, row_in_panel + 32, half * 2 + 1] = a_packed0[3]

        a_base1 = (block_row + row) * K + BLOCK_K + half * 8
        a_offset1 = S.convert(a_base1 * BF16_BYTES, S.i32)
        a_packed1 = S.amdgpu.raw_buffer_load_x4(a_rsrc, zero, a_offset1, 0)
        shared_a_words[1, panel, row_in_panel, half * 2 + 0] = a_packed1[0]
        shared_a_words[1, panel, row_in_panel, half * 2 + 1] = a_packed1[1]
        shared_a_words[1, panel, row_in_panel + 32, half * 2 + 0] = a_packed1[2]
        shared_a_words[1, panel, row_in_panel + 32, half * 2 + 1] = a_packed1[3]
    else:
        b_tid = tid - 128
        k_local = b_tid >> 3
        col_chunk = b_tid & 7

        b_base0 = k_local * N + block_col + col_chunk * 8
        b_offset0 = S.convert(b_base0 * BF16_BYTES, S.i32)
        b_packed0 = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero, b_offset0, 0)
        b_vals0 = S.view(b_packed0, S.Tensor((8,), S.bf16))
        for c in S.range(8):
            col_local = col_chunk * 8 + c
            panel = col_local >> 5
            col_in_panel = col_local & 31
            lane_base = col_in_panel
            if k_local < 4:
                shared_b[0, panel, lane_base, k_local] = b_vals0[c]
            elif k_local < 8:
                shared_b[0, panel, lane_base + 32, k_local - 4] = b_vals0[c]
            elif k_local < 12:
                shared_b[0, panel, lane_base, k_local - 4] = b_vals0[c]
            else:
                shared_b[0, panel, lane_base + 32, k_local - 8] = b_vals0[c]

        b_base1 = (BLOCK_K + k_local) * N + block_col + col_chunk * 8
        b_offset1 = S.convert(b_base1 * BF16_BYTES, S.i32)
        b_packed1 = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero, b_offset1, 0)
        b_vals1 = S.view(b_packed1, S.Tensor((8,), S.bf16))
        for c in S.range(8):
            col_local = col_chunk * 8 + c
            panel = col_local >> 5
            col_in_panel = col_local & 31
            lane_base = col_in_panel
            if k_local < 4:
                shared_b[1, panel, lane_base, k_local] = b_vals1[c]
            elif k_local < 8:
                shared_b[1, panel, lane_base + 32, k_local - 4] = b_vals1[c]
            elif k_local < 12:
                shared_b[1, panel, lane_base, k_local - 4] = b_vals1[c]
            else:
                shared_b[1, panel, lane_base + 32, k_local - 8] = b_vals1[c]
    S.syncthreads()

    for kp in S.range(K_PAIRS - 1):
        even_k = kp * 2
        odd_k = even_k + 1
        next_even_base = (even_k + 2) * BLOCK_K
        a_frag = S.view(shared_a_words[0, wave_m, lane], S.Tensor((2, 4, 1), S.bf16))
        b_frag = S.view(shared_b_words[0, wave_n, lane], S.Tensor((2, 4, 1), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[1], b_frag[1], acc)

        if tid < 128:
            row = tid >> 1
            half = tid & 1
            panel = row >> 5
            row_in_panel = row & 31
            next_a_even_offset = S.convert(((block_row + row) * K + next_even_base + half * 8) * BF16_BYTES, S.i32)
            next_a_even = S.amdgpu.raw_buffer_load_x4(a_rsrc, zero, next_a_even_offset, 0)
            shared_a_words[0, panel, row_in_panel, half * 2 + 0] = next_a_even[0]
            shared_a_words[0, panel, row_in_panel, half * 2 + 1] = next_a_even[1]
            shared_a_words[0, panel, row_in_panel + 32, half * 2 + 0] = next_a_even[2]
            shared_a_words[0, panel, row_in_panel + 32, half * 2 + 1] = next_a_even[3]
        else:
            b_tid = tid - 128
            k_local = b_tid >> 3
            col_chunk = b_tid & 7
            next_b_even_offset = S.convert(((next_even_base + k_local) * N + block_col + col_chunk * 8) * BF16_BYTES, S.i32)
            next_b_even_vals = S.view(S.amdgpu.raw_buffer_load_x4(b_rsrc, zero, next_b_even_offset, 0), S.Tensor((8,), S.bf16))
            for c in S.range(8):
                col_local = col_chunk * 8 + c
                panel = col_local >> 5
                col_in_panel = col_local & 31
                lane_base = col_in_panel
                if k_local < 4:
                    shared_b[0, panel, lane_base, k_local] = next_b_even_vals[c]
                elif k_local < 8:
                    shared_b[0, panel, lane_base + 32, k_local - 4] = next_b_even_vals[c]
                elif k_local < 12:
                    shared_b[0, panel, lane_base, k_local - 4] = next_b_even_vals[c]
                else:
                    shared_b[0, panel, lane_base + 32, k_local - 8] = next_b_even_vals[c]

        next_odd_base = (odd_k + 2) * BLOCK_K
        a_frag = S.view(shared_a_words[1, wave_m, lane], S.Tensor((2, 4, 1), S.bf16))
        b_frag = S.view(shared_b_words[1, wave_n, lane], S.Tensor((2, 4, 1), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[1], b_frag[1], acc)

        if tid < 128:
            row = tid >> 1
            half = tid & 1
            panel = row >> 5
            row_in_panel = row & 31
            next_a_odd_offset = S.convert(((block_row + row) * K + next_odd_base + half * 8) * BF16_BYTES, S.i32)
            next_a_odd = S.amdgpu.raw_buffer_load_x4(a_rsrc, zero, next_a_odd_offset, 0)
            shared_a_words[1, panel, row_in_panel, half * 2 + 0] = next_a_odd[0]
            shared_a_words[1, panel, row_in_panel, half * 2 + 1] = next_a_odd[1]
            shared_a_words[1, panel, row_in_panel + 32, half * 2 + 0] = next_a_odd[2]
            shared_a_words[1, panel, row_in_panel + 32, half * 2 + 1] = next_a_odd[3]
        else:
            b_tid = tid - 128
            k_local = b_tid >> 3
            col_chunk = b_tid & 7
            next_b_odd_offset = S.convert(((next_odd_base + k_local) * N + block_col + col_chunk * 8) * BF16_BYTES, S.i32)
            next_b_odd_vals = S.view(S.amdgpu.raw_buffer_load_x4(b_rsrc, zero, next_b_odd_offset, 0), S.Tensor((8,), S.bf16))
            for c in S.range(8):
                col_local = col_chunk * 8 + c
                panel = col_local >> 5
                col_in_panel = col_local & 31
                lane_base = col_in_panel
                if k_local < 4:
                    shared_b[1, panel, lane_base, k_local] = next_b_odd_vals[c]
                elif k_local < 8:
                    shared_b[1, panel, lane_base + 32, k_local - 4] = next_b_odd_vals[c]
                elif k_local < 12:
                    shared_b[1, panel, lane_base, k_local - 4] = next_b_odd_vals[c]
                else:
                    shared_b[1, panel, lane_base + 32, k_local - 8] = next_b_odd_vals[c]

        S.syncthreads()

    a_frag = S.view(shared_a_words[0, wave_m, lane], S.Tensor((2, 4, 1), S.bf16))
    b_frag = S.view(shared_b_words[0, wave_n, lane], S.Tensor((2, 4, 1), S.bf16))
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], acc)
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[1], b_frag[1], acc)

    a_frag = S.view(shared_a_words[1, wave_m, lane], S.Tensor((2, 4, 1), S.bf16))
    b_frag = S.view(shared_b_words[1, wave_n, lane], S.Tensor((2, 4, 1), S.bf16))
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], acc)
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[1], b_frag[1], acc)

    row_group = lane >> 5
    col = block_col + wave_n * 32 + (lane & 31)

    for r in S.range(16):
        row = block_row + wave_m * 32 + (r >> 2) * 8 + row_group * 4 + (r & 3)
        C[row * N + col] = S.convert(acc[r], S.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if tuple(A.shape) != (M, K) or tuple(B.shape) != (N, K):
            raise ValueError(f"expected A[{M}, {K}] and B[{N}, {K}], got {tuple(A.shape)} and {tuple(B.shape)}")
        if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
            raise ValueError("expected bfloat16 inputs")
        if A.device.type != "cuda" or B.device.type != "cuda":
            raise ValueError("expected CUDA inputs")

        A_flat = A.contiguous().view(M * K)
        B_flat = B.transpose(-2, -1).contiguous().view(K * N)
        C_flat = torch.empty((M * N,), device=A.device, dtype=A.dtype)
        gemm_kernel[lambda: ((N // BLOCK_N, M // BLOCK_M, 1), (THREADS, 1, 1))](A_flat, B_flat, C_flat)
        return C_flat.view(M, N)
