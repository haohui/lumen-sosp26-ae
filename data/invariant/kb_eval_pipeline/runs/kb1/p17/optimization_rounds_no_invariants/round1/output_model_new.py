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

    shared_a_words = S.make_shared((2, 64, 4), S.u32)
    shared_b_words = S.make_shared((2, 64, 4), S.u32)
    shared_a_layout = S.make_layout((2, 64, 8), (512, 8, 1))
    shared_b_layout = S.make_layout((2, 64, 8), (512, 8, 1))
    shared_a = S.view(shared_a_words, S.bf16, shared_a_layout)
    shared_b = S.view(shared_b_words, S.bf16, shared_b_layout)

    acc = S.full((16,), 0.0, S.f32)

    block_row = block_m * BLOCK_M
    block_col = block_n * BLOCK_N

    for ko in S.range(K // BLOCK_K):
        k_base = ko * BLOCK_K

        if tid < 128:
            row = tid >> 1
            half = tid & 1
            row_base = (block_row + row) * K + k_base + half * 8
            a_offset = S.convert(row_base * BF16_BYTES, S.i32)
            a_packed = S.amdgpu.raw_buffer_load_x4(a_rsrc, zero, a_offset, 0)
            panel = row >> 5
            row_in_panel = row & 31
            shared_a_words[panel, row_in_panel, half * 2 + 0] = a_packed[0]
            shared_a_words[panel, row_in_panel, half * 2 + 1] = a_packed[1]
            shared_a_words[panel, row_in_panel + 32, half * 2 + 0] = a_packed[2]
            shared_a_words[panel, row_in_panel + 32, half * 2 + 1] = a_packed[3]
        else:
            b_tid = tid - 128
            k_local = b_tid >> 3
            col_chunk = b_tid & 7
            b_base = (k_base + k_local) * N + block_col + col_chunk * 8
            b_offset = S.convert(b_base * BF16_BYTES, S.i32)
            b_packed = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero, b_offset, 0)
            b_vals = S.view(b_packed, S.Tensor((8,), S.bf16))

            for c in S.range(8):
                col_local = col_chunk * 8 + c
                panel = col_local >> 5
                col_in_panel = col_local & 31
                lane_base = col_in_panel
                elem = k_local
                if k_local < 4:
                    shared_b[panel, lane_base, elem] = b_vals[c]
                elif k_local < 8:
                    shared_b[panel, lane_base + 32, elem - 4] = b_vals[c]
                elif k_local < 12:
                    shared_b[panel, lane_base, elem - 4] = b_vals[c]
                else:
                    shared_b[panel, lane_base + 32, elem - 8] = b_vals[c]

        S.syncthreads()

        a_frag = S.view(shared_a_words[wave_m, lane], S.Tensor((2, 4, 1), S.bf16))
        b_frag = S.view(shared_b_words[wave_n, lane], S.Tensor((2, 4, 1), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[1], b_frag[1], acc)

        S.syncthreads()

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
