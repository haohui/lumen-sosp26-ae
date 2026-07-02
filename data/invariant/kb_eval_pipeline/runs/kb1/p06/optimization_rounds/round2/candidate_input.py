import torch
import torch.nn as nn

import substrate
import substrate.language as S


M = 256
K = 524288
N = 256
BLOCK_M = 64
BLOCK_N = 64
WAVE_SIZE = 64
WAVES_M = 2
WAVES_N = 2
WAVES_PER_BLOCK = WAVES_M * WAVES_N
THREADS = WAVE_SIZE * WAVES_PER_BLOCK
K_TILE = 16
A_RANGE_BYTES = M * K * 2
B_RANGE_BYTES = K * N * 2


@substrate.jit
def gemm_kernel(
    A: S.Tensor((M, K), S.bf16),
    B: S.Tensor((K, N), S.bf16),
    C: S.Tensor((M, N), S.bf16),
):
    tid = S.thread_id(0)
    wave = tid // WAVE_SIZE
    lane = tid % WAVE_SIZE

    block_m = S.block_id(1)
    block_n = S.block_id(0)
    wave_m = wave // WAVES_N
    wave_n = wave % WAVES_N

    tile_row_base = block_m * BLOCK_M + wave_m * 32
    tile_col_base = block_n * BLOCK_N + wave_n * 32

    a_rsrc = S.amdgpu.make_rsrc(A, S.convert(A_RANGE_BYTES, S.i32))
    b_rsrc = S.amdgpu.make_rsrc(B, S.convert(B_RANGE_BYTES, S.i32))
    zero = S.convert(0, S.i32)

    a_words = S.make_shared((WAVES_PER_BLOCK, WAVE_SIZE, 4), S.u32)
    b_words = S.make_shared((WAVES_PER_BLOCK, WAVE_SIZE, 4), S.u32)

    c_lane = S.full((16,), 0.0, S.f32)

    a_row = tile_row_base + (lane % 32)
    a_group = lane // 32

    b_k = lane % 8
    b_col_group = lane // 8

    for k_tile in S.range(K // K_TILE):
        kk = k_tile * K_TILE

        a_col0 = kk + a_group * 4
        a_col1 = kk + 8 + a_group * 4
        a_offset0 = S.convert((a_row * K + a_col0) * 2, S.i32)
        a_offset1 = S.convert((a_row * K + a_col1) * 2, S.i32)
        a_raw0 = S.amdgpu.raw_buffer_load_x4(a_rsrc, zero, a_offset0, 0)
        a_raw1 = S.amdgpu.raw_buffer_load_x4(a_rsrc, zero, a_offset1, 0)

        a_pack = S.full((4,), 0, S.u32)
        a_pack[0] = a_raw0[0]
        a_pack[1] = a_raw0[1]
        a_pack[2] = a_raw1[0]
        a_pack[3] = a_raw1[1]
        a_words[wave, lane] = a_pack

        b_col = tile_col_base + b_col_group * 4
        b_row0 = kk + b_k
        b_row1 = kk + 8 + b_k
        b_offset0 = S.convert((b_row0 * N + b_col) * 2, S.i32)
        b_offset1 = S.convert((b_row1 * N + b_col) * 2, S.i32)
        b_raw0 = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero, b_offset0, 0)
        b_raw1 = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero, b_offset1, 0)

        b_pack = S.full((4,), 0, S.u32)
        b_pack[0] = b_raw0[0]
        b_pack[1] = b_raw0[1]
        b_pack[2] = b_raw1[0]
        b_pack[3] = b_raw1[1]
        b_words[wave, lane] = b_pack

        S.syncthreads()

        m_a = S.view(a_words[wave, lane], S.Tensor((2, 4, 1), S.bf16))
        m_b = S.view(b_words[wave, lane], S.Tensor((2, 4, 1), S.bf16))

        c_lane = S.amdgpu.mfma_32x32x8_bf16_f32(m_a[0], m_b[0], c_lane)
        c_lane = S.amdgpu.mfma_32x32x8_bf16_f32(m_a[1], m_b[1], c_lane)

        S.syncthreads()

    for acc_idx in S.range(16):
        col = tile_col_base + (lane % 32)
        row = tile_row_base + 8 * (acc_idx // 4) + 4 * (lane // 32) + (acc_idx % 4)
        C[row, col] = S.convert(c_lane[acc_idx], S.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if tuple(A.shape) != (M, K) or tuple(B.shape) != (K, N):
            raise ValueError(f"Expected A {(M, K)} and B {(K, N)}, got {tuple(A.shape)} and {tuple(B.shape)}")
        if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
            raise ValueError(f"Expected bf16 inputs, got {A.dtype} and {B.dtype}")

        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)
        gemm_kernel[lambda: ((N // BLOCK_N, M // BLOCK_M, 1), (THREADS, 1, 1))](A, B, C)
        return C
