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
WAVES_PER_BLOCK = 4
THREADS_PER_BLOCK = WAVES_PER_BLOCK * 64

WAVE_TILE_M = 32
WAVE_TILE_N = 32
LANE_TILE_M = 4
LANE_TILE_N = 4


@substrate.jit
def gemm_kernel(
    A: S.Tensor((M, K), S.bf16),
    B: S.Tensor((K, N), S.bf16),
    C: S.Tensor((M, N), S.bf16),
):
    tid = S.thread_id(0)
    lane = tid % 64
    wave = tid // 64

    wave_m = wave // 2
    wave_n = wave % 2
    lane_row = lane // 8
    lane_col = lane % 8

    block_m = S.block_id(1)
    block_n = S.block_id(0)
    m0 = block_m * BLOCK_M
    n0 = block_n * BLOCK_N

    a_tile = S.make_shared((2, BLOCK_M, BLOCK_K), S.bf16)
    b_tile = S.make_shared((2, BLOCK_K, BLOCK_N), S.bf16)
    c_reg = S.full((16,), 0.0, S.f32)

    if tid < 128:
        a_elem = tid * 8
        a_row = a_elem // BLOCK_K
        a_col = a_elem % BLOCK_K
        a_row_view = S.subview(A, (m0 + a_row, 0), (1, K), (1, 1))
        a_row_rsrc = S.amdgpu.make_rsrc(a_row_view, K * 2)
        a_offset = a_col * 2
        a_vec = S.amdgpu.raw_buffer_load_x4(a_row_rsrc, a_offset, 0, 0)
        a_frag = S.view(a_vec, S.Tensor((2, 4, 1), S.bf16))
        for i in S.range(2):
            for j in S.range(4):
                a_tile[0, a_row, a_col + i * 4 + j] = a_frag[i, j, 0]
    else:
        b_tid = tid - 128
        b_elem = b_tid * 8
        b_row = b_elem // BLOCK_N
        b_col = b_elem % BLOCK_N
        b_row_view = S.subview(B, (b_row, 0), (1, N), (1, 1))
        b_row_rsrc = S.amdgpu.make_rsrc(b_row_view, N * 2)
        b_offset = (n0 + b_col) * 2
        b_vec = S.amdgpu.raw_buffer_load_x4(b_row_rsrc, b_offset, 0, 0)
        b_frag = S.view(b_vec, S.Tensor((2, 4, 1), S.bf16))
        for i in S.range(2):
            for j in S.range(4):
                b_tile[0, b_row, b_col + i * 4 + j] = b_frag[i, j, 0]

    S.syncthreads()

    for k0 in S.range(0, K, 2 * BLOCK_K):
        next_k0 = k0 + BLOCK_K

        if tid < 128:
            a_elem = tid * 8
            a_row = a_elem // BLOCK_K
            a_col = a_elem % BLOCK_K
            a_row_view = S.subview(A, (m0 + a_row, 0), (1, K), (1, 1))
            a_row_rsrc = S.amdgpu.make_rsrc(a_row_view, K * 2)
            a_offset = (next_k0 + a_col) * 2
            a_vec = S.amdgpu.raw_buffer_load_x4(a_row_rsrc, a_offset, 0, 0)
            a_frag = S.view(a_vec, S.Tensor((2, 4, 1), S.bf16))
            for i in S.range(2):
                for j in S.range(4):
                    a_tile[1, a_row, a_col + i * 4 + j] = a_frag[i, j, 0]
        else:
            b_tid = tid - 128
            b_elem = b_tid * 8
            b_row = b_elem // BLOCK_N
            b_col = b_elem % BLOCK_N
            b_row_view = S.subview(B, (next_k0 + b_row, 0), (1, N), (1, 1))
            b_row_rsrc = S.amdgpu.make_rsrc(b_row_view, N * 2)
            b_offset = (n0 + b_col) * 2
            b_vec = S.amdgpu.raw_buffer_load_x4(b_row_rsrc, b_offset, 0, 0)
            b_frag = S.view(b_vec, S.Tensor((2, 4, 1), S.bf16))
            for i in S.range(2):
                for j in S.range(4):
                    b_tile[1, b_row, b_col + i * 4 + j] = b_frag[i, j, 0]

        for kk in S.range(0, BLOCK_K, 2):
            for i in S.range(LANE_TILE_M):
                a_val0 = S.convert(
                    a_tile[0, wave_m * WAVE_TILE_M + lane_row + i * 8, kk + 0], S.f32
                )
                a_val1 = S.convert(
                    a_tile[0, wave_m * WAVE_TILE_M + lane_row + i * 8, kk + 1], S.f32
                )
                for j in S.range(LANE_TILE_N):
                    idx = i * LANE_TILE_N + j
                    b_val0 = S.convert(
                        b_tile[0, kk + 0, wave_n * WAVE_TILE_N + lane_col + j * 8], S.f32
                    )
                    b_val1 = S.convert(
                        b_tile[0, kk + 1, wave_n * WAVE_TILE_N + lane_col + j * 8], S.f32
                    )
                    c_old = c_reg[idx]
                    c_reg[idx] = c_old + a_val0 * b_val0 + a_val1 * b_val1

        S.syncthreads()

        load_k0 = k0 + 2 * BLOCK_K

        if tid < 128:
            a_elem = tid * 8
            a_row = a_elem // BLOCK_K
            a_col = a_elem % BLOCK_K
            a_row_view = S.subview(A, (m0 + a_row, 0), (1, K), (1, 1))
            a_row_rsrc = S.amdgpu.make_rsrc(a_row_view, K * 2)
            a_offset = (load_k0 + a_col) * 2
            a_vec = S.amdgpu.raw_buffer_load_x4(a_row_rsrc, a_offset, 0, 0)
            a_frag = S.view(a_vec, S.Tensor((2, 4, 1), S.bf16))
            for i in S.range(2):
                for j in S.range(4):
                    a_tile[0, a_row, a_col + i * 4 + j] = a_frag[i, j, 0]
        else:
            b_tid = tid - 128
            b_elem = b_tid * 8
            b_row = b_elem // BLOCK_N
            b_col = b_elem % BLOCK_N
            b_row_view = S.subview(B, (load_k0 + b_row, 0), (1, N), (1, 1))
            b_row_rsrc = S.amdgpu.make_rsrc(b_row_view, N * 2)
            b_offset = (n0 + b_col) * 2
            b_vec = S.amdgpu.raw_buffer_load_x4(b_row_rsrc, b_offset, 0, 0)
            b_frag = S.view(b_vec, S.Tensor((2, 4, 1), S.bf16))
            for i in S.range(2):
                for j in S.range(4):
                    b_tile[0, b_row, b_col + i * 4 + j] = b_frag[i, j, 0]

        for kk in S.range(0, BLOCK_K, 2):
            for i in S.range(LANE_TILE_M):
                a_val0 = S.convert(
                    a_tile[1, wave_m * WAVE_TILE_M + lane_row + i * 8, kk + 0], S.f32
                )
                a_val1 = S.convert(
                    a_tile[1, wave_m * WAVE_TILE_M + lane_row + i * 8, kk + 1], S.f32
                )
                for j in S.range(LANE_TILE_N):
                    idx = i * LANE_TILE_N + j
                    b_val0 = S.convert(
                        b_tile[1, kk + 0, wave_n * WAVE_TILE_N + lane_col + j * 8], S.f32
                    )
                    b_val1 = S.convert(
                        b_tile[1, kk + 1, wave_n * WAVE_TILE_N + lane_col + j * 8], S.f32
                    )
                    c_old = c_reg[idx]
                    c_reg[idx] = c_old + a_val0 * b_val0 + a_val1 * b_val1

        S.syncthreads()

    for i in S.range(LANE_TILE_M):
        row = m0 + wave_m * WAVE_TILE_M + lane_row + i * 8
        for j in S.range(LANE_TILE_N):
            idx = i * LANE_TILE_N + j
            col = n0 + wave_n * WAVE_TILE_N + lane_col + j * 8
            C[row, col] = S.convert(c_reg[idx], S.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if tuple(A.shape) != (M, K) or tuple(B.shape) != (N, K):
            raise ValueError("ModelNew expects A=(2048, 8192) and B=(4096, 8192)")

        A2 = A.contiguous()
        B2 = B.transpose(-2, -1).contiguous()
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)
        gemm_kernel[lambda: ((N // BLOCK_N, M // BLOCK_M, 1), (THREADS_PER_BLOCK, 1, 1))](
            A2, B2, C
        )
        return C
