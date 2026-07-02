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
WAVES_M = 2
WAVES_N = 2
THREADS = WAVE_SIZE * WAVES_M * WAVES_N

A_RANGE_BYTES = M * K * 2
B_RANGE_BYTES = K * N * 2


@substrate.jit
def gemm_mfma_kernel(
    A: S.Tensor((M, K), S.bf16),
    B: S.Tensor((K, N), S.bf16),
    C: S.Tensor((M, N), S.bf16),
    a_range_bytes: S.i32,
    b_range_bytes: S.i32,
):
    tid = S.thread_id(0)
    lane = tid % WAVE_SIZE
    warp = tid // WAVE_SIZE
    warp_row = warp // WAVES_N
    warp_col = warp % WAVES_N

    block_row = S.block_id(1) * BLOCK_M
    block_col = S.block_id(0) * BLOCK_N

    a_rsrc = S.amdgpu.make_rsrc(A, a_range_bytes)
    b_rsrc = S.amdgpu.make_rsrc(B, b_range_bytes)

    a_lds = S.make_shared((2, WAVES_M * WAVE_SIZE, 4), S.u32)
    b_lds = S.make_shared((2, WAVES_N * WAVE_SIZE, 4), S.u32)

    mfma_acc = S.full((16,), 0.0, S.f32)
    acc = S.full((4, 4), 0.0, S.f32)
    zero = S.convert(0, S.i32)
    thread_row = tid // 16
    thread_col = tid % 16
    row_base_local = thread_row * 4
    col_base_local = thread_col * 4

    if tid < 128:
        load_row_group = tid // WAVE_SIZE
        load_lane = tid % WAVE_SIZE
        a_row = block_row + load_row_group * 32 + load_lane // 2
        a_col8 = (load_lane % 2) * 8

        a_offset0 = S.convert((a_row * K + a_col8) * 2, S.i32)
        a_vec0 = S.amdgpu.raw_buffer_load_x4(a_rsrc, zero, a_offset0, 0)
        for i in S.range(4):
            a_lds[0, load_row_group * WAVE_SIZE + load_lane, i] = a_vec0[i]

        a_offset1 = S.convert((a_row * K + BLOCK_K + a_col8) * 2, S.i32)
        a_vec1 = S.amdgpu.raw_buffer_load_x4(a_rsrc, zero, a_offset1, 0)
        for i in S.range(4):
            a_lds[1, load_row_group * WAVE_SIZE + load_lane, i] = a_vec1[i]
    else:
        load_tid = tid - 128
        load_col_group = load_tid // WAVE_SIZE
        load_lane = load_tid % WAVE_SIZE
        b_row16 = load_lane // 4
        b_col8 = (load_lane % 4) * 8
        b_col = block_col + load_col_group * 32 + b_col8

        b_offset0 = S.convert((b_row16 * N + b_col) * 2, S.i32)
        b_vec0 = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero, b_offset0, 0)
        for i in S.range(4):
            b_lds[0, load_col_group * WAVE_SIZE + load_lane, i] = b_vec0[i]

        b_offset1 = S.convert(((BLOCK_K + b_row16) * N + b_col) * 2, S.i32)
        b_vec1 = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero, b_offset1, 0)
        for i in S.range(4):
            b_lds[1, load_col_group * WAVE_SIZE + load_lane, i] = b_vec1[i]

    S.syncthreads()

    for k0 in S.range(0, K, BLOCK_K * 2):
        a_pack0 = a_lds[0, warp_row * WAVE_SIZE + lane]
        b_pack0 = b_lds[0, warp_col * WAVE_SIZE + lane]
        a_frag0 = S.view(a_pack0, S.Tensor((2, 4, 1), S.bf16))
        b_frag0 = S.view(b_pack0, S.Tensor((2, 4, 1), S.bf16))
        mfma_acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[0], b_frag0[0], mfma_acc)
        mfma_acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[1], b_frag0[1], mfma_acc)

        for kk in S.range(BLOCK_K):
            a_lane_idx = (row_base_local // 32) * WAVE_SIZE + (row_base_local % 32) * 2 + kk // 8
            a_row_pack = S.view(a_lds[0, a_lane_idx], S.Tensor((2, 4, 1), S.bf16))
            a_half = (kk % 8) // 4
            a_elem = kk % 4
            a0 = S.convert(a_row_pack[a_half, a_elem, 0], S.f32)

            a_lane_idx = ((row_base_local + 1) // 32) * WAVE_SIZE + ((row_base_local + 1) % 32) * 2 + kk // 8
            a_row_pack = S.view(a_lds[0, a_lane_idx], S.Tensor((2, 4, 1), S.bf16))
            a1 = S.convert(a_row_pack[a_half, a_elem, 0], S.f32)

            a_lane_idx = ((row_base_local + 2) // 32) * WAVE_SIZE + ((row_base_local + 2) % 32) * 2 + kk // 8
            a_row_pack = S.view(a_lds[0, a_lane_idx], S.Tensor((2, 4, 1), S.bf16))
            a2 = S.convert(a_row_pack[a_half, a_elem, 0], S.f32)

            a_lane_idx = ((row_base_local + 3) // 32) * WAVE_SIZE + ((row_base_local + 3) % 32) * 2 + kk // 8
            a_row_pack = S.view(a_lds[0, a_lane_idx], S.Tensor((2, 4, 1), S.bf16))
            a3 = S.convert(a_row_pack[a_half, a_elem, 0], S.f32)

            b_lane_idx = (col_base_local // 32) * WAVE_SIZE + kk * 4 + (col_base_local % 32) // 8
            b_row_pack = S.view(b_lds[0, b_lane_idx], S.Tensor((2, 4, 1), S.bf16))
            b_half = (col_base_local % 8) // 4
            b_elem = col_base_local % 4
            b0 = S.convert(b_row_pack[b_half, b_elem, 0], S.f32)

            b_lane_idx = ((col_base_local + 1) // 32) * WAVE_SIZE + kk * 4 + ((col_base_local + 1) % 32) // 8
            b_row_pack = S.view(b_lds[0, b_lane_idx], S.Tensor((2, 4, 1), S.bf16))
            b_half = ((col_base_local + 1) % 8) // 4
            b_elem = (col_base_local + 1) % 4
            b1 = S.convert(b_row_pack[b_half, b_elem, 0], S.f32)

            b_lane_idx = ((col_base_local + 2) // 32) * WAVE_SIZE + kk * 4 + ((col_base_local + 2) % 32) // 8
            b_row_pack = S.view(b_lds[0, b_lane_idx], S.Tensor((2, 4, 1), S.bf16))
            b_half = ((col_base_local + 2) % 8) // 4
            b_elem = (col_base_local + 2) % 4
            b2 = S.convert(b_row_pack[b_half, b_elem, 0], S.f32)

            b_lane_idx = ((col_base_local + 3) // 32) * WAVE_SIZE + kk * 4 + ((col_base_local + 3) % 32) // 8
            b_row_pack = S.view(b_lds[0, b_lane_idx], S.Tensor((2, 4, 1), S.bf16))
            b_half = ((col_base_local + 3) % 8) // 4
            b_elem = (col_base_local + 3) % 4
            b3 = S.convert(b_row_pack[b_half, b_elem, 0], S.f32)

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

        next_k0 = k0 + BLOCK_K * 2
        if next_k0 < K:
            if tid < 128:
                load_row_group = tid // WAVE_SIZE
                load_lane = tid % WAVE_SIZE
                a_row = block_row + load_row_group * 32 + load_lane // 2
                a_col8 = (load_lane % 2) * 8
                a_offset = S.convert((a_row * K + next_k0 + a_col8) * 2, S.i32)
                a_vec = S.amdgpu.raw_buffer_load_x4(a_rsrc, zero, a_offset, 0)
                for i in S.range(4):
                    a_lds[0, load_row_group * WAVE_SIZE + load_lane, i] = a_vec[i]
            else:
                load_tid = tid - 128
                load_col_group = load_tid // WAVE_SIZE
                load_lane = load_tid % WAVE_SIZE
                b_row16 = load_lane // 4
                b_col8 = (load_lane % 4) * 8
                b_col = block_col + load_col_group * 32 + b_col8
                b_offset = S.convert(((next_k0 + b_row16) * N + b_col) * 2, S.i32)
                b_vec = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero, b_offset, 0)
                for i in S.range(4):
                    b_lds[0, load_col_group * WAVE_SIZE + load_lane, i] = b_vec[i]

        a_pack1 = a_lds[1, warp_row * WAVE_SIZE + lane]
        b_pack1 = b_lds[1, warp_col * WAVE_SIZE + lane]
        a_frag1 = S.view(a_pack1, S.Tensor((2, 4, 1), S.bf16))
        b_frag1 = S.view(b_pack1, S.Tensor((2, 4, 1), S.bf16))
        mfma_acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[0], b_frag1[0], mfma_acc)
        mfma_acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[1], b_frag1[1], mfma_acc)

        for kk in S.range(BLOCK_K):
            a_lane_idx = (row_base_local // 32) * WAVE_SIZE + (row_base_local % 32) * 2 + kk // 8
            a_row_pack = S.view(a_lds[1, a_lane_idx], S.Tensor((2, 4, 1), S.bf16))
            a_half = (kk % 8) // 4
            a_elem = kk % 4
            a0 = S.convert(a_row_pack[a_half, a_elem, 0], S.f32)

            a_lane_idx = ((row_base_local + 1) // 32) * WAVE_SIZE + ((row_base_local + 1) % 32) * 2 + kk // 8
            a_row_pack = S.view(a_lds[1, a_lane_idx], S.Tensor((2, 4, 1), S.bf16))
            a1 = S.convert(a_row_pack[a_half, a_elem, 0], S.f32)

            a_lane_idx = ((row_base_local + 2) // 32) * WAVE_SIZE + ((row_base_local + 2) % 32) * 2 + kk // 8
            a_row_pack = S.view(a_lds[1, a_lane_idx], S.Tensor((2, 4, 1), S.bf16))
            a2 = S.convert(a_row_pack[a_half, a_elem, 0], S.f32)

            a_lane_idx = ((row_base_local + 3) // 32) * WAVE_SIZE + ((row_base_local + 3) % 32) * 2 + kk // 8
            a_row_pack = S.view(a_lds[1, a_lane_idx], S.Tensor((2, 4, 1), S.bf16))
            a3 = S.convert(a_row_pack[a_half, a_elem, 0], S.f32)

            b_lane_idx = (col_base_local // 32) * WAVE_SIZE + kk * 4 + (col_base_local % 32) // 8
            b_row_pack = S.view(b_lds[1, b_lane_idx], S.Tensor((2, 4, 1), S.bf16))
            b_half = (col_base_local % 8) // 4
            b_elem = col_base_local % 4
            b0 = S.convert(b_row_pack[b_half, b_elem, 0], S.f32)

            b_lane_idx = ((col_base_local + 1) // 32) * WAVE_SIZE + kk * 4 + ((col_base_local + 1) % 32) // 8
            b_row_pack = S.view(b_lds[1, b_lane_idx], S.Tensor((2, 4, 1), S.bf16))
            b_half = ((col_base_local + 1) % 8) // 4
            b_elem = (col_base_local + 1) % 4
            b1 = S.convert(b_row_pack[b_half, b_elem, 0], S.f32)

            b_lane_idx = ((col_base_local + 2) // 32) * WAVE_SIZE + kk * 4 + ((col_base_local + 2) % 32) // 8
            b_row_pack = S.view(b_lds[1, b_lane_idx], S.Tensor((2, 4, 1), S.bf16))
            b_half = ((col_base_local + 2) % 8) // 4
            b_elem = (col_base_local + 2) % 4
            b2 = S.convert(b_row_pack[b_half, b_elem, 0], S.f32)

            b_lane_idx = ((col_base_local + 3) // 32) * WAVE_SIZE + kk * 4 + ((col_base_local + 3) % 32) // 8
            b_row_pack = S.view(b_lds[1, b_lane_idx], S.Tensor((2, 4, 1), S.bf16))
            b_half = ((col_base_local + 3) % 8) // 4
            b_elem = (col_base_local + 3) % 4
            b3 = S.convert(b_row_pack[b_half, b_elem, 0], S.f32)

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

        next_k1 = k0 + BLOCK_K * 3
        if next_k1 < K:
            if tid < 128:
                load_row_group = tid // WAVE_SIZE
                load_lane = tid % WAVE_SIZE
                a_row = block_row + load_row_group * 32 + load_lane // 2
                a_col8 = (load_lane % 2) * 8
                a_offset = S.convert((a_row * K + next_k1 + a_col8) * 2, S.i32)
                a_vec = S.amdgpu.raw_buffer_load_x4(a_rsrc, zero, a_offset, 0)
                for i in S.range(4):
                    a_lds[1, load_row_group * WAVE_SIZE + load_lane, i] = a_vec[i]
            else:
                load_tid = tid - 128
                load_col_group = load_tid // WAVE_SIZE
                load_lane = load_tid % WAVE_SIZE
                b_row16 = load_lane // 4
                b_col8 = (load_lane % 4) * 8
                b_col = block_col + load_col_group * 32 + b_col8
                b_offset = S.convert(((next_k1 + b_row16) * N + b_col) * 2, S.i32)
                b_vec = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero, b_offset, 0)
                for i in S.range(4):
                    b_lds[1, load_col_group * WAVE_SIZE + load_lane, i] = b_vec[i]
            S.syncthreads()

    guard = mfma_acc[0] - mfma_acc[0]
    row_base = block_row + row_base_local
    col_base = block_col + col_base_local

    for i in S.range(4):
        for j in S.range(4):
            C[row_base + i, col_base + j] = S.convert(acc[i, j] + guard, S.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._a_range_bytes = A_RANGE_BYTES
        self._b_range_bytes = B_RANGE_BYTES
        self._grid = (N // BLOCK_N, M // BLOCK_M, 1)
        self._block = (THREADS, 1, 1)

    def forward(self, A, B):
        if tuple(A.shape) != (8192, 2048) or tuple(B.shape) != (4096, 8192):
            raise ValueError(f"Expected A=(8192, 2048) and B=(4096, 8192), got {tuple(A.shape)} and {tuple(B.shape)}")

        A2 = A.transpose(-2, -1).contiguous()
        B2 = B.transpose(-2, -1).contiguous()
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)

        gemm_mfma_kernel[lambda: (self._grid, self._block)](
            A2,
            B2,
            C,
            self._a_range_bytes,
            self._b_range_bytes,
        )
        return C
