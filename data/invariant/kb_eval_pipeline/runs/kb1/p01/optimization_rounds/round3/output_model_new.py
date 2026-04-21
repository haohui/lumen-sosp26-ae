import torch
import torch.nn as nn

import substrate
import substrate.language as S


M = 4096
K = 4096
N = 4096

WAVE_SIZE = 64
WAVES_M = 2
WAVES_N = 2
NUM_WAVES = WAVES_M * WAVES_N
THREADS = WAVE_SIZE * NUM_WAVES

WAVE_TILE_M = 32
WAVE_TILE_N = 32
BLOCK_TILE_M = WAVES_M * WAVE_TILE_M
BLOCK_TILE_N = WAVES_N * WAVE_TILE_N
BLOCK_TILE_K = 16
K_TILE_PAIRS = K // (2 * BLOCK_TILE_K)

A_NUM_CHUNKS = BLOCK_TILE_M * BLOCK_TILE_K // 8
B_NUM_CHUNKS = BLOCK_TILE_K * BLOCK_TILE_N // 8
COPY_THREADS = WAVE_SIZE * 2

# Raw buffer range is expressed in bytes. With the range encoded in the
# resource descriptor, OOB loads are defined to return zero.
A_BYTE_RANGE = M * K * 2
B_BYTE_RANGE = K * N * 2


@substrate.jit
def gemm_kernel(
    A: S.Tensor((4096, 4096), S.bf16),
    B: S.Tensor((4096, 4096), S.bf16),
    C: S.Tensor((4096, 4096), S.bf16),
):
    tid = S.thread_id(0)
    lane = tid % WAVE_SIZE
    warp_id = tid // WAVE_SIZE
    warp_row = warp_id // WAVES_N
    warp_col = warp_id % WAVES_N

    block_row = S.block_id(1) * BLOCK_TILE_M
    block_col = S.block_id(0) * BLOCK_TILE_N
    wave_row_base = block_row + warp_row * WAVE_TILE_M
    wave_col_base = block_col + warp_col * WAVE_TILE_N

    a_rsrc = S.amdgpu.make_rsrc(A, A_BYTE_RANGE)
    b_rsrc = S.amdgpu.make_rsrc(B, B_BYTE_RANGE)

    a_stage = S.make_shared((2, BLOCK_TILE_M, BLOCK_TILE_K), S.bf16)
    b_stage = S.make_shared((2, BLOCK_TILE_K, BLOCK_TILE_N), S.bf16)
    a_lds = S.make_shared((2, WAVES_M, WAVE_SIZE, 4), S.u32)
    b_lds = S.make_shared((2, WAVES_N, WAVE_SIZE, 4), S.u32)

    a_stage_u32 = S.view(a_stage, S.Tensor((2, BLOCK_TILE_M, BLOCK_TILE_K // 8, 4), S.u32))
    b_stage_u32 = S.view(b_stage, S.Tensor((2, BLOCK_TILE_K, BLOCK_TILE_N // 8, 4), S.u32))
    a_lds_bf16 = S.view(a_lds, S.Tensor((2, WAVES_M, WAVE_SIZE, 2, 4, 1), S.bf16))
    b_lds_bf16 = S.view(b_lds, S.Tensor((2, WAVES_N, WAVE_SIZE, 2, 4, 1), S.bf16))

    acc = S.full((16,), 0.0, S.f32)

    if tid < A_NUM_CHUNKS:
        row_in_block = tid // 2
        k_chunk = tid % 2
        global_row = block_row + row_in_block
        global_col = k_chunk * 8
        byte_offset = S.convert((global_row * K + global_col) * 2, S.i32)
        a_stage_u32[0, row_in_block, k_chunk] = S.amdgpu.raw_buffer_load_x4(
            a_rsrc,
            byte_offset,
            S.convert(0, S.i32),
            S.convert(0, S.i32),
        )

    if tid < B_NUM_CHUNKS:
        k_row = tid // (BLOCK_TILE_N // 8)
        col_chunk = tid % (BLOCK_TILE_N // 8)
        global_row = k_row
        global_col = block_col + col_chunk * 8
        byte_offset = S.convert((global_row * N + global_col) * 2, S.i32)
        b_stage_u32[0, k_row, col_chunk] = S.amdgpu.raw_buffer_load_x4(
            b_rsrc,
            byte_offset,
            S.convert(0, S.i32),
            S.convert(0, S.i32),
        )

    S.syncthreads()

    if tid < COPY_THREADS:
        a_wave = tid // WAVE_SIZE
        a_lane = tid % WAVE_SIZE
        a_row = a_wave * WAVE_TILE_M + (a_lane % WAVE_TILE_M)
        a_k_base = 4 * (a_lane // WAVE_TILE_M)
        for e in S.range(4):
            a_lds_bf16[0, a_wave, a_lane, 0, e, 0] = a_stage[0, a_row, a_k_base + e]
            a_lds_bf16[0, a_wave, a_lane, 1, e, 0] = a_stage[0, a_row, 8 + a_k_base + e]

        b_wave = tid // WAVE_SIZE
        b_lane = tid % WAVE_SIZE
        b_col = b_wave * WAVE_TILE_N + (b_lane % WAVE_TILE_N)
        b_k_base = 4 * (b_lane // WAVE_TILE_N)
        for e in S.range(4):
            b_lds_bf16[0, b_wave, b_lane, 0, e, 0] = b_stage[0, b_k_base + e, b_col]
            b_lds_bf16[0, b_wave, b_lane, 1, e, 0] = b_stage[0, 8 + b_k_base + e, b_col]

    S.syncthreads()

    if tid < A_NUM_CHUNKS:
        row_in_block = tid // 2
        k_chunk = tid % 2
        global_row = block_row + row_in_block
        global_col = BLOCK_TILE_K + k_chunk * 8
        byte_offset = S.convert((global_row * K + global_col) * 2, S.i32)
        a_stage_u32[1, row_in_block, k_chunk] = S.amdgpu.raw_buffer_load_x4(
            a_rsrc,
            byte_offset,
            S.convert(0, S.i32),
            S.convert(0, S.i32),
        )

    if tid < B_NUM_CHUNKS:
        k_row = tid // (BLOCK_TILE_N // 8)
        col_chunk = tid % (BLOCK_TILE_N // 8)
        global_row = BLOCK_TILE_K + k_row
        global_col = block_col + col_chunk * 8
        byte_offset = S.convert((global_row * N + global_col) * 2, S.i32)
        b_stage_u32[1, k_row, col_chunk] = S.amdgpu.raw_buffer_load_x4(
            b_rsrc,
            byte_offset,
            S.convert(0, S.i32),
            S.convert(0, S.i32),
        )

    S.syncthreads()

    if tid < COPY_THREADS:
        a_wave = tid // WAVE_SIZE
        a_lane = tid % WAVE_SIZE
        a_row = a_wave * WAVE_TILE_M + (a_lane % WAVE_TILE_M)
        a_k_base = 4 * (a_lane // WAVE_TILE_M)
        for e in S.range(4):
            a_lds_bf16[1, a_wave, a_lane, 0, e, 0] = a_stage[1, a_row, a_k_base + e]
            a_lds_bf16[1, a_wave, a_lane, 1, e, 0] = a_stage[1, a_row, 8 + a_k_base + e]

        b_wave = tid // WAVE_SIZE
        b_lane = tid % WAVE_SIZE
        b_col = b_wave * WAVE_TILE_N + (b_lane % WAVE_TILE_N)
        b_k_base = 4 * (b_lane // WAVE_TILE_N)
        for e in S.range(4):
            b_lds_bf16[1, b_wave, b_lane, 0, e, 0] = b_stage[1, b_k_base + e, b_col]
            b_lds_bf16[1, b_wave, b_lane, 1, e, 0] = b_stage[1, 8 + b_k_base + e, b_col]

    S.syncthreads()

    for k_pair in S.range(K_TILE_PAIRS - 1):
        next_tile0 = 2 * k_pair + 2
        next_tile1 = next_tile0 + 1

        a_frag_u32 = a_lds[0, warp_row, lane]
        b_frag_u32 = b_lds[0, warp_col, lane]
        a_frag = S.view(a_frag_u32, S.Tensor((2, 4, 1), S.bf16))
        b_frag = S.view(b_frag_u32, S.Tensor((2, 4, 1), S.bf16))

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], acc)

        if tid < A_NUM_CHUNKS:
            row_in_block = tid // 2
            k_chunk = tid % 2
            global_row = block_row + row_in_block
            global_col = next_tile0 * BLOCK_TILE_K + k_chunk * 8
            byte_offset = S.convert((global_row * K + global_col) * 2, S.i32)
            a_stage_u32[0, row_in_block, k_chunk] = S.amdgpu.raw_buffer_load_x4(
                a_rsrc,
                byte_offset,
                S.convert(0, S.i32),
                S.convert(0, S.i32),
            )

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[1], b_frag[1], acc)

        if tid < B_NUM_CHUNKS:
            k_row = tid // (BLOCK_TILE_N // 8)
            col_chunk = tid % (BLOCK_TILE_N // 8)
            global_row = next_tile0 * BLOCK_TILE_K + k_row
            global_col = block_col + col_chunk * 8
            byte_offset = S.convert((global_row * N + global_col) * 2, S.i32)
            b_stage_u32[0, k_row, col_chunk] = S.amdgpu.raw_buffer_load_x4(
                b_rsrc,
                byte_offset,
                S.convert(0, S.i32),
                S.convert(0, S.i32),
            )

        S.syncthreads()

        if tid < COPY_THREADS:
            a_wave = tid // WAVE_SIZE
            a_lane = tid % WAVE_SIZE
            a_row = a_wave * WAVE_TILE_M + (a_lane % WAVE_TILE_M)
            a_k_base = 4 * (a_lane // WAVE_TILE_M)
            for e in S.range(4):
                a_lds_bf16[0, a_wave, a_lane, 0, e, 0] = a_stage[0, a_row, a_k_base + e]
                a_lds_bf16[0, a_wave, a_lane, 1, e, 0] = a_stage[0, a_row, 8 + a_k_base + e]

        S.syncthreads()

        if tid < COPY_THREADS:
            b_wave = tid // WAVE_SIZE
            b_lane = tid % WAVE_SIZE
            b_col = b_wave * WAVE_TILE_N + (b_lane % WAVE_TILE_N)
            b_k_base = 4 * (b_lane // WAVE_TILE_N)
            for e in S.range(4):
                b_lds_bf16[0, b_wave, b_lane, 0, e, 0] = b_stage[0, b_k_base + e, b_col]
                b_lds_bf16[0, b_wave, b_lane, 1, e, 0] = b_stage[0, 8 + b_k_base + e, b_col]

        S.syncthreads()

        a_frag_u32 = a_lds[1, warp_row, lane]
        b_frag_u32 = b_lds[1, warp_col, lane]
        a_frag = S.view(a_frag_u32, S.Tensor((2, 4, 1), S.bf16))
        b_frag = S.view(b_frag_u32, S.Tensor((2, 4, 1), S.bf16))

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], acc)

        if tid < A_NUM_CHUNKS:
            row_in_block = tid // 2
            k_chunk = tid % 2
            global_row = block_row + row_in_block
            global_col = next_tile1 * BLOCK_TILE_K + k_chunk * 8
            byte_offset = S.convert((global_row * K + global_col) * 2, S.i32)
            a_stage_u32[1, row_in_block, k_chunk] = S.amdgpu.raw_buffer_load_x4(
                a_rsrc,
                byte_offset,
                S.convert(0, S.i32),
                S.convert(0, S.i32),
            )

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[1], b_frag[1], acc)

        if tid < B_NUM_CHUNKS:
            k_row = tid // (BLOCK_TILE_N // 8)
            col_chunk = tid % (BLOCK_TILE_N // 8)
            global_row = next_tile1 * BLOCK_TILE_K + k_row
            global_col = block_col + col_chunk * 8
            byte_offset = S.convert((global_row * N + global_col) * 2, S.i32)
            b_stage_u32[1, k_row, col_chunk] = S.amdgpu.raw_buffer_load_x4(
                b_rsrc,
                byte_offset,
                S.convert(0, S.i32),
                S.convert(0, S.i32),
            )

        S.syncthreads()

        if tid < COPY_THREADS:
            a_wave = tid // WAVE_SIZE
            a_lane = tid % WAVE_SIZE
            a_row = a_wave * WAVE_TILE_M + (a_lane % WAVE_TILE_M)
            a_k_base = 4 * (a_lane // WAVE_TILE_M)
            for e in S.range(4):
                a_lds_bf16[1, a_wave, a_lane, 0, e, 0] = a_stage[1, a_row, a_k_base + e]
                a_lds_bf16[1, a_wave, a_lane, 1, e, 0] = a_stage[1, a_row, 8 + a_k_base + e]

        S.syncthreads()

        if tid < COPY_THREADS:
            b_wave = tid // WAVE_SIZE
            b_lane = tid % WAVE_SIZE
            b_col = b_wave * WAVE_TILE_N + (b_lane % WAVE_TILE_N)
            b_k_base = 4 * (b_lane // WAVE_TILE_N)
            for e in S.range(4):
                b_lds_bf16[1, b_wave, b_lane, 0, e, 0] = b_stage[1, b_k_base + e, b_col]
                b_lds_bf16[1, b_wave, b_lane, 1, e, 0] = b_stage[1, 8 + b_k_base + e, b_col]

        S.syncthreads()

    a_frag_u32 = a_lds[0, warp_row, lane]
    b_frag_u32 = b_lds[0, warp_col, lane]
    a_frag = S.view(a_frag_u32, S.Tensor((2, 4, 1), S.bf16))
    b_frag = S.view(b_frag_u32, S.Tensor((2, 4, 1), S.bf16))
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], acc)
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[1], b_frag[1], acc)

    a_frag_u32 = a_lds[1, warp_row, lane]
    b_frag_u32 = b_lds[1, warp_col, lane]
    a_frag = S.view(a_frag_u32, S.Tensor((2, 4, 1), S.bf16))
    b_frag = S.view(b_frag_u32, S.Tensor((2, 4, 1), S.bf16))
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], acc)
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[1], b_frag[1], acc)

    lane_col = lane % 32
    lane_row_group = lane // 32
    for acc_idx in S.range(16):
        out_col = wave_col_base + lane_col
        out_row = (
            wave_row_base
            + 8 * (acc_idx // 4)
            + 4 * lane_row_group
            + (acc_idx % 4)
        )
        C[out_row, out_col] = S.convert(acc[acc_idx], S.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if tuple(A.shape) != (4096, 4096) or tuple(B.shape) != (4096, 4096):
            return torch.matmul(A, B)
        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((4096, 4096), device=A.device, dtype=A.dtype)
        gemm_kernel[lambda: (((N // BLOCK_TILE_N), (M // BLOCK_TILE_M), 1), (THREADS, 1, 1))](
            A, B, C
        )
        return C
