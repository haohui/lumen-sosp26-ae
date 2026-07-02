import torch
import torch.nn as nn

import substrate
import substrate.language as S


N = 4096
BLOCK_M = 64
BLOCK_N = 64
WAVE_SIZE = 64
WAVES_M = 2
WAVES_N = 2
BLOCK_THREADS = WAVE_SIZE * WAVES_M * WAVES_N
K_CHUNK = 16
K_TILES = N // K_CHUNK
K_TILE_PAIRS = K_TILES // 2
RANGE_BYTES = N * N * 2
WAVES_PER_BLOCK = WAVES_M * WAVES_N
LDS_STAGE_WORDS = WAVES_PER_BLOCK * 64 * 4
LDS_STAGES = 2


@substrate.jit
def gemm_kernel(
    A: S.Tensor((N, N), S.bf16),
    B: S.Tensor((N, N), S.bf16),
    C: S.Tensor((N, N), S.bf16),
):
    tid = S.thread_id(0)
    wave_id = tid >> 6
    lane = tid % WAVE_SIZE
    warp_row = wave_id >> 1
    warp_col = wave_id % 2

    block_row = S.block_id(1) * BLOCK_M
    block_col = S.block_id(0) * BLOCK_N
    tile_row_base = block_row + warp_row * 32
    tile_col_base = block_col + warp_col * 32

    zero = S.convert(0, S.i32)
    range_bytes = S.convert(RANGE_BYTES, S.i32)
    a_rsrc = S.amdgpu.make_rsrc(A, range_bytes)
    b_rsrc = S.amdgpu.make_rsrc(B, range_bytes)

    a_smem = S.make_shared((LDS_STAGES * LDS_STAGE_WORDS,), S.u32)
    b_smem = S.make_shared((LDS_STAGES * LDS_STAGE_WORDS,), S.u32)
    acc = S.make_local((16,), S.f32)
    frag_layout = S.make_layout((2, 4), (4, 1))
    b_lane_layout = S.make_layout((LDS_STAGES * WAVES_PER_BLOCK * 64, 8), (8, 1))
    b_smem_bf16 = S.view(b_smem, S.bf16, b_lane_layout)

    for i in S.range(16):
        acc[i] = S.convert(0.0, S.f32)

    a_row = lane >> 1
    a_half = lane & 1
    b_k = lane >> 2
    b_col_frag = lane & 3

    k_base = 0
    a_byte_off = ((tile_row_base + a_row) * N + k_base + a_half * 8) * 2
    a_words = S.amdgpu.raw_buffer_load_x4(a_rsrc, zero, S.convert(a_byte_off, S.i32), 0)

    a_stage_base = 0
    a_base0 = a_stage_base + (wave_id * 64 + a_row) * 4 + a_half * 2
    a_base1 = a_stage_base + (wave_id * 64 + a_row + 32) * 4 + a_half * 2
    a_smem[a_base0] = a_words[0]
    a_smem[a_base0 + 1] = a_words[1]
    a_smem[a_base1] = a_words[2]
    a_smem[a_base1 + 1] = a_words[3]

    b_byte_off = ((k_base + b_k) * N + tile_col_base + b_col_frag * 8) * 2
    b_words = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero, S.convert(b_byte_off, S.i32), 0)
    b_vals = S.view(b_words, S.Tensor((2, 4, 1), S.bf16))
    b_step = b_k >> 3
    b_quarter = (b_k >> 2) & 1
    b_elem = b_k & 3
    b_dst_idx = b_step * 4 + b_elem
    b_lane_base = wave_id * 64 + b_quarter * 32 + b_col_frag * 8
    for s in S.range(4):
        b_smem_bf16[b_lane_base + s, b_dst_idx] = b_vals[0, s, 0]
        b_smem_bf16[b_lane_base + 4 + s, b_dst_idx] = b_vals[1, s, 0]

    S.syncthreads()

    for pair_idx in S.range(K_TILE_PAIRS - 1):
        even_stage = 0
        odd_stage = LDS_STAGE_WORDS

        odd_k_base = (pair_idx * 2 + 1) * K_CHUNK
        a_odd_byte_off = ((tile_row_base + a_row) * N + odd_k_base + a_half * 8) * 2
        a_odd_words = S.amdgpu.raw_buffer_load_x4(
            a_rsrc, zero, S.convert(a_odd_byte_off, S.i32), 0
        )

        b_odd_byte_off = ((odd_k_base + b_k) * N + tile_col_base + b_col_frag * 8) * 2
        b_odd_words = S.amdgpu.raw_buffer_load_x4(
            b_rsrc, zero, S.convert(b_odd_byte_off, S.i32), 0
        )
        b_odd_vals = S.view(b_odd_words, S.Tensor((2, 4, 1), S.bf16))

        a_base0 = odd_stage + (wave_id * 64 + a_row) * 4 + a_half * 2
        a_base1 = odd_stage + (wave_id * 64 + a_row + 32) * 4 + a_half * 2
        a_smem[a_base0] = a_odd_words[0]
        a_smem[a_base0 + 1] = a_odd_words[1]
        a_smem[a_base1] = a_odd_words[2]
        a_smem[a_base1 + 1] = a_odd_words[3]
        b_lane_base = (odd_stage // 4) + wave_id * 64 + b_quarter * 32 + b_col_frag * 8
        for s in S.range(4):
            b_smem_bf16[b_lane_base + s, b_dst_idx] = b_odd_vals[0, s, 0]
            b_smem_bf16[b_lane_base + 4 + s, b_dst_idx] = b_odd_vals[1, s, 0]

        lane_base = even_stage + (wave_id * 64 + lane) * 4
        a_even_lane_words = S.subview(a_smem, (lane_base,), (4,), (1,))
        b_even_lane_words = S.subview(b_smem, (lane_base,), (4,), (1,))
        a_even_frag = S.view(a_even_lane_words, S.bf16, frag_layout)
        b_even_frag = S.view(b_even_lane_words, S.bf16, frag_layout)

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_even_frag[0], b_even_frag[0], acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_even_frag[1], b_even_frag[1], acc)

        S.syncthreads()

        next_even_k_base = (pair_idx * 2 + 2) * K_CHUNK
        a_next_even_byte_off = ((tile_row_base + a_row) * N + next_even_k_base + a_half * 8) * 2
        a_next_even_words = S.amdgpu.raw_buffer_load_x4(
            a_rsrc, zero, S.convert(a_next_even_byte_off, S.i32), 0
        )

        b_next_even_byte_off = ((next_even_k_base + b_k) * N + tile_col_base + b_col_frag * 8) * 2
        b_next_even_words = S.amdgpu.raw_buffer_load_x4(
            b_rsrc, zero, S.convert(b_next_even_byte_off, S.i32), 0
        )
        b_next_even_vals = S.view(b_next_even_words, S.Tensor((2, 4, 1), S.bf16))

        a_base0 = even_stage + (wave_id * 64 + a_row) * 4 + a_half * 2
        a_base1 = even_stage + (wave_id * 64 + a_row + 32) * 4 + a_half * 2
        a_smem[a_base0] = a_next_even_words[0]
        a_smem[a_base0 + 1] = a_next_even_words[1]
        a_smem[a_base1] = a_next_even_words[2]
        a_smem[a_base1 + 1] = a_next_even_words[3]
        b_lane_base = (even_stage // 4) + wave_id * 64 + b_quarter * 32 + b_col_frag * 8
        for s in S.range(4):
            b_smem_bf16[b_lane_base + s, b_dst_idx] = b_next_even_vals[0, s, 0]
            b_smem_bf16[b_lane_base + 4 + s, b_dst_idx] = b_next_even_vals[1, s, 0]

        lane_base = odd_stage + (wave_id * 64 + lane) * 4
        a_odd_lane_words = S.subview(a_smem, (lane_base,), (4,), (1,))
        b_odd_lane_words = S.subview(b_smem, (lane_base,), (4,), (1,))
        a_odd_frag = S.view(a_odd_lane_words, S.bf16, frag_layout)
        b_odd_frag = S.view(b_odd_lane_words, S.bf16, frag_layout)

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_odd_frag[0], b_odd_frag[0], acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_odd_frag[1], b_odd_frag[1], acc)

        S.syncthreads()

    final_pair = K_TILE_PAIRS - 1
    even_stage = 0
    odd_stage = LDS_STAGE_WORDS

    odd_k_base = (final_pair * 2 + 1) * K_CHUNK
    a_odd_byte_off = ((tile_row_base + a_row) * N + odd_k_base + a_half * 8) * 2
    a_odd_words = S.amdgpu.raw_buffer_load_x4(a_rsrc, zero, S.convert(a_odd_byte_off, S.i32), 0)

    b_odd_byte_off = ((odd_k_base + b_k) * N + tile_col_base + b_col_frag * 8) * 2
    b_odd_words = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero, S.convert(b_odd_byte_off, S.i32), 0)
    b_odd_vals = S.view(b_odd_words, S.Tensor((2, 4, 1), S.bf16))

    a_base0 = odd_stage + (wave_id * 64 + a_row) * 4 + a_half * 2
    a_base1 = odd_stage + (wave_id * 64 + a_row + 32) * 4 + a_half * 2
    a_smem[a_base0] = a_odd_words[0]
    a_smem[a_base0 + 1] = a_odd_words[1]
    a_smem[a_base1] = a_odd_words[2]
    a_smem[a_base1 + 1] = a_odd_words[3]
    b_lane_base = (odd_stage // 4) + wave_id * 64 + b_quarter * 32 + b_col_frag * 8
    for s in S.range(4):
        b_smem_bf16[b_lane_base + s, b_dst_idx] = b_odd_vals[0, s, 0]
        b_smem_bf16[b_lane_base + 4 + s, b_dst_idx] = b_odd_vals[1, s, 0]

    lane_base = even_stage + (wave_id * 64 + lane) * 4
    a_final_even_lane_words = S.subview(a_smem, (lane_base,), (4,), (1,))
    b_final_even_lane_words = S.subview(b_smem, (lane_base,), (4,), (1,))
    a_final_even_frag = S.view(a_final_even_lane_words, S.bf16, frag_layout)
    b_final_even_frag = S.view(b_final_even_lane_words, S.bf16, frag_layout)

    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_final_even_frag[0], b_final_even_frag[0], acc)
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_final_even_frag[1], b_final_even_frag[1], acc)

    S.syncthreads()

    lane_base = odd_stage + (wave_id * 64 + lane) * 4
    a_final_odd_lane_words = S.subview(a_smem, (lane_base,), (4,), (1,))
    b_final_odd_lane_words = S.subview(b_smem, (lane_base,), (4,), (1,))
    a_final_odd_frag = S.view(a_final_odd_lane_words, S.bf16, frag_layout)
    b_final_odd_frag = S.view(b_final_odd_lane_words, S.bf16, frag_layout)

    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_final_odd_frag[0], b_final_odd_frag[0], acc)
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_final_odd_frag[1], b_final_odd_frag[1], acc)

    col = tile_col_base + (lane % 32)
    lane_row_quad = 4 * (lane >> 5)
    for acc_idx in S.range(16):
        row = tile_row_base + 8 * (acc_idx >> 2) + lane_row_quad + (acc_idx & 3)
        C[row, col] = S.convert(acc[acc_idx], S.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        if tuple(A.shape) != (N, N) or tuple(B.shape) != (N, N):
            raise ValueError(f"expected {(N, N)} inputs, got {tuple(A.shape)} and {tuple(B.shape)}")
        if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
            raise TypeError(f"expected bfloat16 inputs, got {A.dtype} and {B.dtype}")
        if A.device != B.device:
            raise ValueError("A and B must be on the same device")

        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((N, N), device=A.device, dtype=torch.bfloat16)
        gemm_kernel[lambda: ((N // BLOCK_N, N // BLOCK_M, 1), (BLOCK_THREADS, 1, 1))](A, B, C)
        return C
