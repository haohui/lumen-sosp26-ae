import torch
import torch.nn as nn

import substrate
import substrate.language as S


M = 4096
K = 4096
N = 4096

BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 16
WAVE_SIZE = 64
WAVES_PER_BLOCK = 4
THREADS = WAVE_SIZE * WAVES_PER_BLOCK
RSRC_RANGE_BYTES = M * K * 2


@substrate.jit
def tri_gemm_kernel(
    A: S.Tensor((M, K), S.bf16),
    B: S.Tensor((K, N), S.bf16),
    C: S.Tensor((M, N), S.bf16),
):
    tid = S.thread_id(0)
    lane = tid % WAVE_SIZE
    wave = tid // WAVE_SIZE
    warp_row = wave // 2
    warp_col = wave % 2

    block_row = S.block_id(1) * BLOCK_M
    block_col = S.block_id(0) * BLOCK_N

    tile_row_base = block_row + warp_row * 32
    tile_col_base = block_col + warp_col * 32

    c_lane = S.full((16,), 0.0, S.f32)

    shared_a = S.make_shared((2, 2, WAVE_SIZE, 8), S.bf16)
    shared_b = S.make_shared((2, 2, WAVE_SIZE, 8), S.bf16)

    a_rsrc = S.amdgpu.make_rsrc(A, RSRC_RANGE_BYTES)
    b_rsrc = S.amdgpu.make_rsrc(B, RSRC_RANGE_BYTES)

    whole_tile_zero = block_col > block_row
    diag_tile = block_col == block_row

    if not whole_tile_zero:
        k_end = block_row + BLOCK_M

        if tid < 128:
            a_row_local = tid // 2
            a_seg = tid % 2
            a_lane = a_row_local % 32
            a_wave_row = a_row_local // 32
            a_elem_base = a_seg * 4

            a_byte_offset_0 = ((block_row + a_row_local) * K + block_col + a_seg * 8) * 2
            a_vec_0 = S.amdgpu.raw_buffer_load_x4(a_rsrc, a_byte_offset_0, 0, 0)
            a_vals_0 = S.view(a_vec_0, S.Tensor((8,), S.bf16))
            for t in S.range(4):
                shared_a[0, a_wave_row, a_lane + 0, a_elem_base + t] = a_vals_0[t]
                shared_a[0, a_wave_row, a_lane + 32, a_elem_base + t] = a_vals_0[4 + t]

            a_byte_offset_1 = ((block_row + a_row_local) * K + block_col + BLOCK_K + a_seg * 8) * 2
            a_vec_1 = S.amdgpu.raw_buffer_load_x4(a_rsrc, a_byte_offset_1, 0, 0)
            a_vals_1 = S.view(a_vec_1, S.Tensor((8,), S.bf16))
            for t in S.range(4):
                shared_a[1, a_wave_row, a_lane + 0, a_elem_base + t] = a_vals_1[t]
                shared_a[1, a_wave_row, a_lane + 32, a_elem_base + t] = a_vals_1[4 + t]
        else:
            b_tid = tid - 128
            b_row_local = b_tid // 8
            b_seg = b_tid % 8
            b_wave_col = b_seg // 4
            b_seg_local = b_seg % 4
            b_lane_offset = ((b_row_local % 8) // 4) * 32
            b_elem = (b_row_local // 8) * 4 + (b_row_local % 4)

            b_byte_offset_0 = ((block_col + b_row_local) * N + block_col + b_seg * 8) * 2
            b_vec_0 = S.amdgpu.raw_buffer_load_x4(b_rsrc, b_byte_offset_0, 0, 0)
            b_vals_0 = S.view(b_vec_0, S.Tensor((8,), S.bf16))
            for t in S.range(8):
                shared_b[0, b_wave_col, b_lane_offset + b_seg_local * 8 + t, b_elem] = b_vals_0[t]

            b_byte_offset_1 = ((block_col + BLOCK_K + b_row_local) * N + block_col + b_seg * 8) * 2
            b_vec_1 = S.amdgpu.raw_buffer_load_x4(b_rsrc, b_byte_offset_1, 0, 0)
            b_vals_1 = S.view(b_vec_1, S.Tensor((8,), S.bf16))
            for t in S.range(8):
                shared_b[1, b_wave_col, b_lane_offset + b_seg_local * 8 + t, b_elem] = b_vals_1[t]

        S.syncthreads()

        for k0 in S.range(block_col, k_end, BLOCK_K * 2):
            a_frag_0 = S.view(shared_a[0, warp_row, lane], S.Tensor((2, 4, 1), S.bf16))
            b_frag_0 = S.view(shared_b[0, warp_col, lane], S.Tensor((2, 4, 1), S.bf16))
            a0_lo = a_frag_0[0]
            a0_hi = a_frag_0[1]
            b0_lo = b_frag_0[0]
            b0_hi = b_frag_0[1]

            c_lane = S.amdgpu.mfma_32x32x8_bf16_f32(a0_lo, b0_lo, c_lane)
            c_lane = S.amdgpu.mfma_32x32x8_bf16_f32(a0_hi, b0_hi, c_lane)

            S.syncthreads()

            if k0 + BLOCK_K * 2 < k_end:
                if tid < 128:
                    a_row_local = tid // 2
                    a_seg = tid % 2
                    a_lane = a_row_local % 32
                    a_wave_row = a_row_local // 32
                    a_elem_base = a_seg * 4
                    a_byte_offset = ((block_row + a_row_local) * K + k0 + BLOCK_K * 2 + a_seg * 8) * 2
                    a_vec = S.amdgpu.raw_buffer_load_x4(a_rsrc, a_byte_offset, 0, 0)
                    a_vals = S.view(a_vec, S.Tensor((8,), S.bf16))
                    for t in S.range(4):
                        shared_a[0, a_wave_row, a_lane + 0, a_elem_base + t] = a_vals[t]
                        shared_a[0, a_wave_row, a_lane + 32, a_elem_base + t] = a_vals[4 + t]
                else:
                    b_tid = tid - 128
                    b_row_local = b_tid // 8
                    b_seg = b_tid % 8
                    b_wave_col = b_seg // 4
                    b_seg_local = b_seg % 4
                    b_lane_offset = ((b_row_local % 8) // 4) * 32
                    b_elem = (b_row_local // 8) * 4 + (b_row_local % 4)
                    b_byte_offset = ((k0 + BLOCK_K * 2 + b_row_local) * N + block_col + b_seg * 8) * 2
                    b_vec = S.amdgpu.raw_buffer_load_x4(b_rsrc, b_byte_offset, 0, 0)
                    b_vals = S.view(b_vec, S.Tensor((8,), S.bf16))
                    for t in S.range(8):
                        shared_b[0, b_wave_col, b_lane_offset + b_seg_local * 8 + t, b_elem] = b_vals[t]

            a_frag_1 = S.view(shared_a[1, warp_row, lane], S.Tensor((2, 4, 1), S.bf16))
            b_frag_1 = S.view(shared_b[1, warp_col, lane], S.Tensor((2, 4, 1), S.bf16))
            a1_lo = a_frag_1[0]
            a1_hi = a_frag_1[1]
            b1_lo = b_frag_1[0]
            b1_hi = b_frag_1[1]

            c_lane = S.amdgpu.mfma_32x32x8_bf16_f32(a1_lo, b1_lo, c_lane)
            c_lane = S.amdgpu.mfma_32x32x8_bf16_f32(a1_hi, b1_hi, c_lane)

            S.syncthreads()

            if k0 + BLOCK_K * 3 < k_end:
                if tid < 128:
                    a_row_local = tid // 2
                    a_seg = tid % 2
                    a_lane = a_row_local % 32
                    a_wave_row = a_row_local // 32
                    a_elem_base = a_seg * 4
                    a_byte_offset = ((block_row + a_row_local) * K + k0 + BLOCK_K * 3 + a_seg * 8) * 2
                    a_vec = S.amdgpu.raw_buffer_load_x4(a_rsrc, a_byte_offset, 0, 0)
                    a_vals = S.view(a_vec, S.Tensor((8,), S.bf16))
                    for t in S.range(4):
                        shared_a[1, a_wave_row, a_lane + 0, a_elem_base + t] = a_vals[t]
                        shared_a[1, a_wave_row, a_lane + 32, a_elem_base + t] = a_vals[4 + t]
                else:
                    b_tid = tid - 128
                    b_row_local = b_tid // 8
                    b_seg = b_tid % 8
                    b_wave_col = b_seg // 4
                    b_seg_local = b_seg % 4
                    b_lane_offset = ((b_row_local % 8) // 4) * 32
                    b_elem = (b_row_local // 8) * 4 + (b_row_local % 4)
                    b_byte_offset = ((k0 + BLOCK_K * 3 + b_row_local) * N + block_col + b_seg * 8) * 2
                    b_vec = S.amdgpu.raw_buffer_load_x4(b_rsrc, b_byte_offset, 0, 0)
                    b_vals = S.view(b_vec, S.Tensor((8,), S.bf16))
                    for t in S.range(8):
                        shared_b[1, b_wave_col, b_lane_offset + b_seg_local * 8 + t, b_elem] = b_vals[t]

            if k0 + BLOCK_K * 3 < k_end:
                S.syncthreads()

    for acc_idx in S.range(16):
        col = tile_col_base + (lane % 32)
        row = tile_row_base + 8 * (acc_idx // 4) + 4 * (lane // 32) + (acc_idx % 4)
        value = S.convert(0.0, S.bf16)
        if not whole_tile_zero:
            if (not diag_tile) or (col <= row):
                value = S.convert(c_lane[acc_idx], S.bf16)
        C[row, col] = value


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if tuple(A.shape) != (M, K) or tuple(B.shape) != (K, N):
            raise ValueError("ModelNew only supports 4096x4096 inputs")
        if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
            raise ValueError("ModelNew expects bf16 inputs")

        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)
        tri_gemm_kernel[lambda: ((N // BLOCK_N, M // BLOCK_M, 1), (THREADS, 1, 1))](A, B, C)
        return C
