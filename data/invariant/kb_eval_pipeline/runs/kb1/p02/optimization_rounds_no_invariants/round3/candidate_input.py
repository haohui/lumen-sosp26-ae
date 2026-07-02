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
K_TILES = K // BLOCK_K
K_TILE_PAIRS = K_TILES // 2
WAVES_PER_BLOCK = 4
THREADS_PER_BLOCK = 256
A_RANGE_BYTES = M * K * 2
B_RANGE_BYTES = K * N * 2


@substrate.jit
def gemm_kernel(
    A: S.Tensor((M * K,), S.bf16),
    BT: S.Tensor((N * K,), S.bf16),
    C: S.Tensor((M * N,), S.bf16),
):
    tid = S.thread_id(0)
    lane = tid % 64
    warp = tid >> 6
    warp_row = warp >> 1
    warp_col = warp & 1

    block_row = S.block_id(1) * BLOCK_M
    block_col = S.block_id(0) * BLOCK_N

    zero = S.convert(0, S.i32)
    a_rsrc = S.amdgpu.make_rsrc(A, S.convert(A_RANGE_BYTES, S.i32))
    bt_rsrc = S.amdgpu.make_rsrc(BT, S.convert(B_RANGE_BYTES, S.i32))

    a_shared = S.make_shared((2, 2, 64, 4), S.u32)
    b_shared = S.make_shared((2, 2, 64, 4), S.u32)
    acc = S.full((16,), 0.0, S.f32)

    if tid < 128:
        frag = tid
        wave_row_id = frag >> 6
        row_in_wave = frag & 31
        chunk = (frag >> 5) & 1
        global_row = block_row + wave_row_id * 32 + row_in_wave
        lane_lo = row_in_wave
        lane_hi = row_in_wave + 32
        base = chunk * 2

        global_col_0 = chunk * 8
        elem_offset_0 = global_row * K + global_col_0
        byte_offset_0 = S.convert(elem_offset_0 * 2, S.i32)
        packed_0 = S.amdgpu.raw_buffer_load_x4(a_rsrc, zero, byte_offset_0, 0)
        a_shared[0, wave_row_id, lane_lo, base + 0] = packed_0[0]
        a_shared[0, wave_row_id, lane_lo, base + 1] = packed_0[1]
        a_shared[0, wave_row_id, lane_hi, base + 0] = packed_0[2]
        a_shared[0, wave_row_id, lane_hi, base + 1] = packed_0[3]

        global_col_1 = BLOCK_K + chunk * 8
        elem_offset_1 = global_row * K + global_col_1
        byte_offset_1 = S.convert(elem_offset_1 * 2, S.i32)
        packed_1 = S.amdgpu.raw_buffer_load_x4(a_rsrc, zero, byte_offset_1, 0)
        a_shared[1, wave_row_id, lane_lo, base + 0] = packed_1[0]
        a_shared[1, wave_row_id, lane_lo, base + 1] = packed_1[1]
        a_shared[1, wave_row_id, lane_hi, base + 0] = packed_1[2]
        a_shared[1, wave_row_id, lane_hi, base + 1] = packed_1[3]
    else:
        frag = tid - 128
        wave_col_id = frag >> 6
        col_in_wave = frag & 31
        chunk = (frag >> 5) & 1
        global_col = block_col + wave_col_id * 32 + col_in_wave
        lane_lo = col_in_wave
        lane_hi = col_in_wave + 32
        base = chunk * 2

        global_k_0 = chunk * 8
        elem_offset_0 = global_col * K + global_k_0
        byte_offset_0 = S.convert(elem_offset_0 * 2, S.i32)
        packed_0 = S.amdgpu.raw_buffer_load_x4(bt_rsrc, zero, byte_offset_0, 0)
        b_shared[0, wave_col_id, lane_lo, base + 0] = packed_0[0]
        b_shared[0, wave_col_id, lane_lo, base + 1] = packed_0[1]
        b_shared[0, wave_col_id, lane_hi, base + 0] = packed_0[2]
        b_shared[0, wave_col_id, lane_hi, base + 1] = packed_0[3]

        global_k_1 = BLOCK_K + chunk * 8
        elem_offset_1 = global_col * K + global_k_1
        byte_offset_1 = S.convert(elem_offset_1 * 2, S.i32)
        packed_1 = S.amdgpu.raw_buffer_load_x4(bt_rsrc, zero, byte_offset_1, 0)
        b_shared[1, wave_col_id, lane_lo, base + 0] = packed_1[0]
        b_shared[1, wave_col_id, lane_lo, base + 1] = packed_1[1]
        b_shared[1, wave_col_id, lane_hi, base + 0] = packed_1[2]
        b_shared[1, wave_col_id, lane_hi, base + 1] = packed_1[3]

    S.syncthreads()

    for pair_idx in S.range(K_TILE_PAIRS):
        even_words_a = a_shared[0, warp_row, lane]
        even_words_b = b_shared[0, warp_col, lane]
        even_frag_a = S.view(even_words_a, S.Tensor((2, 4, 1), S.bf16))
        even_frag_b = S.view(even_words_b, S.Tensor((2, 4, 1), S.bf16))

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(even_frag_a[0], even_frag_b[0], acc)

        next_even_tile = (pair_idx + 1) * 2
        if next_even_tile < K_TILES:
            if tid < 128:
                frag = tid
                wave_row_id = frag >> 6
                row_in_wave = frag & 31
                chunk = (frag >> 5) & 1
                global_row = block_row + wave_row_id * 32 + row_in_wave
                global_col = next_even_tile * BLOCK_K + chunk * 8
                elem_offset = global_row * K + global_col
                byte_offset = S.convert(elem_offset * 2, S.i32)
                packed = S.amdgpu.raw_buffer_load_x4(a_rsrc, zero, byte_offset, 0)
                lane_lo = row_in_wave
                lane_hi = row_in_wave + 32
                base = chunk * 2
                a_shared[0, wave_row_id, lane_lo, base + 0] = packed[0]
                a_shared[0, wave_row_id, lane_lo, base + 1] = packed[1]
                a_shared[0, wave_row_id, lane_hi, base + 0] = packed[2]
                a_shared[0, wave_row_id, lane_hi, base + 1] = packed[3]
            else:
                frag = tid - 128
                wave_col_id = frag >> 6
                col_in_wave = frag & 31
                chunk = (frag >> 5) & 1
                global_col = block_col + wave_col_id * 32 + col_in_wave
                global_k = next_even_tile * BLOCK_K + chunk * 8
                elem_offset = global_col * K + global_k
                byte_offset = S.convert(elem_offset * 2, S.i32)
                packed = S.amdgpu.raw_buffer_load_x4(bt_rsrc, zero, byte_offset, 0)
                lane_lo = col_in_wave
                lane_hi = col_in_wave + 32
                base = chunk * 2
                b_shared[0, wave_col_id, lane_lo, base + 0] = packed[0]
                b_shared[0, wave_col_id, lane_lo, base + 1] = packed[1]
                b_shared[0, wave_col_id, lane_hi, base + 0] = packed[2]
                b_shared[0, wave_col_id, lane_hi, base + 1] = packed[3]

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(even_frag_a[1], even_frag_b[1], acc)

        odd_words_a = a_shared[1, warp_row, lane]
        odd_words_b = b_shared[1, warp_col, lane]
        odd_frag_a = S.view(odd_words_a, S.Tensor((2, 4, 1), S.bf16))
        odd_frag_b = S.view(odd_words_b, S.Tensor((2, 4, 1), S.bf16))

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(odd_frag_a[0], odd_frag_b[0], acc)

        next_odd_tile = next_even_tile + 1
        if next_odd_tile < K_TILES:
            if tid < 128:
                frag = tid
                wave_row_id = frag >> 6
                row_in_wave = frag & 31
                chunk = (frag >> 5) & 1
                global_row = block_row + wave_row_id * 32 + row_in_wave
                global_col = next_odd_tile * BLOCK_K + chunk * 8
                elem_offset = global_row * K + global_col
                byte_offset = S.convert(elem_offset * 2, S.i32)
                packed = S.amdgpu.raw_buffer_load_x4(a_rsrc, zero, byte_offset, 0)
                lane_lo = row_in_wave
                lane_hi = row_in_wave + 32
                base = chunk * 2
                a_shared[1, wave_row_id, lane_lo, base + 0] = packed[0]
                a_shared[1, wave_row_id, lane_lo, base + 1] = packed[1]
                a_shared[1, wave_row_id, lane_hi, base + 0] = packed[2]
                a_shared[1, wave_row_id, lane_hi, base + 1] = packed[3]
            else:
                frag = tid - 128
                wave_col_id = frag >> 6
                col_in_wave = frag & 31
                chunk = (frag >> 5) & 1
                global_col = block_col + wave_col_id * 32 + col_in_wave
                global_k = next_odd_tile * BLOCK_K + chunk * 8
                elem_offset = global_col * K + global_k
                byte_offset = S.convert(elem_offset * 2, S.i32)
                packed = S.amdgpu.raw_buffer_load_x4(bt_rsrc, zero, byte_offset, 0)
                lane_lo = col_in_wave
                lane_hi = col_in_wave + 32
                base = chunk * 2
                b_shared[1, wave_col_id, lane_lo, base + 0] = packed[0]
                b_shared[1, wave_col_id, lane_lo, base + 1] = packed[1]
                b_shared[1, wave_col_id, lane_hi, base + 0] = packed[2]
                b_shared[1, wave_col_id, lane_hi, base + 1] = packed[3]

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(odd_frag_a[1], odd_frag_b[1], acc)
        S.syncthreads()

    out_col = block_col + warp_col * 32 + (lane & 31)
    row_base = block_row + warp_row * 32 + ((lane >> 5) * 4)
    for i in S.range(16):
        out_row = row_base + (i >> 2) * 8 + (i & 3)
        C[out_row * N + out_col] = S.convert(acc[i], S.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._grid = ((N // BLOCK_N, M // BLOCK_M, 1), (THREADS_PER_BLOCK, 1, 1))

    def forward(self, A, B):
        if tuple(A.shape) != (M, K) or tuple(B.shape) != (K, N):
            raise ValueError(f"Expected A={(M, K)} and B={(K, N)}, got {tuple(A.shape)} and {tuple(B.shape)}")
        if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
            raise ValueError(f"Expected bf16 inputs, got {A.dtype} and {B.dtype}")

        A_flat = A.contiguous().view(-1)
        B_t_flat = B.transpose(0, 1).contiguous().view(-1)
        C = torch.empty((M, N), device=A.device, dtype=torch.bfloat16)
        gemm_kernel[lambda: self._grid](A_flat, B_t_flat, C.view(-1))
        return C
