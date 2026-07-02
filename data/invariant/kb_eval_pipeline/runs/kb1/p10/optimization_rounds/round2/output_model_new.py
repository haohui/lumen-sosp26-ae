import torch
import torch.nn as nn

import substrate
import substrate.language as S


BATCH = 16
M = 1024
K = 2048
N = 768

WARP_SIZE = 64
WARPS_M = 2
WARPS_N = 2
BLOCK_M = 32 * WARPS_M
BLOCK_N = 32 * WARPS_N
K_TILE = 16
BF16_BYTES = 2

A_NUMEL = BATCH * M * K
B_NUMEL = K * N


@substrate.jit
def matmul3d_mfma_kernel(
    A: S.Tensor((BATCH, M, K), S.bf16),
    B: S.Tensor((K, N), S.bf16),
    C: S.Tensor((BATCH, M, N), S.bf16),
):
    tid = S.thread_id(0)
    warp_id = tid >> 6
    lane = tid & 63

    warp_row = warp_id >> 1
    warp_col = warp_id & 1

    batch = S.block_id(2)
    block_row = S.block_id(1) * BLOCK_M
    block_col = S.block_id(0) * BLOCK_N

    a_rsrc = S.amdgpu.make_rsrc(A, A_NUMEL * BF16_BYTES)
    b_rsrc = S.amdgpu.make_rsrc(B, B_NUMEL * BF16_BYTES)

    a_smem = S.make_shared((2, WARPS_M, 64, 4), S.u32)
    b_smem = S.make_shared((2, WARPS_N, 64, 4), S.u32)

    acc = S.full((16,), 0.0, S.f32)

    zero = S.convert(0, S.i32)
    k_tiles = K // K_TILE
    k_pairs = k_tiles // 2

    if tid < 128:
        a_load_id = tid
        a_row_in_block = a_load_id >> 1
        a_chunk = a_load_id & 1
        a_warp_row = a_row_in_block >> 5
        a_local_row = a_row_in_block & 31
        a_row = block_row + a_row_in_block
        a_k = a_chunk * 8
        a_elem_offset = ((batch * M + a_row) * K + a_k) * BF16_BYTES
        a_words = S.amdgpu.raw_buffer_load_x4(
            a_rsrc,
            zero,
            S.convert(a_elem_offset, S.i32),
            0,
        )
        a_lane_lo = a_local_row
        a_lane_hi = a_local_row + 32
        a_word_base = a_chunk * 2
        a_smem[0, a_warp_row, a_lane_lo, a_word_base + 0] = a_words[0]
        a_smem[0, a_warp_row, a_lane_lo, a_word_base + 1] = a_words[1]
        a_smem[0, a_warp_row, a_lane_hi, a_word_base + 0] = a_words[2]
        a_smem[0, a_warp_row, a_lane_hi, a_word_base + 1] = a_words[3]

    if 128 <= tid and tid < 192:
        b_load_id = tid - 128
        b_pair_id = b_load_id >> 3
        b_chunk = b_load_id & 7
        b_warp_col = b_chunk >> 2
        b_local_chunk = b_chunk & 3
        b_k0 = b_pair_id * 2
        b_k1 = b_k0 + 1
        b_col = block_col + b_chunk * 8
        b_elem_offset0 = (b_k0 * N + b_col) * BF16_BYTES
        b_elem_offset1 = (b_k1 * N + b_col) * BF16_BYTES
        b_words0 = S.amdgpu.raw_buffer_load_x4(
            b_rsrc,
            zero,
            S.convert(b_elem_offset0, S.i32),
            0,
        )
        b_words1 = S.amdgpu.raw_buffer_load_x4(
            b_rsrc,
            zero,
            S.convert(b_elem_offset1, S.i32),
            0,
        )
        b_vals0 = S.view(b_words0, S.Tensor((2, 4, 1), S.bf16))
        b_vals1 = S.view(b_words1, S.Tensor((2, 4, 1), S.bf16))
        pair0 = S.make_local((2,), S.bf16)
        pair1 = S.make_local((2,), S.bf16)
        pair2 = S.make_local((2,), S.bf16)
        pair3 = S.make_local((2,), S.bf16)
        pair4 = S.make_local((2,), S.bf16)
        pair5 = S.make_local((2,), S.bf16)
        pair6 = S.make_local((2,), S.bf16)
        pair7 = S.make_local((2,), S.bf16)

        pair0[0] = b_vals0[0, 0, 0]
        pair0[1] = b_vals1[0, 0, 0]
        pair1[0] = b_vals0[0, 1, 0]
        pair1[1] = b_vals1[0, 1, 0]
        pair2[0] = b_vals0[0, 2, 0]
        pair2[1] = b_vals1[0, 2, 0]
        pair3[0] = b_vals0[0, 3, 0]
        pair3[1] = b_vals1[0, 3, 0]
        pair4[0] = b_vals0[1, 0, 0]
        pair4[1] = b_vals1[1, 0, 0]
        pair5[0] = b_vals0[1, 1, 0]
        pair5[1] = b_vals1[1, 1, 0]
        pair6[0] = b_vals0[1, 2, 0]
        pair6[1] = b_vals1[1, 2, 0]
        pair7[0] = b_vals0[1, 3, 0]
        pair7[1] = b_vals1[1, 3, 0]

        if b_pair_id == 0:
            b_lane_base = b_local_chunk * 8
            b_smem[0, b_warp_col, b_lane_base + 0, 0] = S.view(pair0, S.Tensor((1,), S.u32))[0]
            b_smem[0, b_warp_col, b_lane_base + 1, 0] = S.view(pair1, S.Tensor((1,), S.u32))[0]
            b_smem[0, b_warp_col, b_lane_base + 2, 0] = S.view(pair2, S.Tensor((1,), S.u32))[0]
            b_smem[0, b_warp_col, b_lane_base + 3, 0] = S.view(pair3, S.Tensor((1,), S.u32))[0]
            b_smem[0, b_warp_col, b_lane_base + 4, 0] = S.view(pair4, S.Tensor((1,), S.u32))[0]
            b_smem[0, b_warp_col, b_lane_base + 5, 0] = S.view(pair5, S.Tensor((1,), S.u32))[0]
            b_smem[0, b_warp_col, b_lane_base + 6, 0] = S.view(pair6, S.Tensor((1,), S.u32))[0]
            b_smem[0, b_warp_col, b_lane_base + 7, 0] = S.view(pair7, S.Tensor((1,), S.u32))[0]
        elif b_pair_id == 1:
            b_lane_base = b_local_chunk * 8
            b_smem[0, b_warp_col, b_lane_base + 0, 1] = S.view(pair0, S.Tensor((1,), S.u32))[0]
            b_smem[0, b_warp_col, b_lane_base + 1, 1] = S.view(pair1, S.Tensor((1,), S.u32))[0]
            b_smem[0, b_warp_col, b_lane_base + 2, 1] = S.view(pair2, S.Tensor((1,), S.u32))[0]
            b_smem[0, b_warp_col, b_lane_base + 3, 1] = S.view(pair3, S.Tensor((1,), S.u32))[0]
            b_smem[0, b_warp_col, b_lane_base + 4, 1] = S.view(pair4, S.Tensor((1,), S.u32))[0]
            b_smem[0, b_warp_col, b_lane_base + 5, 1] = S.view(pair5, S.Tensor((1,), S.u32))[0]
            b_smem[0, b_warp_col, b_lane_base + 6, 1] = S.view(pair6, S.Tensor((1,), S.u32))[0]
            b_smem[0, b_warp_col, b_lane_base + 7, 1] = S.view(pair7, S.Tensor((1,), S.u32))[0]
        elif b_pair_id == 2:
            b_lane_base = b_local_chunk * 8 + 32
            b_smem[0, b_warp_col, b_lane_base + 0, 0] = S.view(pair0, S.Tensor((1,), S.u32))[0]
            b_smem[0, b_warp_col, b_lane_base + 1, 0] = S.view(pair1, S.Tensor((1,), S.u32))[0]
            b_smem[0, b_warp_col, b_lane_base + 2, 0] = S.view(pair2, S.Tensor((1,), S.u32))[0]
            b_smem[0, b_warp_col, b_lane_base + 3, 0] = S.view(pair3, S.Tensor((1,), S.u32))[0]
            b_smem[0, b_warp_col, b_lane_base + 4, 0] = S.view(pair4, S.Tensor((1,), S.u32))[0]
            b_smem[0, b_warp_col, b_lane_base + 5, 0] = S.view(pair5, S.Tensor((1,), S.u32))[0]
            b_smem[0, b_warp_col, b_lane_base + 6, 0] = S.view(pair6, S.Tensor((1,), S.u32))[0]
            b_smem[0, b_warp_col, b_lane_base + 7, 0] = S.view(pair7, S.Tensor((1,), S.u32))[0]
        elif b_pair_id == 3:
            b_lane_base = b_local_chunk * 8 + 32
            b_smem[0, b_warp_col, b_lane_base + 0, 1] = S.view(pair0, S.Tensor((1,), S.u32))[0]
            b_smem[0, b_warp_col, b_lane_base + 1, 1] = S.view(pair1, S.Tensor((1,), S.u32))[0]
            b_smem[0, b_warp_col, b_lane_base + 2, 1] = S.view(pair2, S.Tensor((1,), S.u32))[0]
            b_smem[0, b_warp_col, b_lane_base + 3, 1] = S.view(pair3, S.Tensor((1,), S.u32))[0]
            b_smem[0, b_warp_col, b_lane_base + 4, 1] = S.view(pair4, S.Tensor((1,), S.u32))[0]
            b_smem[0, b_warp_col, b_lane_base + 5, 1] = S.view(pair5, S.Tensor((1,), S.u32))[0]
            b_smem[0, b_warp_col, b_lane_base + 6, 1] = S.view(pair6, S.Tensor((1,), S.u32))[0]
            b_smem[0, b_warp_col, b_lane_base + 7, 1] = S.view(pair7, S.Tensor((1,), S.u32))[0]
        elif b_pair_id == 4:
            b_lane_base = b_local_chunk * 8
            b_smem[0, b_warp_col, b_lane_base + 0, 2] = S.view(pair0, S.Tensor((1,), S.u32))[0]
            b_smem[0, b_warp_col, b_lane_base + 1, 2] = S.view(pair1, S.Tensor((1,), S.u32))[0]
            b_smem[0, b_warp_col, b_lane_base + 2, 2] = S.view(pair2, S.Tensor((1,), S.u32))[0]
            b_smem[0, b_warp_col, b_lane_base + 3, 2] = S.view(pair3, S.Tensor((1,), S.u32))[0]
            b_smem[0, b_warp_col, b_lane_base + 4, 2] = S.view(pair4, S.Tensor((1,), S.u32))[0]
            b_smem[0, b_warp_col, b_lane_base + 5, 2] = S.view(pair5, S.Tensor((1,), S.u32))[0]
            b_smem[0, b_warp_col, b_lane_base + 6, 2] = S.view(pair6, S.Tensor((1,), S.u32))[0]
            b_smem[0, b_warp_col, b_lane_base + 7, 2] = S.view(pair7, S.Tensor((1,), S.u32))[0]
        elif b_pair_id == 5:
            b_lane_base = b_local_chunk * 8
            b_smem[0, b_warp_col, b_lane_base + 0, 3] = S.view(pair0, S.Tensor((1,), S.u32))[0]
            b_smem[0, b_warp_col, b_lane_base + 1, 3] = S.view(pair1, S.Tensor((1,), S.u32))[0]
            b_smem[0, b_warp_col, b_lane_base + 2, 3] = S.view(pair2, S.Tensor((1,), S.u32))[0]
            b_smem[0, b_warp_col, b_lane_base + 3, 3] = S.view(pair3, S.Tensor((1,), S.u32))[0]
            b_smem[0, b_warp_col, b_lane_base + 4, 3] = S.view(pair4, S.Tensor((1,), S.u32))[0]
            b_smem[0, b_warp_col, b_lane_base + 5, 3] = S.view(pair5, S.Tensor((1,), S.u32))[0]
            b_smem[0, b_warp_col, b_lane_base + 6, 3] = S.view(pair6, S.Tensor((1,), S.u32))[0]
            b_smem[0, b_warp_col, b_lane_base + 7, 3] = S.view(pair7, S.Tensor((1,), S.u32))[0]
        elif b_pair_id == 6:
            b_lane_base = b_local_chunk * 8 + 32
            b_smem[0, b_warp_col, b_lane_base + 0, 2] = S.view(pair0, S.Tensor((1,), S.u32))[0]
            b_smem[0, b_warp_col, b_lane_base + 1, 2] = S.view(pair1, S.Tensor((1,), S.u32))[0]
            b_smem[0, b_warp_col, b_lane_base + 2, 2] = S.view(pair2, S.Tensor((1,), S.u32))[0]
            b_smem[0, b_warp_col, b_lane_base + 3, 2] = S.view(pair3, S.Tensor((1,), S.u32))[0]
            b_smem[0, b_warp_col, b_lane_base + 4, 2] = S.view(pair4, S.Tensor((1,), S.u32))[0]
            b_smem[0, b_warp_col, b_lane_base + 5, 2] = S.view(pair5, S.Tensor((1,), S.u32))[0]
            b_smem[0, b_warp_col, b_lane_base + 6, 2] = S.view(pair6, S.Tensor((1,), S.u32))[0]
            b_smem[0, b_warp_col, b_lane_base + 7, 2] = S.view(pair7, S.Tensor((1,), S.u32))[0]
        else:
            b_lane_base = b_local_chunk * 8 + 32
            b_smem[0, b_warp_col, b_lane_base + 0, 3] = S.view(pair0, S.Tensor((1,), S.u32))[0]
            b_smem[0, b_warp_col, b_lane_base + 1, 3] = S.view(pair1, S.Tensor((1,), S.u32))[0]
            b_smem[0, b_warp_col, b_lane_base + 2, 3] = S.view(pair2, S.Tensor((1,), S.u32))[0]
            b_smem[0, b_warp_col, b_lane_base + 3, 3] = S.view(pair3, S.Tensor((1,), S.u32))[0]
            b_smem[0, b_warp_col, b_lane_base + 4, 3] = S.view(pair4, S.Tensor((1,), S.u32))[0]
            b_smem[0, b_warp_col, b_lane_base + 5, 3] = S.view(pair5, S.Tensor((1,), S.u32))[0]
            b_smem[0, b_warp_col, b_lane_base + 6, 3] = S.view(pair6, S.Tensor((1,), S.u32))[0]
            b_smem[0, b_warp_col, b_lane_base + 7, 3] = S.view(pair7, S.Tensor((1,), S.u32))[0]

    if tid < 128:
        a_load_id = tid
        a_row_in_block = a_load_id >> 1
        a_chunk = a_load_id & 1
        a_warp_row = a_row_in_block >> 5
        a_local_row = a_row_in_block & 31
        a_row = block_row + a_row_in_block
        a_k = K_TILE + a_chunk * 8
        a_elem_offset = ((batch * M + a_row) * K + a_k) * BF16_BYTES
        a_words = S.amdgpu.raw_buffer_load_x4(
            a_rsrc,
            zero,
            S.convert(a_elem_offset, S.i32),
            0,
        )
        a_lane_lo = a_local_row
        a_lane_hi = a_local_row + 32
        a_word_base = a_chunk * 2
        a_smem[1, a_warp_row, a_lane_lo, a_word_base + 0] = a_words[0]
        a_smem[1, a_warp_row, a_lane_lo, a_word_base + 1] = a_words[1]
        a_smem[1, a_warp_row, a_lane_hi, a_word_base + 0] = a_words[2]
        a_smem[1, a_warp_row, a_lane_hi, a_word_base + 1] = a_words[3]

    if 128 <= tid and tid < 192:
        b_load_id = tid - 128
        b_pair_id = b_load_id >> 3
        b_chunk = b_load_id & 7
        b_warp_col = b_chunk >> 2
        b_local_chunk = b_chunk & 3
        b_k0 = K_TILE + b_pair_id * 2
        b_k1 = b_k0 + 1
        b_col = block_col + b_chunk * 8
        b_elem_offset0 = (b_k0 * N + b_col) * BF16_BYTES
        b_elem_offset1 = (b_k1 * N + b_col) * BF16_BYTES
        b_words0 = S.amdgpu.raw_buffer_load_x4(
            b_rsrc,
            zero,
            S.convert(b_elem_offset0, S.i32),
            0,
        )
        b_words1 = S.amdgpu.raw_buffer_load_x4(
            b_rsrc,
            zero,
            S.convert(b_elem_offset1, S.i32),
            0,
        )
        b_vals0 = S.view(b_words0, S.Tensor((2, 4, 1), S.bf16))
        b_vals1 = S.view(b_words1, S.Tensor((2, 4, 1), S.bf16))
        pair0 = S.make_local((2,), S.bf16)
        pair1 = S.make_local((2,), S.bf16)
        pair2 = S.make_local((2,), S.bf16)
        pair3 = S.make_local((2,), S.bf16)
        pair4 = S.make_local((2,), S.bf16)
        pair5 = S.make_local((2,), S.bf16)
        pair6 = S.make_local((2,), S.bf16)
        pair7 = S.make_local((2,), S.bf16)

        pair0[0] = b_vals0[0, 0, 0]
        pair0[1] = b_vals1[0, 0, 0]
        pair1[0] = b_vals0[0, 1, 0]
        pair1[1] = b_vals1[0, 1, 0]
        pair2[0] = b_vals0[0, 2, 0]
        pair2[1] = b_vals1[0, 2, 0]
        pair3[0] = b_vals0[0, 3, 0]
        pair3[1] = b_vals1[0, 3, 0]
        pair4[0] = b_vals0[1, 0, 0]
        pair4[1] = b_vals1[1, 0, 0]
        pair5[0] = b_vals0[1, 1, 0]
        pair5[1] = b_vals1[1, 1, 0]
        pair6[0] = b_vals0[1, 2, 0]
        pair6[1] = b_vals1[1, 2, 0]
        pair7[0] = b_vals0[1, 3, 0]
        pair7[1] = b_vals1[1, 3, 0]

        if b_pair_id == 0:
            b_lane_base = b_local_chunk * 8
            b_smem[1, b_warp_col, b_lane_base + 0, 0] = S.view(pair0, S.Tensor((1,), S.u32))[0]
            b_smem[1, b_warp_col, b_lane_base + 1, 0] = S.view(pair1, S.Tensor((1,), S.u32))[0]
            b_smem[1, b_warp_col, b_lane_base + 2, 0] = S.view(pair2, S.Tensor((1,), S.u32))[0]
            b_smem[1, b_warp_col, b_lane_base + 3, 0] = S.view(pair3, S.Tensor((1,), S.u32))[0]
            b_smem[1, b_warp_col, b_lane_base + 4, 0] = S.view(pair4, S.Tensor((1,), S.u32))[0]
            b_smem[1, b_warp_col, b_lane_base + 5, 0] = S.view(pair5, S.Tensor((1,), S.u32))[0]
            b_smem[1, b_warp_col, b_lane_base + 6, 0] = S.view(pair6, S.Tensor((1,), S.u32))[0]
            b_smem[1, b_warp_col, b_lane_base + 7, 0] = S.view(pair7, S.Tensor((1,), S.u32))[0]
        elif b_pair_id == 1:
            b_lane_base = b_local_chunk * 8
            b_smem[1, b_warp_col, b_lane_base + 0, 1] = S.view(pair0, S.Tensor((1,), S.u32))[0]
            b_smem[1, b_warp_col, b_lane_base + 1, 1] = S.view(pair1, S.Tensor((1,), S.u32))[0]
            b_smem[1, b_warp_col, b_lane_base + 2, 1] = S.view(pair2, S.Tensor((1,), S.u32))[0]
            b_smem[1, b_warp_col, b_lane_base + 3, 1] = S.view(pair3, S.Tensor((1,), S.u32))[0]
            b_smem[1, b_warp_col, b_lane_base + 4, 1] = S.view(pair4, S.Tensor((1,), S.u32))[0]
            b_smem[1, b_warp_col, b_lane_base + 5, 1] = S.view(pair5, S.Tensor((1,), S.u32))[0]
            b_smem[1, b_warp_col, b_lane_base + 6, 1] = S.view(pair6, S.Tensor((1,), S.u32))[0]
            b_smem[1, b_warp_col, b_lane_base + 7, 1] = S.view(pair7, S.Tensor((1,), S.u32))[0]
        elif b_pair_id == 2:
            b_lane_base = b_local_chunk * 8 + 32
            b_smem[1, b_warp_col, b_lane_base + 0, 0] = S.view(pair0, S.Tensor((1,), S.u32))[0]
            b_smem[1, b_warp_col, b_lane_base + 1, 0] = S.view(pair1, S.Tensor((1,), S.u32))[0]
            b_smem[1, b_warp_col, b_lane_base + 2, 0] = S.view(pair2, S.Tensor((1,), S.u32))[0]
            b_smem[1, b_warp_col, b_lane_base + 3, 0] = S.view(pair3, S.Tensor((1,), S.u32))[0]
            b_smem[1, b_warp_col, b_lane_base + 4, 0] = S.view(pair4, S.Tensor((1,), S.u32))[0]
            b_smem[1, b_warp_col, b_lane_base + 5, 0] = S.view(pair5, S.Tensor((1,), S.u32))[0]
            b_smem[1, b_warp_col, b_lane_base + 6, 0] = S.view(pair6, S.Tensor((1,), S.u32))[0]
            b_smem[1, b_warp_col, b_lane_base + 7, 0] = S.view(pair7, S.Tensor((1,), S.u32))[0]
        elif b_pair_id == 3:
            b_lane_base = b_local_chunk * 8 + 32
            b_smem[1, b_warp_col, b_lane_base + 0, 1] = S.view(pair0, S.Tensor((1,), S.u32))[0]
            b_smem[1, b_warp_col, b_lane_base + 1, 1] = S.view(pair1, S.Tensor((1,), S.u32))[0]
            b_smem[1, b_warp_col, b_lane_base + 2, 1] = S.view(pair2, S.Tensor((1,), S.u32))[0]
            b_smem[1, b_warp_col, b_lane_base + 3, 1] = S.view(pair3, S.Tensor((1,), S.u32))[0]
            b_smem[1, b_warp_col, b_lane_base + 4, 1] = S.view(pair4, S.Tensor((1,), S.u32))[0]
            b_smem[1, b_warp_col, b_lane_base + 5, 1] = S.view(pair5, S.Tensor((1,), S.u32))[0]
            b_smem[1, b_warp_col, b_lane_base + 6, 1] = S.view(pair6, S.Tensor((1,), S.u32))[0]
            b_smem[1, b_warp_col, b_lane_base + 7, 1] = S.view(pair7, S.Tensor((1,), S.u32))[0]
        elif b_pair_id == 4:
            b_lane_base = b_local_chunk * 8
            b_smem[1, b_warp_col, b_lane_base + 0, 2] = S.view(pair0, S.Tensor((1,), S.u32))[0]
            b_smem[1, b_warp_col, b_lane_base + 1, 2] = S.view(pair1, S.Tensor((1,), S.u32))[0]
            b_smem[1, b_warp_col, b_lane_base + 2, 2] = S.view(pair2, S.Tensor((1,), S.u32))[0]
            b_smem[1, b_warp_col, b_lane_base + 3, 2] = S.view(pair3, S.Tensor((1,), S.u32))[0]
            b_smem[1, b_warp_col, b_lane_base + 4, 2] = S.view(pair4, S.Tensor((1,), S.u32))[0]
            b_smem[1, b_warp_col, b_lane_base + 5, 2] = S.view(pair5, S.Tensor((1,), S.u32))[0]
            b_smem[1, b_warp_col, b_lane_base + 6, 2] = S.view(pair6, S.Tensor((1,), S.u32))[0]
            b_smem[1, b_warp_col, b_lane_base + 7, 2] = S.view(pair7, S.Tensor((1,), S.u32))[0]
        elif b_pair_id == 5:
            b_lane_base = b_local_chunk * 8
            b_smem[1, b_warp_col, b_lane_base + 0, 3] = S.view(pair0, S.Tensor((1,), S.u32))[0]
            b_smem[1, b_warp_col, b_lane_base + 1, 3] = S.view(pair1, S.Tensor((1,), S.u32))[0]
            b_smem[1, b_warp_col, b_lane_base + 2, 3] = S.view(pair2, S.Tensor((1,), S.u32))[0]
            b_smem[1, b_warp_col, b_lane_base + 3, 3] = S.view(pair3, S.Tensor((1,), S.u32))[0]
            b_smem[1, b_warp_col, b_lane_base + 4, 3] = S.view(pair4, S.Tensor((1,), S.u32))[0]
            b_smem[1, b_warp_col, b_lane_base + 5, 3] = S.view(pair5, S.Tensor((1,), S.u32))[0]
            b_smem[1, b_warp_col, b_lane_base + 6, 3] = S.view(pair6, S.Tensor((1,), S.u32))[0]
            b_smem[1, b_warp_col, b_lane_base + 7, 3] = S.view(pair7, S.Tensor((1,), S.u32))[0]
        elif b_pair_id == 6:
            b_lane_base = b_local_chunk * 8 + 32
            b_smem[1, b_warp_col, b_lane_base + 0, 2] = S.view(pair0, S.Tensor((1,), S.u32))[0]
            b_smem[1, b_warp_col, b_lane_base + 1, 2] = S.view(pair1, S.Tensor((1,), S.u32))[0]
            b_smem[1, b_warp_col, b_lane_base + 2, 2] = S.view(pair2, S.Tensor((1,), S.u32))[0]
            b_smem[1, b_warp_col, b_lane_base + 3, 2] = S.view(pair3, S.Tensor((1,), S.u32))[0]
            b_smem[1, b_warp_col, b_lane_base + 4, 2] = S.view(pair4, S.Tensor((1,), S.u32))[0]
            b_smem[1, b_warp_col, b_lane_base + 5, 2] = S.view(pair5, S.Tensor((1,), S.u32))[0]
            b_smem[1, b_warp_col, b_lane_base + 6, 2] = S.view(pair6, S.Tensor((1,), S.u32))[0]
            b_smem[1, b_warp_col, b_lane_base + 7, 2] = S.view(pair7, S.Tensor((1,), S.u32))[0]
        else:
            b_lane_base = b_local_chunk * 8 + 32
            b_smem[1, b_warp_col, b_lane_base + 0, 3] = S.view(pair0, S.Tensor((1,), S.u32))[0]
            b_smem[1, b_warp_col, b_lane_base + 1, 3] = S.view(pair1, S.Tensor((1,), S.u32))[0]
            b_smem[1, b_warp_col, b_lane_base + 2, 3] = S.view(pair2, S.Tensor((1,), S.u32))[0]
            b_smem[1, b_warp_col, b_lane_base + 3, 3] = S.view(pair3, S.Tensor((1,), S.u32))[0]
            b_smem[1, b_warp_col, b_lane_base + 4, 3] = S.view(pair4, S.Tensor((1,), S.u32))[0]
            b_smem[1, b_warp_col, b_lane_base + 5, 3] = S.view(pair5, S.Tensor((1,), S.u32))[0]
            b_smem[1, b_warp_col, b_lane_base + 6, 3] = S.view(pair6, S.Tensor((1,), S.u32))[0]
            b_smem[1, b_warp_col, b_lane_base + 7, 3] = S.view(pair7, S.Tensor((1,), S.u32))[0]

    S.syncthreads()

    for k_pair in S.range(k_pairs):
        a_frag0 = S.view(a_smem[0, warp_row, lane], S.Tensor((2, 4, 1), S.bf16))
        b_frag0 = S.view(b_smem[0, warp_col, lane], S.Tensor((2, 4, 1), S.bf16))

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[0], b_frag0[0], acc)

        if k_pair + 1 < k_pairs:
            next_k_base0 = (k_pair * 2 + 2) * K_TILE

            if tid < 128:
                a_load_id = tid
                a_row_in_block = a_load_id >> 1
                a_chunk = a_load_id & 1
                a_warp_row = a_row_in_block >> 5
                a_local_row = a_row_in_block & 31
                a_row = block_row + a_row_in_block
                a_k = next_k_base0 + a_chunk * 8
                a_elem_offset = ((batch * M + a_row) * K + a_k) * BF16_BYTES
                a_words = S.amdgpu.raw_buffer_load_x4(
                    a_rsrc,
                    zero,
                    S.convert(a_elem_offset, S.i32),
                    0,
                )
                a_lane_lo = a_local_row
                a_lane_hi = a_local_row + 32
                a_word_base = a_chunk * 2
                a_smem[0, a_warp_row, a_lane_lo, a_word_base + 0] = a_words[0]
                a_smem[0, a_warp_row, a_lane_lo, a_word_base + 1] = a_words[1]
                a_smem[0, a_warp_row, a_lane_hi, a_word_base + 0] = a_words[2]
                a_smem[0, a_warp_row, a_lane_hi, a_word_base + 1] = a_words[3]

            if 128 <= tid and tid < 192:
                b_load_id = tid - 128
                b_pair_id = b_load_id >> 3
                b_chunk = b_load_id & 7
                b_warp_col = b_chunk >> 2
                b_local_chunk = b_chunk & 3
                b_k0 = next_k_base0 + b_pair_id * 2
                b_k1 = b_k0 + 1
                b_col = block_col + b_chunk * 8
                b_elem_offset0 = (b_k0 * N + b_col) * BF16_BYTES
                b_elem_offset1 = (b_k1 * N + b_col) * BF16_BYTES
                b_words0 = S.amdgpu.raw_buffer_load_x4(
                    b_rsrc,
                    zero,
                    S.convert(b_elem_offset0, S.i32),
                    0,
                )
                b_words1 = S.amdgpu.raw_buffer_load_x4(
                    b_rsrc,
                    zero,
                    S.convert(b_elem_offset1, S.i32),
                    0,
                )
                b_vals0 = S.view(b_words0, S.Tensor((2, 4, 1), S.bf16))
                b_vals1 = S.view(b_words1, S.Tensor((2, 4, 1), S.bf16))
                pair0 = S.make_local((2,), S.bf16)
                pair1 = S.make_local((2,), S.bf16)
                pair2 = S.make_local((2,), S.bf16)
                pair3 = S.make_local((2,), S.bf16)
                pair4 = S.make_local((2,), S.bf16)
                pair5 = S.make_local((2,), S.bf16)
                pair6 = S.make_local((2,), S.bf16)
                pair7 = S.make_local((2,), S.bf16)

                pair0[0] = b_vals0[0, 0, 0]
                pair0[1] = b_vals1[0, 0, 0]
                pair1[0] = b_vals0[0, 1, 0]
                pair1[1] = b_vals1[0, 1, 0]
                pair2[0] = b_vals0[0, 2, 0]
                pair2[1] = b_vals1[0, 2, 0]
                pair3[0] = b_vals0[0, 3, 0]
                pair3[1] = b_vals1[0, 3, 0]
                pair4[0] = b_vals0[1, 0, 0]
                pair4[1] = b_vals1[1, 0, 0]
                pair5[0] = b_vals0[1, 1, 0]
                pair5[1] = b_vals1[1, 1, 0]
                pair6[0] = b_vals0[1, 2, 0]
                pair6[1] = b_vals1[1, 2, 0]
                pair7[0] = b_vals0[1, 3, 0]
                pair7[1] = b_vals1[1, 3, 0]

                if b_pair_id == 0:
                    b_lane_base = b_local_chunk * 8
                    b_smem[0, b_warp_col, b_lane_base + 0, 0] = S.view(pair0, S.Tensor((1,), S.u32))[0]
                    b_smem[0, b_warp_col, b_lane_base + 1, 0] = S.view(pair1, S.Tensor((1,), S.u32))[0]
                    b_smem[0, b_warp_col, b_lane_base + 2, 0] = S.view(pair2, S.Tensor((1,), S.u32))[0]
                    b_smem[0, b_warp_col, b_lane_base + 3, 0] = S.view(pair3, S.Tensor((1,), S.u32))[0]
                    b_smem[0, b_warp_col, b_lane_base + 4, 0] = S.view(pair4, S.Tensor((1,), S.u32))[0]
                    b_smem[0, b_warp_col, b_lane_base + 5, 0] = S.view(pair5, S.Tensor((1,), S.u32))[0]
                    b_smem[0, b_warp_col, b_lane_base + 6, 0] = S.view(pair6, S.Tensor((1,), S.u32))[0]
                    b_smem[0, b_warp_col, b_lane_base + 7, 0] = S.view(pair7, S.Tensor((1,), S.u32))[0]
                elif b_pair_id == 1:
                    b_lane_base = b_local_chunk * 8
                    b_smem[0, b_warp_col, b_lane_base + 0, 1] = S.view(pair0, S.Tensor((1,), S.u32))[0]
                    b_smem[0, b_warp_col, b_lane_base + 1, 1] = S.view(pair1, S.Tensor((1,), S.u32))[0]
                    b_smem[0, b_warp_col, b_lane_base + 2, 1] = S.view(pair2, S.Tensor((1,), S.u32))[0]
                    b_smem[0, b_warp_col, b_lane_base + 3, 1] = S.view(pair3, S.Tensor((1,), S.u32))[0]
                    b_smem[0, b_warp_col, b_lane_base + 4, 1] = S.view(pair4, S.Tensor((1,), S.u32))[0]
                    b_smem[0, b_warp_col, b_lane_base + 5, 1] = S.view(pair5, S.Tensor((1,), S.u32))[0]
                    b_smem[0, b_warp_col, b_lane_base + 6, 1] = S.view(pair6, S.Tensor((1,), S.u32))[0]
                    b_smem[0, b_warp_col, b_lane_base + 7, 1] = S.view(pair7, S.Tensor((1,), S.u32))[0]
                elif b_pair_id == 2:
                    b_lane_base = b_local_chunk * 8 + 32
                    b_smem[0, b_warp_col, b_lane_base + 0, 0] = S.view(pair0, S.Tensor((1,), S.u32))[0]
                    b_smem[0, b_warp_col, b_lane_base + 1, 0] = S.view(pair1, S.Tensor((1,), S.u32))[0]
                    b_smem[0, b_warp_col, b_lane_base + 2, 0] = S.view(pair2, S.Tensor((1,), S.u32))[0]
                    b_smem[0, b_warp_col, b_lane_base + 3, 0] = S.view(pair3, S.Tensor((1,), S.u32))[0]
                    b_smem[0, b_warp_col, b_lane_base + 4, 0] = S.view(pair4, S.Tensor((1,), S.u32))[0]
                    b_smem[0, b_warp_col, b_lane_base + 5, 0] = S.view(pair5, S.Tensor((1,), S.u32))[0]
                    b_smem[0, b_warp_col, b_lane_base + 6, 0] = S.view(pair6, S.Tensor((1,), S.u32))[0]
                    b_smem[0, b_warp_col, b_lane_base + 7, 0] = S.view(pair7, S.Tensor((1,), S.u32))[0]
                elif b_pair_id == 3:
                    b_lane_base = b_local_chunk * 8 + 32
                    b_smem[0, b_warp_col, b_lane_base + 0, 1] = S.view(pair0, S.Tensor((1,), S.u32))[0]
                    b_smem[0, b_warp_col, b_lane_base + 1, 1] = S.view(pair1, S.Tensor((1,), S.u32))[0]
                    b_smem[0, b_warp_col, b_lane_base + 2, 1] = S.view(pair2, S.Tensor((1,), S.u32))[0]
                    b_smem[0, b_warp_col, b_lane_base + 3, 1] = S.view(pair3, S.Tensor((1,), S.u32))[0]
                    b_smem[0, b_warp_col, b_lane_base + 4, 1] = S.view(pair4, S.Tensor((1,), S.u32))[0]
                    b_smem[0, b_warp_col, b_lane_base + 5, 1] = S.view(pair5, S.Tensor((1,), S.u32))[0]
                    b_smem[0, b_warp_col, b_lane_base + 6, 1] = S.view(pair6, S.Tensor((1,), S.u32))[0]
                    b_smem[0, b_warp_col, b_lane_base + 7, 1] = S.view(pair7, S.Tensor((1,), S.u32))[0]
                elif b_pair_id == 4:
                    b_lane_base = b_local_chunk * 8
                    b_smem[0, b_warp_col, b_lane_base + 0, 2] = S.view(pair0, S.Tensor((1,), S.u32))[0]
                    b_smem[0, b_warp_col, b_lane_base + 1, 2] = S.view(pair1, S.Tensor((1,), S.u32))[0]
                    b_smem[0, b_warp_col, b_lane_base + 2, 2] = S.view(pair2, S.Tensor((1,), S.u32))[0]
                    b_smem[0, b_warp_col, b_lane_base + 3, 2] = S.view(pair3, S.Tensor((1,), S.u32))[0]
                    b_smem[0, b_warp_col, b_lane_base + 4, 2] = S.view(pair4, S.Tensor((1,), S.u32))[0]
                    b_smem[0, b_warp_col, b_lane_base + 5, 2] = S.view(pair5, S.Tensor((1,), S.u32))[0]
                    b_smem[0, b_warp_col, b_lane_base + 6, 2] = S.view(pair6, S.Tensor((1,), S.u32))[0]
                    b_smem[0, b_warp_col, b_lane_base + 7, 2] = S.view(pair7, S.Tensor((1,), S.u32))[0]
                elif b_pair_id == 5:
                    b_lane_base = b_local_chunk * 8
                    b_smem[0, b_warp_col, b_lane_base + 0, 3] = S.view(pair0, S.Tensor((1,), S.u32))[0]
                    b_smem[0, b_warp_col, b_lane_base + 1, 3] = S.view(pair1, S.Tensor((1,), S.u32))[0]
                    b_smem[0, b_warp_col, b_lane_base + 2, 3] = S.view(pair2, S.Tensor((1,), S.u32))[0]
                    b_smem[0, b_warp_col, b_lane_base + 3, 3] = S.view(pair3, S.Tensor((1,), S.u32))[0]
                    b_smem[0, b_warp_col, b_lane_base + 4, 3] = S.view(pair4, S.Tensor((1,), S.u32))[0]
                    b_smem[0, b_warp_col, b_lane_base + 5, 3] = S.view(pair5, S.Tensor((1,), S.u32))[0]
                    b_smem[0, b_warp_col, b_lane_base + 6, 3] = S.view(pair6, S.Tensor((1,), S.u32))[0]
                    b_smem[0, b_warp_col, b_lane_base + 7, 3] = S.view(pair7, S.Tensor((1,), S.u32))[0]
                elif b_pair_id == 6:
                    b_lane_base = b_local_chunk * 8 + 32
                    b_smem[0, b_warp_col, b_lane_base + 0, 2] = S.view(pair0, S.Tensor((1,), S.u32))[0]
                    b_smem[0, b_warp_col, b_lane_base + 1, 2] = S.view(pair1, S.Tensor((1,), S.u32))[0]
                    b_smem[0, b_warp_col, b_lane_base + 2, 2] = S.view(pair2, S.Tensor((1,), S.u32))[0]
                    b_smem[0, b_warp_col, b_lane_base + 3, 2] = S.view(pair3, S.Tensor((1,), S.u32))[0]
                    b_smem[0, b_warp_col, b_lane_base + 4, 2] = S.view(pair4, S.Tensor((1,), S.u32))[0]
                    b_smem[0, b_warp_col, b_lane_base + 5, 2] = S.view(pair5, S.Tensor((1,), S.u32))[0]
                    b_smem[0, b_warp_col, b_lane_base + 6, 2] = S.view(pair6, S.Tensor((1,), S.u32))[0]
                    b_smem[0, b_warp_col, b_lane_base + 7, 2] = S.view(pair7, S.Tensor((1,), S.u32))[0]
                else:
                    b_lane_base = b_local_chunk * 8 + 32
                    b_smem[0, b_warp_col, b_lane_base + 0, 3] = S.view(pair0, S.Tensor((1,), S.u32))[0]
                    b_smem[0, b_warp_col, b_lane_base + 1, 3] = S.view(pair1, S.Tensor((1,), S.u32))[0]
                    b_smem[0, b_warp_col, b_lane_base + 2, 3] = S.view(pair2, S.Tensor((1,), S.u32))[0]
                    b_smem[0, b_warp_col, b_lane_base + 3, 3] = S.view(pair3, S.Tensor((1,), S.u32))[0]
                    b_smem[0, b_warp_col, b_lane_base + 4, 3] = S.view(pair4, S.Tensor((1,), S.u32))[0]
                    b_smem[0, b_warp_col, b_lane_base + 5, 3] = S.view(pair5, S.Tensor((1,), S.u32))[0]
                    b_smem[0, b_warp_col, b_lane_base + 6, 3] = S.view(pair6, S.Tensor((1,), S.u32))[0]
                    b_smem[0, b_warp_col, b_lane_base + 7, 3] = S.view(pair7, S.Tensor((1,), S.u32))[0]

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[1], b_frag0[1], acc)
        S.syncthreads()

        a_frag1 = S.view(a_smem[1, warp_row, lane], S.Tensor((2, 4, 1), S.bf16))
        b_frag1 = S.view(b_smem[1, warp_col, lane], S.Tensor((2, 4, 1), S.bf16))

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[0], b_frag1[0], acc)

        if k_pair + 1 < k_pairs:
            next_k_base1 = (k_pair * 2 + 3) * K_TILE

            if tid < 128:
                a_load_id = tid
                a_row_in_block = a_load_id >> 1
                a_chunk = a_load_id & 1
                a_warp_row = a_row_in_block >> 5
                a_local_row = a_row_in_block & 31
                a_row = block_row + a_row_in_block
                a_k = next_k_base1 + a_chunk * 8
                a_elem_offset = ((batch * M + a_row) * K + a_k) * BF16_BYTES
                a_words = S.amdgpu.raw_buffer_load_x4(
                    a_rsrc,
                    zero,
                    S.convert(a_elem_offset, S.i32),
                    0,
                )
                a_lane_lo = a_local_row
                a_lane_hi = a_local_row + 32
                a_word_base = a_chunk * 2
                a_smem[1, a_warp_row, a_lane_lo, a_word_base + 0] = a_words[0]
                a_smem[1, a_warp_row, a_lane_lo, a_word_base + 1] = a_words[1]
                a_smem[1, a_warp_row, a_lane_hi, a_word_base + 0] = a_words[2]
                a_smem[1, a_warp_row, a_lane_hi, a_word_base + 1] = a_words[3]

            if 128 <= tid and tid < 192:
                b_load_id = tid - 128
                b_pair_id = b_load_id >> 3
                b_chunk = b_load_id & 7
                b_warp_col = b_chunk >> 2
                b_local_chunk = b_chunk & 3
                b_k0 = next_k_base1 + b_pair_id * 2
                b_k1 = b_k0 + 1
                b_col = block_col + b_chunk * 8
                b_elem_offset0 = (b_k0 * N + b_col) * BF16_BYTES
                b_elem_offset1 = (b_k1 * N + b_col) * BF16_BYTES
                b_words0 = S.amdgpu.raw_buffer_load_x4(
                    b_rsrc,
                    zero,
                    S.convert(b_elem_offset0, S.i32),
                    0,
                )
                b_words1 = S.amdgpu.raw_buffer_load_x4(
                    b_rsrc,
                    zero,
                    S.convert(b_elem_offset1, S.i32),
                    0,
                )
                b_vals0 = S.view(b_words0, S.Tensor((2, 4, 1), S.bf16))
                b_vals1 = S.view(b_words1, S.Tensor((2, 4, 1), S.bf16))
                pair0 = S.make_local((2,), S.bf16)
                pair1 = S.make_local((2,), S.bf16)
                pair2 = S.make_local((2,), S.bf16)
                pair3 = S.make_local((2,), S.bf16)
                pair4 = S.make_local((2,), S.bf16)
                pair5 = S.make_local((2,), S.bf16)
                pair6 = S.make_local((2,), S.bf16)
                pair7 = S.make_local((2,), S.bf16)

                pair0[0] = b_vals0[0, 0, 0]
                pair0[1] = b_vals1[0, 0, 0]
                pair1[0] = b_vals0[0, 1, 0]
                pair1[1] = b_vals1[0, 1, 0]
                pair2[0] = b_vals0[0, 2, 0]
                pair2[1] = b_vals1[0, 2, 0]
                pair3[0] = b_vals0[0, 3, 0]
                pair3[1] = b_vals1[0, 3, 0]
                pair4[0] = b_vals0[1, 0, 0]
                pair4[1] = b_vals1[1, 0, 0]
                pair5[0] = b_vals0[1, 1, 0]
                pair5[1] = b_vals1[1, 1, 0]
                pair6[0] = b_vals0[1, 2, 0]
                pair6[1] = b_vals1[1, 2, 0]
                pair7[0] = b_vals0[1, 3, 0]
                pair7[1] = b_vals1[1, 3, 0]

                if b_pair_id == 0:
                    b_lane_base = b_local_chunk * 8
                    b_smem[1, b_warp_col, b_lane_base + 0, 0] = S.view(pair0, S.Tensor((1,), S.u32))[0]
                    b_smem[1, b_warp_col, b_lane_base + 1, 0] = S.view(pair1, S.Tensor((1,), S.u32))[0]
                    b_smem[1, b_warp_col, b_lane_base + 2, 0] = S.view(pair2, S.Tensor((1,), S.u32))[0]
                    b_smem[1, b_warp_col, b_lane_base + 3, 0] = S.view(pair3, S.Tensor((1,), S.u32))[0]
                    b_smem[1, b_warp_col, b_lane_base + 4, 0] = S.view(pair4, S.Tensor((1,), S.u32))[0]
                    b_smem[1, b_warp_col, b_lane_base + 5, 0] = S.view(pair5, S.Tensor((1,), S.u32))[0]
                    b_smem[1, b_warp_col, b_lane_base + 6, 0] = S.view(pair6, S.Tensor((1,), S.u32))[0]
                    b_smem[1, b_warp_col, b_lane_base + 7, 0] = S.view(pair7, S.Tensor((1,), S.u32))[0]
                elif b_pair_id == 1:
                    b_lane_base = b_local_chunk * 8
                    b_smem[1, b_warp_col, b_lane_base + 0, 1] = S.view(pair0, S.Tensor((1,), S.u32))[0]
                    b_smem[1, b_warp_col, b_lane_base + 1, 1] = S.view(pair1, S.Tensor((1,), S.u32))[0]
                    b_smem[1, b_warp_col, b_lane_base + 2, 1] = S.view(pair2, S.Tensor((1,), S.u32))[0]
                    b_smem[1, b_warp_col, b_lane_base + 3, 1] = S.view(pair3, S.Tensor((1,), S.u32))[0]
                    b_smem[1, b_warp_col, b_lane_base + 4, 1] = S.view(pair4, S.Tensor((1,), S.u32))[0]
                    b_smem[1, b_warp_col, b_lane_base + 5, 1] = S.view(pair5, S.Tensor((1,), S.u32))[0]
                    b_smem[1, b_warp_col, b_lane_base + 6, 1] = S.view(pair6, S.Tensor((1,), S.u32))[0]
                    b_smem[1, b_warp_col, b_lane_base + 7, 1] = S.view(pair7, S.Tensor((1,), S.u32))[0]
                elif b_pair_id == 2:
                    b_lane_base = b_local_chunk * 8 + 32
                    b_smem[1, b_warp_col, b_lane_base + 0, 0] = S.view(pair0, S.Tensor((1,), S.u32))[0]
                    b_smem[1, b_warp_col, b_lane_base + 1, 0] = S.view(pair1, S.Tensor((1,), S.u32))[0]
                    b_smem[1, b_warp_col, b_lane_base + 2, 0] = S.view(pair2, S.Tensor((1,), S.u32))[0]
                    b_smem[1, b_warp_col, b_lane_base + 3, 0] = S.view(pair3, S.Tensor((1,), S.u32))[0]
                    b_smem[1, b_warp_col, b_lane_base + 4, 0] = S.view(pair4, S.Tensor((1,), S.u32))[0]
                    b_smem[1, b_warp_col, b_lane_base + 5, 0] = S.view(pair5, S.Tensor((1,), S.u32))[0]
                    b_smem[1, b_warp_col, b_lane_base + 6, 0] = S.view(pair6, S.Tensor((1,), S.u32))[0]
                    b_smem[1, b_warp_col, b_lane_base + 7, 0] = S.view(pair7, S.Tensor((1,), S.u32))[0]
                elif b_pair_id == 3:
                    b_lane_base = b_local_chunk * 8 + 32
                    b_smem[1, b_warp_col, b_lane_base + 0, 1] = S.view(pair0, S.Tensor((1,), S.u32))[0]
                    b_smem[1, b_warp_col, b_lane_base + 1, 1] = S.view(pair1, S.Tensor((1,), S.u32))[0]
                    b_smem[1, b_warp_col, b_lane_base + 2, 1] = S.view(pair2, S.Tensor((1,), S.u32))[0]
                    b_smem[1, b_warp_col, b_lane_base + 3, 1] = S.view(pair3, S.Tensor((1,), S.u32))[0]
                    b_smem[1, b_warp_col, b_lane_base + 4, 1] = S.view(pair4, S.Tensor((1,), S.u32))[0]
                    b_smem[1, b_warp_col, b_lane_base + 5, 1] = S.view(pair5, S.Tensor((1,), S.u32))[0]
                    b_smem[1, b_warp_col, b_lane_base + 6, 1] = S.view(pair6, S.Tensor((1,), S.u32))[0]
                    b_smem[1, b_warp_col, b_lane_base + 7, 1] = S.view(pair7, S.Tensor((1,), S.u32))[0]
                elif b_pair_id == 4:
                    b_lane_base = b_local_chunk * 8
                    b_smem[1, b_warp_col, b_lane_base + 0, 2] = S.view(pair0, S.Tensor((1,), S.u32))[0]
                    b_smem[1, b_warp_col, b_lane_base + 1, 2] = S.view(pair1, S.Tensor((1,), S.u32))[0]
                    b_smem[1, b_warp_col, b_lane_base + 2, 2] = S.view(pair2, S.Tensor((1,), S.u32))[0]
                    b_smem[1, b_warp_col, b_lane_base + 3, 2] = S.view(pair3, S.Tensor((1,), S.u32))[0]
                    b_smem[1, b_warp_col, b_lane_base + 4, 2] = S.view(pair4, S.Tensor((1,), S.u32))[0]
                    b_smem[1, b_warp_col, b_lane_base + 5, 2] = S.view(pair5, S.Tensor((1,), S.u32))[0]
                    b_smem[1, b_warp_col, b_lane_base + 6, 2] = S.view(pair6, S.Tensor((1,), S.u32))[0]
                    b_smem[1, b_warp_col, b_lane_base + 7, 2] = S.view(pair7, S.Tensor((1,), S.u32))[0]
                elif b_pair_id == 5:
                    b_lane_base = b_local_chunk * 8
                    b_smem[1, b_warp_col, b_lane_base + 0, 3] = S.view(pair0, S.Tensor((1,), S.u32))[0]
                    b_smem[1, b_warp_col, b_lane_base + 1, 3] = S.view(pair1, S.Tensor((1,), S.u32))[0]
                    b_smem[1, b_warp_col, b_lane_base + 2, 3] = S.view(pair2, S.Tensor((1,), S.u32))[0]
                    b_smem[1, b_warp_col, b_lane_base + 3, 3] = S.view(pair3, S.Tensor((1,), S.u32))[0]
                    b_smem[1, b_warp_col, b_lane_base + 4, 3] = S.view(pair4, S.Tensor((1,), S.u32))[0]
                    b_smem[1, b_warp_col, b_lane_base + 5, 3] = S.view(pair5, S.Tensor((1,), S.u32))[0]
                    b_smem[1, b_warp_col, b_lane_base + 6, 3] = S.view(pair6, S.Tensor((1,), S.u32))[0]
                    b_smem[1, b_warp_col, b_lane_base + 7, 3] = S.view(pair7, S.Tensor((1,), S.u32))[0]
                elif b_pair_id == 6:
                    b_lane_base = b_local_chunk * 8 + 32
                    b_smem[1, b_warp_col, b_lane_base + 0, 2] = S.view(pair0, S.Tensor((1,), S.u32))[0]
                    b_smem[1, b_warp_col, b_lane_base + 1, 2] = S.view(pair1, S.Tensor((1,), S.u32))[0]
                    b_smem[1, b_warp_col, b_lane_base + 2, 2] = S.view(pair2, S.Tensor((1,), S.u32))[0]
                    b_smem[1, b_warp_col, b_lane_base + 3, 2] = S.view(pair3, S.Tensor((1,), S.u32))[0]
                    b_smem[1, b_warp_col, b_lane_base + 4, 2] = S.view(pair4, S.Tensor((1,), S.u32))[0]
                    b_smem[1, b_warp_col, b_lane_base + 5, 2] = S.view(pair5, S.Tensor((1,), S.u32))[0]
                    b_smem[1, b_warp_col, b_lane_base + 6, 2] = S.view(pair6, S.Tensor((1,), S.u32))[0]
                    b_smem[1, b_warp_col, b_lane_base + 7, 2] = S.view(pair7, S.Tensor((1,), S.u32))[0]
                else:
                    b_lane_base = b_local_chunk * 8 + 32
                    b_smem[1, b_warp_col, b_lane_base + 0, 3] = S.view(pair0, S.Tensor((1,), S.u32))[0]
                    b_smem[1, b_warp_col, b_lane_base + 1, 3] = S.view(pair1, S.Tensor((1,), S.u32))[0]
                    b_smem[1, b_warp_col, b_lane_base + 2, 3] = S.view(pair2, S.Tensor((1,), S.u32))[0]
                    b_smem[1, b_warp_col, b_lane_base + 3, 3] = S.view(pair3, S.Tensor((1,), S.u32))[0]
                    b_smem[1, b_warp_col, b_lane_base + 4, 3] = S.view(pair4, S.Tensor((1,), S.u32))[0]
                    b_smem[1, b_warp_col, b_lane_base + 5, 3] = S.view(pair5, S.Tensor((1,), S.u32))[0]
                    b_smem[1, b_warp_col, b_lane_base + 6, 3] = S.view(pair6, S.Tensor((1,), S.u32))[0]
                    b_smem[1, b_warp_col, b_lane_base + 7, 3] = S.view(pair7, S.Tensor((1,), S.u32))[0]

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[1], b_frag1[1], acc)
        S.syncthreads()

    tile_row_base = block_row + warp_row * 32
    tile_col_base = block_col + warp_col * 32

    for acc_idx in S.range(16):
        col = tile_col_base + (lane & 31)
        row = tile_row_base + 8 * (acc_idx >> 2) + 4 * (lane >> 5) + (acc_idx & 3)
        C[batch, row, col] = S.convert(acc[acc_idx], S.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if tuple(A.shape) != (BATCH, M, K):
            raise ValueError(f"expected A shape {(BATCH, M, K)}, got {tuple(A.shape)}")
        if tuple(B.shape) != (K, N):
            raise ValueError(f"expected B shape {(K, N)}, got {tuple(B.shape)}")
        if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
            raise ValueError("expected bf16 inputs")

        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((BATCH, M, N), device=A.device, dtype=torch.bfloat16)
        matmul3d_mfma_kernel[lambda: ((N // BLOCK_N, M // BLOCK_M, BATCH), (256, 1, 1))](
            A, B, C
        )
        return C
