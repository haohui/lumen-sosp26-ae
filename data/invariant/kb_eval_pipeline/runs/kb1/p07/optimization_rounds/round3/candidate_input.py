import torch
import torch.nn as nn

import substrate
import substrate.language as S


M = 32768
K = 64
N = 32768
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 16
WAVE_M = 32
WAVE_N = 32
WAVES_PER_BLOCK = 4
THREADS_PER_BLOCK = 64 * WAVES_PER_BLOCK
A_RANGE_BYTES = M * K * 2
B_RANGE_BYTES = K * N * 2
@substrate.jit
def gemm_kernel(
    A: S.Tensor((M, K), S.bf16),
    B: S.Tensor((K, N), S.bf16),
    C: S.Tensor((M, N), S.bf16),
):
    tid = S.thread_id(0)
    lane = tid % 64
    warp_id = tid // 64
    warp_row = warp_id // 2
    warp_col = warp_id % 2
    block_row = S.block_id(1)
    block_col = S.block_id(0)

    a_rsrc = S.amdgpu.make_rsrc(A, S.convert(A_RANGE_BYTES, S.i32))
    b_rsrc = S.amdgpu.make_rsrc(B, S.convert(B_RANGE_BYTES, S.i32))
    zero = S.convert(0, S.i32)

    a_words = S.make_shared((1024,), S.u32)
    b_words = S.make_shared((1024,), S.u32)
    a_lane_packs = S.view(a_words, S.u32, S.make_layout((256, 4), (4, 1)))
    b_lane_packs = S.view(b_words, S.u32, S.make_layout((256, 4), (4, 1)))
    b_lane_vals = S.view(b_words, S.bf16, S.make_layout((256, 8), (8, 1)))
    acc = S.full((16,), 0.0, S.f32)

    block_row_base = block_row * BLOCK_M
    block_col_base = block_col * BLOCK_N
    warp_row_base = warp_row * WAVE_M
    warp_col_base = warp_col * WAVE_N
    a_row = warp_row_base + (lane % 32)
    a_half = lane // 32
    b_col = warp_col_base + (lane % 32)
    a_idx = a_row * 2 + a_half
    b_idx = b_col * 2 + a_half

    if tid < 128:
        a_linear = tid * 8
        a_load_row = a_linear // BLOCK_K
        a_col = a_linear % BLOCK_K
        load_group = tid % 2
        lane_base0 = (a_load_row * 2 + 0) * 4 + load_group * 2
        lane_base1 = (a_load_row * 2 + 1) * 4 + load_group * 2

        a_offset0 = ((block_row_base + a_load_row) * K + 0 * BLOCK_K + a_col) * 2
        a_pack0 = S.amdgpu.raw_buffer_load_x4(a_rsrc, zero, S.convert(a_offset0, S.i32), 0)
        a_words[lane_base0 + 0] = a_pack0[0]
        a_words[lane_base0 + 1] = a_pack0[1]
        a_words[lane_base1 + 0] = a_pack0[2]
        a_words[lane_base1 + 1] = a_pack0[3]
    else:
        b_tid = tid - 128
        b_linear = b_tid * 8
        b_row = b_linear // BLOCK_N
        b_load_col = b_linear % BLOCK_N
        b_offset0 = ((0 * BLOCK_K + b_row) * N + block_col_base + b_load_col) * 2
        b_pack0 = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero, S.convert(b_offset0, S.i32), 0)
        b_vals0 = S.view(b_pack0, S.Tensor((2, 4, 1), S.bf16))
        row_mod8 = b_row % 8
        row_half = row_mod8 // 4
        if b_row < 8:
            row_slot = row_mod8 % 4
            for e in S.range(4):
                pack_idx0 = (b_load_col + e) * 2 + row_half
                pack_idx1 = (b_load_col + 4 + e) * 2 + row_half
                b_lane_vals[pack_idx0, row_slot] = b_vals0[0, e, 0]
                b_lane_vals[pack_idx1, row_slot] = b_vals0[1, e, 0]
        else:
            row_slot = 4 + (row_mod8 % 4)
            for e in S.range(4):
                pack_idx0 = (b_load_col + e) * 2 + row_half
                pack_idx1 = (b_load_col + 4 + e) * 2 + row_half
                b_lane_vals[pack_idx0, row_slot] = b_vals0[0, e, 0]
                b_lane_vals[pack_idx1, row_slot] = b_vals0[1, e, 0]

    S.syncthreads()

    if tid < 128:
        a_linear = tid * 8
        a_load_row = a_linear // BLOCK_K
        a_col = a_linear % BLOCK_K
        load_group = tid % 2
        lane_base0 = 512 + (a_load_row * 2 + 0) * 4 + load_group * 2
        lane_base1 = 512 + (a_load_row * 2 + 1) * 4 + load_group * 2

        a_offset1 = ((block_row_base + a_load_row) * K + 1 * BLOCK_K + a_col) * 2
        a_pack1 = S.amdgpu.raw_buffer_load_x4(a_rsrc, zero, S.convert(a_offset1, S.i32), 0)
        a_words[lane_base0 + 0] = a_pack1[0]
        a_words[lane_base0 + 1] = a_pack1[1]
        a_words[lane_base1 + 0] = a_pack1[2]
        a_words[lane_base1 + 1] = a_pack1[3]
    else:
        b_tid = tid - 128
        b_linear = b_tid * 8
        b_row = b_linear // BLOCK_N
        b_load_col = b_linear % BLOCK_N
        b_offset1 = ((1 * BLOCK_K + b_row) * N + block_col_base + b_load_col) * 2
        b_pack1 = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero, S.convert(b_offset1, S.i32), 0)
        b_vals1 = S.view(b_pack1, S.Tensor((2, 4, 1), S.bf16))
        row_mod8 = b_row % 8
        row_half = row_mod8 // 4
        if b_row < 8:
            row_slot = row_mod8 % 4
            for e in S.range(4):
                pack_idx0 = 128 + (b_load_col + e) * 2 + row_half
                pack_idx1 = 128 + (b_load_col + 4 + e) * 2 + row_half
                b_lane_vals[pack_idx0, row_slot] = b_vals1[0, e, 0]
                b_lane_vals[pack_idx1, row_slot] = b_vals1[1, e, 0]
        else:
            row_slot = 4 + (row_mod8 % 4)
            for e in S.range(4):
                pack_idx0 = 128 + (b_load_col + e) * 2 + row_half
                pack_idx1 = 128 + (b_load_col + 4 + e) * 2 + row_half
                b_lane_vals[pack_idx0, row_slot] = b_vals1[0, e, 0]
                b_lane_vals[pack_idx1, row_slot] = b_vals1[1, e, 0]

    a_pack = a_lane_packs[a_idx]
    b_pack = b_lane_packs[b_idx]
    a_vec = S.view(a_pack, S.Tensor((2, 4, 1), S.bf16))
    b_vec = S.view(b_pack, S.Tensor((2, 4, 1), S.bf16))
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_vec[0], b_vec[0], acc)
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_vec[1], b_vec[1], acc)

    S.syncthreads()

    if tid < 128:
        a_linear = tid * 8
        a_load_row = a_linear // BLOCK_K
        a_col = a_linear % BLOCK_K
        load_group = tid % 2
        lane_base0 = (a_load_row * 2 + 0) * 4 + load_group * 2
        lane_base1 = (a_load_row * 2 + 1) * 4 + load_group * 2

        a_offset2 = ((block_row_base + a_load_row) * K + 2 * BLOCK_K + a_col) * 2
        a_pack2 = S.amdgpu.raw_buffer_load_x4(a_rsrc, zero, S.convert(a_offset2, S.i32), 0)
        a_words[lane_base0 + 0] = a_pack2[0]
        a_words[lane_base0 + 1] = a_pack2[1]
        a_words[lane_base1 + 0] = a_pack2[2]
        a_words[lane_base1 + 1] = a_pack2[3]
    else:
        b_tid = tid - 128
        b_linear = b_tid * 8
        b_row = b_linear // BLOCK_N
        b_load_col = b_linear % BLOCK_N
        b_offset2 = ((2 * BLOCK_K + b_row) * N + block_col_base + b_load_col) * 2
        b_pack2 = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero, S.convert(b_offset2, S.i32), 0)
        b_vals2 = S.view(b_pack2, S.Tensor((2, 4, 1), S.bf16))
        row_mod8 = b_row % 8
        row_half = row_mod8 // 4
        if b_row < 8:
            row_slot = row_mod8 % 4
            for e in S.range(4):
                pack_idx0 = (b_load_col + e) * 2 + row_half
                pack_idx1 = (b_load_col + 4 + e) * 2 + row_half
                b_lane_vals[pack_idx0, row_slot] = b_vals2[0, e, 0]
                b_lane_vals[pack_idx1, row_slot] = b_vals2[1, e, 0]
        else:
            row_slot = 4 + (row_mod8 % 4)
            for e in S.range(4):
                pack_idx0 = (b_load_col + e) * 2 + row_half
                pack_idx1 = (b_load_col + 4 + e) * 2 + row_half
                b_lane_vals[pack_idx0, row_slot] = b_vals2[0, e, 0]
                b_lane_vals[pack_idx1, row_slot] = b_vals2[1, e, 0]

    a_pack = a_lane_packs[128 + a_idx]
    b_pack = b_lane_packs[128 + b_idx]
    a_vec = S.view(a_pack, S.Tensor((2, 4, 1), S.bf16))
    b_vec = S.view(b_pack, S.Tensor((2, 4, 1), S.bf16))
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_vec[0], b_vec[0], acc)
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_vec[1], b_vec[1], acc)

    S.syncthreads()

    if tid < 128:
        a_linear = tid * 8
        a_load_row = a_linear // BLOCK_K
        a_col = a_linear % BLOCK_K
        load_group = tid % 2
        lane_base0 = 512 + (a_load_row * 2 + 0) * 4 + load_group * 2
        lane_base1 = 512 + (a_load_row * 2 + 1) * 4 + load_group * 2

        a_offset3 = ((block_row_base + a_load_row) * K + 3 * BLOCK_K + a_col) * 2
        a_pack3 = S.amdgpu.raw_buffer_load_x4(a_rsrc, zero, S.convert(a_offset3, S.i32), 0)
        a_words[lane_base0 + 0] = a_pack3[0]
        a_words[lane_base0 + 1] = a_pack3[1]
        a_words[lane_base1 + 0] = a_pack3[2]
        a_words[lane_base1 + 1] = a_pack3[3]
    else:
        b_tid = tid - 128
        b_linear = b_tid * 8
        b_row = b_linear // BLOCK_N
        b_load_col = b_linear % BLOCK_N
        b_offset3 = ((3 * BLOCK_K + b_row) * N + block_col_base + b_load_col) * 2
        b_pack3 = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero, S.convert(b_offset3, S.i32), 0)
        b_vals3 = S.view(b_pack3, S.Tensor((2, 4, 1), S.bf16))
        row_mod8 = b_row % 8
        row_half = row_mod8 // 4
        if b_row < 8:
            row_slot = row_mod8 % 4
            for e in S.range(4):
                pack_idx0 = 128 + (b_load_col + e) * 2 + row_half
                pack_idx1 = 128 + (b_load_col + 4 + e) * 2 + row_half
                b_lane_vals[pack_idx0, row_slot] = b_vals3[0, e, 0]
                b_lane_vals[pack_idx1, row_slot] = b_vals3[1, e, 0]
        else:
            row_slot = 4 + (row_mod8 % 4)
            for e in S.range(4):
                pack_idx0 = 128 + (b_load_col + e) * 2 + row_half
                pack_idx1 = 128 + (b_load_col + 4 + e) * 2 + row_half
                b_lane_vals[pack_idx0, row_slot] = b_vals3[0, e, 0]
                b_lane_vals[pack_idx1, row_slot] = b_vals3[1, e, 0]

    a_pack = a_lane_packs[a_idx]
    b_pack = b_lane_packs[b_idx]
    a_vec = S.view(a_pack, S.Tensor((2, 4, 1), S.bf16))
    b_vec = S.view(b_pack, S.Tensor((2, 4, 1), S.bf16))
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_vec[0], b_vec[0], acc)
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_vec[1], b_vec[1], acc)

    S.syncthreads()

    a_pack = a_lane_packs[128 + a_idx]
    b_pack = b_lane_packs[128 + b_idx]
    a_vec = S.view(a_pack, S.Tensor((2, 4, 1), S.bf16))
    b_vec = S.view(b_pack, S.Tensor((2, 4, 1), S.bf16))
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_vec[0], b_vec[0], acc)
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_vec[1], b_vec[1], acc)

    S.syncthreads()

    out_col = block_col_base + warp_col_base + (lane % 32)
    out_row_base = block_row_base + warp_row_base + 4 * (lane // 32)
    for acc_idx in S.range(16):
        out_row = out_row_base + 8 * (acc_idx // 4) + (acc_idx % 4)
        C[out_row, out_col] = S.convert(acc[acc_idx], S.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if tuple(A.shape) != (M, K) or tuple(B.shape) != (K, N):
            raise ValueError(f"Expected A={(M, K)} and B={(K, N)}, got {tuple(A.shape)} and {tuple(B.shape)}")
        if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
            raise ValueError(f"Expected bf16 inputs, got {A.dtype} and {B.dtype}")

        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)
        gemm_kernel[lambda: ((N // BLOCK_N, M // BLOCK_M, 1), (THREADS_PER_BLOCK, 1, 1))](A, B, C)
        return C
