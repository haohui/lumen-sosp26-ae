import torch
import torch.nn as nn

import substrate
import substrate.language as S


BATCH = 128
M = 512
K = 1024
N = 2048
BLOCK_M = 64
BLOCK_N = 64
WARP_SIZE = 64
WARPS_PER_BLOCK = 4
THREADS_PER_BLOCK = WARP_SIZE * WARPS_PER_BLOCK
BF16_BYTES = 2


@substrate.jit
def bmm_kernel_mfma(
    A: S.Tensor((BATCH, M, K), S.bf16),
    B: S.Tensor((BATCH, N, K), S.bf16),
    C: S.Tensor((BATCH, M, N), S.bf16),
):
    tid = S.thread_id(0)
    lane = tid % WARP_SIZE
    warp = tid // WARP_SIZE

    warp_row = warp // 2
    warp_col = warp % 2

    batch = S.block_id(2)
    block_row = S.block_id(1) * BLOCK_M
    block_col = S.block_id(0) * BLOCK_N

    tile_row_base = block_row + warp_row * 32
    tile_col_base = block_col + warp_col * 32

    lane_base = lane % 32
    lane_group = lane // 32
    lane_hi = lane_base + 32

    a_words = S.make_shared((2, WARPS_PER_BLOCK, WARP_SIZE, 4), S.u32)
    b_words = S.make_shared((2, WARPS_PER_BLOCK, WARP_SIZE, 4), S.u32)

    a_rsrc = S.amdgpu.make_rsrc(A, BATCH * M * K * BF16_BYTES)
    b_rsrc = S.amdgpu.make_rsrc(B, BATCH * N * K * BF16_BYTES)
    zero = S.convert(0, S.i32)

    acc = S.full((16,), 0.0, S.f32)

    a_row = tile_row_base + lane_base
    b_col = tile_col_base + lane_base

    a_k0 = lane_group * 8
    a_elem_offset0 = ((batch * M + a_row) * K + a_k0) * BF16_BYTES
    a_vec0 = S.amdgpu.raw_buffer_load_x4(a_rsrc, zero, a_elem_offset0, 0)
    if lane_group == 0:
        a_words[0, warp, lane_base, 0] = a_vec0[0]
        a_words[0, warp, lane_base, 1] = a_vec0[1]
        a_words[0, warp, lane_hi, 0] = a_vec0[2]
        a_words[0, warp, lane_hi, 1] = a_vec0[3]
    else:
        a_words[0, warp, lane_base, 2] = a_vec0[0]
        a_words[0, warp, lane_base, 3] = a_vec0[1]
        a_words[0, warp, lane_hi, 2] = a_vec0[2]
        a_words[0, warp, lane_hi, 3] = a_vec0[3]

    b_k0 = lane_group * 8
    b_elem_offset0 = ((batch * N + b_col) * K + b_k0) * BF16_BYTES
    b_vec0 = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero, b_elem_offset0, 0)
    if lane_group == 0:
        b_words[0, warp, lane_base, 0] = b_vec0[0]
        b_words[0, warp, lane_base, 1] = b_vec0[1]
        b_words[0, warp, lane_hi, 0] = b_vec0[2]
        b_words[0, warp, lane_hi, 1] = b_vec0[3]
    else:
        b_words[0, warp, lane_base, 2] = b_vec0[0]
        b_words[0, warp, lane_base, 3] = b_vec0[1]
        b_words[0, warp, lane_hi, 2] = b_vec0[2]
        b_words[0, warp, lane_hi, 3] = b_vec0[3]

    a_k1 = 16 + lane_group * 8
    a_elem_offset1 = ((batch * M + a_row) * K + a_k1) * BF16_BYTES
    a_vec1 = S.amdgpu.raw_buffer_load_x4(a_rsrc, zero, a_elem_offset1, 0)
    if lane_group == 0:
        a_words[1, warp, lane_base, 0] = a_vec1[0]
        a_words[1, warp, lane_base, 1] = a_vec1[1]
        a_words[1, warp, lane_hi, 0] = a_vec1[2]
        a_words[1, warp, lane_hi, 1] = a_vec1[3]
    else:
        a_words[1, warp, lane_base, 2] = a_vec1[0]
        a_words[1, warp, lane_base, 3] = a_vec1[1]
        a_words[1, warp, lane_hi, 2] = a_vec1[2]
        a_words[1, warp, lane_hi, 3] = a_vec1[3]

    b_k1 = 16 + lane_group * 8
    b_elem_offset1 = ((batch * N + b_col) * K + b_k1) * BF16_BYTES
    b_vec1 = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero, b_elem_offset1, 0)
    if lane_group == 0:
        b_words[1, warp, lane_base, 0] = b_vec1[0]
        b_words[1, warp, lane_base, 1] = b_vec1[1]
        b_words[1, warp, lane_hi, 0] = b_vec1[2]
        b_words[1, warp, lane_hi, 1] = b_vec1[3]
    else:
        b_words[1, warp, lane_base, 2] = b_vec1[0]
        b_words[1, warp, lane_base, 3] = b_vec1[1]
        b_words[1, warp, lane_hi, 2] = b_vec1[2]
        b_words[1, warp, lane_hi, 3] = b_vec1[3]

    S.syncthreads()

    for k0 in S.range(0, K - 32, 32):
        a_frag0 = S.view(a_words[0, warp, lane], S.Tensor((2, 4, 1), S.bf16))
        b_frag0 = S.view(b_words[0, warp, lane], S.Tensor((2, 4, 1), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[0], b_frag0[0], acc)

        next_k0 = k0 + 32 + lane_group * 8
        next_a_offset0 = ((batch * M + a_row) * K + next_k0) * BF16_BYTES
        next_b_offset0 = ((batch * N + b_col) * K + next_k0) * BF16_BYTES
        next_a_vec0 = S.amdgpu.raw_buffer_load_x4(a_rsrc, zero, next_a_offset0, 0)
        next_b_vec0 = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero, next_b_offset0, 0)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[1], b_frag0[1], acc)

        if lane_group == 0:
            a_words[0, warp, lane_base, 0] = next_a_vec0[0]
            a_words[0, warp, lane_base, 1] = next_a_vec0[1]
            a_words[0, warp, lane_hi, 0] = next_a_vec0[2]
            a_words[0, warp, lane_hi, 1] = next_a_vec0[3]
            b_words[0, warp, lane_base, 0] = next_b_vec0[0]
            b_words[0, warp, lane_base, 1] = next_b_vec0[1]
            b_words[0, warp, lane_hi, 0] = next_b_vec0[2]
            b_words[0, warp, lane_hi, 1] = next_b_vec0[3]
        else:
            a_words[0, warp, lane_base, 2] = next_a_vec0[0]
            a_words[0, warp, lane_base, 3] = next_a_vec0[1]
            a_words[0, warp, lane_hi, 2] = next_a_vec0[2]
            a_words[0, warp, lane_hi, 3] = next_a_vec0[3]
            b_words[0, warp, lane_base, 2] = next_b_vec0[0]
            b_words[0, warp, lane_base, 3] = next_b_vec0[1]
            b_words[0, warp, lane_hi, 2] = next_b_vec0[2]
            b_words[0, warp, lane_hi, 3] = next_b_vec0[3]

        a_frag1 = S.view(a_words[1, warp, lane], S.Tensor((2, 4, 1), S.bf16))
        b_frag1 = S.view(b_words[1, warp, lane], S.Tensor((2, 4, 1), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[0], b_frag1[0], acc)

        next_k1 = k0 + 48 + lane_group * 8
        next_a_offset1 = ((batch * M + a_row) * K + next_k1) * BF16_BYTES
        next_b_offset1 = ((batch * N + b_col) * K + next_k1) * BF16_BYTES
        next_a_vec1 = S.amdgpu.raw_buffer_load_x4(a_rsrc, zero, next_a_offset1, 0)
        next_b_vec1 = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero, next_b_offset1, 0)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[1], b_frag1[1], acc)

        if lane_group == 0:
            a_words[1, warp, lane_base, 0] = next_a_vec1[0]
            a_words[1, warp, lane_base, 1] = next_a_vec1[1]
            a_words[1, warp, lane_hi, 0] = next_a_vec1[2]
            a_words[1, warp, lane_hi, 1] = next_a_vec1[3]
            b_words[1, warp, lane_base, 0] = next_b_vec1[0]
            b_words[1, warp, lane_base, 1] = next_b_vec1[1]
            b_words[1, warp, lane_hi, 0] = next_b_vec1[2]
            b_words[1, warp, lane_hi, 1] = next_b_vec1[3]
        else:
            a_words[1, warp, lane_base, 2] = next_a_vec1[0]
            a_words[1, warp, lane_base, 3] = next_a_vec1[1]
            a_words[1, warp, lane_hi, 2] = next_a_vec1[2]
            a_words[1, warp, lane_hi, 3] = next_a_vec1[3]
            b_words[1, warp, lane_base, 2] = next_b_vec1[0]
            b_words[1, warp, lane_base, 3] = next_b_vec1[1]
            b_words[1, warp, lane_hi, 2] = next_b_vec1[2]
            b_words[1, warp, lane_hi, 3] = next_b_vec1[3]

        S.syncthreads()

    a_frag0 = S.view(a_words[0, warp, lane], S.Tensor((2, 4, 1), S.bf16))
    b_frag0 = S.view(b_words[0, warp, lane], S.Tensor((2, 4, 1), S.bf16))
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[0], b_frag0[0], acc)
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[1], b_frag0[1], acc)

    a_frag1 = S.view(a_words[1, warp, lane], S.Tensor((2, 4, 1), S.bf16))
    b_frag1 = S.view(b_words[1, warp, lane], S.Tensor((2, 4, 1), S.bf16))
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[0], b_frag1[0], acc)
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[1], b_frag1[1], acc)

    lane_col = tile_col_base + lane_base
    lane_row_group = 4 * (lane // 32)
    for acc_idx in S.range(16):
        row = tile_row_base + 8 * (acc_idx // 4) + lane_row_group + (acc_idx % 4)
        C[batch, row, lane_col] = S.convert(acc[acc_idx], S.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if tuple(A.shape) != (BATCH, M, K) or tuple(B.shape) != (BATCH, K, N):
            raise ValueError(
                f"Expected A={(BATCH, M, K)} and B={(BATCH, K, N)}, "
                f"got A={tuple(A.shape)} and B={tuple(B.shape)}"
            )
        if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
            raise ValueError(f"Expected bf16 inputs, got A={A.dtype}, B={B.dtype}")

        A = A.contiguous()
        B = B.transpose(1, 2).contiguous()
        C = torch.empty((BATCH, M, N), device=A.device, dtype=torch.bfloat16)
        bmm_kernel_mfma[lambda: ((N // BLOCK_N, M // BLOCK_M, BATCH), (THREADS_PER_BLOCK, 1, 1))](
            A, B, C
        )
        return C
