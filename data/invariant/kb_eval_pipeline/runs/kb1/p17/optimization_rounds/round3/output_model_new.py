import torch
import torch.nn as nn

import substrate
import substrate.language as S


M = 2048
K = 8192
N = 4096

WAVE_SIZE = 64
WAVES_M = 2
WAVES_N = 2
WARPS_PER_BLOCK = WAVES_M * WAVES_N
BLOCK_M = WAVES_M * 32
BLOCK_N = WAVES_N * 32
BLOCK_K = 16
OUTER_K = BLOCK_K * 2

A_RSRC_RANGE = M * K * 2
B_RSRC_RANGE = N * K * 2


@substrate.jit
def gemm_kernel(
    A: S.Tensor((M, K), S.bf16),
    B: S.Tensor((N, K), S.bf16),
    C: S.Tensor((M, N), S.bf16),
):
    lane = S.thread_id(0)
    warp_id = S.thread_id(1)

    block_row = S.block_id(1) * BLOCK_M
    block_col = S.block_id(0) * BLOCK_N

    warp_row = warp_id // WAVES_N
    warp_col = warp_id % WAVES_N

    a_rsrc = S.amdgpu.make_rsrc(A, A_RSRC_RANGE)
    b_rsrc = S.amdgpu.make_rsrc(B, B_RSRC_RANGE)

    a_shared = S.make_shared((2, WAVES_M, WAVE_SIZE, 4), S.u32)
    b_shared = S.make_shared((2, WAVES_N, WAVE_SIZE, 4), S.u32)

    acc = S.full((16,), 0.0, S.f32)

    if warp_id < WAVES_M:
        a_row = block_row + warp_id * 32 + (lane % 32)
        a_k = (lane // 32) * 8
        a_byte_offset = (a_row * K + a_k) * 2
        a_shared[0, warp_id, lane] = S.amdgpu.raw_buffer_load_x4(
            a_rsrc, a_byte_offset, 0, 0, range=A_RSRC_RANGE
        )
    else:
        b_group = warp_id - WAVES_M
        b_col = block_col + b_group * 32 + (lane % 32)
        b_k = (lane // 32) * 8
        b_byte_offset = (b_col * K + b_k) * 2
        b_shared[0, b_group, lane] = S.amdgpu.raw_buffer_load_x4(
            b_rsrc, b_byte_offset, 0, 0, range=B_RSRC_RANGE
        )

    if warp_id < WAVES_M:
        a_row = block_row + warp_id * 32 + (lane % 32)
        a_k = BLOCK_K + (lane // 32) * 8
        a_byte_offset = (a_row * K + a_k) * 2
        a_shared[1, warp_id, lane] = S.amdgpu.raw_buffer_load_x4(
            a_rsrc, a_byte_offset, 0, 0, range=A_RSRC_RANGE
        )
    else:
        b_group = warp_id - WAVES_M
        b_col = block_col + b_group * 32 + (lane % 32)
        b_k = BLOCK_K + (lane // 32) * 8
        b_byte_offset = (b_col * K + b_k) * 2
        b_shared[1, b_group, lane] = S.amdgpu.raw_buffer_load_x4(
            b_rsrc, b_byte_offset, 0, 0, range=B_RSRC_RANGE
        )

    S.syncthreads()

    for outer_iter in S.range(K // OUTER_K):
        a_frag0 = S.view(a_shared[0, warp_row, lane], S.Tensor((2, 4, 1), S.bf16))
        b_frag0 = S.view(b_shared[0, warp_col, lane], S.Tensor((2, 4, 1), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[0], b_frag0[0], acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[1], b_frag0[1], acc)

        next_k0 = (outer_iter + 1) * OUTER_K
        if warp_id < WAVES_M:
            a_row = block_row + warp_id * 32 + (lane % 32)
            a_k = next_k0 + (lane // 32) * 8
            a_byte_offset = (a_row * K + a_k) * 2
            a_shared[0, warp_id, lane] = S.amdgpu.raw_buffer_load_x4(
                a_rsrc, a_byte_offset, 0, 0, range=A_RSRC_RANGE
            )
        else:
            b_group = warp_id - WAVES_M
            b_col = block_col + b_group * 32 + (lane % 32)
            b_k = next_k0 + (lane // 32) * 8
            b_byte_offset = (b_col * K + b_k) * 2
            b_shared[0, b_group, lane] = S.amdgpu.raw_buffer_load_x4(
                b_rsrc, b_byte_offset, 0, 0, range=B_RSRC_RANGE
            )

        a_frag1 = S.view(a_shared[1, warp_row, lane], S.Tensor((2, 4, 1), S.bf16))
        b_frag1 = S.view(b_shared[1, warp_col, lane], S.Tensor((2, 4, 1), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[0], b_frag1[0], acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[1], b_frag1[1], acc)

        next_k1 = next_k0 + BLOCK_K
        if warp_id < WAVES_M:
            a_row = block_row + warp_id * 32 + (lane % 32)
            a_k = next_k1 + (lane // 32) * 8
            a_byte_offset = (a_row * K + a_k) * 2
            a_shared[1, warp_id, lane] = S.amdgpu.raw_buffer_load_x4(
                a_rsrc, a_byte_offset, 0, 0, range=A_RSRC_RANGE
            )
        else:
            b_group = warp_id - WAVES_M
            b_col = block_col + b_group * 32 + (lane % 32)
            b_k = next_k1 + (lane // 32) * 8
            b_byte_offset = (b_col * K + b_k) * 2
            b_shared[1, b_group, lane] = S.amdgpu.raw_buffer_load_x4(
                b_rsrc, b_byte_offset, 0, 0, range=B_RSRC_RANGE
            )

        S.syncthreads()

    out_col = block_col + warp_col * 32 + (lane % 32)
    out_row_base = block_row + warp_row * 32 + 4 * (lane // 32)

    for acc_idx in S.range(16):
        out_row = out_row_base + 8 * (acc_idx // 4) + (acc_idx % 4)
        C[out_row, out_col] = S.convert(acc[acc_idx], S.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._launch = lambda: ((N // BLOCK_N, M // BLOCK_M, 1), (WAVE_SIZE, WARPS_PER_BLOCK, 1))

    def forward(self, A, B):
        if tuple(A.shape) != (M, K) or tuple(B.shape) != (N, K):
            raise ValueError(f"Expected A={(M, K)} and B={(N, K)}, got {tuple(A.shape)} and {tuple(B.shape)}")
        if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
            raise ValueError(f"Expected bf16 inputs, got {A.dtype} and {B.dtype}")

        A2 = A.contiguous()
        B2 = B.contiguous()
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)
        gemm_kernel[self._launch](A2, B2, C)
        return C
