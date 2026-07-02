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

    a_shared = S.make_shared((WARPS_PER_BLOCK, 32, 16), S.bf16)
    b_shared = S.make_shared((WARPS_PER_BLOCK, 16, 32), S.bf16)

    a_rsrc = S.amdgpu.make_rsrc(A, BATCH * M * K * BF16_BYTES)
    b_rsrc = S.amdgpu.make_rsrc(B, BATCH * N * K * BF16_BYTES)
    zero = S.convert(0, S.i32)

    acc = S.full((16,), 0.0, S.f32)

    for k0 in S.range(0, K, 16):
        a_row = lane % 32
        a_group = lane // 32
        a0_k = k0 + a_group * 4
        a1_k = k0 + 8 + a_group * 4
        a0_elem_offset = ((batch * M + tile_row_base + a_row) * K + a0_k) * BF16_BYTES
        a1_elem_offset = ((batch * M + tile_row_base + a_row) * K + a1_k) * BF16_BYTES
        a0_vec = S.amdgpu.raw_buffer_load_x4(a_rsrc, zero, a0_elem_offset, 0)
        a1_vec = S.amdgpu.raw_buffer_load_x4(a_rsrc, zero, a1_elem_offset, 0)
        a0_loaded = S.view(a0_vec, S.Tensor((2, 4, 1), S.bf16))
        a1_loaded = S.view(a1_vec, S.Tensor((2, 4, 1), S.bf16))

        b_col = lane % 32
        b_group = lane // 32
        b0_k = k0 + b_group * 4
        b1_k = k0 + 8 + b_group * 4
        b0_elem_offset = ((batch * N + tile_col_base + b_col) * K + b0_k) * BF16_BYTES
        b1_elem_offset = ((batch * N + tile_col_base + b_col) * K + b1_k) * BF16_BYTES
        b0_vec = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero, b0_elem_offset, 0)
        b1_vec = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero, b1_elem_offset, 0)
        b0_loaded = S.view(b0_vec, S.Tensor((2, 4, 1), S.bf16))
        b1_loaded = S.view(b1_vec, S.Tensor((2, 4, 1), S.bf16))

        for i in S.range(4):
            a_shared[warp, a_row, a_group * 8 + i] = a0_loaded[0, i, 0]
            a_shared[warp, a_row, a_group * 8 + 4 + i] = a1_loaded[0, i, 0]
            b_shared[warp, b_group * 4 + i, b_col] = b0_loaded[0, i, 0]
            b_shared[warp, 8 + b_group * 4 + i, b_col] = b1_loaded[0, i, 0]

        S.syncthreads()

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a0_loaded[0], b0_loaded[0], acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a1_loaded[0], b1_loaded[0], acc)

        S.syncthreads()

    lane_col = tile_col_base + (lane % 32)
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
