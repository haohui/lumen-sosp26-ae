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
NUM_WAVES = 4
THREADS = WAVE_SIZE * NUM_WAVES
A_RANGE_BYTES = M * K * 2
B_RANGE_BYTES = K * N * 2


@substrate.jit
def tri_gemm_kernel(
    A: S.Tensor((4096, 4096), S.bf16),
    B: S.Tensor((4096, 4096), S.bf16),
    C: S.Tensor((4096, 4096), S.bf16),
):
    tid = S.thread_id(0)
    lane = tid % WAVE_SIZE
    warp = tid // WAVE_SIZE
    warp_m = warp // 2
    warp_n = warp % 2

    block_m = S.block_id(1) * BLOCK_M
    block_n = S.block_id(0) * BLOCK_N

    a_rsrc = S.amdgpu.make_rsrc(A, A_RANGE_BYTES)
    b_rsrc = S.amdgpu.make_rsrc(B, B_RANGE_BYTES)
    zero = S.convert(0, S.i32)

    acc = S.full((16,), 0.0, S.f32)

    a_lane_row = lane % 32
    a_lane_col_group = lane // 32
    b_lane_row = lane // 8
    b_lane_col_group = lane % 8
    a_row = block_m + warp_m * 32 + a_lane_row
    b_col = block_n + warp_n * 32 + b_lane_col_group * 4

    for k0 in S.range(K // BLOCK_K):
        k_base = k0 * BLOCK_K

        a_col_base = k_base + a_lane_col_group * 8
        a_offset = (a_row * K + a_col_base) * 2
        a_pack = S.amdgpu.raw_buffer_load_x4(
            a_rsrc,
            zero,
            S.convert(a_offset, S.i32),
            0,
        )
        a_frag = S.view(a_pack, S.Tensor((2, 4, 1), S.bf16))

        b_row_0 = k_base + b_lane_row
        b_row_1 = b_row_0 + 8
        b_offset_0 = (b_row_0 * N + b_col) * 2
        b_offset_1 = (b_row_1 * N + b_col) * 2
        b_pack_0 = S.amdgpu.raw_buffer_load_x2(
            b_rsrc,
            zero,
            S.convert(b_offset_0, S.i32),
            0,
        )
        b_pack_1 = S.amdgpu.raw_buffer_load_x2(
            b_rsrc,
            zero,
            S.convert(b_offset_1, S.i32),
            0,
        )
        b_frag_0 = S.view(b_pack_0, S.Tensor((1, 4, 1), S.bf16))
        b_frag_1 = S.view(b_pack_1, S.Tensor((1, 4, 1), S.bf16))

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag_0[0], acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[1], b_frag_1[0], acc)

    lane_row_group = lane % 8
    lane_col_group = lane // 8
    tile_row = block_m + warp_m * 32 + lane_row_group * 4
    tile_col = block_n + warp_n * 32 + lane_col_group * 4

    for i in S.range(16):
        out_row = tile_row + i // 4
        out_col = tile_col + i % 4
        if out_col >= out_row:
            C[out_row, out_col] = S.convert(acc[i], S.bf16)
        else:
            C[out_row, out_col] = S.convert(0.0, S.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if tuple(A.shape) != (4096, 4096) or tuple(B.shape) != (4096, 4096):
            raise ValueError("ModelNew only supports 4096x4096 bf16 inputs")
        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((4096, 4096), device=A.device, dtype=A.dtype)
        tri_gemm_kernel[lambda: ((N // BLOCK_N, M // BLOCK_M, 1), (THREADS, 1, 1))](A, B, C)
        return C
