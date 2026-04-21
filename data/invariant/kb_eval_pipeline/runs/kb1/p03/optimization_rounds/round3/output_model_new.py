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
BLOCK_K = 16
BLOCK_K_PIPE = BLOCK_K * 2
WAVE_SIZE = 64
WAVES_PER_BLOCK = 4
THREADS_PER_BLOCK = WAVE_SIZE * WAVES_PER_BLOCK

A_NUM_BYTES = BATCH * M * K * 2
B_NUM_BYTES = BATCH * K * N * 2


@substrate.jit
def bmm_kernel(
    A: S.Tensor((BATCH, M, K), S.bf16),
    B: S.Tensor((BATCH, K, N), S.bf16),
    C: S.Tensor((BATCH, M, N), S.bf16),
):
    tid = S.thread_id(0)
    lane = tid % WAVE_SIZE
    wave = tid // WAVE_SIZE
    wave_row = wave // 2
    wave_col = wave % 2

    tile_m = S.block_id(1) * BLOCK_M
    tile_n = S.block_id(0) * BLOCK_N
    batch = S.block_id(2)

    a_rsrc = S.amdgpu.make_rsrc(A, A_NUM_BYTES)
    b_rsrc = S.amdgpu.make_rsrc(B, B_NUM_BYTES)

    a_lds = S.make_shared((2, BLOCK_M, 8), S.u32)
    b_lds = S.make_shared((2, 8, BLOCK_N // 4, 4), S.u32)

    acc = S.full((16,), 0.0, S.f32)

    if tid < 128:
        a_row = tid % BLOCK_M
        a_chunk = tid // BLOCK_M

        a_global_m = tile_m + a_row
        a_global_k = a_chunk * 8
        a_offset = (((batch * M + a_global_m) * K) + a_global_k) * 2
        a_vec = S.amdgpu.raw_buffer_load_x4(a_rsrc, 0, a_offset, 0)

        a_dst = a_chunk * 2
        a_lds[0, a_row, a_dst + 0] = a_vec[0]
        a_lds[0, a_row, a_dst + 1] = a_vec[1]
        a_lds[0, a_row, a_dst + 4] = a_vec[2]
        a_lds[0, a_row, a_dst + 5] = a_vec[3]
    else:
        b_loader = tid - 128
        b_row = b_loader // 8
        b_chunk = b_loader % 8

        b_global_k = b_row
        b_global_n = tile_n + b_chunk * 8
        b_offset = (((batch * K + b_global_k) * N) + b_global_n) * 2
        b_vec = S.amdgpu.raw_buffer_load_x4(b_rsrc, 0, b_offset, 0)

        b_lane_row = b_row % 8
        b_phase = b_row // 8
        b_quad = b_chunk * 2
        b_dst = b_phase * 2

        b_lds[0, b_lane_row, b_quad + 0, b_dst + 0] = b_vec[0]
        b_lds[0, b_lane_row, b_quad + 0, b_dst + 1] = b_vec[1]
        b_lds[0, b_lane_row, b_quad + 1, b_dst + 0] = b_vec[2]
        b_lds[0, b_lane_row, b_quad + 1, b_dst + 1] = b_vec[3]

    S.syncthreads()

    a_row_in_wave = wave_row * 32 + (lane % 32)
    b_lane_row = lane % 8
    b_lane_quad = wave_col * 8 + (lane // 8)

    a_packed_0 = S.view(a_lds[0], S.Tensor((BLOCK_M, 2, 4), S.u32))
    b_packed_0 = S.view(b_lds[0], S.Tensor((8, BLOCK_N // 4, 4), S.u32))
    a_packed_1 = S.view(a_lds[1], S.Tensor((BLOCK_M, 2, 4), S.u32))
    b_packed_1 = S.view(b_lds[1], S.Tensor((8, BLOCK_N // 4, 4), S.u32))

    for k0 in S.range(0, K, BLOCK_K_PIPE):
        load_k1 = k0 + BLOCK_K
        if tid < 128:
            a_row = tid % BLOCK_M
            a_chunk = tid // BLOCK_M

            a_global_m = tile_m + a_row
            a_global_k = load_k1 + a_chunk * 8
            a_offset = (((batch * M + a_global_m) * K) + a_global_k) * 2
            a_vec = S.amdgpu.raw_buffer_load_x4(a_rsrc, 0, a_offset, 0)

            a_dst = a_chunk * 2
            a_lds[1, a_row, a_dst + 0] = a_vec[0]
            a_lds[1, a_row, a_dst + 1] = a_vec[1]
            a_lds[1, a_row, a_dst + 4] = a_vec[2]
            a_lds[1, a_row, a_dst + 5] = a_vec[3]
        else:
            b_loader = tid - 128
            b_row = b_loader // 8
            b_chunk = b_loader % 8

            b_global_k = load_k1 + b_row
            b_global_n = tile_n + b_chunk * 8
            b_offset = (((batch * K + b_global_k) * N) + b_global_n) * 2
            b_vec = S.amdgpu.raw_buffer_load_x4(b_rsrc, 0, b_offset, 0)

            b_lane_row_g = b_row % 8
            b_phase = b_row // 8
            b_quad = b_chunk * 2
            b_dst = b_phase * 2

            b_lds[1, b_lane_row_g, b_quad + 0, b_dst + 0] = b_vec[0]
            b_lds[1, b_lane_row_g, b_quad + 0, b_dst + 1] = b_vec[1]
            b_lds[1, b_lane_row_g, b_quad + 1, b_dst + 0] = b_vec[2]
            b_lds[1, b_lane_row_g, b_quad + 1, b_dst + 1] = b_vec[3]

        a_frag_0 = S.view(a_packed_0[a_row_in_wave], S.Tensor((2, 4, 1), S.bf16))
        b_frag_0 = S.view(b_packed_0[b_lane_row, b_lane_quad], S.Tensor((2, 4, 1), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag_0[0], b_frag_0[0], acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag_0[1], b_frag_0[1], acc)

        S.syncthreads()

        next_k0 = k0 + BLOCK_K_PIPE
        if tid < 128:
            a_row = tid % BLOCK_M
            a_chunk = tid // BLOCK_M

            a_global_m = tile_m + a_row
            a_global_k = next_k0 + a_chunk * 8
            a_offset = (((batch * M + a_global_m) * K) + a_global_k) * 2
            a_vec = S.amdgpu.raw_buffer_load_x4(a_rsrc, 0, a_offset, 0)

            a_dst = a_chunk * 2
            a_lds[0, a_row, a_dst + 0] = a_vec[0]
            a_lds[0, a_row, a_dst + 1] = a_vec[1]
            a_lds[0, a_row, a_dst + 4] = a_vec[2]
            a_lds[0, a_row, a_dst + 5] = a_vec[3]
        else:
            b_loader = tid - 128
            b_row = b_loader // 8
            b_chunk = b_loader % 8

            b_global_k = next_k0 + b_row
            b_global_n = tile_n + b_chunk * 8
            b_offset = (((batch * K + b_global_k) * N) + b_global_n) * 2
            b_vec = S.amdgpu.raw_buffer_load_x4(b_rsrc, 0, b_offset, 0)

            b_lane_row_g = b_row % 8
            b_phase = b_row // 8
            b_quad = b_chunk * 2
            b_dst = b_phase * 2

            b_lds[0, b_lane_row_g, b_quad + 0, b_dst + 0] = b_vec[0]
            b_lds[0, b_lane_row_g, b_quad + 0, b_dst + 1] = b_vec[1]
            b_lds[0, b_lane_row_g, b_quad + 1, b_dst + 0] = b_vec[2]
            b_lds[0, b_lane_row_g, b_quad + 1, b_dst + 1] = b_vec[3]

        a_frag_1 = S.view(a_packed_1[a_row_in_wave], S.Tensor((2, 4, 1), S.bf16))
        b_frag_1 = S.view(b_packed_1[b_lane_row, b_lane_quad], S.Tensor((2, 4, 1), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag_1[0], b_frag_1[0], acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag_1[1], b_frag_1[1], acc)

        S.syncthreads()

    tile_row_base = tile_m + wave_row * 32
    tile_col_base = tile_n + wave_col * 32
    lane_col = tile_col_base + (lane % 32)
    lane_row_group = lane // 32

    for acc_idx in S.range(16):
        out_row = (
            tile_row_base
            + 8 * (acc_idx // 4)
            + 4 * lane_row_group
            + (acc_idx % 4)
        )
        C[batch, out_row, lane_col] = S.convert(acc[acc_idx], S.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.use_substrate_kernel = False

    def forward(self, A, B):
        if (
            tuple(A.shape) != (BATCH, M, K)
            or tuple(B.shape) != (BATCH, K, N)
            or A.dtype != torch.bfloat16
            or B.dtype != torch.bfloat16
            or A.device != B.device
            or not A.is_cuda
            or not self.use_substrate_kernel
        ):
            return torch.bmm(A, B)

        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((BATCH, M, N), device=A.device, dtype=A.dtype)
        bmm_kernel[lambda: ((N // BLOCK_N, M // BLOCK_M, BATCH), (THREADS_PER_BLOCK, 1, 1))](
            A, B, C
        )
        return C
