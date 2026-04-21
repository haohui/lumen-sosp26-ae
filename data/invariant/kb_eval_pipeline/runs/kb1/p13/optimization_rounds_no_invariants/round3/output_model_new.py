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
THREADS = 256
WAVES = 4
A_ROW_RANGE_BYTES = K * 2
B_ROW_RANGE_BYTES = N * 2
TILES = K // BLOCK_K


@substrate.jit
def gemm_kernel(
    A: S.Tensor((M, K), S.bf16),
    B: S.Tensor((K, N), S.bf16),
    C: S.Tensor((M, N), S.bf16),
):
    tid = S.thread_id(0)
    lane = tid % 64
    wave = tid // 64
    wave_m = wave // 2
    wave_n = wave % 2
    lane_row = lane // 32
    lane_col = lane % 32
    k_group = lane_row
    a_row = tid % 64
    a_group = tid // 64
    b_loader = tid - 128
    b_k = b_loader // 8
    b_col_group = b_loader % 8

    block_m = S.block_id(1)
    block_n = S.block_id(0)
    m_base = block_m * BLOCK_M
    n_base = block_n * BLOCK_N

    a_sh_0 = S.make_shared((64, 8), S.u32)
    a_sh_1 = S.make_shared((64, 8), S.u32)
    b_sh_0 = S.make_shared((64, 8), S.u32)
    b_sh_1 = S.make_shared((64, 8), S.u32)
    a_sh_0_packed = S.view(a_sh_0, S.Tensor((64, 2, 4), S.u32))
    a_sh_1_packed = S.view(a_sh_1, S.Tensor((64, 2, 4), S.u32))
    b_sh_0_packed = S.view(b_sh_0, S.Tensor((64, 2, 4), S.u32))
    b_sh_1_packed = S.view(b_sh_1, S.Tensor((64, 2, 4), S.u32))
    b_sh_0_bf16 = S.view(b_sh_0, S.Tensor((64, 16), S.bf16))
    b_sh_1_bf16 = S.view(b_sh_1, S.Tensor((64, 16), S.bf16))

    a_rsrc = S.amdgpu.make_rsrc(A, A_ROW_RANGE_BYTES)
    b_rsrc = S.amdgpu.make_rsrc(B, B_ROW_RANGE_BYTES)

    acc = S.full((16,), 0.0, S.f32)

    if tid < 128:
        a_vindex = (m_base + a_row) * A_ROW_RANGE_BYTES
        a_soffset = a_group * 16
        a_vec = S.amdgpu.raw_buffer_load_x4(a_rsrc, a_vindex, a_soffset, 0)
        for i in S.range(4):
            a_sh_0_packed[a_row, a_group, i] = a_vec[i]
    else:
        b_vindex = b_k * B_ROW_RANGE_BYTES
        b_soffset = (n_base + b_col_group * 8) * 2
        b_vec = S.amdgpu.raw_buffer_load_x4(b_rsrc, b_vindex, b_soffset, 0)
        b_frag = S.view(b_vec, S.Tensor((2, 4, 1), S.bf16))
        for i in S.range(4):
            b_sh_0_bf16[b_col_group * 8 + i, b_k] = b_frag[0, i, 0]
            b_sh_0_bf16[b_col_group * 8 + 4 + i, b_k] = b_frag[1, i, 0]

    S.syncthreads()

    for pair in S.range(TILES // 2):
        even_tile = pair * 2
        odd_tile = even_tile + 1

        if tid < 128:
            a_vindex = (m_base + a_row) * A_ROW_RANGE_BYTES
            a_soffset = (odd_tile * BLOCK_K + a_group * 8) * 2
            a_vec = S.amdgpu.raw_buffer_load_x4(a_rsrc, a_vindex, a_soffset, 0)
            for i in S.range(4):
                a_sh_1_packed[a_row, a_group, i] = a_vec[i]
        else:
            b_vindex = (odd_tile * BLOCK_K + b_k) * B_ROW_RANGE_BYTES
            b_soffset = (n_base + b_col_group * 8) * 2
            b_vec = S.amdgpu.raw_buffer_load_x4(b_rsrc, b_vindex, b_soffset, 0)
            b_frag = S.view(b_vec, S.Tensor((2, 4, 1), S.bf16))
            for i in S.range(4):
                b_sh_1_bf16[b_col_group * 8 + i, b_k] = b_frag[0, i, 0]
                b_sh_1_bf16[b_col_group * 8 + 4 + i, b_k] = b_frag[1, i, 0]

        S.syncthreads()

        a_even_vec = a_sh_0_packed[wave_m * 32 + lane_col, k_group]
        b_even_vec = b_sh_0_packed[wave_n * 32 + lane_col, k_group]
        a_even_frag = S.view(a_even_vec, S.Tensor((2, 4, 1), S.bf16))
        b_even_frag = S.view(b_even_vec, S.Tensor((2, 4, 1), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_even_frag[0], b_even_frag[0], acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_even_frag[1], b_even_frag[1], acc)

        next_even_tile = even_tile + 2
        if tid < 128:
            a_vindex = (m_base + a_row) * A_ROW_RANGE_BYTES
            a_soffset = (next_even_tile * BLOCK_K + a_group * 8) * 2
            a_vec = S.amdgpu.raw_buffer_load_x4(a_rsrc, a_vindex, a_soffset, 0)
            for i in S.range(4):
                a_sh_0_packed[a_row, a_group, i] = a_vec[i]
        else:
            b_vindex = (next_even_tile * BLOCK_K + b_k) * B_ROW_RANGE_BYTES
            b_soffset = (n_base + b_col_group * 8) * 2
            b_vec = S.amdgpu.raw_buffer_load_x4(b_rsrc, b_vindex, b_soffset, 0)
            b_frag = S.view(b_vec, S.Tensor((2, 4, 1), S.bf16))
            for i in S.range(4):
                b_sh_0_bf16[b_col_group * 8 + i, b_k] = b_frag[0, i, 0]
                b_sh_0_bf16[b_col_group * 8 + 4 + i, b_k] = b_frag[1, i, 0]

        S.syncthreads()

        a_odd_vec = a_sh_1_packed[wave_m * 32 + lane_col, k_group]
        b_odd_vec = b_sh_1_packed[wave_n * 32 + lane_col, k_group]
        a_odd_frag = S.view(a_odd_vec, S.Tensor((2, 4, 1), S.bf16))
        b_odd_frag = S.view(b_odd_vec, S.Tensor((2, 4, 1), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_odd_frag[0], b_odd_frag[0], acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_odd_frag[1], b_odd_frag[1], acc)

        S.syncthreads()

    c_col = n_base + wave_n * 32 + lane_col
    c_row_base = m_base + wave_m * 32 + lane_row * 4
    for reg in S.range(16):
        c_row = c_row_base + (reg % 4) + 8 * (reg // 4)
        C[c_row, c_col] = S.convert(acc[reg], S.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if tuple(A.shape) != (M, K) or tuple(B.shape) != (K, N):
            raise NotImplementedError("ModelNew only supports 4096x4096 BF16 GEMM.")
        if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
            raise NotImplementedError("ModelNew expects BF16 inputs.")
        if not A.is_cuda or not B.is_cuda:
            raise NotImplementedError("ModelNew expects CUDA/HIP tensors.")

        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((M, N), device=A.device, dtype=torch.bfloat16)
        gemm_kernel[lambda: ((N // BLOCK_N, M // BLOCK_M, 1), (THREADS, 1, 1))](
            A,
            B,
            C,
            num_warps=WAVES,
        )
        return C
