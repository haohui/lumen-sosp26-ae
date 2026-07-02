import torch
import torch.nn as nn

import substrate
import substrate.language as S


M = 4096
N = 4096
K = 4096

BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 16
WAVE_SIZE = 64
WAVES_PER_BLOCK = 4
THREADS_PER_BLOCK = WAVE_SIZE * WAVES_PER_BLOCK
RANGE_BYTES = M * K * 2


@substrate.jit
def tri_gemm_mfma_kernel(
    A: S.Tensor((M, K), S.bf16),
    B: S.Tensor((K, N), S.bf16),
    C: S.Tensor((M, N), S.bf16),
):
    tid = S.thread_id(0)
    lane = tid % WAVE_SIZE
    warp = tid // WAVE_SIZE

    block_row = S.block_id(1) * BLOCK_M
    block_col = S.block_id(0) * BLOCK_N
    warp_row = warp // 2
    warp_col = warp % 2
    warp_row_base = warp_row * 32
    warp_col_base = warp_col * 32

    a_rsrc = S.amdgpu.make_rsrc(A, S.convert(RANGE_BYTES, S.i32))
    b_rsrc = S.amdgpu.make_rsrc(B, S.convert(RANGE_BYTES, S.i32))
    zero = S.convert(0, S.i32)

    shared_words = S.make_shared((1024,), S.u32)
    a_words = S.subview(shared_words, (0,), (512,), (1,))
    b_words = S.subview(shared_words, (512,), (512,), (1,))
    frag_layout = S.make_layout((128, 4), (4, 1))
    a_frags = S.view(a_words, S.u32, frag_layout)
    b_frags = S.view(b_words, S.u32, frag_layout)

    c_lane = S.full((16,), 0.0, S.f32)

    for k_tile in S.range(K // BLOCK_K):
        k0 = k_tile * BLOCK_K

        if tid < 128:
            a_frag = tid
            a_row = a_frag // 2
            a_half = a_frag % 2
            a_byte_offset = S.convert(((block_row + a_row) * K + k0 + a_half * 8) * 2, S.i32)
            a_pack = S.amdgpu.raw_buffer_load_x4(a_rsrc, zero, a_byte_offset, 0)
            a_base = a_row * 8 + a_half * 4
            for i in S.range(4):
                a_words[a_base + i] = a_pack[i]
        else:
            b_frag = tid - 128
            b_row = b_frag // 8
            b_seg = b_frag % 8
            b_byte_offset = S.convert(((k0 + b_row) * N + block_col + b_seg * 8) * 2, S.i32)
            b_pack = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero, b_byte_offset, 0)
            b_base = b_row * 32 + b_seg * 4
            for i in S.range(4):
                b_words[b_base + i] = b_pack[i]

        S.syncthreads()

        a_frag_idx = (
            warp_row * 64
            + (lane % 8) * 2
            + (lane // 32)
            + ((lane // 16) % 2) * 32
        )
        b_row = (lane % 8) + (lane // 32) * 8
        b_seg = warp_col * 4 + ((lane // 8) % 4)
        b_frag_idx = b_row * 8 + b_seg

        m_a = S.view(a_frags[a_frag_idx], S.Tensor((2, 4, 1), S.bf16))
        m_b = S.view(b_frags[b_frag_idx], S.Tensor((2, 4, 1), S.bf16))

        c_lane = S.amdgpu.mfma_32x32x8_bf16_f32(m_a[0], m_b[0], c_lane)
        c_lane = S.amdgpu.mfma_32x32x8_bf16_f32(m_a[1], m_b[1], c_lane)

        S.syncthreads()

    for i in S.range(16):
        row_local = (lane % 8) * 2 + ((i // 4) % 2) + (i // 8) * 16
        col_local = (lane // 8) * 4 + (i % 4)
        global_row = block_row + warp_row_base + row_local
        global_col = block_col + warp_col_base + col_local
        if global_col <= global_row:
            C[global_row, global_col] = S.convert(c_lane[i], S.bf16)
        else:
            C[global_row, global_col] = S.convert(0.0, S.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if tuple(A.shape) != (M, K) or tuple(B.shape) != (K, N):
            raise RuntimeError("ModelNew expects A and B with shape (4096, 4096)")
        if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
            raise RuntimeError("ModelNew expects bfloat16 inputs")

        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((M, N), device=A.device, dtype=torch.bfloat16)
        tri_gemm_mfma_kernel[lambda: ((N // BLOCK_N, M // BLOCK_M, 1), (THREADS_PER_BLOCK, 1, 1))](
            A, B, C
        )
        return C
