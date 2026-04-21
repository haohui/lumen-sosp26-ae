import torch
import torch.nn as nn

import substrate
import substrate.language as S


M = 32768
K = 32
N = 32768
BLOCK_M = 64
BLOCK_N = 64
WAVE_SIZE = 64
WARPS_PER_BLOCK = 4
THREADS_PER_BLOCK = WAVE_SIZE * WARPS_PER_BLOCK
PIPE_STAGES = 2
K_STAGE = 16


@substrate.jit
def gemm_kernel(
    A: S.Tensor((M, K), S.bf16),
    B: S.Tensor((K, N), S.bf16),
    C: S.Tensor((M, N), S.bf16),
):
    tid = S.thread_id(0)
    lane = tid % WAVE_SIZE
    warp = tid // WAVE_SIZE
    warp_m = warp // 2
    warp_n = warp % 2

    block_m = S.block_id(1)
    block_n = S.block_id(0)

    a_words = S.make_shared((PIPE_STAGES, BLOCK_M, K_STAGE // 8, 4), S.u32)
    b_words = S.make_shared((PIPE_STAGES, K_STAGE, BLOCK_N // 8, 4), S.u32)

    load_id = tid % 128
    load_a = tid < 128

    a_stage_row = load_id // (K_STAGE // 8)
    a_stage_chunk = load_id % (K_STAGE // 8)
    a_global_row = block_m * BLOCK_M + a_stage_row

    b_stage_row = load_id // (BLOCK_N // 8)
    b_stage_chunk = load_id % (BLOCK_N // 8)
    b_global_col = block_n * BLOCK_N + b_stage_chunk * 8

    if load_a:
        a_row_rsrc = S.amdgpu.make_rsrc(A[a_global_row], K * 2)
        a_vec0 = S.amdgpu.raw_buffer_load_x4(a_row_rsrc, a_stage_chunk * 16, 0, 0)
        a_words[0, a_stage_row, a_stage_chunk] = a_vec0
    else:
        b_row_rsrc0 = S.amdgpu.make_rsrc(B[b_stage_row], N * 2)
        b_vec0 = S.amdgpu.raw_buffer_load_x4(b_row_rsrc0, b_global_col * 2, 0, 0)
        b_words[0, b_stage_row, b_stage_chunk] = b_vec0

    S.syncthreads()

    acc = S.full((16,), 0.0, S.f32)

    a_row = warp_m * 32 + lane // 2
    a_chunk = lane % 2
    b_row = lane // 4
    b_chunk = warp_n * 4 + (lane % 4)

    if load_a:
        a_vec1 = S.amdgpu.raw_buffer_load_x4(a_row_rsrc, (K_STAGE + a_stage_chunk * 8) * 2, 0, 0)
    else:
        b_row_rsrc1 = S.amdgpu.make_rsrc(B[K_STAGE + b_stage_row], N * 2)
        b_vec1 = S.amdgpu.raw_buffer_load_x4(b_row_rsrc1, b_global_col * 2, 0, 0)

    a_frag0 = S.view(a_words[0, a_row, a_chunk], S.Tensor((2, 4, 1), S.bf16))
    b_frag0 = S.view(b_words[0, b_row, b_chunk], S.Tensor((2, 4, 1), S.bf16))

    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[0], b_frag0[0], acc)

    if load_a:
        a_words[1, a_stage_row, a_stage_chunk] = a_vec1
    else:
        b_words[1, b_stage_row, b_stage_chunk] = b_vec1

    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[1], b_frag0[1], acc)

    S.syncthreads()

    a_frag1 = S.view(a_words[1, a_row, a_chunk], S.Tensor((2, 4, 1), S.bf16))
    b_frag1 = S.view(b_words[1, b_row, b_chunk], S.Tensor((2, 4, 1), S.bf16))

    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[0], b_frag1[0], acc)
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[1], b_frag1[1], acc)

    c_row = block_m * BLOCK_M + warp_m * 32 + (lane // 8) * 4
    c_col = block_n * BLOCK_N + warp_n * 32 + (lane % 8) * 4

    c_tile = S.make_local((4, 4), S.bf16)
    c_tile[0, 0] = S.convert(acc[0], S.bf16)
    c_tile[0, 1] = S.convert(acc[1], S.bf16)
    c_tile[0, 2] = S.convert(acc[2], S.bf16)
    c_tile[0, 3] = S.convert(acc[3], S.bf16)
    c_tile[1, 0] = S.convert(acc[4], S.bf16)
    c_tile[1, 1] = S.convert(acc[5], S.bf16)
    c_tile[1, 2] = S.convert(acc[6], S.bf16)
    c_tile[1, 3] = S.convert(acc[7], S.bf16)
    c_tile[2, 0] = S.convert(acc[8], S.bf16)
    c_tile[2, 1] = S.convert(acc[9], S.bf16)
    c_tile[2, 2] = S.convert(acc[10], S.bf16)
    c_tile[2, 3] = S.convert(acc[11], S.bf16)
    c_tile[3, 0] = S.convert(acc[12], S.bf16)
    c_tile[3, 1] = S.convert(acc[13], S.bf16)
    c_tile[3, 2] = S.convert(acc[14], S.bf16)
    c_tile[3, 3] = S.convert(acc[15], S.bf16)

    c_words = S.view(c_tile, S.Tensor((4, 2), S.u32))

    c_row_rsrc0 = S.amdgpu.make_rsrc(C[c_row + 0], N * 2)
    c_row_rsrc1 = S.amdgpu.make_rsrc(C[c_row + 1], N * 2)
    c_row_rsrc2 = S.amdgpu.make_rsrc(C[c_row + 2], N * 2)
    c_row_rsrc3 = S.amdgpu.make_rsrc(C[c_row + 3], N * 2)

    S.amdgpu.raw_buffer_store_x2(c_words[0], c_row_rsrc0, c_col * 2, 0, 0)
    S.amdgpu.raw_buffer_store_x2(c_words[1], c_row_rsrc1, c_col * 2, 0, 0)
    S.amdgpu.raw_buffer_store_x2(c_words[2], c_row_rsrc2, c_col * 2, 0, 0)
    S.amdgpu.raw_buffer_store_x2(c_words[3], c_row_rsrc3, c_col * 2, 0, 0)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if tuple(A.shape) != (M, K) or tuple(B.shape) != (K, N):
            raise ValueError(f"Expected A {(M, K)} and B {(K, N)}, got {tuple(A.shape)} and {tuple(B.shape)}")
        if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
            raise ValueError(f"Expected bfloat16 inputs, got {A.dtype} and {B.dtype}")

        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)
        gemm_kernel[lambda: ((N // BLOCK_N, M // BLOCK_M, 1), (THREADS_PER_BLOCK, 1, 1))](A, B, C)
        return C
