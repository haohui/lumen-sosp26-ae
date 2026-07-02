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
WARP_M = 32
WARP_N = 32
THREADS = 256
A_CHUNKS = (BLOCK_M * BLOCK_K) // 8
B_CHUNKS = (BLOCK_K * BLOCK_N) // 8
RANGE_BYTES = M * K * 2


@substrate.jit
def gemm_kernel(
    A: S.Tensor((M, K), S.bf16),
    B: S.Tensor((N, K), S.bf16),
    C: S.Tensor((M, N), S.bf16),
    range_bytes: S.i32,
):
    tid = S.thread_id(0)
    lane = tid % 64
    warp = tid // 64

    block_row = S.block_id(1) * BLOCK_M
    block_col = S.block_id(0) * BLOCK_N
    warp_row = warp // 2
    warp_col = warp % 2

    a_rsrc = S.amdgpu.make_rsrc(A, range_bytes)
    b_rsrc = S.amdgpu.make_rsrc(B, range_bytes)
    zero = S.convert(0, S.i32)

    shared_words = S.make_shared((A_CHUNKS + B_CHUNKS, 4), S.u32)
    a_words = S.subview(shared_words, (0, 0), (A_CHUNKS, 4), (1, 1))
    b_words = S.subview(shared_words, (A_CHUNKS, 0), (B_CHUNKS, 4), (1, 1))

    acc = S.full((16,), 0.0, S.f32)

    for ko in S.range(K // BLOCK_K):
        if tid < A_CHUNKS:
            a_chunk = tid
            a_row = a_chunk // 2
            a_col_group = a_chunk % 2
            a_offset = ((block_row + a_row) * K + ko * BLOCK_K + a_col_group * 8) * 2
            a_pack = S.amdgpu.raw_buffer_load_x4(
                a_rsrc,
                zero,
                S.convert(a_offset, S.i32),
                0,
            )
            for i in S.range(4):
                a_words[a_chunk, i] = a_pack[i]

        if tid >= A_CHUNKS:
            b_chunk = tid - A_CHUNKS
            b_row = b_chunk // 2
            b_col_group = b_chunk % 2
            b_offset = ((block_col + b_row) * K + ko * BLOCK_K + b_col_group * 8) * 2
            b_pack = S.amdgpu.raw_buffer_load_x4(
                b_rsrc,
                zero,
                S.convert(b_offset, S.i32),
                0,
            )
            for i in S.range(4):
                b_words[b_chunk, i] = b_pack[i]

        S.syncthreads()

        a_chunk_idx = warp_row * 64 + (lane % 32) * 2 + (lane // 32)
        b_chunk_idx = warp_col * 64 + (lane % 32) * 2 + (lane // 32)

        a_frag = S.view(a_words[a_chunk_idx], S.Tensor((2, 4, 1), S.bf16))
        b_frag = S.view(b_words[b_chunk_idx], S.Tensor((2, 4, 1), S.bf16))

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[1], b_frag[1], acc)

        S.syncthreads()

    tile_row_base = block_row + warp_row * WARP_M
    tile_col_base = block_col + warp_col * WARP_N
    out_col = tile_col_base + (lane % 32)
    lane_row_base = tile_row_base + 4 * (lane // 32)

    for acc_idx in S.range(16):
        out_row = lane_row_base + 8 * (acc_idx // 4) + (acc_idx % 4)
        C[out_row, out_col] = S.convert(acc[acc_idx], S.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.range_bytes = RANGE_BYTES

    def forward(self, A, B):
        if tuple(A.shape) != (M, K) or tuple(B.shape) != (K, N):
            raise ValueError("ModelNew only supports 4096x4096 bf16 inputs")
        if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
            raise ValueError("ModelNew requires bfloat16 inputs")

        A = A.contiguous()
        B = B.transpose(0, 1).contiguous()
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)
        gemm_kernel[lambda: ((N // BLOCK_N, M // BLOCK_M, 1), (THREADS, 1, 1))](
            A, B, C, self.range_bytes
        )
        return C
