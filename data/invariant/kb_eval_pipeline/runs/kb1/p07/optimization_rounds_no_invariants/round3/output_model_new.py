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
THREADS = 256
A_BYTES = M * K * 2
B_BYTES = K * N * 2
C_BYTES = M * N * 2


@substrate.jit
def gemm_kernel(
    A: S.Tensor((32768, 64), S.bf16),
    B: S.Tensor((64, 32768), S.bf16),
    C: S.Tensor((32768, 32768), S.bf16),
):
    tid = S.thread_id(0)
    lane = tid % 64
    wave = tid // 64
    wave_m = wave // 2
    wave_n = wave % 2

    block_m = S.block_id(1)
    block_n = S.block_id(0)
    tile_m = block_m * BLOCK_M
    tile_n = block_n * BLOCK_N

    a_rsrc = S.amdgpu.make_rsrc(A, A_BYTES)
    b_rsrc = S.amdgpu.make_rsrc(B, B_BYTES)
    c_rsrc = S.amdgpu.make_rsrc(C, C_BYTES)

    a_stage = S.make_shared((BLOCK_M, BLOCK_K), S.bf16)
    b_stage = S.make_shared((BLOCK_K, BLOCK_N), S.bf16)
    a_packed = S.make_shared((BLOCK_M, 2, 4), S.u32)
    b_packed = S.make_shared((BLOCK_N, 2, 4), S.u32)
    c_stage = S.make_shared((BLOCK_M, BLOCK_N), S.bf16)

    acc = S.full((16,), 0.0, S.f32)

    for k_tile in S.range(K // BLOCK_K):
        if tid < 128:
            a_load = tid
            row = a_load // 2
            chunk = a_load % 2
            a_offset = ((tile_m + row) * K + k_tile * BLOCK_K + chunk * 8) * 2
            vec = S.amdgpu.raw_buffer_load_x4(a_rsrc, a_offset, 0, 0)
            frag = S.view(vec, S.Tensor((8,), S.bf16))
            for i in S.range(8):
                a_stage[row, chunk * 8 + i] = frag[i]
        else:
            b_load = tid - 128
            k_row = b_load // 8
            col_chunk = b_load % 8
            b_offset = ((k_tile * BLOCK_K + k_row) * N + tile_n + col_chunk * 8) * 2
            vec = S.amdgpu.raw_buffer_load_x4(b_rsrc, b_offset, 0, 0)
            frag = S.view(vec, S.Tensor((8,), S.bf16))
            for i in S.range(8):
                b_stage[k_row, col_chunk * 8 + i] = frag[i]

        S.syncthreads()

        if tid < 128:
            a_pack = tid
            row = a_pack % BLOCK_M
            pack = a_pack // BLOCK_M
            frag = S.make_local((8,), S.bf16)
            for i in S.range(4):
                frag[i] = a_stage[row, pack * 4 + i]
                frag[4 + i] = a_stage[row, 8 + pack * 4 + i]
            a_packed[row, pack] = S.view(frag, S.Tensor((4,), S.u32))
        else:
            b_pack = tid - 128
            col = b_pack % BLOCK_N
            pack = b_pack // BLOCK_N
            frag = S.make_local((8,), S.bf16)
            for i in S.range(4):
                frag[i] = b_stage[pack * 4 + i, col]
                frag[4 + i] = b_stage[8 + pack * 4 + i, col]
            b_packed[col, pack] = S.view(frag, S.Tensor((4,), S.u32))

        S.syncthreads()

        a_row = wave_m * WAVE_M + (lane % WAVE_M)
        b_col = wave_n * WAVE_N + (lane % WAVE_N)
        half = lane // 32

        a_frag = S.view(a_packed[a_row, half], S.Tensor((2, 4, 1), S.bf16))
        b_frag = S.view(b_packed[b_col, half], S.Tensor((2, 4, 1), S.bf16))

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[1], b_frag[1], acc)

        S.syncthreads()

    col = wave_n * WAVE_N + (lane % WAVE_N)
    row_base = wave_m * WAVE_M + (lane // 32) * 4
    for slot in S.range(16):
        row = row_base + (slot // 4) * 8 + (slot % 4)
        c_stage[row, col] = S.convert(acc[slot], S.bf16)

    S.syncthreads()

    for store_iter in S.range(2):
        store_idx = tid + store_iter * THREADS
        row = store_idx // (BLOCK_N // 8)
        col_chunk = store_idx % (BLOCK_N // 8)
        frag = S.make_local((8,), S.bf16)
        for i in S.range(8):
            frag[i] = c_stage[row, col_chunk * 8 + i]
        packed = S.view(frag, S.Tensor((4,), S.u32))
        c_offset = ((tile_m + row) * N + tile_n + col_chunk * 8) * 2
        S.amdgpu.raw_buffer_store_x4(packed, c_rsrc, c_offset, 0, 0)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if (
            tuple(A.shape) != (M, K)
            or tuple(B.shape) != (K, N)
            or A.dtype != torch.bfloat16
            or B.dtype != torch.bfloat16
            or not A.is_cuda
            or not B.is_cuda
            or torch.version.hip is None
        ):
            return torch.matmul(A, B)

        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)
        gemm_kernel[lambda: ((N // BLOCK_N, M // BLOCK_M, 1), (THREADS, 1, 1))](
            A,
            B,
            C,
            num_warps=4,
        )
        return C
