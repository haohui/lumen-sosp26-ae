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
WAVES_M = 2
WAVES_N = 2
THREADS = WAVE_SIZE * WAVES_M * WAVES_N

BYTE_SIZE_BF16 = 2
TOTAL_A_BYTES = M * K * BYTE_SIZE_BF16
TOTAL_B_BYTES = K * N * BYTE_SIZE_BF16


@substrate.jit
def gemm_kernel(
    A: S.Tensor((M, K), S.bf16),
    B: S.Tensor((K, N), S.bf16),
    C: S.Tensor((M, N), S.bf16),
):
    tid = S.thread_id(0)
    lane = tid % WAVE_SIZE
    warp = tid // WAVE_SIZE
    warp_row = warp // WAVES_N
    warp_col = warp % WAVES_N

    block_row = S.block_id(1) * BLOCK_M
    block_col = S.block_id(0) * BLOCK_N

    # Double buffered LDS for software pipelining
    a_tile0 = S.make_shared((BLOCK_M, BLOCK_K), S.bf16)
    a_tile1 = S.make_shared((BLOCK_M, BLOCK_K), S.bf16)
    b_tile0 = S.make_shared((BLOCK_K, BLOCK_N), S.bf16)
    b_tile1 = S.make_shared((BLOCK_K, BLOCK_N), S.bf16)

    acc = S.full((16,), 0.0, S.f32)

    # range parameter in make_rsrc handles OOB access automatically
    a_rsrc = S.amdgpu.make_rsrc(A, TOTAL_A_BYTES)
    b_rsrc = S.amdgpu.make_rsrc(B, TOTAL_B_BYTES)

    # Prolog: load first chunk to buffer 0
    k_base = 0
    if tid < 128:
        load_id = tid
        row = load_id // (BLOCK_K // 8)
        chunk = load_id % (BLOCK_K // 8)
        byte_offset = ((block_row + row) * K + k_base + chunk * 8) * BYTE_SIZE_BF16
        vec = S.amdgpu.raw_buffer_load_x4(a_rsrc, 0, byte_offset, 0)
        vals = S.view(vec, S.Tensor((8,), S.bf16))
        for t in S.range(8):
            a_tile0[row, chunk * 8 + t] = vals[t]
    else:
        load_id = tid - 128
        row = load_id // (BLOCK_N // 8)
        chunk = load_id % (BLOCK_N // 8)
        byte_offset = ((k_base + row) * N + block_col + chunk * 8) * BYTE_SIZE_BF16
        vec = S.amdgpu.raw_buffer_load_x4(b_rsrc, 0, byte_offset, 0)
        vals = S.view(vec, S.Tensor((8,), S.bf16))
        for t in S.range(8):
            b_tile0[row, chunk * 8 + t] = vals[t]

    S.syncthreads()

    # Main loop with double buffering
    # K-loop unrolled by 2: process 2 chunks per iteration
    for k_idx in S.range(0, K // BLOCK_K, 2):
        a_frag = S.make_local((2, 4), S.bf16)
        b_frag = S.make_local((2, 4), S.bf16)

        a_row = warp_row * 32 + (lane % 32)
        a_col_group = lane // 32

        b_col = warp_col * 32 + (lane % 32)
        b_k_group = lane // 32

        # Compute on buffer determined by iteration
        # k_idx even: buffer 0, k_idx odd: buffer 1
        if k_idx % 2 == 0:
            # Compute on buffer 0
            for t in S.range(4):
                a_frag[0, t] = a_tile0[a_row, 4 * a_col_group + t]
                a_frag[1, t] = a_tile0[a_row, 8 + 4 * a_col_group + t]
                b_frag[0, t] = b_tile0[4 * b_k_group + t, b_col]
                b_frag[1, t] = b_tile0[8 + 4 * b_k_group + t, b_col]
            acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], acc)
            acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[1], b_frag[1], acc)

            # Load second chunk of pair to buffer 1
            k_base = (k_idx + 1) * BLOCK_K
            if tid < 128:
                load_id = tid
                row = load_id // (BLOCK_K // 8)
                chunk = load_id % (BLOCK_K // 8)
                byte_offset = ((block_row + row) * K + k_base + chunk * 8) * BYTE_SIZE_BF16
                vec = S.amdgpu.raw_buffer_load_x4(a_rsrc, 0, byte_offset, 0)
                vals = S.view(vec, S.Tensor((8,), S.bf16))
                for t in S.range(8):
                    a_tile1[row, chunk * 8 + t] = vals[t]
            else:
                load_id = tid - 128
                row = load_id // (BLOCK_N // 8)
                chunk = load_id % (BLOCK_N // 8)
                byte_offset = ((k_base + row) * N + block_col + chunk * 8) * BYTE_SIZE_BF16
                vec = S.amdgpu.raw_buffer_load_x4(b_rsrc, 0, byte_offset, 0)
                vals = S.view(vec, S.Tensor((8,), S.bf16))
                for t in S.range(8):
                    b_tile1[row, chunk * 8 + t] = vals[t]

            S.syncthreads()

            # Compute on buffer 1
            for t in S.range(4):
                a_frag[0, t] = a_tile1[a_row, 4 * a_col_group + t]
                a_frag[1, t] = a_tile1[a_row, 8 + 4 * a_col_group + t]
                b_frag[0, t] = b_tile1[4 * b_k_group + t, b_col]
                b_frag[1, t] = b_tile1[8 + 4 * b_k_group + t, b_col]
            acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], acc)
            acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[1], b_frag[1], acc)

            # Load next pair to buffer 0
            # OOB access handled by range in make_rsrc - no branch needed
            k_base = (k_idx + 2) * BLOCK_K
            if tid < 128:
                load_id = tid
                row = load_id // (BLOCK_K // 8)
                chunk = load_id % (BLOCK_K // 8)
                byte_offset = ((block_row + row) * K + k_base + chunk * 8) * BYTE_SIZE_BF16
                vec = S.amdgpu.raw_buffer_load_x4(a_rsrc, 0, byte_offset, 0)
                vals = S.view(vec, S.Tensor((8,), S.bf16))
                for t in S.range(8):
                    a_tile0[row, chunk * 8 + t] = vals[t]
            else:
                load_id = tid - 128
                row = load_id // (BLOCK_N // 8)
                chunk = load_id % (BLOCK_N // 8)
                byte_offset = ((k_base + row) * N + block_col + chunk * 8) * BYTE_SIZE_BF16
                vec = S.amdgpu.raw_buffer_load_x4(b_rsrc, 0, byte_offset, 0)
                vals = S.view(vec, S.Tensor((8,), S.bf16))
                for t in S.range(8):
                    b_tile0[row, chunk * 8 + t] = vals[t]
        else:
            # Compute on buffer 1
            for t in S.range(4):
                a_frag[0, t] = a_tile1[a_row, 4 * a_col_group + t]
                a_frag[1, t] = a_tile1[a_row, 8 + 4 * a_col_group + t]
                b_frag[0, t] = b_tile1[4 * b_k_group + t, b_col]
                b_frag[1, t] = b_tile1[8 + 4 * b_k_group + t, b_col]
            acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], acc)
            acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[1], b_frag[1], acc)

            # Load second chunk of pair to buffer 0
            k_base = (k_idx + 1) * BLOCK_K
            if tid < 128:
                load_id = tid
                row = load_id // (BLOCK_K // 8)
                chunk = load_id % (BLOCK_K // 8)
                byte_offset = ((block_row + row) * K + k_base + chunk * 8) * BYTE_SIZE_BF16
                vec = S.amdgpu.raw_buffer_load_x4(a_rsrc, 0, byte_offset, 0)
                vals = S.view(vec, S.Tensor((8,), S.bf16))
                for t in S.range(8):
                    a_tile0[row, chunk * 8 + t] = vals[t]
            else:
                load_id = tid - 128
                row = load_id // (BLOCK_N // 8)
                chunk = load_id % (BLOCK_N // 8)
                byte_offset = ((k_base + row) * N + block_col + chunk * 8) * BYTE_SIZE_BF16
                vec = S.amdgpu.raw_buffer_load_x4(b_rsrc, 0, byte_offset, 0)
                vals = S.view(vec, S.Tensor((8,), S.bf16))
                for t in S.range(8):
                    b_tile0[row, chunk * 8 + t] = vals[t]

            S.syncthreads()

            # Compute on buffer 0
            for t in S.range(4):
                a_frag[0, t] = a_tile0[a_row, 4 * a_col_group + t]
                a_frag[1, t] = a_tile0[a_row, 8 + 4 * a_col_group + t]
                b_frag[0, t] = b_tile0[4 * b_k_group + t, b_col]
                b_frag[1, t] = b_tile0[8 + 4 * b_k_group + t, b_col]
            acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], acc)
            acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[1], b_frag[1], acc)

            # Load next pair to buffer 1
            # OOB access handled by range in make_rsrc - no branch needed
            k_base = (k_idx + 2) * BLOCK_K
            if tid < 128:
                load_id = tid
                row = load_id // (BLOCK_K // 8)
                chunk = load_id % (BLOCK_K // 8)
                byte_offset = ((block_row + row) * K + k_base + chunk * 8) * BYTE_SIZE_BF16
                vec = S.amdgpu.raw_buffer_load_x4(a_rsrc, 0, byte_offset, 0)
                vals = S.view(vec, S.Tensor((8,), S.bf16))
                for t in S.range(8):
                    a_tile1[row, chunk * 8 + t] = vals[t]
            else:
                load_id = tid - 128
                row = load_id // (BLOCK_N // 8)
                chunk = load_id % (BLOCK_N // 8)
                byte_offset = ((k_base + row) * N + block_col + chunk * 8) * BYTE_SIZE_BF16
                vec = S.amdgpu.raw_buffer_load_x4(b_rsrc, 0, byte_offset, 0)
                vals = S.view(vec, S.Tensor((8,), S.bf16))
                for t in S.range(8):
                    b_tile1[row, chunk * 8 + t] = vals[t]

        S.syncthreads()

    # Write output with lower triangular mask
    # Grid perfectly tiles the matrix (M=N=4096, BLOCK_M=BLOCK_N=64), so no OOB for stores
    tile_row_base = block_row + warp_row * 32
    tile_col_base = block_col + warp_col * 32
    col = tile_col_base + (lane % 32)
    row_group = 4 * (lane // 32)
    for acc_idx in S.range(16):
        row = tile_row_base + 8 * (acc_idx // 4) + row_group + (acc_idx % 4)
        if col <= row:
            C[row, col] = S.convert(acc[acc_idx], S.bf16)
        else:
            C[row, col] = S.convert(0.0, S.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if tuple(A.shape) != (M, K) or tuple(B.shape) != (K, N):
            return torch.tril(torch.matmul(A, B))

        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)
        grid = (N // BLOCK_N, M // BLOCK_M, 1)
        block = (THREADS, 1, 1)
        gemm_kernel[lambda: (grid, block)](A, B, C, num_warps=4)
        return C
