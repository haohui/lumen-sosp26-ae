import torch
import torch.nn as nn

import substrate
import substrate.language as S


@substrate.jit
def einsum4d_mfma_kernel(
    A: S.Tensor((8, 256, 512, 256), S.bf16),
    B: S.Tensor((256, 768), S.bf16),
    C: S.Tensor((8, 256, 512, 768), S.bf16),
):
    tid = S.thread_id(0)
    lane = tid % 64
    wave_id = tid // 64
    wr = wave_id // 2
    wc = wave_id % 2

    block_x = S.block_id(0)

    k_tile = block_x % 12
    remainder = block_x // 12
    j_tile = remainder % 8
    remainder = remainder // 8
    i_idx = remainder % 256
    b_idx = remainder // 256

    j_base = j_tile * 64
    n_base = k_tile * 64
    warp_row = j_base + wr * 32
    warp_col = n_base + wc * 32

    acc = S.full((16,), 0.0, S.f32)

    # Double-buffered LDS
    A_s0 = S.make_shared((64, 16), S.bf16)
    B_s0 = S.make_shared((64, 16), S.bf16)
    A_s1 = S.make_shared((64, 16), S.bf16)
    B_s1 = S.make_shared((64, 16), S.bf16)

    A_s0_v = S.view(A_s0, S.Tensor((64, 4, 2), S.u32))
    B_s0_v = S.view(B_s0, S.Tensor((64, 4, 2), S.u32))
    A_s1_v = S.view(A_s1, S.Tensor((64, 4, 2), S.u32))
    B_s1_v = S.view(B_s1, S.Tensor((64, 4, 2), S.u32))

    a_row_idx = wr * 32 + (lane % 32)
    b_col_idx = wc * 32 + (lane % 32)
    kg = lane // 32

    # Resource descriptors with range for OOB-safe raw buffer loads
    A_range = S.convert(8 * 256 * 512 * 256 * 2, S.u32)
    B_range = S.convert(256 * 768 * 2, S.u32)
    A_rsrc = S.amdgpu.make_rsrc(A, A_range)
    B_rsrc = S.amdgpu.make_rsrc(B, B_range)

    # A byte strides for (8, 256, 512, 256) bf16 row-major
    a_stride_b = S.convert(256 * 512 * 256 * 2, S.u32)
    a_stride_i = S.convert(512 * 256 * 2, S.u32)
    a_stride_j = S.convert(256 * 2, S.u32)
    a_stride_k = S.convert(2, S.u32)

    # B byte strides for (256, 768) bf16 row-major
    b_stride_k = S.convert(768 * 2, S.u32)
    b_stride_n = S.convert(2, S.u32)

    # Pre-compute A base offset (common across all k_steps for this block)
    a_base = (S.convert(b_idx, S.u32) * a_stride_b +
              S.convert(i_idx, S.u32) * a_stride_i +
              S.convert(j_base, S.u32) * a_stride_j)

    # --- Prologue: load k_step 0 into buf0 ---
    k_off_0 = S.convert(0, S.u32)
    if tid < 128:
        chunk = tid
        row = chunk // 2
        col_group = chunk % 2
        offset = a_base + S.convert(row, S.u32) * a_stride_j + (k_off_0 + S.convert(col_group * 8, S.u32)) * a_stride_k
        val = S.amdgpu.raw_buffer_load_x4(A_rsrc, offset, 0, 0)
        val_bf16 = S.view(val, S.Tensor((8,), S.bf16))
        for v in S.range(8):
            A_s0[row, col_group * 8 + v] = val_bf16[v]
    if tid >= 128:
        chunk = tid - 128
        brow = chunk // 8
        col_chunk = chunk % 8
        n_col = col_chunk * 8
        offset = (k_off_0 + S.convert(brow, S.u32)) * b_stride_k + S.convert(n_base + n_col, S.u32) * b_stride_n
        val = S.amdgpu.raw_buffer_load_x4(B_rsrc, offset, 0, 0)
        val_bf16 = S.view(val, S.Tensor((8,), S.bf16))
        for v in S.range(8):
            B_s0[n_col + v, brow] = val_bf16[v]
    S.syncthreads()

    # --- Pipelined main loop: 7 iterations, each processes 2 k_steps ---
    for iter in S.range(7):
        # Compute MFMA on buf0 (k_step = iter * 2)
        a_vec1 = A_s0_v[a_row_idx, kg]
        b_vec1 = B_s0_v[b_col_idx, kg]
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(
            S.view(a_vec1, S.Tensor((1, 4, 1), S.bf16))[0],
            S.view(b_vec1, S.Tensor((1, 4, 1), S.bf16))[0],
            acc,
        )
        a_vec2 = A_s0_v[a_row_idx, kg + 2]
        b_vec2 = B_s0_v[b_col_idx, kg + 2]
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(
            S.view(a_vec2, S.Tensor((1, 4, 1), S.bf16))[0],
            S.view(b_vec2, S.Tensor((1, 4, 1), S.bf16))[0],
            acc,
        )

        # Load k_step = 2*iter+1 into buf1
        k_start_b = S.convert((iter * 2 + 1) * 16, S.u32)
        if tid < 128:
            chunk = tid
            row = chunk // 2
            col_group = chunk % 2
            offset = a_base + S.convert(row, S.u32) * a_stride_j + (k_start_b + S.convert(col_group * 8, S.u32)) * a_stride_k
            val = S.amdgpu.raw_buffer_load_x4(A_rsrc, offset, 0, 0)
            val_bf16 = S.view(val, S.Tensor((8,), S.bf16))
            for v in S.range(8):
                A_s1[row, col_group * 8 + v] = val_bf16[v]
        if tid >= 128:
            chunk = tid - 128
            brow = chunk // 8
            col_chunk = chunk % 8
            n_col = col_chunk * 8
            offset = (k_start_b + S.convert(brow, S.u32)) * b_stride_k + S.convert(n_base + n_col, S.u32) * b_stride_n
            val = S.amdgpu.raw_buffer_load_x4(B_rsrc, offset, 0, 0)
            val_bf16 = S.view(val, S.Tensor((8,), S.bf16))
            for v in S.range(8):
                B_s1[n_col + v, brow] = val_bf16[v]
        S.syncthreads()

        # Compute MFMA on buf1 (k_step = 2*iter+1)
        a_vec1 = A_s1_v[a_row_idx, kg]
        b_vec1 = B_s1_v[b_col_idx, kg]
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(
            S.view(a_vec1, S.Tensor((1, 4, 1), S.bf16))[0],
            S.view(b_vec1, S.Tensor((1, 4, 1), S.bf16))[0],
            acc,
        )
        a_vec2 = A_s1_v[a_row_idx, kg + 2]
        b_vec2 = B_s1_v[b_col_idx, kg + 2]
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(
            S.view(a_vec2, S.Tensor((1, 4, 1), S.bf16))[0],
            S.view(b_vec2, S.Tensor((1, 4, 1), S.bf16))[0],
            acc,
        )

        # Load k_step = 2*(iter+1) into buf0 for next iteration
        k_start_a_next = S.convert((iter + 1) * 2 * 16, S.u32)
        if tid < 128:
            chunk = tid
            row = chunk // 2
            col_group = chunk % 2
            offset = a_base + S.convert(row, S.u32) * a_stride_j + (k_start_a_next + S.convert(col_group * 8, S.u32)) * a_stride_k
            val = S.amdgpu.raw_buffer_load_x4(A_rsrc, offset, 0, 0)
            val_bf16 = S.view(val, S.Tensor((8,), S.bf16))
            for v in S.range(8):
                A_s0[row, col_group * 8 + v] = val_bf16[v]
        if tid >= 128:
            chunk = tid - 128
            brow = chunk // 8
            col_chunk = chunk % 8
            n_col = col_chunk * 8
            offset = (k_start_a_next + S.convert(brow, S.u32)) * b_stride_k + S.convert(n_base + n_col, S.u32) * b_stride_n
            val = S.amdgpu.raw_buffer_load_x4(B_rsrc, offset, 0, 0)
            val_bf16 = S.view(val, S.Tensor((8,), S.bf16))
            for v in S.range(8):
                B_s0[n_col + v, brow] = val_bf16[v]
        S.syncthreads()

    # --- Epilogue: compute k_step 14 on buf0 ---
    a_vec1 = A_s0_v[a_row_idx, kg]
    b_vec1 = B_s0_v[b_col_idx, kg]
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(
        S.view(a_vec1, S.Tensor((1, 4, 1), S.bf16))[0],
        S.view(b_vec1, S.Tensor((1, 4, 1), S.bf16))[0],
        acc,
    )
    a_vec2 = A_s0_v[a_row_idx, kg + 2]
    b_vec2 = B_s0_v[b_col_idx, kg + 2]
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(
        S.view(a_vec2, S.Tensor((1, 4, 1), S.bf16))[0],
        S.view(b_vec2, S.Tensor((1, 4, 1), S.bf16))[0],
        acc,
    )

    # Load k_step 15 into buf1
    k_start_15 = S.convert(15 * 16, S.u32)
    if tid < 128:
        chunk = tid
        row = chunk // 2
        col_group = chunk % 2
        offset = a_base + S.convert(row, S.u32) * a_stride_j + (k_start_15 + S.convert(col_group * 8, S.u32)) * a_stride_k
        val = S.amdgpu.raw_buffer_load_x4(A_rsrc, offset, 0, 0)
        val_bf16 = S.view(val, S.Tensor((8,), S.bf16))
        for v in S.range(8):
            A_s1[row, col_group * 8 + v] = val_bf16[v]
    if tid >= 128:
        chunk = tid - 128
        brow = chunk // 8
        col_chunk = chunk % 8
        n_col = col_chunk * 8
        offset = (k_start_15 + S.convert(brow, S.u32)) * b_stride_k + S.convert(n_base + n_col, S.u32) * b_stride_n
        val = S.amdgpu.raw_buffer_load_x4(B_rsrc, offset, 0, 0)
        val_bf16 = S.view(val, S.Tensor((8,), S.bf16))
        for v in S.range(8):
            B_s1[n_col + v, brow] = val_bf16[v]
    S.syncthreads()

    # Compute k_step 15 on buf1
    a_vec1 = A_s1_v[a_row_idx, kg]
    b_vec1 = B_s1_v[b_col_idx, kg]
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(
        S.view(a_vec1, S.Tensor((1, 4, 1), S.bf16))[0],
        S.view(b_vec1, S.Tensor((1, 4, 1), S.bf16))[0],
        acc,
    )
    a_vec2 = A_s1_v[a_row_idx, kg + 2]
    b_vec2 = B_s1_v[b_col_idx, kg + 2]
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(
        S.view(a_vec2, S.Tensor((1, 4, 1), S.bf16))[0],
        S.view(b_vec2, S.Tensor((1, 4, 1), S.bf16))[0],
        acc,
    )

    # Write results
    for acc_idx in S.range(16):
        out_col = warp_col + (lane % 32)
        out_row = warp_row + 8 * (acc_idx // 4) + 4 * (lane // 32) + (acc_idx % 4)
        C[b_idx, i_idx, out_row, out_col] = S.convert(acc[acc_idx], S.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if tuple(A.shape) != (8, 256, 512, 256) or tuple(B.shape) != (256, 768):
            return torch.einsum("bijl,lk->bijk", A, B)
        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((8, 256, 512, 768), device=A.device, dtype=A.dtype)
        num_blocks = 8 * 256 * 8 * 12
        einsum4d_mfma_kernel[lambda: ((num_blocks, 1, 1), (256, 1, 1))](A, B, C)
        return C
