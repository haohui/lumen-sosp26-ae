import torch
import torch.nn as nn

import avelang
import avelang.language as al


@avelang.jit
def bmm_kernel(
    A: al.Tensor((128, 512, 1024), al.bf16),
    B: al.Tensor((128, 1024, 2048), al.bf16),
    C: al.Tensor((128, 512, 2048), al.bf16),
):
    bx = al.block_id(0)
    by = al.block_id(1)
    bz = al.block_id(2)

    tid = al.thread_id(0)
    warp_id = tid // 64
    lane_id = tid % 64
    warp_y = warp_id // 2
    warp_x = warp_id % 2

    m_base = by * 64
    n_base = bz * 64

    # Fine-grain double-buffered shared memory: half-sized tiles (8 K per buffer)
    A_buf0 = al.make_shared((64, 8), al.bf16)
    A_buf1 = al.make_shared((64, 8), al.bf16)
    B_buf0 = al.make_shared((64, 8), al.bf16)
    B_buf1 = al.make_shared((64, 8), al.bf16)

    # Accumulator: 16 f32 values per thread (mfma_32x32x8_bf16_f32 output)
    acc_mem = al.make_local((16,), al.f32)
    for i in al.range(16):
        acc_mem[i] = al.convert(0.0, al.f32)

    # Prologue: preload first K-slice (K=0) into buf0
    if tid < 128:
        row = tid % 64
        col_grp = tid // 64
        a_row = m_base + row
        a_col = col_grp * 4
        for j in al.range(4):
            A_buf0[row, col_grp * 4 + j] = A[bx, a_row, a_col + j]

    if tid < 128:
        i = tid // 16
        j0 = (tid % 16) * 4
        b_row = i
        b_col = n_base + j0
        for j in al.range(4):
            B_buf0[j0 + j, i] = B[bx, b_row, b_col + j]

    al.syncthreads()

    # Main loop: K unrolled by 2, double-buffered software pipeline
    # 128 K-slices of 8 each, processed as 63 pairs (slices 0..125), epilogue handles 126,127
    for pair in al.range(0, 63):
        k_even = pair * 16
        k_odd = pair * 16 + 8
        k_next = (pair + 1) * 16

        # Phase 1: load odd slice (K=k_odd) into buf1
        if tid < 128:
            row = tid % 64
            col_grp = tid // 64
            a_row = m_base + row
            a_col = k_odd + col_grp * 4
            for j in al.range(4):
                A_buf1[row, col_grp * 4 + j] = A[bx, a_row, a_col + j]

        if tid < 128:
            i = tid // 16
            j0 = (tid % 16) * 4
            b_row = k_odd + i
            b_col = n_base + j0
            for j in al.range(4):
                B_buf1[j0 + j, i] = B[bx, b_row, b_col + j]

        al.syncthreads()

        # Phase 2: compute even slice (K=k_even) on buf0
        if warp_id < 4:
            w_row = warp_y * 32
            w_col = warp_x * 32
            a_row = w_row + (lane_id % 32)
            gA = lane_id // 32
            b_row = w_col + (lane_id % 32)
            gB = lane_id // 32

            a_u32 = al.make_local((2,), al.u32)
            b_u32 = al.make_local((2,), al.u32)

            for p in al.range(2):
                lo_a = al.bitcast(A_buf0[a_row, gA * 4 + p * 2], al.u16)
                hi_a = al.bitcast(A_buf0[a_row, gA * 4 + p * 2 + 1], al.u16)
                a_u32[p] = al.convert(lo_a, al.u32) | (al.convert(hi_a, al.u32) << 16)
                lo_b = al.bitcast(B_buf0[b_row, gB * 4 + p * 2], al.u16)
                hi_b = al.bitcast(B_buf0[b_row, gB * 4 + p * 2 + 1], al.u16)
                b_u32[p] = al.convert(lo_b, al.u32) | (al.convert(hi_b, al.u32) << 16)

            a_vec = al.view(a_u32, al.Tensor((2,), al.u32))
            b_vec = al.view(b_u32, al.Tensor((2,), al.u32))
            c_vec = al.view(acc_mem, al.Tensor((16,), al.f32))

            c_vec = al.amdgpu.mfma_32x32x8_bf16_f32(a_vec, b_vec, c_vec)

            for i in al.range(16):
                acc_mem[i] = c_vec[i]

        al.syncthreads()

        # Phase 3: load next even slice (K=k_next) into buf0
        if tid < 128:
            row = tid % 64
            col_grp = tid // 64
            a_row = m_base + row
            a_col = k_next + col_grp * 4
            for j in al.range(4):
                A_buf0[row, col_grp * 4 + j] = A[bx, a_row, a_col + j]

        if tid < 128:
            i = tid // 16
            j0 = (tid % 16) * 4
            b_row = k_next + i
            b_col = n_base + j0
            for j in al.range(4):
                B_buf0[j0 + j, i] = B[bx, b_row, b_col + j]

        al.syncthreads()

        # Phase 4: compute odd slice (K=k_odd) on buf1
        if warp_id < 4:
            w_row = warp_y * 32
            w_col = warp_x * 32
            a_row = w_row + (lane_id % 32)
            gA = lane_id // 32
            b_row = w_col + (lane_id % 32)
            gB = lane_id // 32

            a_u32 = al.make_local((2,), al.u32)
            b_u32 = al.make_local((2,), al.u32)

            for p in al.range(2):
                lo_a = al.bitcast(A_buf1[a_row, gA * 4 + p * 2], al.u16)
                hi_a = al.bitcast(A_buf1[a_row, gA * 4 + p * 2 + 1], al.u16)
                a_u32[p] = al.convert(lo_a, al.u32) | (al.convert(hi_a, al.u32) << 16)
                lo_b = al.bitcast(B_buf1[b_row, gB * 4 + p * 2], al.u16)
                hi_b = al.bitcast(B_buf1[b_row, gB * 4 + p * 2 + 1], al.u16)
                b_u32[p] = al.convert(lo_b, al.u32) | (al.convert(hi_b, al.u32) << 16)

            a_vec = al.view(a_u32, al.Tensor((2,), al.u32))
            b_vec = al.view(b_u32, al.Tensor((2,), al.u32))
            c_vec = al.view(acc_mem, al.Tensor((16,), al.f32))

            c_vec = al.amdgpu.mfma_32x32x8_bf16_f32(a_vec, b_vec, c_vec)

            for i in al.range(16):
                acc_mem[i] = c_vec[i]

        al.syncthreads()

    # Epilogue: compute slice 126 (K=1008) on buf0
    if warp_id < 4:
        w_row = warp_y * 32
        w_col = warp_x * 32
        a_row = w_row + (lane_id % 32)
        gA = lane_id // 32
        b_row = w_col + (lane_id % 32)
        gB = lane_id // 32

        a_u32 = al.make_local((2,), al.u32)
        b_u32 = al.make_local((2,), al.u32)

        for p in al.range(2):
            lo_a = al.bitcast(A_buf0[a_row, gA * 4 + p * 2], al.u16)
            hi_a = al.bitcast(A_buf0[a_row, gA * 4 + p * 2 + 1], al.u16)
            a_u32[p] = al.convert(lo_a, al.u32) | (al.convert(hi_a, al.u32) << 16)
            lo_b = al.bitcast(B_buf0[b_row, gB * 4 + p * 2], al.u16)
            hi_b = al.bitcast(B_buf0[b_row, gB * 4 + p * 2 + 1], al.u16)
            b_u32[p] = al.convert(lo_b, al.u32) | (al.convert(hi_b, al.u32) << 16)

        a_vec = al.view(a_u32, al.Tensor((2,), al.u32))
        b_vec = al.view(b_u32, al.Tensor((2,), al.u32))
        c_vec = al.view(acc_mem, al.Tensor((16,), al.f32))

        c_vec = al.amdgpu.mfma_32x32x8_bf16_f32(a_vec, b_vec, c_vec)

        for i in al.range(16):
            acc_mem[i] = c_vec[i]

    al.syncthreads()

    # Load final slice 127 (K=1016) into buf1
    k_last = 1016
    if tid < 128:
        row = tid % 64
        col_grp = tid // 64
        a_row = m_base + row
        a_col = k_last + col_grp * 4
        for j in al.range(4):
            A_buf1[row, col_grp * 4 + j] = A[bx, a_row, a_col + j]

    if tid < 128:
        i = tid // 16
        j0 = (tid % 16) * 4
        b_row = k_last + i
        b_col = n_base + j0
        for j in al.range(4):
            B_buf1[j0 + j, i] = B[bx, b_row, b_col + j]

    al.syncthreads()

    # Compute slice 127 (K=1016) on buf1
    if warp_id < 4:
        w_row = warp_y * 32
        w_col = warp_x * 32
        a_row = w_row + (lane_id % 32)
        gA = lane_id // 32
        b_row = w_col + (lane_id % 32)
        gB = lane_id // 32

        a_u32 = al.make_local((2,), al.u32)
        b_u32 = al.make_local((2,), al.u32)

        for p in al.range(2):
            lo_a = al.bitcast(A_buf1[a_row, gA * 4 + p * 2], al.u16)
            hi_a = al.bitcast(A_buf1[a_row, gA * 4 + p * 2 + 1], al.u16)
            a_u32[p] = al.convert(lo_a, al.u32) | (al.convert(hi_a, al.u32) << 16)
            lo_b = al.bitcast(B_buf1[b_row, gB * 4 + p * 2], al.u16)
            hi_b = al.bitcast(B_buf1[b_row, gB * 4 + p * 2 + 1], al.u16)
            b_u32[p] = al.convert(lo_b, al.u32) | (al.convert(hi_b, al.u32) << 16)

        a_vec = al.view(a_u32, al.Tensor((2,), al.u32))
        b_vec = al.view(b_u32, al.Tensor((2,), al.u32))
        c_vec = al.view(acc_mem, al.Tensor((16,), al.f32))

        c_vec = al.amdgpu.mfma_32x32x8_bf16_f32(a_vec, b_vec, c_vec)

        for i in al.range(16):
            acc_mem[i] = c_vec[i]

    al.syncthreads()

    # Store results: column-major MFMA output layout
    # Each thread outputs one N column and 16 M rows within its 32x32 warp tile
    if warp_id < 4:
        lane_col = lane_id % 32
        lane_group = lane_id // 32
        w_row = warp_y * 32
        w_col = warp_x * 32
        col = n_base + w_col + lane_col
        for t in al.range(16):
            row = m_base + w_row + (t // 4) * 8 + lane_group * 4 + (t % 4)
            C[bx, row, col] = al.convert(acc_mem[t], al.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((128, 512, 2048), device=A.device, dtype=A.dtype)
        grid = (128, 8, 32)
        block = (256, 1, 1)
        bmm_kernel[lambda: (grid, block)](A, B, C)
        return C
