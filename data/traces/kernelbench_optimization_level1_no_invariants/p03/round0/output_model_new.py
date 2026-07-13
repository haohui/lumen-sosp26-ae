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

    A_shared = al.make_shared((64, 16), al.bf16)
    B_shared = al.make_shared((64, 16), al.bf16)

    # Initialize accumulator via a dummy MFMA on zero inputs
    acc_mem = al.make_local((16,), al.f32)
    for i in al.range(16):
        acc_mem[i] = al.convert(0.0, al.f32)

    for k_block in al.range(64):
        k_base = k_block * 16

        if tid < 128:
            row = tid % 64
            col_grp = tid // 64
            a_row = m_base + row
            a_col = k_base + col_grp * 8
            for j in al.range(8):
                A_shared[row, col_grp * 8 + j] = A[bx, a_row, a_col + j]

        if tid < 128:
            i = tid // 8
            j0 = (tid % 8) * 8
            b_row = k_base + i
            b_col = n_base + j0
            for j in al.range(8):
                B_shared[j0 + j, i] = B[bx, b_row, b_col + j]

        al.syncthreads()

        if warp_id < 4:
            w_row = warp_y * 32
            w_col = warp_x * 32
            rA = lane_id % 32
            gA = lane_id // 32
            cB = lane_id % 32
            gB = lane_id // 32
            a_row = w_row + rA
            b_row = w_col + cB

            a0_u32 = al.make_local((2,), al.u32)
            b0_u32 = al.make_local((2,), al.u32)
            a1_u32 = al.make_local((2,), al.u32)
            b1_u32 = al.make_local((2,), al.u32)

            for p in al.range(2):
                lo_a0 = al.bitcast(A_shared[a_row, gA * 4 + p * 2], al.u16)
                hi_a0 = al.bitcast(A_shared[a_row, gA * 4 + p * 2 + 1], al.u16)
                a0_u32[p] = al.convert(lo_a0, al.u32) | (al.convert(hi_a0, al.u32) << 16)
                lo_b0 = al.bitcast(B_shared[b_row, gB * 4 + p * 2], al.u16)
                hi_b0 = al.bitcast(B_shared[b_row, gB * 4 + p * 2 + 1], al.u16)
                b0_u32[p] = al.convert(lo_b0, al.u32) | (al.convert(hi_b0, al.u32) << 16)
                lo_a1 = al.bitcast(A_shared[a_row, gA * 4 + 8 + p * 2], al.u16)
                hi_a1 = al.bitcast(A_shared[a_row, gA * 4 + 8 + p * 2 + 1], al.u16)
                a1_u32[p] = al.convert(lo_a1, al.u32) | (al.convert(hi_a1, al.u32) << 16)
                lo_b1 = al.bitcast(B_shared[b_row, gB * 4 + 8 + p * 2], al.u16)
                hi_b1 = al.bitcast(B_shared[b_row, gB * 4 + 8 + p * 2 + 1], al.u16)
                b1_u32[p] = al.convert(lo_b1, al.u32) | (al.convert(hi_b1, al.u32) << 16)

            a0 = al.view(a0_u32, al.Tensor((2,), al.u32))
            a1 = al.view(a1_u32, al.Tensor((2,), al.u32))
            b0 = al.view(b0_u32, al.Tensor((2,), al.u32))
            b1 = al.view(b1_u32, al.Tensor((2,), al.u32))
            c_vec = al.view(acc_mem, al.Tensor((16,), al.f32))

            c_vec = al.amdgpu.mfma_32x32x8_bf16_f32(a0, b0, c_vec)
            c_vec = al.amdgpu.mfma_32x32x8_bf16_f32(a1, b1, c_vec)

            for i in al.range(16):
                acc_mem[i] = c_vec[i]

        al.syncthreads()

    if warp_id < 4:
        w_row = warp_y * 32
        w_col = warp_x * 32
        r0 = (lane_id % 8) * 2
        c0 = (lane_id // 8) * 2
        rr = m_base + w_row + r0
        cc = n_base + w_col + c0

        C[bx, rr,     cc]      = al.convert(acc_mem[0], al.bf16)
        C[bx, rr,     cc + 1]  = al.convert(acc_mem[1], al.bf16)
        C[bx, rr + 1, cc]      = al.convert(acc_mem[2], al.bf16)
        C[bx, rr + 1, cc + 1]  = al.convert(acc_mem[3], al.bf16)
        C[bx, rr,     cc + 16] = al.convert(acc_mem[4], al.bf16)
        C[bx, rr,     cc + 17] = al.convert(acc_mem[5], al.bf16)
        C[bx, rr + 1, cc + 16] = al.convert(acc_mem[6], al.bf16)
        C[bx, rr + 1, cc + 17] = al.convert(acc_mem[7], al.bf16)
        C[bx, rr + 16, cc]     = al.convert(acc_mem[8], al.bf16)
        C[bx, rr + 16, cc + 1] = al.convert(acc_mem[9], al.bf16)
        C[bx, rr + 17, cc]     = al.convert(acc_mem[10], al.bf16)
        C[bx, rr + 17, cc + 1] = al.convert(acc_mem[11], al.bf16)
        C[bx, rr + 16, cc + 16] = al.convert(acc_mem[12], al.bf16)
        C[bx, rr + 16, cc + 17] = al.convert(acc_mem[13], al.bf16)
        C[bx, rr + 17, cc + 16] = al.convert(acc_mem[14], al.bf16)
        C[bx, rr + 17, cc + 17] = al.convert(acc_mem[15], al.bf16)


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
