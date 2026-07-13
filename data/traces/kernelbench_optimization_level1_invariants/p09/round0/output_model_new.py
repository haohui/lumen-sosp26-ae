import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def gemm_kernel(
    A: al.Tensor((32768, 32), al.bf16),
    B: al.Tensor((32, 32768), al.bf16),
    C: al.Tensor((32768, 32768), al.bf16),
):
    lane = al.thread_id(0) % 64
    block_m = al.block_id(1)
    block_n = al.block_id(0)

    acc = al.make_local((16,), al.f32)
    for i in al.range(16):
        acc[i] = al.convert(0.0, al.f32)

    a_row = block_m * 32 + (lane % 32)
    b_col = block_n * 32 + (lane % 32)

    for k_block in al.range(0, 32, 16):
        for ks in al.range(0, 16, 8):
            a0 = al.make_local((4,), al.bf16)
            a1 = al.make_local((4,), al.bf16)
            b0 = al.make_local((4,), al.bf16)
            b1 = al.make_local((4,), al.bf16)

            if lane < 32:
                a0[0] = A[a_row, k_block + ks + 0]
                a0[1] = A[a_row, k_block + ks + 1]
                a0[2] = A[a_row, k_block + ks + 2]
                a0[3] = A[a_row, k_block + ks + 3]
                a1[0] = A[a_row, k_block + ks + 8]
                a1[1] = A[a_row, k_block + ks + 9]
                a1[2] = A[a_row, k_block + ks + 10]
                a1[3] = A[a_row, k_block + ks + 11]
                b0[0] = B[k_block + ks + 0, b_col]
                b0[1] = B[k_block + ks + 1, b_col]
                b0[2] = B[k_block + ks + 2, b_col]
                b0[3] = B[k_block + ks + 3, b_col]
                b1[0] = B[k_block + ks + 8, b_col]
                b1[1] = B[k_block + ks + 9, b_col]
                b1[2] = B[k_block + ks + 10, b_col]
                b1[3] = B[k_block + ks + 11, b_col]
            else:
                a0[0] = A[a_row, k_block + ks + 4]
                a0[1] = A[a_row, k_block + ks + 5]
                a0[2] = A[a_row, k_block + ks + 6]
                a0[3] = A[a_row, k_block + ks + 7]
                a1[0] = A[a_row, k_block + ks + 12]
                a1[1] = A[a_row, k_block + ks + 13]
                a1[2] = A[a_row, k_block + ks + 14]
                a1[3] = A[a_row, k_block + ks + 15]
                b0[0] = B[k_block + ks + 4, b_col]
                b0[1] = B[k_block + ks + 5, b_col]
                b0[2] = B[k_block + ks + 6, b_col]
                b0[3] = B[k_block + ks + 7, b_col]
                b1[0] = B[k_block + ks + 12, b_col]
                b1[1] = B[k_block + ks + 13, b_col]
                b1[2] = B[k_block + ks + 14, b_col]
                b1[3] = B[k_block + ks + 15, b_col]

            au0 = al.view(a0, al.Tensor((2,), al.u32))
            bu0 = al.view(b0, al.Tensor((2,), al.u32))
            acc = al.amdgpu.mfma_32x32x8_bf16_f32(au0, bu0, acc)
            au1 = al.view(a1, al.Tensor((2,), al.u32))
            bu1 = al.view(b1, al.Tensor((2,), al.u32))
            acc = al.amdgpu.mfma_32x32x8_bf16_f32(au1, bu1, acc)

    for t in al.range(16):
        row = block_m * 32 + 8 * (t // 4) + 4 * (lane // 32) + (t % 4)
        col = block_n * 32 + (lane % 32)
        C[row, col] = al.convert(acc[t], al.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
            raise RuntimeError('Kernel requires bfloat16 inputs.')
        if A.device != B.device:
            raise RuntimeError('A and B must be on the same device.')

        M_val = int(A.shape[0])
        N_val = int(B.shape[1])

        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((M_val, N_val), device=A.device, dtype=torch.bfloat16)

        grid_n = N_val // 32
        grid_m = M_val // 32

        gemm_kernel[lambda: ((grid_n, grid_m, 1), (64, 1, 1))](A, B, C)
        return C
