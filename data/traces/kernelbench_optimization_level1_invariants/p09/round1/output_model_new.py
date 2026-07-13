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
    wid = al.thread_id(0) // 64
    warp_row = wid // 2
    warp_col = wid % 2
    block_m = al.block_id(1)
    block_n = al.block_id(0)

    acc00 = al.make_local((16,), al.f32)
    acc01 = al.make_local((16,), al.f32)
    acc10 = al.make_local((16,), al.f32)
    acc11 = al.make_local((16,), al.f32)
    for i in al.range(16):
        acc00[i] = al.convert(0.0, al.f32)
        acc01[i] = al.convert(0.0, al.f32)
        acc10[i] = al.convert(0.0, al.f32)
        acc11[i] = al.convert(0.0, al.f32)

    a_row0 = block_m * 128 + warp_row * 64 + 0 + (lane % 32)
    a_row1 = block_m * 128 + warp_row * 64 + 32 + (lane % 32)
    b_col0 = block_n * 128 + warp_col * 64 + 0 + (lane % 32)
    b_col1 = block_n * 128 + warp_col * 64 + 32 + (lane % 32)

    for k_block in al.range(0, 32, 16):
        a0_s0 = al.make_local((4,), al.bf16)
        a0_s1 = al.make_local((4,), al.bf16)
        a1_s0 = al.make_local((4,), al.bf16)
        a1_s1 = al.make_local((4,), al.bf16)
        b0_s0 = al.make_local((4,), al.bf16)
        b0_s1 = al.make_local((4,), al.bf16)
        b1_s0 = al.make_local((4,), al.bf16)
        b1_s1 = al.make_local((4,), al.bf16)

        if lane < 32:
            a0_s0[0] = A[a_row0, k_block + 0]; a0_s0[1] = A[a_row0, k_block + 1]
            a0_s0[2] = A[a_row0, k_block + 2]; a0_s0[3] = A[a_row0, k_block + 3]
            a0_s1[0] = A[a_row0, k_block + 8]; a0_s1[1] = A[a_row0, k_block + 9]
            a0_s1[2] = A[a_row0, k_block + 10]; a0_s1[3] = A[a_row0, k_block + 11]
            a1_s0[0] = A[a_row1, k_block + 0]; a1_s0[1] = A[a_row1, k_block + 1]
            a1_s0[2] = A[a_row1, k_block + 2]; a1_s0[3] = A[a_row1, k_block + 3]
            a1_s1[0] = A[a_row1, k_block + 8]; a1_s1[1] = A[a_row1, k_block + 9]
            a1_s1[2] = A[a_row1, k_block + 10]; a1_s1[3] = A[a_row1, k_block + 11]
        else:
            a0_s0[0] = A[a_row0, k_block + 4]; a0_s0[1] = A[a_row0, k_block + 5]
            a0_s0[2] = A[a_row0, k_block + 6]; a0_s0[3] = A[a_row0, k_block + 7]
            a0_s1[0] = A[a_row0, k_block + 12]; a0_s1[1] = A[a_row0, k_block + 13]
            a0_s1[2] = A[a_row0, k_block + 14]; a0_s1[3] = A[a_row0, k_block + 15]
            a1_s0[0] = A[a_row1, k_block + 4]; a1_s0[1] = A[a_row1, k_block + 5]
            a1_s0[2] = A[a_row1, k_block + 6]; a1_s0[3] = A[a_row1, k_block + 7]
            a1_s1[0] = A[a_row1, k_block + 12]; a1_s1[1] = A[a_row1, k_block + 13]
            a1_s1[2] = A[a_row1, k_block + 14]; a1_s1[3] = A[a_row1, k_block + 15]

        if lane < 32:
            b0_s0[0] = B[k_block + 0, b_col0]; b0_s0[1] = B[k_block + 1, b_col0]
            b0_s0[2] = B[k_block + 2, b_col0]; b0_s0[3] = B[k_block + 3, b_col0]
            b0_s1[0] = B[k_block + 8, b_col0]; b0_s1[1] = B[k_block + 9, b_col0]
            b0_s1[2] = B[k_block + 10, b_col0]; b0_s1[3] = B[k_block + 11, b_col0]
            b1_s0[0] = B[k_block + 0, b_col1]; b1_s0[1] = B[k_block + 1, b_col1]
            b1_s0[2] = B[k_block + 2, b_col1]; b1_s0[3] = B[k_block + 3, b_col1]
            b1_s1[0] = B[k_block + 8, b_col1]; b1_s1[1] = B[k_block + 9, b_col1]
            b1_s1[2] = B[k_block + 10, b_col1]; b1_s1[3] = B[k_block + 11, b_col1]
        else:
            b0_s0[0] = B[k_block + 4, b_col0]; b0_s0[1] = B[k_block + 5, b_col0]
            b0_s0[2] = B[k_block + 6, b_col0]; b0_s0[3] = B[k_block + 7, b_col0]
            b0_s1[0] = B[k_block + 12, b_col0]; b0_s1[1] = B[k_block + 13, b_col0]
            b0_s1[2] = B[k_block + 14, b_col0]; b0_s1[3] = B[k_block + 15, b_col0]
            b1_s0[0] = B[k_block + 4, b_col1]; b1_s0[1] = B[k_block + 5, b_col1]
            b1_s0[2] = B[k_block + 6, b_col1]; b1_s0[3] = B[k_block + 7, b_col1]
            b1_s1[0] = B[k_block + 12, b_col1]; b1_s1[1] = B[k_block + 13, b_col1]
            b1_s1[2] = B[k_block + 14, b_col1]; b1_s1[3] = B[k_block + 15, b_col1]

        au0 = al.view(a0_s0, al.Tensor((2,), al.u32))
        au1 = al.view(a0_s1, al.Tensor((2,), al.u32))
        bu0 = al.view(b0_s0, al.Tensor((2,), al.u32))
        bu1 = al.view(b0_s1, al.Tensor((2,), al.u32))
        acc00 = al.amdgpu.mfma_32x32x8_bf16_f32(au0, bu0, acc00)
        acc00 = al.amdgpu.mfma_32x32x8_bf16_f32(au1, bu1, acc00)

        bu0 = al.view(b1_s0, al.Tensor((2,), al.u32))
        bu1 = al.view(b1_s1, al.Tensor((2,), al.u32))
        acc01 = al.amdgpu.mfma_32x32x8_bf16_f32(au0, bu0, acc01)
        acc01 = al.amdgpu.mfma_32x32x8_bf16_f32(au1, bu1, acc01)

        au0 = al.view(a1_s0, al.Tensor((2,), al.u32))
        au1 = al.view(a1_s1, al.Tensor((2,), al.u32))
        bu0 = al.view(b0_s0, al.Tensor((2,), al.u32))
        bu1 = al.view(b0_s1, al.Tensor((2,), al.u32))
        acc10 = al.amdgpu.mfma_32x32x8_bf16_f32(au0, bu0, acc10)
        acc10 = al.amdgpu.mfma_32x32x8_bf16_f32(au1, bu1, acc10)

        bu0 = al.view(b1_s0, al.Tensor((2,), al.u32))
        bu1 = al.view(b1_s1, al.Tensor((2,), al.u32))
        acc11 = al.amdgpu.mfma_32x32x8_bf16_f32(au0, bu0, acc11)
        acc11 = al.amdgpu.mfma_32x32x8_bf16_f32(au1, bu1, acc11)

    for t in al.range(16):
        r0 = block_m * 128 + warp_row * 64 + 0 + 8 * (t // 4) + 4 * (lane // 32) + (t % 4)
        r1 = block_m * 128 + warp_row * 64 + 32 + 8 * (t // 4) + 4 * (lane // 32) + (t % 4)
        c0 = block_n * 128 + warp_col * 64 + 0 + (lane % 32)
        c1 = block_n * 128 + warp_col * 64 + 32 + (lane % 32)
        C[r0, c0] = al.convert(acc00[t], al.bf16)
        C[r0, c1] = al.convert(acc01[t], al.bf16)
        C[r1, c0] = al.convert(acc10[t], al.bf16)
        C[r1, c1] = al.convert(acc11[t], al.bf16)


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

        grid_n = N_val // 128
        grid_m = M_val // 128

        gemm_kernel[lambda: ((grid_n, grid_m, 1), (256, 1, 1))](A, B, C)
        return C
