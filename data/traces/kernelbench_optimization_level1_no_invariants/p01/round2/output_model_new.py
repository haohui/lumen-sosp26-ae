import torch
import torch.nn as nn

import avelang
import avelang.language as al

N = 4096

TILE_M = 32
TILE_N = 32
TILE_K = 32


@avelang.jit
def gemm_kernel(
    A_ptr: al.Pointer(al.bf16),
    B_ptr: al.Pointer(al.bf16),
    C_ptr: al.Pointer(al.bf16),
    M: al.i32,
    N_val: al.i32,
    K_val: al.i32,
    lda: al.i32,
    ldb: al.i32,
    ldc: al.i32,
):
    layout_a = al.make_layout((M, K_val), (lda, 1))
    A = al.make_tensor(A_ptr, al.bf16, layout_a)
    layout_b = al.make_layout((K_val, N_val), (ldb, 1))
    B = al.make_tensor(B_ptr, al.bf16, layout_b)
    layout_c = al.make_layout((M, N_val), (ldc, 1))
    C = al.make_tensor(C_ptr, al.bf16, layout_c)

    block_m = al.block_id(0)
    block_n = al.block_id(1)
    lane_id = al.thread_id(0) % 64

    As = al.make_shared((32, 32), al.bf16)
    Bs = al.make_shared((32, 32), al.bf16)

    a_m_base = block_m * 32
    b_n_base = block_n * 32

    row_16 = ((lane_id // 4) % 8) * 2
    col_16 = (lane_id % 4) * 2
    k_off16 = (lane_id % 4) * 4

    acc00 = al.make_local((4,), al.f32)
    acc01 = al.make_local((4,), al.f32)
    acc10 = al.make_local((4,), al.f32)
    acc11 = al.make_local((4,), al.f32)
    for i in al.range(4):
        acc00[i] = 0.0
        acc01[i] = 0.0
        acc10[i] = 0.0
        acc11[i] = 0.0

    for k_block in al.range(0, K_val, 32):
        # Global -> shared: A tile. No OOB guards needed -- N=4096 is tile-aligned.
        row_a = lane_id // 2
        col_a = (lane_id % 2) * 16
        a_r = a_m_base + row_a
        a_c = k_block + col_a
        for i in al.range(16):
            As[row_a, col_a + i] = A[a_r, a_c + i]

        # Global -> shared: B tile.
        row_b = lane_id // 2
        col_b = (lane_id % 2) * 16
        b_r = k_block + row_b
        b_c = b_n_base + col_b
        for i in al.range(16):
            Bs[row_b, col_b + i] = B[b_r, b_c + i]

        al.syncthreads()

        for ks in al.range(0, 32, 16):
            a0_bf16 = al.make_local((4,), al.bf16)
            a0_bf16[0] = As[row_16 + 0, ks + k_off16 + 0]
            a0_bf16[1] = As[row_16 + 1, ks + k_off16 + 0]
            a0_bf16[2] = As[row_16 + 0, ks + k_off16 + 1]
            a0_bf16[3] = As[row_16 + 1, ks + k_off16 + 1]
            a0_u32 = al.view(a0_bf16, al.Tensor((2,), al.u32))

            a1_bf16 = al.make_local((4,), al.bf16)
            a1_bf16[0] = As[16 + row_16 + 0, ks + k_off16 + 0]
            a1_bf16[1] = As[16 + row_16 + 1, ks + k_off16 + 0]
            a1_bf16[2] = As[16 + row_16 + 0, ks + k_off16 + 1]
            a1_bf16[3] = As[16 + row_16 + 1, ks + k_off16 + 1]
            a1_u32 = al.view(a1_bf16, al.Tensor((2,), al.u32))

            b0_bf16 = al.make_local((4,), al.bf16)
            b0_bf16[0] = Bs[ks + k_off16 + 0, col_16 + 0]
            b0_bf16[1] = Bs[ks + k_off16 + 0, col_16 + 1]
            b0_bf16[2] = Bs[ks + k_off16 + 1, col_16 + 0]
            b0_bf16[3] = Bs[ks + k_off16 + 1, col_16 + 1]
            b0_u32 = al.view(b0_bf16, al.Tensor((2,), al.u32))

            b1_bf16 = al.make_local((4,), al.bf16)
            b1_bf16[0] = Bs[ks + k_off16 + 0, 16 + col_16 + 0]
            b1_bf16[1] = Bs[ks + k_off16 + 0, 16 + col_16 + 1]
            b1_bf16[2] = Bs[ks + k_off16 + 1, 16 + col_16 + 0]
            b1_bf16[3] = Bs[ks + k_off16 + 1, 16 + col_16 + 1]
            b1_u32 = al.view(b1_bf16, al.Tensor((2,), al.u32))

            acc00 = al.amdgpu.mfma_16x16x16_bf16_f32(a0_u32, b0_u32, acc00)
            acc01 = al.amdgpu.mfma_16x16x16_bf16_f32(a0_u32, b1_u32, acc01)
            acc10 = al.amdgpu.mfma_16x16x16_bf16_f32(a1_u32, b0_u32, acc10)
            acc11 = al.amdgpu.mfma_16x16x16_bf16_f32(a1_u32, b1_u32, acc11)

        al.syncthreads()

    C[a_m_base + row_16 + 0, b_n_base + col_16 + 0] = al.convert(acc00[0], al.bf16)
    C[a_m_base + row_16 + 0, b_n_base + col_16 + 1] = al.convert(acc00[1], al.bf16)
    C[a_m_base + row_16 + 1, b_n_base + col_16 + 0] = al.convert(acc00[2], al.bf16)
    C[a_m_base + row_16 + 1, b_n_base + col_16 + 1] = al.convert(acc00[3], al.bf16)
    C[a_m_base + row_16 + 0, b_n_base + 16 + col_16 + 0] = al.convert(acc01[0], al.bf16)
    C[a_m_base + row_16 + 0, b_n_base + 16 + col_16 + 1] = al.convert(acc01[1], al.bf16)
    C[a_m_base + row_16 + 1, b_n_base + 16 + col_16 + 0] = al.convert(acc01[2], al.bf16)
    C[a_m_base + row_16 + 1, b_n_base + 16 + col_16 + 1] = al.convert(acc01[3], al.bf16)
    C[a_m_base + 16 + row_16 + 0, b_n_base + col_16 + 0] = al.convert(acc10[0], al.bf16)
    C[a_m_base + 16 + row_16 + 0, b_n_base + col_16 + 1] = al.convert(acc10[1], al.bf16)
    C[a_m_base + 16 + row_16 + 1, b_n_base + col_16 + 0] = al.convert(acc10[2], al.bf16)
    C[a_m_base + 16 + row_16 + 1, b_n_base + col_16 + 1] = al.convert(acc10[3], al.bf16)
    C[a_m_base + 16 + row_16 + 0, b_n_base + 16 + col_16 + 0] = al.convert(acc11[0], al.bf16)
    C[a_m_base + 16 + row_16 + 0, b_n_base + 16 + col_16 + 1] = al.convert(acc11[1], al.bf16)
    C[a_m_base + 16 + row_16 + 1, b_n_base + 16 + col_16 + 0] = al.convert(acc11[2], al.bf16)
    C[a_m_base + 16 + row_16 + 1, b_n_base + 16 + col_16 + 1] = al.convert(acc11[3], al.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
            raise RuntimeError("AveLang kernel expects bf16 inputs")
        M, K_val = A.shape
        K_b, N_val = B.shape
        if K_val != K_b:
            raise RuntimeError("Inner dimension mismatch")
        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((M, N_val), device=A.device, dtype=A.dtype)
        grid_m = (M + TILE_M - 1) // TILE_M
        grid_n = (N_val + TILE_N - 1) // TILE_N
        gemm_kernel[lambda: ((grid_m, grid_n, 1), (64, 1, 1))](
            A, B, C, M, N_val, K_val,
            A.stride(0), B.stride(0), C.stride(0),
        )
        return C
