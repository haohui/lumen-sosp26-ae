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

    rsrc_a = al.amdgpu.make_rsrc(A, al.convert(M * K_val, al.i32) * 2)
    rsrc_b = al.amdgpu.make_rsrc(B, al.convert(K_val * N_val, al.i32) * 2)

    block_m = al.block_id(0)
    block_n = al.block_id(1)
    lane_id = al.thread_id(0) % 64

    As = al.make_shared((32, 32), al.bf16)
    Bs = al.make_shared((32, 32), al.bf16)

    a_m_base = block_m * 32
    b_n_base = block_n * 32

    row_g = lane_id // 8
    col_g = lane_id % 8
    k_g = lane_id % 4

    acc = al.make_local((16,), al.f32)
    for i in al.range(16):
        acc[i] = 0.0

    for k_block in al.range(0, K_val, 32):
        # Cooperative global -> LDS load for A (32x32)
        row_a = lane_id // 2
        col_a = (lane_id % 2) * 16
        if row_a < 32:
            a_byte_off = ((a_m_base + row_a) * lda + (k_block + col_a)) * 2
            a_loaded = al.amdgpu.raw_buffer_load_x4(rsrc_a, a_byte_off, 0, 0)
            a_bf16_load = al.view(a_loaded, al.Tensor((8,), al.bf16))
            for i in al.range(8):
                As[row_a, col_a + i] = a_bf16_load[i]
            a_byte_off2 = ((a_m_base + row_a) * lda + (k_block + col_a + 8)) * 2
            a_loaded2 = al.amdgpu.raw_buffer_load_x4(rsrc_a, a_byte_off2, 0, 0)
            a_bf16_load2 = al.view(a_loaded2, al.Tensor((8,), al.bf16))
            for i in al.range(8):
                As[row_a, col_a + 8 + i] = a_bf16_load2[i]

        # Cooperative global -> LDS load for B (32x32)
        row_b = lane_id // 2
        col_b = (lane_id % 2) * 16
        if row_b < 32:
            b_byte_off = ((k_block + row_b) * ldb + (b_n_base + col_b)) * 2
            b_loaded = al.amdgpu.raw_buffer_load_x4(rsrc_b, b_byte_off, 0, 0)
            b_bf16_load = al.view(b_loaded, al.Tensor((8,), al.bf16))
            for i in al.range(8):
                Bs[row_b, col_b + i] = b_bf16_load[i]
            b_byte_off2 = ((k_block + row_b) * ldb + (b_n_base + col_b + 8)) * 2
            b_loaded2 = al.amdgpu.raw_buffer_load_x4(rsrc_b, b_byte_off2, 0, 0)
            b_bf16_load2 = al.view(b_loaded2, al.Tensor((8,), al.bf16))
            for i in al.range(8):
                Bs[row_b, col_b + 8 + i] = b_bf16_load2[i]

        al.syncthreads()

        for ks in al.range(0, 32, 8):
            a_bf16 = al.make_local((4,), al.bf16)
            a_bf16[0] = As[row_g * 4 + 0, ks + k_g * 2 + 0]
            a_bf16[1] = As[row_g * 4 + 0, ks + k_g * 2 + 1]
            a_bf16[2] = As[row_g * 4 + 1, ks + k_g * 2 + 0]
            a_bf16[3] = As[row_g * 4 + 1, ks + k_g * 2 + 1]
            a_u32 = al.view(a_bf16, al.Tensor((2,), al.u32))

            b_bf16 = al.make_local((4,), al.bf16)
            b_bf16[0] = Bs[ks + k_g * 2 + 0, col_g * 4 + 0]
            b_bf16[1] = Bs[ks + k_g * 2 + 1, col_g * 4 + 0]
            b_bf16[2] = Bs[ks + k_g * 2 + 0, col_g * 4 + 1]
            b_bf16[3] = Bs[ks + k_g * 2 + 1, col_g * 4 + 1]
            b_u32 = al.view(b_bf16, al.Tensor((2,), al.u32))

            acc = al.amdgpu.mfma_32x32x8_bf16_f32(a_u32, b_u32, acc)

        al.syncthreads()

    out_row = a_m_base + row_g * 4
    out_col = b_n_base + col_g * 4
    for mi in al.range(4):
        for ni in al.range(4):
            C[out_row + mi, out_col + ni] = al.convert(
                acc[mi * 4 + ni], al.bf16
            )


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
