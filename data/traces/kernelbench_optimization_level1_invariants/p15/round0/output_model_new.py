import torch
import torch.nn as nn

import avelang
import avelang.language as al


@avelang.jit
def tri_gemm_kernel(
    A_ptr: al.Pointer(al.u8),
    B_ptr: al.Pointer(al.u8),
    C_ptr: al.Pointer(al.u8),
    M: al.i32,
):
    """Tiled GEMM: 64x64 block, 256 threads, LDS staging, raw_buffer_load_x4, tril mask."""

    layout_M = al.make_layout((M, M), (M, 1))
    A = al.make_tensor(A_ptr, al.bf16, layout_M)
    B = al.make_tensor(B_ptr, al.bf16, layout_M)
    C = al.make_tensor(C_ptr, al.bf16, layout_M)

    tid = al.thread_id(0)
    block_m = al.block_id(0)
    block_n = al.block_id(1)

    thr_m = tid // 16
    thr_n = tid % 16

    out_row_base = block_m * 64 + thr_m * 4
    out_col_base = block_n * 64 + thr_n * 4

    # 4x4 FP32 accumulators
    acc = al.make_local((4, 4), al.f32)
    zero = al.convert(0.0, al.f32)
    for i in al.range(4):
        for j in al.range(4):
            acc[i, j] = zero

    # Shared memory
    A_lds = al.make_shared((64, 32), al.bf16)
    B_lds = al.make_shared((32, 64), al.bf16)

    A_rsrc = al.amdgpu.make_rsrc(A, 4096 * 4096 * 2)
    B_rsrc = al.amdgpu.make_rsrc(B, 4096 * 4096 * 2)

    a_lds_row = thr_m * 4
    b_lds_col = thr_n * 4

    for k in al.range(0, M, 32):
        # --- Global -> LDS: A tile [64 x 32] ---
        a_ld_row = tid // 4
        a_col_b16 = (tid % 4) * 8
        a_global_row = block_m * 64 + a_ld_row
        a_global_col = k + a_col_b16
        a_byte_off = (a_global_row * M + a_global_col) * 2

        a_vec = al.amdgpu.raw_buffer_load_x4(A_rsrc, a_byte_off, 0, 0)
        a_v8 = al.view(a_vec, al.Tensor((8,), al.bf16))
        A_lds[a_ld_row, a_col_b16 + 0] = a_v8[0]
        A_lds[a_ld_row, a_col_b16 + 1] = a_v8[1]
        A_lds[a_ld_row, a_col_b16 + 2] = a_v8[2]
        A_lds[a_ld_row, a_col_b16 + 3] = a_v8[3]
        A_lds[a_ld_row, a_col_b16 + 4] = a_v8[4]
        A_lds[a_ld_row, a_col_b16 + 5] = a_v8[5]
        A_lds[a_ld_row, a_col_b16 + 6] = a_v8[6]
        A_lds[a_ld_row, a_col_b16 + 7] = a_v8[7]

        # --- Global -> LDS: B tile [32 x 64] ---
        b_ld_row = tid // 8
        b_col_b16 = (tid % 8) * 8
        b_global_row = k + b_ld_row
        b_global_col = block_n * 64 + b_col_b16
        b_byte_off = (b_global_row * M + b_global_col) * 2

        b_vec = al.amdgpu.raw_buffer_load_x4(B_rsrc, b_byte_off, 0, 0)
        b_v8 = al.view(b_vec, al.Tensor((8,), al.bf16))
        B_lds[b_ld_row, b_col_b16 + 0] = b_v8[0]
        B_lds[b_ld_row, b_col_b16 + 1] = b_v8[1]
        B_lds[b_ld_row, b_col_b16 + 2] = b_v8[2]
        B_lds[b_ld_row, b_col_b16 + 3] = b_v8[3]
        B_lds[b_ld_row, b_col_b16 + 4] = b_v8[4]
        B_lds[b_ld_row, b_col_b16 + 5] = b_v8[5]
        B_lds[b_ld_row, b_col_b16 + 6] = b_v8[6]
        B_lds[b_ld_row, b_col_b16 + 7] = b_v8[7]

        al.syncthreads()

        # --- Per-thread accumulation: 4x4 output, K-unrolled by 4 ---
        for i in al.range(4):
            a_row = a_lds_row + i
            for j in al.range(4):
                b_col = b_lds_col + j
                dot = al.convert(0.0, al.f32)
                for kk in al.range(0, 32, 4):
                    a0 = A_lds[a_row, kk + 0]
                    a1 = A_lds[a_row, kk + 1]
                    a2 = A_lds[a_row, kk + 2]
                    a3 = A_lds[a_row, kk + 3]
                    b0 = B_lds[kk + 0, b_col]
                    b1 = B_lds[kk + 1, b_col]
                    b2 = B_lds[kk + 2, b_col]
                    b3 = B_lds[kk + 3, b_col]
                    dot = dot + al.convert(a0, al.f32) * al.convert(b0, al.f32)
                    dot = dot + al.convert(a1, al.f32) * al.convert(b1, al.f32)
                    dot = dot + al.convert(a2, al.f32) * al.convert(b2, al.f32)
                    dot = dot + al.convert(a3, al.f32) * al.convert(b3, al.f32)
                acc[i, j] = acc[i, j] + dot

        al.syncthreads()

    # Writeback with tril mask
    for i in al.range(4):
        g_row = out_row_base + i
        for j in al.range(4):
            g_col = out_col_base + j
            if g_col <= g_row:
                C[g_row, g_col] = al.convert(acc[i, j], al.bf16)
            else:
                C[g_row, g_col] = al.convert(0.0, al.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if tuple(A.shape) != (4096, 4096) or tuple(B.shape) != (4096, 4096):
            raise RuntimeError("Shape mismatch: expected (4096, 4096)")
        A = A.contiguous()
        B = B.contiguous()
        if A.dtype != torch.bfloat16:
            A = A.to(torch.bfloat16)
        if B.dtype != torch.bfloat16:
            B = B.to(torch.bfloat16)
        C = torch.empty((4096, 4096), device=A.device, dtype=torch.bfloat16)
        M_val = 4096
        grid_dim = M_val // 64
        tri_gemm_kernel[lambda: ((grid_dim, grid_dim, 1), (256, 1, 1))](
            A, B, C, M_val
        )
        return C
