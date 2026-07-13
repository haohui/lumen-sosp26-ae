import torch
import torch.nn as nn

import avelang
import avelang.language as al


@avelang.jit
def gemm_kernel(
    A_ptr: al.Pointer(al.bf16),
    B_ptr: al.Pointer(al.bf16),
    C_ptr: al.Pointer(al.bf16),
    M: al.i32,
    L_dim: al.i32,
    K_dim: al.i32,
):
    block_m = al.block_id(0) * 64
    block_n = al.block_id(1) * 64
    tid = al.thread_id(0)
    wave_id = tid // 64
    wm = wave_id // 2
    wn = wave_id - wm * 2
    lane = tid - wave_id * 64

    A_layout = al.make_layout((M, L_dim), (L_dim, 1))
    A = al.make_tensor(A_ptr, al.bf16, A_layout)
    B_layout = al.make_layout((L_dim, K_dim), (K_dim, 1))
    B = al.make_tensor(B_ptr, al.bf16, B_layout)
    C_layout = al.make_layout((M, K_dim), (K_dim, 1))
    C = al.make_tensor(C_ptr, al.bf16, C_layout)

    # LDS as u32 for raw_buffer_load_x4
    A_lds = al.make_shared((64, 8), al.u32)
    B_lds = al.make_shared((16, 32), al.u32)
    A_rsrc = al.amdgpu.make_rsrc(A, M * L_dim * 2)
    B_rsrc = al.amdgpu.make_rsrc(B, L_dim * K_dim * 2)

    # Per-thread accumulator: 4x4 f32 (16 elements)
    acc = al.make_local((4, 4), al.f32)
    for r in al.range(4):
        for c in al.range(4):
            acc[r, c] = al.convert(0.0, al.f32)

    for k_block in al.range(0, L_dim, 16):
        # --- Global -> LDS: load A tile (threads 0-127) ---
        if tid < 128:
            a_row = tid // 2
            a_col_group = tid - a_row * 2
            g_row = block_m + a_row
            g_col = k_block + a_col_group * 8
            if g_row < M:
                bo = (g_row * L_dim + g_col) * 2
                d = al.amdgpu.raw_buffer_load_x4(A_rsrc, bo, 0, 0)
                A_lds[a_row, a_col_group * 4 + 0] = d[0]
                A_lds[a_row, a_col_group * 4 + 1] = d[1]
                A_lds[a_row, a_col_group * 4 + 2] = d[2]
                A_lds[a_row, a_col_group * 4 + 3] = d[3]

        # --- Global -> LDS: load B tile (threads 128-255) ---
        if tid >= 128:
            tidx = tid - 128
            b_row = tidx // 8
            b_col_group = tidx - b_row * 8
            g_row = k_block + b_row
            g_col = block_n + b_col_group * 8
            if g_col < K_dim:
                bo = (g_row * K_dim + g_col) * 2
                d = al.amdgpu.raw_buffer_load_x4(B_rsrc, bo, 0, 0)
                B_lds[b_row, b_col_group * 4 + 0] = d[0]
                B_lds[b_row, b_col_group * 4 + 1] = d[1]
                B_lds[b_row, b_col_group * 4 + 2] = d[2]
                B_lds[b_row, b_col_group * 4 + 3] = d[3]

        al.syncthreads()

        # bf16 view of u32 LDS
        A_bf = al.view(A_lds, al.bf16, al.make_layout((64, 16), (16, 1)))
        B_bf = al.view(B_lds, al.bf16, al.make_layout((16, 64), (64, 1)))

        # Thread's 4x4 output sub-tile
        t_row = lane // 8
        t_col = lane - t_row * 8
        a_row_off = wm * 32 + t_row * 4
        b_col_off = wn * 32 + t_col * 4

        for ki in al.range(16):
            for r in al.range(4):
                a_val = al.convert(A_bf[a_row_off + r, ki], al.f32)
                for c in al.range(4):
                    b_val = al.convert(B_bf[ki, b_col_off + c], al.f32)
                    acc[r, c] = acc[r, c] + a_val * b_val

        al.syncthreads()

    # --- Writeback ---
    t_row = lane // 8
    t_col = lane - t_row * 8
    c_row_off = block_m + wm * 32 + t_row * 4
    c_col_off = block_n + wn * 32 + t_col * 4
    for r in al.range(4):
        c_row = c_row_off + r
        if c_row < M:
            for c in al.range(4):
                c_col = c_col_off + c
                if c_col < K_dim:
                    C[c_row, c_col] = al.convert(acc[r, c], al.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
            raise RuntimeError("AveLang kernel expects bf16 inputs")
        A = A.contiguous()
        B = B.contiguous()

        b_dim, i_dim, j_dim, l_dim = A.shape
        l_dim_b, k_dim = B.shape
        M_dim = b_dim * i_dim * j_dim

        C = torch.empty((b_dim, i_dim, j_dim, k_dim), device=A.device, dtype=torch.bfloat16)

        grid_m = M_dim // 64
        grid_n = k_dim // 64

        gemm_kernel[lambda: ((grid_m, grid_n, 1), (256, 1, 1))](
            A.data_ptr(), B.data_ptr(), C.data_ptr(),
            M_dim, l_dim, k_dim,
        )

        return C
