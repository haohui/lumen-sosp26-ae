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
    K_dim: al.i32,
    N: al.i32,
):
    A_layout = al.make_layout((M, K_dim), (K_dim, al.convert(1, al.i32)))
    A = al.make_tensor(A_ptr, al.bf16, A_layout)
    B_layout = al.make_layout((K_dim, N), (N, al.convert(1, al.i32)))
    B = al.make_tensor(B_ptr, al.bf16, B_layout)
    C_layout = al.make_layout((M, N), (N, al.convert(1, al.i32)))
    C = al.make_tensor(C_ptr, al.bf16, C_layout)

    block_m = al.block_id(0)
    block_n = al.block_id(1)
    tid = al.thread_id(0)

    wave_id = tid // 64
    lane_id = tid - wave_id * 64
    warp_row = wave_id // 2
    warp_col = wave_id - warp_row * 2

    A_lds = al.make_shared((1024,), al.u32)
    B_lds = al.make_shared((1024,), al.u32)

    two = al.convert(2, al.i32)
    A_rsrc = al.amdgpu.make_rsrc(A, M * K_dim * two)
    B_rsrc = al.amdgpu.make_rsrc(B, K_dim * N * two)

    a_row_local = tid // 4
    a_chunk = tid - a_row_local * 4
    global_a_row = block_m * al.convert(64, al.i32) + a_row_local
    a_byte_off = (global_a_row * K_dim + a_chunk * al.convert(8, al.i32)) * two
    a_data = al.amdgpu.raw_buffer_load_x4(A_rsrc, a_byte_off, 0, 0)
    a_data_vec = al.view(a_data, al.Tensor((4,), al.u32))
    a_base = tid * al.convert(4, al.i32)
    for i in al.range(4):
        A_lds[a_base + i] = a_data_vec[i]

    b_row_local = tid // 8
    b_col_local = (tid - b_row_local * 8) * al.convert(8, al.i32)
    global_b_k = b_row_local
    global_b_n = block_n * al.convert(64, al.i32) + b_col_local
    b_byte_off = (global_b_k * N + global_b_n) * two
    b_data = al.amdgpu.raw_buffer_load_x4(B_rsrc, b_byte_off, 0, 0)
    b_data_vec = al.view(b_data, al.Tensor((4,), al.u32))
    b_base = tid * al.convert(4, al.i32)
    for i in al.range(4):
        B_lds[b_base + i] = b_data_vec[i]

    al.syncthreads()

    A_lds_view = al.view(A_lds, al.u32, al.make_layout((64, 8, 2), (16, 2, 1)))
    B_lds_view = al.view(B_lds, al.u32, al.make_layout((32, 16, 2), (32, 2, 1)))

    acc = al.make_local((16,), al.f32)
    for i in al.range(16):
        acc[i] = al.convert(0.0, al.f32)

    c32 = al.convert(32, al.i32)
    c64 = al.convert(64, al.i32)
    c8 = al.convert(8, al.i32)
    c4 = al.convert(4, al.i32)
    c2 = al.convert(2, al.i32)

    for k_step in al.range(4):
        a_row_l = warp_row * c32 + (lane_id % c32)
        a_k_chunk = k_step * c2 + (lane_id // c32)
        a_frag = A_lds_view[a_row_l, a_k_chunk]

        b_k_row = k_step * c8 + (lane_id // c8)
        b_n_chunk = warp_col * c8 + (lane_id % c8)
        b_frag = B_lds_view[b_k_row, b_n_chunk]

        acc = al.amdgpu.mfma_32x32x8_bf16_f32(a_frag, b_frag, acc)

    r0 = (lane_id % c8) * c4
    c0_col = (lane_id // c8) * c4

    for i in al.range(16):
        out_row = block_m * c64 + warp_row * c32 + r0 + (i // c4)
        out_col = block_n * c64 + warp_col * c32 + c0_col + (i % c4)
        C[out_row, out_col] = al.convert(acc[i], al.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        M_val, K_val = A.shape
        K_val2, N_val = B.shape

        if M_val != 32768 or K_val != 32:
            raise RuntimeError("Expected A (32768, 32)")
        if K_val2 != 32 or N_val != 32768:
            raise RuntimeError("Expected B (32, 32768)")

        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((M_val, N_val), device=A.device, dtype=A.dtype)

        grid_m = (M_val + 63) // 64
        grid_n = (N_val + 63) // 64

        gemm_kernel[lambda: ((grid_m, grid_n, 1), (256, 1, 1))](
            A, B, C,
            M_val,
            K_val,
            N_val,
        )
        return C
