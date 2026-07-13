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
    N: al.i32,
    K: al.i32,
    stride_a: al.i32,
    stride_b: al.i32,
    stride_c: al.i32,
):
    tid = al.thread_id(0)
    bid_m = al.block_id(0)
    bid_n = al.block_id(1)

    lane_id = tid % 64
    wave_id = tid // 64
    wave_m = wave_id // 2
    wave_n = wave_id % 2

    block_m_start = bid_m * 64
    block_n_start = bid_n * 64
    wave_m_start = wave_m * 32
    wave_n_start = wave_n * 32

    A = al.make_tensor(A_ptr, al.bf16, al.make_layout((M, K), (stride_a, 1)))
    B = al.make_tensor(B_ptr, al.bf16, al.make_layout((K, N), (stride_b, 1)))
    C = al.make_tensor(C_ptr, al.bf16, al.make_layout((M, N), (stride_c, 1)))

    A_rsrc = al.amdgpu.make_rsrc(A, M * K * 2)
    B_rsrc = al.amdgpu.make_rsrc(B, K * N * 2)

    A_lds = al.make_shared((64, 32), al.bf16)
    B_lds = al.make_shared((32, 64), al.bf16)

    acc = al.make_local((16,), al.f32)
    for i in al.range(16):
        acc[i] = al.convert(0.0, al.f32)

    a_row_mfma = wave_m_start + lane_id % 32
    a_col4_mfma = (lane_id // 32) * 4
    b_col_mfma = wave_n_start + lane_id % 32
    b_row_base_mfma = (lane_id // 32) * 4

    for k_block in al.range(0, K, 32):
        a_row = tid // 4
        a_chunk = tid % 4
        a_global_row = block_m_start + a_row
        a_global_k = k_block + a_chunk * 8
        a_offset = (a_global_row * stride_a + a_global_k) * 2
        a_data = al.amdgpu.raw_buffer_load_x4(A_rsrc, a_offset, 0, 0)
        a_bf16 = al.view(a_data, al.Tensor((8,), al.bf16))
        for j in al.range(8):
            A_lds[a_row, a_chunk * 8 + j] = a_bf16[j]

        b_k = tid // 8
        b_n_chunk = tid % 8
        b_global_k = k_block + b_k
        b_global_n = block_n_start + b_n_chunk * 8
        b_offset = (b_global_k * stride_b + b_global_n) * 2
        b_data = al.amdgpu.raw_buffer_load_x4(B_rsrc, b_offset, 0, 0)
        b_bf16 = al.view(b_data, al.Tensor((8,), al.bf16))
        for j in al.range(8):
            B_lds[b_k, b_n_chunk * 8 + j] = b_bf16[j]

        al.syncthreads()

        # MFMA step 0: K [0, 8)
        a0 = al.make_local((4,), al.bf16)
        for i in al.range(4):
            a0[i] = A_lds[a_row_mfma, a_col4_mfma + i]
        b0 = al.make_local((4,), al.bf16)
        for i in al.range(4):
            b0[i] = B_lds[b_row_base_mfma + i, b_col_mfma]
        a0p = al.view(a0, al.Tensor((2,), al.i32))
        b0p = al.view(b0, al.Tensor((2,), al.i32))
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(a0p, b0p, acc)

        # MFMA step 1: K [8, 16)
        a1 = al.make_local((4,), al.bf16)
        for i in al.range(4):
            a1[i] = A_lds[a_row_mfma, 8 + a_col4_mfma + i]
        b1 = al.make_local((4,), al.bf16)
        for i in al.range(4):
            b1[i] = B_lds[8 + b_row_base_mfma + i, b_col_mfma]
        a1p = al.view(a1, al.Tensor((2,), al.i32))
        b1p = al.view(b1, al.Tensor((2,), al.i32))
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(a1p, b1p, acc)

        # MFMA step 2: K [16, 24)
        a2 = al.make_local((4,), al.bf16)
        for i in al.range(4):
            a2[i] = A_lds[a_row_mfma, 16 + a_col4_mfma + i]
        b2 = al.make_local((4,), al.bf16)
        for i in al.range(4):
            b2[i] = B_lds[16 + b_row_base_mfma + i, b_col_mfma]
        a2p = al.view(a2, al.Tensor((2,), al.i32))
        b2p = al.view(b2, al.Tensor((2,), al.i32))
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(a2p, b2p, acc)

        # MFMA step 3: K [24, 32)
        a3 = al.make_local((4,), al.bf16)
        for i in al.range(4):
            a3[i] = A_lds[a_row_mfma, 24 + a_col4_mfma + i]
        b3 = al.make_local((4,), al.bf16)
        for i in al.range(4):
            b3[i] = B_lds[24 + b_row_base_mfma + i, b_col_mfma]
        a3p = al.view(a3, al.Tensor((2,), al.i32))
        b3p = al.view(b3, al.Tensor((2,), al.i32))
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(a3p, b3p, acc)

        al.syncthreads()

    c_col = block_n_start + wave_n_start + lane_id % 32
    c_row_base = block_m_start + wave_m_start
    lane_hi = lane_id // 32

    for acc_idx in al.range(16):
        c_row = c_row_base + 8 * (acc_idx // 4) + 4 * lane_hi + (acc_idx % 4)
        C[c_row, c_col] = al.convert(acc[acc_idx], al.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        A2 = A.contiguous()
        B2 = B.transpose(-2, -1).contiguous()
        M_val, K_val = A2.shape
        K_val2, N_val = B2.shape

        C = torch.empty((M_val, N_val), device=A.device, dtype=A.dtype)

        grid_m = (M_val + 63) // 64
        grid_n = (N_val + 63) // 64

        gemm_kernel[lambda: ((grid_m, grid_n, 1), (256, 1, 1))](
            A2.data_ptr(),
            B2.data_ptr(),
            C.data_ptr(),
            M_val,
            N_val,
            K_val,
            K_val,
            N_val,
            N_val,
        )
        return C
