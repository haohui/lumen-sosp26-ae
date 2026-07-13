import torch
import torch.nn as nn
import avelang
import avelang.language as al

M = 2048
K = 8192
N = 4096

BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 16


@avelang.jit
def gemm_kernel(
    A_ptr: al.Pointer(al.bf16),
    B_ptr: al.Pointer(al.bf16),
    C_ptr: al.Pointer(al.bf16),
    stride_ak: al.i32,
    stride_bn: al.i32,
    stride_cn: al.i32,
):
    A = al.make_tensor(A_ptr, al.bf16, al.make_layout((M, stride_ak), (stride_ak, 1)))
    B = al.make_tensor(B_ptr, al.bf16, al.make_layout((stride_ak, stride_bn), (stride_bn, 1)))
    C = al.make_tensor(C_ptr, al.bf16, al.make_layout((M, stride_cn), (stride_cn, 1)))

    block_m = al.block_id(0)
    block_n = al.block_id(1)

    tid = al.thread_id(0)
    wave_id = tid // 64
    lane_id = tid % 64
    wave_row = wave_id // 2
    wave_col = wave_id % 2

    thr_row = lane_id // 8
    thr_col = lane_id % 8

    m_base = block_m * BLOCK_M + wave_row * 32
    n_base = block_n * BLOCK_N + wave_col * 32

    A_lds = al.make_shared((BLOCK_M, BLOCK_K), al.bf16)
    B_lds = al.make_shared((BLOCK_K, BLOCK_N), al.bf16)

    A_rsrc = al.amdgpu.make_rsrc(A, M * stride_ak * 2)
    B_rsrc = al.amdgpu.make_rsrc(B, stride_ak * stride_bn * 2)

    acc = al.make_local((16,), al.f32)
    for i in al.range(16):
        acc[i] = al.convert(0.0, al.f32)

    for k_block in al.range(0, stride_ak, BLOCK_K):
        if tid < 128:
            ld_row = (tid // 4) * 2
            ld_col = (tid % 4) * 4
            g_row = block_m * BLOCK_M + ld_row
            g_col = k_block + ld_col
            byte_off = al.convert((g_row * stride_ak + g_col) * 2, al.i32)
            data = al.amdgpu.raw_buffer_load_x4(A_rsrc, byte_off, 0, 0)
            d = al.view(data, al.Tensor((8,), al.bf16))
            A_lds[ld_row + 0, ld_col + 0] = d[0]
            A_lds[ld_row + 0, ld_col + 1] = d[1]
            A_lds[ld_row + 0, ld_col + 2] = d[2]
            A_lds[ld_row + 0, ld_col + 3] = d[3]
            A_lds[ld_row + 1, ld_col + 0] = d[4]
            A_lds[ld_row + 1, ld_col + 1] = d[5]
            A_lds[ld_row + 1, ld_col + 2] = d[6]
            A_lds[ld_row + 1, ld_col + 3] = d[7]
        else:
            t = tid - 128
            ld_row = (t // 16) * 2
            ld_col = (t % 16) * 4
            g_row = k_block + ld_row
            g_col = block_n * BLOCK_N + ld_col
            byte_off = al.convert((g_row * stride_bn + g_col) * 2, al.i32)
            data = al.amdgpu.raw_buffer_load_x4(B_rsrc, byte_off, 0, 0)
            d = al.view(data, al.Tensor((8,), al.bf16))
            B_lds[ld_row + 0, ld_col + 0] = d[0]
            B_lds[ld_row + 0, ld_col + 1] = d[1]
            B_lds[ld_row + 0, ld_col + 2] = d[2]
            B_lds[ld_row + 0, ld_col + 3] = d[3]
            B_lds[ld_row + 1, ld_col + 0] = d[4]
            B_lds[ld_row + 1, ld_col + 1] = d[5]
            B_lds[ld_row + 1, ld_col + 2] = d[6]
            B_lds[ld_row + 1, ld_col + 3] = d[7]

        al.syncthreads()

        a_row = wave_row * 32 + thr_row * 4
        a_k0 = al.make_local((4,), al.bf16)
        a_k0[0] = A_lds[a_row + 0, thr_col]
        a_k0[1] = A_lds[a_row + 1, thr_col]
        a_k0[2] = A_lds[a_row + 2, thr_col]
        a_k0[3] = A_lds[a_row + 3, thr_col]

        a_k8 = al.make_local((4,), al.bf16)
        a_k8[0] = A_lds[a_row + 0, 8 + thr_col]
        a_k8[1] = A_lds[a_row + 1, 8 + thr_col]
        a_k8[2] = A_lds[a_row + 2, 8 + thr_col]
        a_k8[3] = A_lds[a_row + 3, 8 + thr_col]

        b_col = wave_col * 32 + thr_col * 4
        b_k0 = al.make_local((4,), al.bf16)
        b_k0[0] = B_lds[thr_row, b_col + 0]
        b_k0[1] = B_lds[thr_row, b_col + 1]
        b_k0[2] = B_lds[thr_row, b_col + 2]
        b_k0[3] = B_lds[thr_row, b_col + 3]

        b_k8 = al.make_local((4,), al.bf16)
        b_k8[0] = B_lds[8 + thr_row, b_col + 0]
        b_k8[1] = B_lds[8 + thr_row, b_col + 1]
        b_k8[2] = B_lds[8 + thr_row, b_col + 2]
        b_k8[3] = B_lds[8 + thr_row, b_col + 3]

        a_k0_u32 = al.view(a_k0, al.Tensor((2,), al.u32))
        a_k8_u32 = al.view(a_k8, al.Tensor((2,), al.u32))
        b_k0_u32 = al.view(b_k0, al.Tensor((2,), al.u32))
        b_k8_u32 = al.view(b_k8, al.Tensor((2,), al.u32))

        acc = al.amdgpu.mfma_32x32x8_bf16_f32(a_k0_u32, b_k0_u32, acc)
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(a_k8_u32, b_k8_u32, acc)

        al.syncthreads()

    for ri in al.range(4):
        for ci in al.range(4):
            idx = ri * 4 + ci
            c_val = al.convert(acc[idx], al.bf16)
            C[m_base + thr_row * 4 + ri, n_base + thr_col * 4 + ci] = c_val


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA/HIP device."
        M_act = A.shape[0]
        K_act = A.shape[1]
        N_act = B.shape[1]

        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((M_act, N_act), device=A.device, dtype=A.dtype)

        grid_m = M_act // BLOCK_M
        grid_n = N_act // BLOCK_N

        gemm_kernel[lambda: ((grid_m, grid_n, 1), (256, 1, 1))](
            A.data_ptr(),
            B.data_ptr(),
            C.data_ptr(),
            K_act,
            N_act,
            N_act,
        )
        return C
