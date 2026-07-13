import torch
import torch.nn as nn

import avelang
import avelang.language as al


@avelang.jit
def tri_gemm_kernel(
    A_ptr: al.Pointer(al.bf16),
    B_ptr: al.Pointer(al.bf16),
    C_ptr: al.Pointer(al.bf16),
    M: al.i32,
    K_dim: al.i32,
    N: al.i32,
):
    BM = al.convert(64, al.i32)
    BK = al.convert(16, al.i32)
    WM = al.convert(32, al.i32)
    WARP_SIZE = al.convert(64, al.i32)
    two = al.convert(2, al.i32)
    four = al.convert(4, al.i32)
    eight = al.convert(8, al.i32)

    A = al.make_tensor(A_ptr, al.bf16, al.make_layout((M, K_dim), (K_dim, 1)))
    B = al.make_tensor(B_ptr, al.bf16, al.make_layout((K_dim, N), (N, 1)))
    C = al.make_tensor(C_ptr, al.bf16, al.make_layout((M, N), (N, 1)))

    A_rsrc = al.amdgpu.make_rsrc(A, M * K_dim * two)
    B_rsrc = al.amdgpu.make_rsrc(B, K_dim * N * two)

    tid = al.thread_id(0)
    lane_id = tid % WARP_SIZE
    warp_id = tid / WARP_SIZE
    warp_row = warp_id / two
    warp_col = warp_id % two

    block_m = al.block_id(0) * BM
    block_n = al.block_id(1) * BM

    if block_n + BM <= block_m:
        return

    wave_m = block_m + warp_row * WM
    wave_n = block_n + warp_col * WM

    As_bf16 = al.make_shared((128, BK), al.bf16)
    Bs_bf16 = al.make_shared((64, WM), al.bf16)
    As_u32 = al.view(As_bf16, al.u32, al.make_layout((128, BK // 2), (BK // 2, 1)))

    as_base = warp_id * WM
    bs_base = warp_id * BK

    acc = al.make_local((16,), al.f32)
    for ai in al.range(16):
        acc[ai] = al.convert(0.0, al.f32)

    lane_mod_wm = lane_id % WM
    lane_div_wm = lane_id / WM
    lane_mod_bk = lane_id % BK
    lane_div_bk = lane_id / BK
    lane_mod_8  = lane_id % eight

    for k in al.range(0, K_dim, BK):
        # A: interleaved store [0,2,1,3,4,6,5,7,8,10,9,11,12,14,13,15]
        a_row = wave_m + lane_mod_wm
        a_col = k + lane_div_wm * eight
        a_off = (a_row * K_dim + a_col) * two
        a_packed = al.amdgpu.raw_buffer_load_x4(A_rsrc, a_off, 0, 0)
        a_bf16 = al.view(a_packed, al.Tensor((8,), al.bf16))
        for e in al.range(8):
            le = e
            pe = (le % four) * two + (le / four)
            As_bf16[as_base + lane_mod_wm, lane_div_wm * eight + pe] = a_bf16[e]

        # B: row-major
        b_row = k + lane_mod_bk
        b_col = wave_n + lane_div_bk * eight
        b_off = ((b_row * N) + b_col) * two
        b_packed = al.amdgpu.raw_buffer_load_x4(B_rsrc, b_off, 0, 0)
        b_bf16 = al.view(b_packed, al.Tensor((8,), al.bf16))
        for e in al.range(8):
            Bs_bf16[bs_base + lane_mod_bk, lane_div_bk * eight + e] = b_bf16[e]

        al.syncthreads()

        a_col_u32 = lane_div_wm * two
        a0 = al.make_local((2,), al.u32)
        a1 = al.make_local((2,), al.u32)
        for u in al.range(2):
            a0[u] = As_u32[as_base + lane_mod_wm, a_col_u32 + u]
            a1[u] = As_u32[as_base + lane_mod_wm, a_col_u32 + four + u]

        b_n0 = (lane_div_wm * four) + (lane_mod_8 % four)
        b4_0 = al.make_local((4,), al.bf16)
        b4_0[0] = Bs_bf16[bs_base + lane_mod_8, b_n0 + 0]
        b4_0[1] = Bs_bf16[bs_base + lane_mod_8, b_n0 + 8]
        b4_0[2] = Bs_bf16[bs_base + lane_mod_8, b_n0 + 16]
        b4_0[3] = Bs_bf16[bs_base + lane_mod_8, b_n0 + 24]
        b0 = al.view(b4_0, al.Tensor((2,), al.u32))

        b4_1 = al.make_local((4,), al.bf16)
        b4_1[0] = Bs_bf16[bs_base + eight + lane_mod_8, b_n0 + 0]
        b4_1[1] = Bs_bf16[bs_base + eight + lane_mod_8, b_n0 + 8]
        b4_1[2] = Bs_bf16[bs_base + eight + lane_mod_8, b_n0 + 16]
        b4_1[3] = Bs_bf16[bs_base + eight + lane_mod_8, b_n0 + 24]
        b1 = al.view(b4_1, al.Tensor((2,), al.u32))

        acc = al.amdgpu.mfma_32x32x8_bf16_f32(a0, b0, acc)
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(a1, b1, acc)

        al.syncthreads()

    c_col = wave_n + lane_mod_wm
    lane_half = lane_id / WM

    for idx in al.range(16):
        c_row = wave_m + eight * (idx / four) + four * lane_half + (idx % four)
        if c_col >= c_row:
            C[c_row, c_col] = al.convert(acc[idx], al.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if tuple(A.shape) != (4096, 4096) or tuple(B.shape) != (4096, 4096):
            raise RuntimeError("Expected (4096, 4096) inputs")
        A = A.contiguous(); B = B.contiguous()
        M_val = A.shape[0]; K_val = A.shape[1]; N_val = B.shape[1]
        C = torch.zeros((M_val, N_val), device=A.device, dtype=torch.bfloat16)
        tri_gemm_kernel[lambda: ((64, 64, 1), (256, 1, 1))](
            A.data_ptr(), B.data_ptr(), C.data_ptr(),
            M_val, K_val, N_val, num_warps=4)
        return C
