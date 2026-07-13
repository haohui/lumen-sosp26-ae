import torch
import torch.nn as nn

import avelang
import avelang.language as al


@avelang.jit
def diag_left_gemm_kernel(
    A_ptr: al.Pointer(al.bf16),
    B_ptr: al.Pointer(al.bf16),
    C_ptr: al.Pointer(al.bf16),
    N: al.i32,
    M: al.i32,
    BLOCK_M: al.constexpr,
    BLOCK_N: al.constexpr,
    BLOCK_K: al.constexpr,
):
    # Tensor views
    A_bf16 = al.make_tensor(A_ptr, al.bf16, al.make_layout((N,), (1,)))
    B_bf16 = al.make_tensor(B_ptr, al.bf16, al.make_layout((N, M), (M, 1)))
    C_bf16 = al.make_tensor(C_ptr, al.bf16, al.make_layout((N, M), (M, 1)))

    # Grid / lane / warp identification
    tid = al.thread_id(0)
    lane = tid & 63
    lane_col = lane & 31
    lane_group = lane >> 5

    warp_id = tid >> 6
    warp_m = warp_id >> 1
    warp_n = warp_id & 1

    block_m = al.block_id(1) * (BLOCK_M * 2)
    block_n = al.block_id(0) * (BLOCK_N * 2)

    m_base = block_m + warp_m * BLOCK_M
    n_base = block_n + warp_n * BLOCK_N

    # LDS tiles per warp
    a_smem_all = al.make_shared((4 * BLOCK_M * (BLOCK_K >> 3), BLOCK_K >> 2), al.i32)
    b_smem_all = al.make_shared((4 * BLOCK_N * (BLOCK_K >> 3), BLOCK_K >> 2), al.i32)
    c_smem_all = al.make_shared((4 * BLOCK_M, BLOCK_N), al.f32)

    a_smem = al.subview(
        a_smem_all,
        (warp_id * (BLOCK_M * (BLOCK_K >> 3)), 0),
        (BLOCK_M * (BLOCK_K >> 3), BLOCK_K >> 2),
        (1, 1),
    )
    b_smem = al.subview(
        b_smem_all,
        (warp_id * (BLOCK_N * (BLOCK_K >> 3)), 0),
        (BLOCK_N * (BLOCK_K >> 3), BLOCK_K >> 2),
        (1, 1),
    )
    c_smem = al.subview(
        c_smem_all,
        (warp_id * BLOCK_M, 0),
        (BLOCK_M, BLOCK_N),
        (1, 1),
    )

    acc = al.full((16,), 0.0, al.f32)

    s16 = al.convert(16, al.u32)
    half_k = BLOCK_K >> 1

    # Bounded K loop: only iterate over K steps intersecting the diagonal
    kt_start = m_base >> 4
    kt_end = (m_base + BLOCK_M - 1) >> 4
    max_kt = (N - 1) >> 4
    if kt_end > max_kt:
        kt_end = max_kt

    for kt in al.range(kt_start, kt_end + 1):
        k_base = kt * BLOCK_K

        # ---- Load B tile ----
        k_start = k_base + lane_group * half_k
        b_words = al.make_local((4,), al.i32)
        for bi in al.range(4):
            k0 = k_start + bi * 2
            k1 = k0 + 1
            b0 = B_bf16[k0, n_base + lane_col]
            b1 = B_bf16[k1, n_base + lane_col]
            u0 = al.convert(al.bitcast(b0, al.u16), al.u32)
            u1 = al.convert(al.bitcast(b1, al.u16), al.u32)
            b_words[bi] = al.convert(u0 + (u1 << s16), al.i32)
        b_smem[lane] = b_words

        # ---- Load A tile (diagonal) ----
        g_row = m_base + lane_col
        diag_kpos = g_row - (k_base + lane_group * half_k)

        a_words = al.make_local((4,), al.i32)
        for i in al.range(4):
            a_words[i] = al.convert(0, al.i32)

        if diag_kpos >= 0:
            if diag_kpos < half_k:
                i32_idx = diag_kpos >> 1
                half = diag_kpos & 1
                a_val = A_bf16[g_row]
                u0 = al.convert(al.bitcast(a_val, al.u16), al.u32)
                if half == 0:
                    a_words[i32_idx] = al.convert(u0, al.i32)
                else:
                    a_words[i32_idx] = al.convert(u0 << s16, al.i32)

        a_smem[lane] = a_words

        al.syncthreads()

        # ---- MFMA ----
        a_loaded = a_smem[lane]
        b_loaded = b_smem[lane]
        a_frag = al.view(a_loaded, al.Tensor((2, 2, 1), al.u32))
        b_frag = al.view(b_loaded, al.Tensor((2, 2, 1), al.u32))

        acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag[0], a_frag[0], acc)
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag[1], a_frag[1], acc)

        al.syncthreads()

    # ---- Store output ----
    for r in al.range(16):
        row_offset = ((r >> 2) << 3) + lane_group * 4 + (r & 3)
        c_smem[lane_col, row_offset] = acc[r]

    al.syncthreads()

    store_row = lane >> 1
    store_idx = lane & 1
    for block_idx in al.range(4):
        base_col = block_idx * 8 + store_idx * 4
        for v in al.range(4):
            col = base_col + v
            C_bf16[m_base + store_row, n_base + col] = al.convert(
                c_smem[store_row, col], al.bf16
            )


def _make_avelang_gemm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    N_val = A.shape[0]
    M_val = B.shape[1]
    A = A.contiguous()
    B = B.contiguous()
    C = torch.empty((N_val, M_val), device=B.device, dtype=torch.bfloat16)
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 16
    grid_m = (N_val + BLOCK_M * 2 - 1) // (BLOCK_M * 2)
    grid_n = (M_val + BLOCK_N * 2 - 1) // (BLOCK_N * 2)
    diag_left_gemm_kernel[lambda: ((grid_n, grid_m, 1), (256, 1, 1))](
        A, B, C, N_val, M_val, BLOCK_M, BLOCK_N, BLOCK_K,
    )
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
            A = A.to(torch.bfloat16)
            B = B.to(torch.bfloat16)
        return _make_avelang_gemm(A, B)
