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
    # Row-major tensor views
    layout_a = al.make_layout((M, K), (stride_a, al.convert(1, al.i32)))
    A = al.make_tensor(A_ptr, al.bf16, layout_a)
    layout_b = al.make_layout((K, N), (stride_b, al.convert(1, al.i32)))
    B = al.make_tensor(B_ptr, al.bf16, layout_b)
    layout_c = al.make_layout((M, N), (stride_c, al.convert(1, al.i32)))
    C = al.make_tensor(C_ptr, al.bf16, layout_c)

    # 256 threads, block tile 64x64, K tile 32
    tid = al.thread_id(0)
    bid_m = al.block_id(1)
    bid_n = al.block_id(0)

    bm_off = bid_m * al.convert(64, al.i32)
    bn_off = bid_n * al.convert(64, al.i32)

    # Warp identification (4 warps, 2x2 grid, each 32x32 output)
    wy = al.convert(0, al.i32)
    wx = al.convert(0, al.i32)
    tloc = tid
    if tid >= al.convert(192, al.i32):
        wy = al.convert(1, al.i32)
        wx = al.convert(1, al.i32)
        tloc = tid - al.convert(192, al.i32)
    else:
        if tid >= al.convert(128, al.i32):
            wy = al.convert(1, al.i32)
            wx = al.convert(0, al.i32)
            tloc = tid - al.convert(128, al.i32)
        else:
            if tid >= al.convert(64, al.i32):
                wy = al.convert(0, al.i32)
                wx = al.convert(1, al.i32)
                tloc = tid - al.convert(64, al.i32)

    # Thread-to-4x4-subtile mapping (tloc//8, tloc%8)
    thr = al.convert(0, al.i32)
    thc = tloc
    if tloc >= al.convert(56, al.i32):
        thr = al.convert(7, al.i32)
        thc = tloc - al.convert(56, al.i32)
    else:
        if tloc >= al.convert(48, al.i32):
            thr = al.convert(6, al.i32)
            thc = tloc - al.convert(48, al.i32)
        else:
            if tloc >= al.convert(40, al.i32):
                thr = al.convert(5, al.i32)
                thc = tloc - al.convert(40, al.i32)
            else:
                if tloc >= al.convert(32, al.i32):
                    thr = al.convert(4, al.i32)
                    thc = tloc - al.convert(32, al.i32)
                else:
                    if tloc >= al.convert(24, al.i32):
                        thr = al.convert(3, al.i32)
                        thc = tloc - al.convert(24, al.i32)
                    else:
                        if tloc >= al.convert(16, al.i32):
                            thr = al.convert(2, al.i32)
                            thc = tloc - al.convert(16, al.i32)
                        else:
                            if tloc >= al.convert(8, al.i32):
                                thr = al.convert(1, al.i32)
                                thc = tloc - al.convert(8, al.i32)

    th_row = wy * al.convert(32, al.i32) + thr * al.convert(4, al.i32)
    th_col = wx * al.convert(32, al.i32) + thc * al.convert(4, al.i32)

    # Accumulator scalars: 16 f32 per thread
    acc00 = al.convert(0.0, al.f32)
    acc01 = al.convert(0.0, al.f32)
    acc02 = al.convert(0.0, al.f32)
    acc03 = al.convert(0.0, al.f32)
    acc10 = al.convert(0.0, al.f32)
    acc11 = al.convert(0.0, al.f32)
    acc12 = al.convert(0.0, al.f32)
    acc13 = al.convert(0.0, al.f32)
    acc20 = al.convert(0.0, al.f32)
    acc21 = al.convert(0.0, al.f32)
    acc22 = al.convert(0.0, al.f32)
    acc23 = al.convert(0.0, al.f32)
    acc30 = al.convert(0.0, al.f32)
    acc31 = al.convert(0.0, al.f32)
    acc32 = al.convert(0.0, al.f32)
    acc33 = al.convert(0.0, al.f32)

    # LDS tiles: A(64,32), B(32,64) — 256 chunks each, perfect for 256 threads
    a_lds = al.make_shared((64, 32), al.bf16)
    b_lds = al.make_shared((32, 64), al.bf16)

    # Buffer resources
    a_rsrc = al.amdgpu.make_rsrc(A, M * K * al.convert(2, al.i32))
    b_rsrc = al.amdgpu.make_rsrc(B, K * N * al.convert(2, al.i32))

    # Main K loop: step by 32
    for k_block in al.range(0, K, 32):
        # ---- Load A tile (64x32) into LDS ----
        # 256 chunks = 64 rows * 4 chunks/row. tid → row=tid//4, chunk=tid%4
        t2 = tid
        row_a = al.convert(0, al.i32)
        if t2 >= al.convert(128, al.i32):
            row_a = al.convert(32, al.i32)
            t2 = t2 - al.convert(128, al.i32)
        if t2 >= al.convert(64, al.i32):
            row_a = row_a + al.convert(16, al.i32)
            t2 = t2 - al.convert(64, al.i32)
        if t2 >= al.convert(32, al.i32):
            row_a = row_a + al.convert(8, al.i32)
            t2 = t2 - al.convert(32, al.i32)
        if t2 >= al.convert(16, al.i32):
            row_a = row_a + al.convert(4, al.i32)
            t2 = t2 - al.convert(16, al.i32)
        if t2 >= al.convert(8, al.i32):
            row_a = row_a + al.convert(2, al.i32)
            t2 = t2 - al.convert(8, al.i32)
        if t2 >= al.convert(4, al.i32):
            row_a = row_a + al.convert(1, al.i32)
            t2 = t2 - al.convert(4, al.i32)
        chunk_a = t2

        ga_row = bm_off + row_a
        ga_col = k_block + chunk_a * al.convert(8, al.i32)
        a_off = (ga_row * stride_a + ga_col) * al.convert(2, al.i32)
        a_data = al.amdgpu.raw_buffer_load_x4(a_rsrc, a_off, al.convert(0, al.i32), al.convert(0, al.i32))
        a_bf16 = al.view(a_data, al.Tensor((8,), al.bf16))
        a_base = chunk_a * al.convert(8, al.i32)
        col_a = a_base
        for i in al.range(8):
            a_lds[row_a, col_a] = a_bf16[i]
            col_a = col_a + al.convert(1, al.i32)

        # ---- Load B tile (32x64) into LDS ----
        # 256 chunks = 32 rows * 8 chunks/row. tid → row=tid//8, chunk=tid%8
        tb = tid
        row_b = al.convert(0, al.i32)
        if tb >= al.convert(128, al.i32):
            row_b = al.convert(16, al.i32)
            tb = tb - al.convert(128, al.i32)
        if tb >= al.convert(64, al.i32):
            row_b = row_b + al.convert(8, al.i32)
            tb = tb - al.convert(64, al.i32)
        if tb >= al.convert(32, al.i32):
            row_b = row_b + al.convert(4, al.i32)
            tb = tb - al.convert(32, al.i32)
        if tb >= al.convert(16, al.i32):
            row_b = row_b + al.convert(2, al.i32)
            tb = tb - al.convert(16, al.i32)
        if tb >= al.convert(8, al.i32):
            row_b = row_b + al.convert(1, al.i32)
            tb = tb - al.convert(8, al.i32)
        ch_b = tb

        gb_row = k_block + row_b
        gb_col = bn_off + ch_b * al.convert(8, al.i32)
        b_off = (gb_row * stride_b + gb_col) * al.convert(2, al.i32)
        b_data = al.amdgpu.raw_buffer_load_x4(b_rsrc, b_off, al.convert(0, al.i32), al.convert(0, al.i32))
        b_bf16 = al.view(b_data, al.Tensor((8,), al.bf16))
        b_base = ch_b * al.convert(8, al.i32)
        col_b = b_base
        for i in al.range(8):
            b_lds[row_b, col_b] = b_bf16[i]
            col_b = col_b + al.convert(1, al.i32)

        al.syncthreads()

        # ---- Compute: unrolled 4x4 MACs across 32 K values ----
        r0 = th_row
        r1 = th_row + al.convert(1, al.i32)
        r2 = th_row + al.convert(2, al.i32)
        r3 = th_row + al.convert(3, al.i32)
        c0 = th_col
        c1 = th_col + al.convert(1, al.i32)
        c2 = th_col + al.convert(2, al.i32)
        c3 = th_col + al.convert(3, al.i32)
        for kk in al.range(32):
            a0 = al.convert(a_lds[r0, kk], al.f32)
            a1 = al.convert(a_lds[r1, kk], al.f32)
            a2 = al.convert(a_lds[r2, kk], al.f32)
            a3 = al.convert(a_lds[r3, kk], al.f32)
            b0 = al.convert(b_lds[kk, c0], al.f32)
            b1 = al.convert(b_lds[kk, c1], al.f32)
            b2 = al.convert(b_lds[kk, c2], al.f32)
            b3 = al.convert(b_lds[kk, c3], al.f32)
            acc00 = acc00 + a0 * b0
            acc01 = acc01 + a0 * b1
            acc02 = acc02 + a0 * b2
            acc03 = acc03 + a0 * b3
            acc10 = acc10 + a1 * b0
            acc11 = acc11 + a1 * b1
            acc12 = acc12 + a1 * b2
            acc13 = acc13 + a1 * b3
            acc20 = acc20 + a2 * b0
            acc21 = acc21 + a2 * b1
            acc22 = acc22 + a2 * b2
            acc23 = acc23 + a2 * b3
            acc30 = acc30 + a3 * b0
            acc31 = acc31 + a3 * b1
            acc32 = acc32 + a3 * b2
            acc33 = acc33 + a3 * b3

        al.syncthreads()

    # ---- Writeback ----
    r0 = bm_off + th_row
    r1 = r0 + al.convert(1, al.i32)
    r2 = r0 + al.convert(2, al.i32)
    r3 = r0 + al.convert(3, al.i32)
    c0 = bn_off + th_col
    c1 = c0 + al.convert(1, al.i32)
    c2 = c0 + al.convert(2, al.i32)
    c3 = c0 + al.convert(3, al.i32)
    C[r0, c0] = al.convert(acc00, al.bf16)
    C[r0, c1] = al.convert(acc01, al.bf16)
    C[r0, c2] = al.convert(acc02, al.bf16)
    C[r0, c3] = al.convert(acc03, al.bf16)
    C[r1, c0] = al.convert(acc10, al.bf16)
    C[r1, c1] = al.convert(acc11, al.bf16)
    C[r1, c2] = al.convert(acc12, al.bf16)
    C[r1, c3] = al.convert(acc13, al.bf16)
    C[r2, c0] = al.convert(acc20, al.bf16)
    C[r2, c1] = al.convert(acc21, al.bf16)
    C[r2, c2] = al.convert(acc22, al.bf16)
    C[r2, c3] = al.convert(acc23, al.bf16)
    C[r3, c0] = al.convert(acc30, al.bf16)
    C[r3, c1] = al.convert(acc31, al.bf16)
    C[r3, c2] = al.convert(acc32, al.bf16)
    C[r3, c3] = al.convert(acc33, al.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        # Reference semantics: torch.matmul(A.T, B.T)
        A2 = A.transpose(-2, -1).contiguous()
        B2 = B.transpose(-2, -1).contiguous()

        m = A2.shape[0]
        n = B2.shape[1]
        k = A2.shape[1]

        C = torch.empty((m, n), device=A.device, dtype=A.dtype)

        grid = (
            (n + 64 - 1) // 64,
            (m + 64 - 1) // 64,
            1,
        )

        gemm_kernel[lambda: (grid, (256, 1, 1))](
            A2, B2, C,
            m, n, k,
            A2.stride(0), B2.stride(0), C.stride(0),
        )
        return C
