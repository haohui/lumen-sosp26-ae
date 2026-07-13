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

    A_lds0 = al.make_shared((BLOCK_M, BLOCK_K), al.bf16)
    A_lds1 = al.make_shared((BLOCK_M, BLOCK_K), al.bf16)
    B_lds0 = al.make_shared((BLOCK_K, BLOCK_N), al.bf16)
    B_lds1 = al.make_shared((BLOCK_K, BLOCK_N), al.bf16)

    A_rsrc = al.amdgpu.make_rsrc(A, M * stride_ak * 2)
    B_rsrc = al.amdgpu.make_rsrc(B, stride_ak * stride_bn * 2)

    acc = al.make_local((16,), al.f32)
    for i in al.range(16):
        acc[i] = al.convert(0.0, al.f32)

    # ===== Prologue: load first K-block (k=0) into buffer 0 =====
    if tid < 128:
        ld_row = (tid // 4) * 2
        ld_col = (tid % 4) * 4
        g_row0 = block_m * BLOCK_M + ld_row
        g_row1 = g_row0 + 1
        g_col = ld_col
        bo0 = al.convert((g_row0 * stride_ak + g_col) * 2, al.i32)
        d0 = al.view(al.amdgpu.raw_buffer_load_x2(A_rsrc, bo0, 0, 0), al.Tensor((4,), al.bf16))
        bo1 = al.convert((g_row1 * stride_ak + g_col) * 2, al.i32)
        d1 = al.view(al.amdgpu.raw_buffer_load_x2(A_rsrc, bo1, 0, 0), al.Tensor((4,), al.bf16))
        A_lds0[ld_row + 0, ld_col + 0] = d0[0]
        A_lds0[ld_row + 0, ld_col + 1] = d0[1]
        A_lds0[ld_row + 0, ld_col + 2] = d0[2]
        A_lds0[ld_row + 0, ld_col + 3] = d0[3]
        A_lds0[ld_row + 1, ld_col + 0] = d1[0]
        A_lds0[ld_row + 1, ld_col + 1] = d1[1]
        A_lds0[ld_row + 1, ld_col + 2] = d1[2]
        A_lds0[ld_row + 1, ld_col + 3] = d1[3]
    else:
        t = tid - 128
        ld_row = (t // 16) * 2
        ld_col = (t % 16) * 4
        g_row0 = ld_row
        g_row1 = ld_row + 1
        g_col = block_n * BLOCK_N + ld_col
        bo0 = al.convert((g_row0 * stride_bn + g_col) * 2, al.i32)
        d0 = al.view(al.amdgpu.raw_buffer_load_x2(B_rsrc, bo0, 0, 0), al.Tensor((4,), al.bf16))
        bo1 = al.convert((g_row1 * stride_bn + g_col) * 2, al.i32)
        d1 = al.view(al.amdgpu.raw_buffer_load_x2(B_rsrc, bo1, 0, 0), al.Tensor((4,), al.bf16))
        B_lds0[ld_row + 0, ld_col + 0] = d0[0]
        B_lds0[ld_row + 0, ld_col + 1] = d0[1]
        B_lds0[ld_row + 0, ld_col + 2] = d0[2]
        B_lds0[ld_row + 0, ld_col + 3] = d0[3]
        B_lds0[ld_row + 1, ld_col + 0] = d1[0]
        B_lds0[ld_row + 1, ld_col + 1] = d1[1]
        B_lds0[ld_row + 1, ld_col + 2] = d1[2]
        B_lds0[ld_row + 1, ld_col + 3] = d1[3]

    al.syncthreads()

    buf = 0

    # ===== Main loop: double-buffered, K unrolled by 2 =====
    for k_block in al.range(BLOCK_K, stride_ak, 2 * BLOCK_K):
        nxt = 1 - buf
        k1 = k_block

        # --- Half 1: load k1 into nxt, compute from buf ---
        if tid < 128:
            ld_row = (tid // 4) * 2
            ld_col = (tid % 4) * 4
            g_row0 = block_m * BLOCK_M + ld_row
            g_row1 = g_row0 + 1
            g_col = k1 + ld_col
            bo0 = al.convert((g_row0 * stride_ak + g_col) * 2, al.i32)
            d0 = al.view(al.amdgpu.raw_buffer_load_x2(A_rsrc, bo0, 0, 0), al.Tensor((4,), al.bf16))
            bo1 = al.convert((g_row1 * stride_ak + g_col) * 2, al.i32)
            d1 = al.view(al.amdgpu.raw_buffer_load_x2(A_rsrc, bo1, 0, 0), al.Tensor((4,), al.bf16))
            if nxt == 0:
                A_lds0[ld_row + 0, ld_col + 0] = d0[0]
                A_lds0[ld_row + 0, ld_col + 1] = d0[1]
                A_lds0[ld_row + 0, ld_col + 2] = d0[2]
                A_lds0[ld_row + 0, ld_col + 3] = d0[3]
                A_lds0[ld_row + 1, ld_col + 0] = d1[0]
                A_lds0[ld_row + 1, ld_col + 1] = d1[1]
                A_lds0[ld_row + 1, ld_col + 2] = d1[2]
                A_lds0[ld_row + 1, ld_col + 3] = d1[3]
            else:
                A_lds1[ld_row + 0, ld_col + 0] = d0[0]
                A_lds1[ld_row + 0, ld_col + 1] = d0[1]
                A_lds1[ld_row + 0, ld_col + 2] = d0[2]
                A_lds1[ld_row + 0, ld_col + 3] = d0[3]
                A_lds1[ld_row + 1, ld_col + 0] = d1[0]
                A_lds1[ld_row + 1, ld_col + 1] = d1[1]
                A_lds1[ld_row + 1, ld_col + 2] = d1[2]
                A_lds1[ld_row + 1, ld_col + 3] = d1[3]
        else:
            t = tid - 128
            ld_row = (t // 16) * 2
            ld_col = (t % 16) * 4
            g_row0 = k1 + ld_row
            g_row1 = g_row0 + 1
            g_col = block_n * BLOCK_N + ld_col
            bo0 = al.convert((g_row0 * stride_bn + g_col) * 2, al.i32)
            d0 = al.view(al.amdgpu.raw_buffer_load_x2(B_rsrc, bo0, 0, 0), al.Tensor((4,), al.bf16))
            bo1 = al.convert((g_row1 * stride_bn + g_col) * 2, al.i32)
            d1 = al.view(al.amdgpu.raw_buffer_load_x2(B_rsrc, bo1, 0, 0), al.Tensor((4,), al.bf16))
            if nxt == 0:
                B_lds0[ld_row + 0, ld_col + 0] = d0[0]
                B_lds0[ld_row + 0, ld_col + 1] = d0[1]
                B_lds0[ld_row + 0, ld_col + 2] = d0[2]
                B_lds0[ld_row + 0, ld_col + 3] = d0[3]
                B_lds0[ld_row + 1, ld_col + 0] = d1[0]
                B_lds0[ld_row + 1, ld_col + 1] = d1[1]
                B_lds0[ld_row + 1, ld_col + 2] = d1[2]
                B_lds0[ld_row + 1, ld_col + 3] = d1[3]
            else:
                B_lds1[ld_row + 0, ld_col + 0] = d0[0]
                B_lds1[ld_row + 0, ld_col + 1] = d0[1]
                B_lds1[ld_row + 0, ld_col + 2] = d0[2]
                B_lds1[ld_row + 0, ld_col + 3] = d0[3]
                B_lds1[ld_row + 1, ld_col + 0] = d1[0]
                B_lds1[ld_row + 1, ld_col + 1] = d1[1]
                B_lds1[ld_row + 1, ld_col + 2] = d1[2]
                B_lds1[ld_row + 1, ld_col + 3] = d1[3]

        # Compute MFMA from buf while loads to nxt are in flight
        a_row = wave_row * 32 + thr_row * 4
        a_k0 = al.make_local((4,), al.bf16)
        a_k8 = al.make_local((4,), al.bf16)
        b_col = wave_col * 32 + thr_col * 4
        b_k0 = al.make_local((4,), al.bf16)
        b_k8 = al.make_local((4,), al.bf16)

        if buf == 0:
            a_k0[0] = A_lds0[a_row + 0, thr_col]
            a_k0[1] = A_lds0[a_row + 1, thr_col]
            a_k0[2] = A_lds0[a_row + 2, thr_col]
            a_k0[3] = A_lds0[a_row + 3, thr_col]
            a_k8[0] = A_lds0[a_row + 0, 8 + thr_col]
            a_k8[1] = A_lds0[a_row + 1, 8 + thr_col]
            a_k8[2] = A_lds0[a_row + 2, 8 + thr_col]
            a_k8[3] = A_lds0[a_row + 3, 8 + thr_col]
            b_k0[0] = B_lds0[thr_row, b_col + 0]
            b_k0[1] = B_lds0[thr_row, b_col + 1]
            b_k0[2] = B_lds0[thr_row, b_col + 2]
            b_k0[3] = B_lds0[thr_row, b_col + 3]
            b_k8[0] = B_lds0[8 + thr_row, b_col + 0]
            b_k8[1] = B_lds0[8 + thr_row, b_col + 1]
            b_k8[2] = B_lds0[8 + thr_row, b_col + 2]
            b_k8[3] = B_lds0[8 + thr_row, b_col + 3]
        else:
            a_k0[0] = A_lds1[a_row + 0, thr_col]
            a_k0[1] = A_lds1[a_row + 1, thr_col]
            a_k0[2] = A_lds1[a_row + 2, thr_col]
            a_k0[3] = A_lds1[a_row + 3, thr_col]
            a_k8[0] = A_lds1[a_row + 0, 8 + thr_col]
            a_k8[1] = A_lds1[a_row + 1, 8 + thr_col]
            a_k8[2] = A_lds1[a_row + 2, 8 + thr_col]
            a_k8[3] = A_lds1[a_row + 3, 8 + thr_col]
            b_k0[0] = B_lds1[thr_row, b_col + 0]
            b_k0[1] = B_lds1[thr_row, b_col + 1]
            b_k0[2] = B_lds1[thr_row, b_col + 2]
            b_k0[3] = B_lds1[thr_row, b_col + 3]
            b_k8[0] = B_lds1[8 + thr_row, b_col + 0]
            b_k8[1] = B_lds1[8 + thr_row, b_col + 1]
            b_k8[2] = B_lds1[8 + thr_row, b_col + 2]
            b_k8[3] = B_lds1[8 + thr_row, b_col + 3]

        a_k0_u32 = al.view(a_k0, al.Tensor((2,), al.u32))
        a_k8_u32 = al.view(a_k8, al.Tensor((2,), al.u32))
        b_k0_u32 = al.view(b_k0, al.Tensor((2,), al.u32))
        b_k8_u32 = al.view(b_k8, al.Tensor((2,), al.u32))

        acc = al.amdgpu.mfma_32x32x8_bf16_f32(a_k0_u32, b_k0_u32, acc)
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(a_k8_u32, b_k8_u32, acc)

        al.syncthreads()
        buf = nxt

        # --- Half 2: load k1 + BLOCK_K into nxt, compute from buf ---
        if k_block + BLOCK_K < stride_ak:
            nxt = 1 - buf
            k2 = k_block + BLOCK_K

            if tid < 128:
                ld_row = (tid // 4) * 2
                ld_col = (tid % 4) * 4
                g_row0 = block_m * BLOCK_M + ld_row
                g_row1 = g_row0 + 1
                g_col = k2 + ld_col
                bo0 = al.convert((g_row0 * stride_ak + g_col) * 2, al.i32)
                d0 = al.view(al.amdgpu.raw_buffer_load_x2(A_rsrc, bo0, 0, 0), al.Tensor((4,), al.bf16))
                bo1 = al.convert((g_row1 * stride_ak + g_col) * 2, al.i32)
                d1 = al.view(al.amdgpu.raw_buffer_load_x2(A_rsrc, bo1, 0, 0), al.Tensor((4,), al.bf16))
                if nxt == 0:
                    A_lds0[ld_row + 0, ld_col + 0] = d0[0]
                    A_lds0[ld_row + 0, ld_col + 1] = d0[1]
                    A_lds0[ld_row + 0, ld_col + 2] = d0[2]
                    A_lds0[ld_row + 0, ld_col + 3] = d0[3]
                    A_lds0[ld_row + 1, ld_col + 0] = d1[0]
                    A_lds0[ld_row + 1, ld_col + 1] = d1[1]
                    A_lds0[ld_row + 1, ld_col + 2] = d1[2]
                    A_lds0[ld_row + 1, ld_col + 3] = d1[3]
                else:
                    A_lds1[ld_row + 0, ld_col + 0] = d0[0]
                    A_lds1[ld_row + 0, ld_col + 1] = d0[1]
                    A_lds1[ld_row + 0, ld_col + 2] = d0[2]
                    A_lds1[ld_row + 0, ld_col + 3] = d0[3]
                    A_lds1[ld_row + 1, ld_col + 0] = d1[0]
                    A_lds1[ld_row + 1, ld_col + 1] = d1[1]
                    A_lds1[ld_row + 1, ld_col + 2] = d1[2]
                    A_lds1[ld_row + 1, ld_col + 3] = d1[3]
            else:
                t = tid - 128
                ld_row = (t // 16) * 2
                ld_col = (t % 16) * 4
                g_row0 = k2 + ld_row
                g_row1 = g_row0 + 1
                g_col = block_n * BLOCK_N + ld_col
                bo0 = al.convert((g_row0 * stride_bn + g_col) * 2, al.i32)
                d0 = al.view(al.amdgpu.raw_buffer_load_x2(B_rsrc, bo0, 0, 0), al.Tensor((4,), al.bf16))
                bo1 = al.convert((g_row1 * stride_bn + g_col) * 2, al.i32)
                d1 = al.view(al.amdgpu.raw_buffer_load_x2(B_rsrc, bo1, 0, 0), al.Tensor((4,), al.bf16))
                if nxt == 0:
                    B_lds0[ld_row + 0, ld_col + 0] = d0[0]
                    B_lds0[ld_row + 0, ld_col + 1] = d0[1]
                    B_lds0[ld_row + 0, ld_col + 2] = d0[2]
                    B_lds0[ld_row + 0, ld_col + 3] = d0[3]
                    B_lds0[ld_row + 1, ld_col + 0] = d1[0]
                    B_lds0[ld_row + 1, ld_col + 1] = d1[1]
                    B_lds0[ld_row + 1, ld_col + 2] = d1[2]
                    B_lds0[ld_row + 1, ld_col + 3] = d1[3]
                else:
                    B_lds1[ld_row + 0, ld_col + 0] = d0[0]
                    B_lds1[ld_row + 0, ld_col + 1] = d0[1]
                    B_lds1[ld_row + 0, ld_col + 2] = d0[2]
                    B_lds1[ld_row + 0, ld_col + 3] = d0[3]
                    B_lds1[ld_row + 1, ld_col + 0] = d1[0]
                    B_lds1[ld_row + 1, ld_col + 1] = d1[1]
                    B_lds1[ld_row + 1, ld_col + 2] = d1[2]
                    B_lds1[ld_row + 1, ld_col + 3] = d1[3]

            a_row = wave_row * 32 + thr_row * 4
            a_k0 = al.make_local((4,), al.bf16)
            a_k8 = al.make_local((4,), al.bf16)
            b_col = wave_col * 32 + thr_col * 4
            b_k0 = al.make_local((4,), al.bf16)
            b_k8 = al.make_local((4,), al.bf16)

            if buf == 0:
                a_k0[0] = A_lds0[a_row + 0, thr_col]
                a_k0[1] = A_lds0[a_row + 1, thr_col]
                a_k0[2] = A_lds0[a_row + 2, thr_col]
                a_k0[3] = A_lds0[a_row + 3, thr_col]
                a_k8[0] = A_lds0[a_row + 0, 8 + thr_col]
                a_k8[1] = A_lds0[a_row + 1, 8 + thr_col]
                a_k8[2] = A_lds0[a_row + 2, 8 + thr_col]
                a_k8[3] = A_lds0[a_row + 3, 8 + thr_col]
                b_k0[0] = B_lds0[thr_row, b_col + 0]
                b_k0[1] = B_lds0[thr_row, b_col + 1]
                b_k0[2] = B_lds0[thr_row, b_col + 2]
                b_k0[3] = B_lds0[thr_row, b_col + 3]
                b_k8[0] = B_lds0[8 + thr_row, b_col + 0]
                b_k8[1] = B_lds0[8 + thr_row, b_col + 1]
                b_k8[2] = B_lds0[8 + thr_row, b_col + 2]
                b_k8[3] = B_lds0[8 + thr_row, b_col + 3]
            else:
                a_k0[0] = A_lds1[a_row + 0, thr_col]
                a_k0[1] = A_lds1[a_row + 1, thr_col]
                a_k0[2] = A_lds1[a_row + 2, thr_col]
                a_k0[3] = A_lds1[a_row + 3, thr_col]
                a_k8[0] = A_lds1[a_row + 0, 8 + thr_col]
                a_k8[1] = A_lds1[a_row + 1, 8 + thr_col]
                a_k8[2] = A_lds1[a_row + 2, 8 + thr_col]
                a_k8[3] = A_lds1[a_row + 3, 8 + thr_col]
                b_k0[0] = B_lds1[thr_row, b_col + 0]
                b_k0[1] = B_lds1[thr_row, b_col + 1]
                b_k0[2] = B_lds1[thr_row, b_col + 2]
                b_k0[3] = B_lds1[thr_row, b_col + 3]
                b_k8[0] = B_lds1[8 + thr_row, b_col + 0]
                b_k8[1] = B_lds1[8 + thr_row, b_col + 1]
                b_k8[2] = B_lds1[8 + thr_row, b_col + 2]
                b_k8[3] = B_lds1[8 + thr_row, b_col + 3]

            a_k0_u32 = al.view(a_k0, al.Tensor((2,), al.u32))
            a_k8_u32 = al.view(a_k8, al.Tensor((2,), al.u32))
            b_k0_u32 = al.view(b_k0, al.Tensor((2,), al.u32))
            b_k8_u32 = al.view(b_k8, al.Tensor((2,), al.u32))

            acc = al.amdgpu.mfma_32x32x8_bf16_f32(a_k0_u32, b_k0_u32, acc)
            acc = al.amdgpu.mfma_32x32x8_bf16_f32(a_k8_u32, b_k8_u32, acc)

            al.syncthreads()
            buf = nxt

    # ===== Epilogue: compute the last loaded K-block from buf =====
    a_row = wave_row * 32 + thr_row * 4
    a_k0 = al.make_local((4,), al.bf16)
    a_k8 = al.make_local((4,), al.bf16)
    b_col = wave_col * 32 + thr_col * 4
    b_k0 = al.make_local((4,), al.bf16)
    b_k8 = al.make_local((4,), al.bf16)

    if buf == 0:
        a_k0[0] = A_lds0[a_row + 0, thr_col]
        a_k0[1] = A_lds0[a_row + 1, thr_col]
        a_k0[2] = A_lds0[a_row + 2, thr_col]
        a_k0[3] = A_lds0[a_row + 3, thr_col]
        a_k8[0] = A_lds0[a_row + 0, 8 + thr_col]
        a_k8[1] = A_lds0[a_row + 1, 8 + thr_col]
        a_k8[2] = A_lds0[a_row + 2, 8 + thr_col]
        a_k8[3] = A_lds0[a_row + 3, 8 + thr_col]
        b_k0[0] = B_lds0[thr_row, b_col + 0]
        b_k0[1] = B_lds0[thr_row, b_col + 1]
        b_k0[2] = B_lds0[thr_row, b_col + 2]
        b_k0[3] = B_lds0[thr_row, b_col + 3]
        b_k8[0] = B_lds0[8 + thr_row, b_col + 0]
        b_k8[1] = B_lds0[8 + thr_row, b_col + 1]
        b_k8[2] = B_lds0[8 + thr_row, b_col + 2]
        b_k8[3] = B_lds0[8 + thr_row, b_col + 3]
    else:
        a_k0[0] = A_lds1[a_row + 0, thr_col]
        a_k0[1] = A_lds1[a_row + 1, thr_col]
        a_k0[2] = A_lds1[a_row + 2, thr_col]
        a_k0[3] = A_lds1[a_row + 3, thr_col]
        a_k8[0] = A_lds1[a_row + 0, 8 + thr_col]
        a_k8[1] = A_lds1[a_row + 1, 8 + thr_col]
        a_k8[2] = A_lds1[a_row + 2, 8 + thr_col]
        a_k8[3] = A_lds1[a_row + 3, 8 + thr_col]
        b_k0[0] = B_lds1[thr_row, b_col + 0]
        b_k0[1] = B_lds1[thr_row, b_col + 1]
        b_k0[2] = B_lds1[thr_row, b_col + 2]
        b_k0[3] = B_lds1[thr_row, b_col + 3]
        b_k8[0] = B_lds1[8 + thr_row, b_col + 0]
        b_k8[1] = B_lds1[8 + thr_row, b_col + 1]
        b_k8[2] = B_lds1[8 + thr_row, b_col + 2]
        b_k8[3] = B_lds1[8 + thr_row, b_col + 3]

    a_k0_u32 = al.view(a_k0, al.Tensor((2,), al.u32))
    a_k8_u32 = al.view(a_k8, al.Tensor((2,), al.u32))
    b_k0_u32 = al.view(b_k0, al.Tensor((2,), al.u32))
    b_k8_u32 = al.view(b_k8, al.Tensor((2,), al.u32))

    acc = al.amdgpu.mfma_32x32x8_bf16_f32(a_k0_u32, b_k0_u32, acc)
    acc = al.amdgpu.mfma_32x32x8_bf16_f32(a_k8_u32, b_k8_u32, acc)

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
