import torch
import torch.nn as nn

import avelang
import avelang.language as al


@avelang.jit
def tri_gemm_kernel(
    A_ptr: al.Pointer(al.bf16),
    Bt_ptr: al.Pointer(al.bf16),
    C_ptr: al.Pointer(al.bf16),
    M: al.i32,
    K_dim: al.i32,
    N: al.i32,
):
    BM = al.convert(64, al.i32)
    BK = al.convert(16, al.i32)
    BK2 = al.convert(32, al.i32)
    MMA_M = al.convert(32, al.i32)
    MMA_N = al.convert(32, al.i32)
    B2 = al.convert(2, al.i32)
    WARP_SIZE = al.convert(64, al.i32)
    WARPS_N = al.convert(2, al.i32)
    zero_u32 = al.convert(0, al.u32)
    one = al.convert(1, al.i32)
    two = al.convert(2, al.i32)
    four = al.convert(4, al.i32)
    eight = al.convert(8, al.i32)
    acc_n = al.convert(16, al.i32)
    threads = al.convert(256, al.i32)
    shm_vecs = al.convert(128, al.i32)
    vec = al.convert(8, al.i32)

    # A: (M, K) — consecutive along K
    A = al.make_tensor(A_ptr, al.bf16, al.make_layout((M, K_dim), (K_dim, 1)))
    # Bt: transposed B — (N, K) — consecutive along K
    Bt = al.make_tensor(Bt_ptr, al.bf16, al.make_layout((N, K_dim), (K_dim, 1)))
    C = al.make_tensor(C_ptr, al.bf16, al.make_layout((M, N), (N, 1)))

    A_rsrc = al.amdgpu.make_rsrc(A, M * K_dim * B2)
    Bt_rsrc = al.amdgpu.make_rsrc(Bt, N * K_dim * B2)

    tid = al.thread_id(0)
    wid = tid // WARP_SIZE
    lane = tid % WARP_SIZE
    warp_row = wid // WARPS_N
    warp_col = wid % WARPS_N

    block_m = al.block_id(0) * BM
    block_n = al.block_id(1) * BM

    if block_n + BM <= block_m:
        return

    # Double-buffered shared memory — flat u32 layout
    shm_a0 = al.make_shared((128, 4), al.u32)
    shm_a1 = al.make_shared((128, 4), al.u32)
    shm_b0 = al.make_shared((128, 4), al.u32)
    shm_b1 = al.make_shared((128, 4), al.u32)

    shm_a0_f = al.view(shm_a0, al.u32, al.make_layout((512,), (1,)))
    shm_a1_f = al.view(shm_a1, al.u32, al.make_layout((512,), (1,)))
    shm_b0_f = al.view(shm_b0, al.u32, al.make_layout((512,), (1,)))
    shm_b1_f = al.view(shm_b1, al.u32, al.make_layout((512,), (1,)))

    # Accumulator — same type as candidate
    acc = al.make_local((16,), al.f32)
    for ai in al.range(16):
        acc[ai] = al.convert(0.0, al.f32)

    # MFMA operand registers — u32 typed like candidate
    a_h0 = al.make_local((2,), al.u32)
    a_h1 = al.make_local((2,), al.u32)
    b_h0 = al.make_local((2,), al.u32)
    b_h1 = al.make_local((2,), al.u32)

    # Fetch bases: ROW_U32 = 2 vecs/row * 4 u32/vec = 8
    a_row = warp_row * MMA_M + (lane % MMA_M)
    a_kg = (lane // MMA_M) * two
    a_rb = a_row * eight

    b_row = warp_col * MMA_M + (lane % MMA_M)
    b_kg = (lane // MMA_M) * two
    b_rb = b_row * eight

    # --- Prologue: load tile 0 into buffer 0 ---
    k0 = al.convert(0, al.i32)
    idx = tid
    for _ in al.range(1):
        if idx < shm_vecs:
            r = idx // two
            cv = idx % two
            off = ((block_m + r) * K_dim + k0 + cv * vec) * B2
            shm_a0[idx] = al.amdgpu.raw_buffer_load_x4(A_rsrc, zero_u32, off, 0)
        idx += threads

    idx = tid
    for _ in al.range(1):
        if idx < shm_vecs:
            r = idx // two
            cv = idx % two
            off = ((block_n + r) * K_dim + k0 + cv * vec) * B2
            shm_b0[idx] = al.amdgpu.raw_buffer_load_x4(Bt_rsrc, zero_u32, off, 0)
        idx += threads

    al.syncthreads()

    # Fetch half 0 from buffer 0
    a_h0[0] = shm_a0_f[a_rb + a_kg]
    a_h0[1] = shm_a0_f[a_rb + a_kg + one]
    b_h0[0] = shm_b0_f[b_rb + b_kg]
    b_h0[1] = shm_b0_f[b_rb + b_kg + one]

    # Fetch half 1 from buffer 0
    a_h1[0] = shm_a0_f[a_rb + four + a_kg]
    a_h1[1] = shm_a0_f[a_rb + four + a_kg + one]
    b_h1[0] = shm_b0_f[b_rb + four + b_kg]
    b_h1[1] = shm_b0_f[b_rb + four + b_kg + one]

    # --- Main loop: software-pipelined, unrolled by 2 ---
    for k in al.range(BK, K_dim - BK, BK2):
        # Phase A: load tile[k] -> buf[1], compute buf[0]
        idx = tid
        for _ in al.range(1):
            if idx < shm_vecs:
                r = idx // two
                cv = idx % two
                off = ((block_m + r) * K_dim + k + cv * vec) * B2
                shm_a1[idx] = al.amdgpu.raw_buffer_load_x4(A_rsrc, zero_u32, off, 0)
            idx += threads

        idx = tid
        for _ in al.range(1):
            if idx < shm_vecs:
                r = idx // two
                cv = idx % two
                off = ((block_n + r) * K_dim + k + cv * vec) * B2
                shm_b1[idx] = al.amdgpu.raw_buffer_load_x4(Bt_rsrc, zero_u32, off, 0)
            idx += threads

        # MFMA on buffer 0 (overlaps with global loads for buffer 1)
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(a_h0, b_h0, acc)
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(a_h1, b_h1, acc)

        al.syncthreads()

        # Fetch from buffer 1
        a_h0[0] = shm_a1_f[a_rb + a_kg]
        a_h0[1] = shm_a1_f[a_rb + a_kg + one]
        b_h0[0] = shm_b1_f[b_rb + b_kg]
        b_h0[1] = shm_b1_f[b_rb + b_kg + one]

        a_h1[0] = shm_a1_f[a_rb + four + a_kg]
        a_h1[1] = shm_a1_f[a_rb + four + a_kg + one]
        b_h1[0] = shm_b1_f[b_rb + four + b_kg]
        b_h1[1] = shm_b1_f[b_rb + four + b_kg + one]

        # Phase B: load tile[k+BK] -> buf[0], compute buf[1]
        k2 = k + BK
        idx = tid
        for _ in al.range(1):
            if idx < shm_vecs:
                r = idx // two
                cv = idx % two
                off = ((block_m + r) * K_dim + k2 + cv * vec) * B2
                shm_a0[idx] = al.amdgpu.raw_buffer_load_x4(A_rsrc, zero_u32, off, 0)
            idx += threads

        idx = tid
        for _ in al.range(1):
            if idx < shm_vecs:
                r = idx // two
                cv = idx % two
                off = ((block_n + r) * K_dim + k2 + cv * vec) * B2
                shm_b0[idx] = al.amdgpu.raw_buffer_load_x4(Bt_rsrc, zero_u32, off, 0)
            idx += threads

        # MFMA on buffer 1 (overlaps with global loads for buffer 0)
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(a_h0, b_h0, acc)
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(a_h1, b_h1, acc)

        al.syncthreads()

        # Fetch from buffer 0 (for next iteration)
        a_h0[0] = shm_a0_f[a_rb + a_kg]
        a_h0[1] = shm_a0_f[a_rb + a_kg + one]
        b_h0[0] = shm_b0_f[b_rb + b_kg]
        b_h0[1] = shm_b0_f[b_rb + b_kg + one]

        a_h1[0] = shm_a0_f[a_rb + four + a_kg]
        a_h1[1] = shm_a0_f[a_rb + four + a_kg + one]
        b_h1[0] = shm_b0_f[b_rb + four + b_kg]
        b_h1[1] = shm_b0_f[b_rb + four + b_kg + one]

    # --- Epilogue: final K-tile ---
    k_last = K_dim - BK

    idx = tid
    for _ in al.range(1):
        if idx < shm_vecs:
            r = idx // two
            cv = idx % two
            off = ((block_m + r) * K_dim + k_last + cv * vec) * B2
            shm_a1[idx] = al.amdgpu.raw_buffer_load_x4(A_rsrc, zero_u32, off, 0)
        idx += threads

    idx = tid
    for _ in al.range(1):
        if idx < shm_vecs:
            r = idx // two
            cv = idx % two
            off = ((block_n + r) * K_dim + k_last + cv * vec) * B2
            shm_b1[idx] = al.amdgpu.raw_buffer_load_x4(Bt_rsrc, zero_u32, off, 0)
        idx += threads

    # MFMA on buffer 0 (pre-fetched at loop tail)
    acc = al.amdgpu.mfma_32x32x8_bf16_f32(a_h0, b_h0, acc)
    acc = al.amdgpu.mfma_32x32x8_bf16_f32(a_h1, b_h1, acc)

    al.syncthreads()

    # Fetch final tile from buffer 1
    a_h0[0] = shm_a1_f[a_rb + a_kg]
    a_h0[1] = shm_a1_f[a_rb + a_kg + one]
    b_h0[0] = shm_b1_f[b_rb + b_kg]
    b_h0[1] = shm_b1_f[b_rb + b_kg + one]

    a_h1[0] = shm_a1_f[a_rb + four + a_kg]
    a_h1[1] = shm_a1_f[a_rb + four + a_kg + one]
    b_h1[0] = shm_b1_f[b_rb + four + b_kg]
    b_h1[1] = shm_b1_f[b_rb + four + b_kg + one]

    # Final MFMA
    acc = al.amdgpu.mfma_32x32x8_bf16_f32(a_h0, b_h0, acc)
    acc = al.amdgpu.mfma_32x32x8_bf16_f32(a_h1, b_h1, acc)

    # --- Writeback ---
    lane_group = lane // MMA_N
    lane_col = lane % MMA_N

    row_base = block_m + warp_row * MMA_M
    col = block_n + warp_col * MMA_N + lane_col

    for t in al.range(acc_n):
        row = row_base + (t // four) * eight + lane_group * four + (t % four)
        if col >= row:
            C[row, col] = al.convert(acc[t], al.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if tuple(A.shape) != (4096, 4096) or tuple(B.shape) != (4096, 4096):
            raise RuntimeError("Expected (4096, 4096) inputs")
        A = A.contiguous()
        Bt = B.contiguous().t().contiguous()
        M_val = A.shape[0]; K_val = A.shape[1]; N_val = Bt.shape[0]
        C = torch.zeros((M_val, N_val), device=A.device, dtype=torch.bfloat16)
        tri_gemm_kernel[lambda: ((64, 64, 1), (256, 1, 1))](
            A.data_ptr(), Bt.data_ptr(), C.data_ptr(),
            M_val, K_val, N_val, num_warps=4)
        return C
