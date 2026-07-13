import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 16
BF16_BYTES = 2


@avelang.jit
def gemm_partial_sum_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    partial_ptr: al.Pointer(al.f32),
    M: al.i32,
    N: al.i32,
    K: al.i32,
    N_BLOCKS: al.i32,
    BM: al.constexpr,
    BN: al.constexpr,
    BK: al.constexpr,
):
    x_layout = al.make_layout((M, K), (K, 1))
    x = al.make_tensor(x_ptr, al.bf16, x_layout)
    w_layout = al.make_layout((K, N), (N, 1))
    w = al.make_tensor(w_ptr, al.bf16, w_layout)
    partial_layout = al.make_layout((M, N_BLOCKS), (N_BLOCKS, 1))
    partial = al.make_tensor(partial_ptr, al.f32, partial_layout)

    x_rsrc = al.amdgpu.make_rsrc(x, M * K * BF16_BYTES)
    w_rsrc = al.amdgpu.make_rsrc(w, K * N * BF16_BYTES)

    # Double-buffered LDS
    A0 = al.make_shared((BM, BK), al.bf16)
    B0 = al.make_shared((BK, BN), al.bf16)
    A1 = al.make_shared((BM, BK), al.bf16)
    B1 = al.make_shared((BK, BN), al.bf16)

    block_m = al.block_id(0)
    block_n = al.block_id(1)
    m_base = block_m * BM
    n_base = block_n * BN

    tid = al.thread_id(0)
    lane = tid % 64
    wave_id = tid // 64
    warp_row = wave_id // 2
    warp_col = wave_id % 2

    acc = al.make_local((16,), al.f32)
    for i in al.range(16):
        acc[i] = al.convert(0.0, al.f32)

    # =========================================================================
    # Prologue: prefetch blocks 0 and BK
    # =========================================================================

    # Prefetch block 0 -> A0, B0
    if tid < 128:
        row_a = (tid * 8) // BK
        col_a = (tid * 8) % BK
        gbl_off = (m_base + row_a) * K * BF16_BYTES + col_a * BF16_BYTES
        loaded = al.amdgpu.raw_buffer_load_x4(x_rsrc, gbl_off, 0, 0)
        bf16_8 = al.view(loaded, al.Tensor((8,), al.bf16))
        for i in al.range(8):
            A0[row_a, col_a + i] = bf16_8[i]
    if tid >= 128:
        local_id = tid - 128
        k_idx = local_id // 8
        n_idx = (local_id % 8) * 8
        gbl_off = k_idx * N * BF16_BYTES + (n_base + n_idx) * BF16_BYTES
        loaded = al.amdgpu.raw_buffer_load_x4(w_rsrc, gbl_off, 0, 0)
        bf16_8 = al.view(loaded, al.Tensor((8,), al.bf16))
        for i in al.range(8):
            B0[k_idx, n_idx + i] = bf16_8[i]
    al.syncthreads()

    # Prefetch block BK -> A1, B1
    if tid < 128:
        row_a = (tid * 8) // BK
        col_a = (tid * 8) % BK
        gbl_off = (m_base + row_a) * K * BF16_BYTES + (BK + col_a) * BF16_BYTES
        loaded = al.amdgpu.raw_buffer_load_x4(x_rsrc, gbl_off, 0, 0)
        bf16_8 = al.view(loaded, al.Tensor((8,), al.bf16))
        for i in al.range(8):
            A1[row_a, col_a + i] = bf16_8[i]
    if tid >= 128:
        local_id = tid - 128
        k_idx = local_id // 8
        n_idx = (local_id % 8) * 8
        gbl_off = (BK + k_idx) * N * BF16_BYTES + (n_base + n_idx) * BF16_BYTES
        loaded = al.amdgpu.raw_buffer_load_x4(w_rsrc, gbl_off, 0, 0)
        bf16_8 = al.view(loaded, al.Tensor((8,), al.bf16))
        for i in al.range(8):
            B1[k_idx, n_idx + i] = bf16_8[i]
    al.syncthreads()

    # Compute block 0 from A0, B0 (fine-grained: 2 sub-tile MFMA ops)
    for k_step in al.range(2):
        k_off = k_step * 8
        a_bf16 = al.make_local((4,), al.bf16)
        for e in al.range(4):
            a_row = lane % 32
            a_col = k_off + (lane // 32) * 4 + e
            a_bf16[e] = A0[warp_row * 32 + a_row, a_col]
        a_u32 = al.view(a_bf16, al.Tensor((2,), al.u32))
        b_bf16 = al.make_local((4,), al.bf16)
        for e in al.range(4):
            b_col = warp_col * 32 + (lane % 32)
            b_row = k_off + (lane // 32) * 4 + e
            b_bf16[e] = B0[b_row, b_col]
        b_u32 = al.view(b_bf16, al.Tensor((2,), al.u32))
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(a_u32, b_u32, acc)

    # =========================================================================
    # Main loop: unrolled by 2 (2*BK stride), double-buffered
    # Each iteration: load k_even->A0,B0 while computing k_even-BK from A1,B1,
    #                 then load k_odd->A1,B1 while computing k_even from A0,B0.
    # =========================================================================
    for k_even in al.range(2 * BK, K, 2 * BK):
        k_odd = k_even + BK

        # ---- Phase 1: Load k_even into A0,B0 (async) + compute k_even-BK from A1,B1 ----
        if tid < 128:
            row_a = (tid * 8) // BK
            col_a = (tid * 8) % BK
            gbl_off = (m_base + row_a) * K * BF16_BYTES + (k_even + col_a) * BF16_BYTES
            loaded = al.amdgpu.raw_buffer_load_x4(x_rsrc, gbl_off, 0, 0)
            bf16_8 = al.view(loaded, al.Tensor((8,), al.bf16))
            for i in al.range(8):
                A0[row_a, col_a + i] = bf16_8[i]
        if tid >= 128:
            local_id = tid - 128
            k_idx = local_id // 8
            n_idx = (local_id % 8) * 8
            gbl_off = (k_even + k_idx) * N * BF16_BYTES + (n_base + n_idx) * BF16_BYTES
            loaded = al.amdgpu.raw_buffer_load_x4(w_rsrc, gbl_off, 0, 0)
            bf16_8 = al.view(loaded, al.Tensor((8,), al.bf16))
            for i in al.range(8):
                B0[k_idx, n_idx + i] = bf16_8[i]

        for k_step in al.range(2):
            k_off = k_step * 8
            a_bf16 = al.make_local((4,), al.bf16)
            for e in al.range(4):
                a_row = lane % 32
                a_col = k_off + (lane // 32) * 4 + e
                a_bf16[e] = A1[warp_row * 32 + a_row, a_col]
            a_u32 = al.view(a_bf16, al.Tensor((2,), al.u32))
            b_bf16 = al.make_local((4,), al.bf16)
            for e in al.range(4):
                b_col = warp_col * 32 + (lane % 32)
                b_row = k_off + (lane // 32) * 4 + e
                b_bf16[e] = B1[b_row, b_col]
            b_u32 = al.view(b_bf16, al.Tensor((2,), al.u32))
            acc = al.amdgpu.mfma_32x32x8_bf16_f32(a_u32, b_u32, acc)
        al.syncthreads()

        # ---- Phase 2: Load k_odd into A1,B1 (async) + compute k_even from A0,B0 ----
        if tid < 128:
            row_a = (tid * 8) // BK
            col_a = (tid * 8) % BK
            gbl_off = (m_base + row_a) * K * BF16_BYTES + (k_odd + col_a) * BF16_BYTES
            loaded = al.amdgpu.raw_buffer_load_x4(x_rsrc, gbl_off, 0, 0)
            bf16_8 = al.view(loaded, al.Tensor((8,), al.bf16))
            for i in al.range(8):
                A1[row_a, col_a + i] = bf16_8[i]
        if tid >= 128:
            local_id = tid - 128
            k_idx = local_id // 8
            n_idx = (local_id % 8) * 8
            gbl_off = (k_odd + k_idx) * N * BF16_BYTES + (n_base + n_idx) * BF16_BYTES
            loaded = al.amdgpu.raw_buffer_load_x4(w_rsrc, gbl_off, 0, 0)
            bf16_8 = al.view(loaded, al.Tensor((8,), al.bf16))
            for i in al.range(8):
                B1[k_idx, n_idx + i] = bf16_8[i]

        for k_step in al.range(2):
            k_off = k_step * 8
            a_bf16 = al.make_local((4,), al.bf16)
            for e in al.range(4):
                a_row = lane % 32
                a_col = k_off + (lane // 32) * 4 + e
                a_bf16[e] = A0[warp_row * 32 + a_row, a_col]
            a_u32 = al.view(a_bf16, al.Tensor((2,), al.u32))
            b_bf16 = al.make_local((4,), al.bf16)
            for e in al.range(4):
                b_col = warp_col * 32 + (lane % 32)
                b_row = k_off + (lane // 32) * 4 + e
                b_bf16[e] = B0[b_row, b_col]
            b_u32 = al.view(b_bf16, al.Tensor((2,), al.u32))
            acc = al.amdgpu.mfma_32x32x8_bf16_f32(a_u32, b_u32, acc)
        al.syncthreads()

    # =========================================================================
    # Epilogue: compute last block (K-BK) from A1, B1
    # =========================================================================
    for k_step in al.range(2):
        k_off = k_step * 8
        a_bf16 = al.make_local((4,), al.bf16)
        for e in al.range(4):
            a_row = lane % 32
            a_col = k_off + (lane // 32) * 4 + e
            a_bf16[e] = A1[warp_row * 32 + a_row, a_col]
        a_u32 = al.view(a_bf16, al.Tensor((2,), al.u32))
        b_bf16 = al.make_local((4,), al.bf16)
        for e in al.range(4):
            b_col = warp_col * 32 + (lane % 32)
            b_row = k_off + (lane // 32) * 4 + e
            b_bf16[e] = B1[b_row, b_col]
        b_u32 = al.view(b_bf16, al.Tensor((2,), al.u32))
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(a_u32, b_u32, acc)

    # =========================================================================
    # Reduction: write acc to shared mem, sum rows, write partial result
    # =========================================================================
    tile_buf = al.make_shared((BM, BN), al.f32)

    for acc_idx in al.range(16):
        grp = acc_idx // 4
        elem = acc_idx % 4
        lgrp = lane // 32
        row = warp_row * 32 + 8 * grp + 4 * lgrp + elem
        col = warp_col * 32 + (lane % 32)
        tile_buf[row, col] = acc[acc_idx]

    al.syncthreads()

    my_row = tid % 64
    col_start = (tid // 64) * 16
    partial_sum = al.convert(0.0, al.f32)
    for c in al.range(16):
        partial_sum = partial_sum + tile_buf[my_row, col_start + c]

    reduce_buf = al.make_shared((256,), al.f32)
    reduce_buf[tid] = partial_sum
    al.syncthreads()

    if tid < 64:
        row_sum = reduce_buf[tid] + reduce_buf[tid + 64] + reduce_buf[tid + 128] + reduce_buf[tid + 192]
        partial[m_base + tid, block_n] = row_sum


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        batch_size = x.shape[0]
        input_size = x.shape[1]
        hidden_size = self.weight.shape[0]

        x_bf16 = x.to(dtype=torch.bfloat16).contiguous()
        w_t = self.weight.t().to(device=x.device, dtype=torch.bfloat16).contiguous()

        n_blocks = (hidden_size + BLOCK_N - 1) // BLOCK_N
        partial = torch.zeros((batch_size, n_blocks), device=x.device, dtype=torch.float32)

        grid_m = (batch_size + BLOCK_M - 1) // BLOCK_M
        grid_n = n_blocks
        gemm_partial_sum_kernel[
            lambda: ((grid_m, grid_n, 1), (256, 1, 1))
        ](
            x_bf16.data_ptr(),
            w_t.data_ptr(),
            partial.data_ptr(),
            batch_size,
            hidden_size,
            input_size,
            n_blocks,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
        )

        scale = self.scaling_factor / 2.0
        row_sums = partial.sum(dim=1, keepdim=True)
        out = (row_sums * scale).to(dtype=torch.bfloat16)
        return out
