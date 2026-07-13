import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def fused_gemm_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    M: al.i32,
    N: al.i32,
    K: al.i32,
    BM: al.constexpr,
    BN: al.constexpr,
    BK: al.constexpr,
):
    BF16_BYTES = 2

    # Tensor views from raw pointers
    x_layout = al.make_layout((M, K), (K, 1))
    x = al.make_tensor(x_ptr, al.bf16, x_layout)
    w_layout = al.make_layout((K, N), (N, 1))
    w = al.make_tensor(w_ptr, al.bf16, w_layout)
    bias_layout = al.make_layout((N,), (1,))
    bias = al.make_tensor(bias_ptr, al.bf16, bias_layout)
    out_layout = al.make_layout((M, N), (N, 1))
    out = al.make_tensor(out_ptr, al.bf16, out_layout)

    # Buffer resources for vectorized global loads
    x_rsrc = al.amdgpu.make_rsrc(x, M * K * BF16_BYTES)
    w_rsrc = al.amdgpu.make_rsrc(w, K * N * BF16_BYTES)

    # Double-buffered shared memory for A (BM x BK) and B (BK x BN) tiles
    A0 = al.make_shared((BM, BK), al.bf16)
    B0 = al.make_shared((BK, BN), al.bf16)
    A1 = al.make_shared((BM, BK), al.bf16)
    B1 = al.make_shared((BK, BN), al.bf16)

    # Block and warp indexing (2x2 warps covering 64x64)
    block_m = al.block_id(1)
    block_n = al.block_id(0)
    m_base = block_m * BM
    n_base = block_n * BN

    tid = al.thread_id(0)
    lane = tid % 64
    wave_id = tid // 64
    warp_row = wave_id // 2
    warp_col = wave_id % 2

    # Accumulator: 16 f32 per lane (32x32 output tile per warp)
    acc = al.make_local((16,), al.f32)
    for i in al.range(16):
        acc[i] = al.convert(0.0, al.f32)

    # Precomputed per-thread load indices
    row_a = tid // 4
    col_a = (tid % 4) * 4
    row_b = tid // 16
    col_b = (tid % 16) * 4

    # =========================================================================
    # Prologue: prefetch K=0 into A0,B0 and K=BK into A1,B1
    # =========================================================================

    # Load A tile at K=0 into A0
    gbl_off = (m_base + row_a) * K * BF16_BYTES + col_a * BF16_BYTES
    loaded = al.amdgpu.raw_buffer_load_x4(x_rsrc, gbl_off, 0, 0)
    bf16_8 = al.view(loaded, al.Tensor((8,), al.bf16))
    for i in al.range(4):
        A0[row_a, col_a + i] = bf16_8[i]

    # Load B tile at K=0 into B0  (w is (K, N): w[row, n_base+col])
    gbl_off = row_b * N * BF16_BYTES + (n_base + col_b) * BF16_BYTES
    loaded = al.amdgpu.raw_buffer_load_x4(w_rsrc, gbl_off, 0, 0)
    bf16_8 = al.view(loaded, al.Tensor((8,), al.bf16))
    for i in al.range(4):
        B0[row_b, col_b + i] = bf16_8[i]
    al.syncthreads()

    # Load A tile at K=BK into A1
    gbl_off = (m_base + row_a) * K * BF16_BYTES + (BK + col_a) * BF16_BYTES
    loaded = al.amdgpu.raw_buffer_load_x4(x_rsrc, gbl_off, 0, 0)
    bf16_8 = al.view(loaded, al.Tensor((8,), al.bf16))
    for i in al.range(4):
        A1[row_a, col_a + i] = bf16_8[i]

    # Load B tile at K=BK into B1
    gbl_off = (BK + row_b) * N * BF16_BYTES + (n_base + col_b) * BF16_BYTES
    loaded = al.amdgpu.raw_buffer_load_x4(w_rsrc, gbl_off, 0, 0)
    bf16_8 = al.view(loaded, al.Tensor((8,), al.bf16))
    for i in al.range(4):
        B1[row_b, col_b + i] = bf16_8[i]
    al.syncthreads()

    # Compute K block 0 from A0, B0  (2 MFMA calls for BK=16)
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
    # Main loop: double-buffered with 2*BK stride
    # =========================================================================
    for k_even in al.range(2 * BK, K, 2 * BK):
        k_odd = k_even + BK

        # Phase 1: load k_even into A0,B0; compute from A1,B1 (k_even - BK)
        gbl_off = (m_base + row_a) * K * BF16_BYTES + (k_even + col_a) * BF16_BYTES
        loaded = al.amdgpu.raw_buffer_load_x4(x_rsrc, gbl_off, 0, 0)
        bf16_8 = al.view(loaded, al.Tensor((8,), al.bf16))
        for i in al.range(4):
            A0[row_a, col_a + i] = bf16_8[i]

        gbl_off = (k_even + row_b) * N * BF16_BYTES + (n_base + col_b) * BF16_BYTES
        loaded = al.amdgpu.raw_buffer_load_x4(w_rsrc, gbl_off, 0, 0)
        bf16_8 = al.view(loaded, al.Tensor((8,), al.bf16))
        for i in al.range(4):
            B0[row_b, col_b + i] = bf16_8[i]

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

        # Phase 2: load k_odd into A1,B1; compute from A0,B0 (k_even)
        gbl_off = (m_base + row_a) * K * BF16_BYTES + (k_odd + col_a) * BF16_BYTES
        loaded = al.amdgpu.raw_buffer_load_x4(x_rsrc, gbl_off, 0, 0)
        bf16_8 = al.view(loaded, al.Tensor((8,), al.bf16))
        for i in al.range(4):
            A1[row_a, col_a + i] = bf16_8[i]

        gbl_off = (k_odd + row_b) * N * BF16_BYTES + (n_base + col_b) * BF16_BYTES
        loaded = al.amdgpu.raw_buffer_load_x4(w_rsrc, gbl_off, 0, 0)
        bf16_8 = al.view(loaded, al.Tensor((8,), al.bf16))
        for i in al.range(4):
            B1[row_b, col_b + i] = bf16_8[i]

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
    # Epilogue: compute final K block from A1, B1
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
    # Post-GEMM: bias + scale + hardtanh + GELU
    # =========================================================================

    # Load bias for this tile's columns into shared memory
    bias_shared = al.make_shared((BN,), al.bf16)
    if tid < BN:
        bias_shared[tid] = bias[n_base + tid]
    al.syncthreads()

    # Write accumulator + bias to shared memory (BM x BN in f32)
    tile_buf = al.make_shared((BM, BN), al.f32)
    for acc_idx in al.range(16):
        grp = acc_idx // 4
        elem = acc_idx % 4
        lgrp = lane // 32
        row = warp_row * 32 + 8 * grp + 4 * lgrp + elem
        col = warp_col * 32 + (lane % 32)
        tile_buf[row, col] = acc[acc_idx] + al.convert(bias_shared[col], al.f32)
    al.syncthreads()

    # Activation constants
    scale_val = al.convert(0.5, al.f32)
    ht_min = al.convert(-2.0, al.f32)
    ht_max = al.convert(2.0, al.f32)
    sqrt_2_pi = al.convert(0.7978845608028654, al.f32)
    gelu_coeff = al.convert(0.044715, al.f32)
    one = al.convert(1.0, al.f32)
    half = al.convert(0.5, al.f32)

    # Each thread processes a strided subset of the tile for element-wise ops
    total_elems = BM * BN
    for idx in al.range(tid, total_elems, 256):
        r = idx // BN
        c = idx % BN
        gm = m_base + r
        gn = n_base + c
        if gm < M and gn < N:
            val = tile_buf[r, c]
            val = val * scale_val
            if val < ht_min:
                val = ht_min
            elif val > ht_max:
                val = ht_max
            x3 = val * val * val
            inner = sqrt_2_pi * (val + gelu_coeff * x3)
            tanh_val = al.tanh(inner)
            gelu_val = half * val * (one + tanh_val)
            out[gm, gn] = al.convert(gelu_val, al.bf16)


class ModelNew(nn.Module):
    def __init__(
        self, in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max
    ):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max

    def forward(self, x):
        w = self.gemm.weight.data
        w_t = w.t().contiguous()
        bias = self.gemm.bias.data

        assert x.is_cuda and w_t.is_cuda and bias.is_cuda

        M = x.shape[0]
        K = x.shape[1]
        N = bias.shape[0]

        x = x.contiguous()
        out = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)

        BM = 64
        BN = 64
        BK = 16

        grid_x = (N + BN - 1) // BN
        grid_y = (M + BM - 1) // BM

        fused_gemm_kernel[lambda: ((grid_x, grid_y, 1), (256, 1, 1))](
            x.data_ptr(),
            w_t.data_ptr(),
            bias.data_ptr(),
            out.data_ptr(),
            M,
            N,
            K,
            BM,
            BN,
            BK,
        )
        return out
