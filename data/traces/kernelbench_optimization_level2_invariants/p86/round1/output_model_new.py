import struct
import torch
import torch.nn as nn

import avelang
import avelang.language as al

BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 8
THREADS_PER_BLOCK = 256


@avelang.jit
def fused_gemm_gelu_kernel(
    X_ptr: al.Pointer(al.bf16),
    W_ptr: al.Pointer(al.bf16),
    Bias_ptr: al.Pointer(al.bf16),
    Y_ptr: al.Pointer(al.bf16),
    M: al.i32,
    K: al.i32,
    N: al.i32,
    divisor_bits: al.u32,
):
    bid_m = al.block_id(0)
    bid_n = al.block_id(1)
    tid = al.thread_id(0)
    lane_id = tid % 64
    warp_id = tid // 64
    warp_m = warp_id // 2
    warp_n = warp_id % 2

    block_m = bid_m * 64
    block_n = bid_n * 64
    warp_row = warp_m * 32
    warp_col = warp_n * 32

    x_layout = al.make_layout((M, K), (K, 1))
    X = al.make_tensor(X_ptr, al.bf16, x_layout)
    w_layout = al.make_layout((K, N), (N, 1))
    W = al.make_tensor(W_ptr, al.bf16, w_layout)
    bias_layout = al.make_layout((N,), (1,))
    Bias = al.make_tensor(Bias_ptr, al.bf16, bias_layout)
    y_layout = al.make_layout((M, N), (N, 1))
    Y = al.make_tensor(Y_ptr, al.bf16, y_layout)

    # Double-buffered shared memory: buffer 0 and buffer 1
    lds_a0 = al.make_shared((64, 8), al.bf16)
    lds_b0 = al.make_shared((8, 64), al.bf16)
    lds_a1 = al.make_shared((64, 8), al.bf16)
    lds_b1 = al.make_shared((8, 64), al.bf16)

    lds_a0_u32 = al.view(lds_a0, al.u32, al.make_layout((64, 4), (4, 1)))
    lds_b0_u32 = al.view(lds_b0, al.u32, al.make_layout((8, 32), (32, 1)))
    lds_a1_u32 = al.view(lds_a1, al.u32, al.make_layout((64, 4), (4, 1)))
    lds_b1_u32 = al.view(lds_b1, al.u32, al.make_layout((8, 32), (32, 1)))

    divisor_val = al.bitcast(divisor_bits, al.f32)

    acc = al.make_local((16,), al.f32)
    for i in al.range(16):
        acc[i] = al.convert(0.0, al.f32)

    a_op = al.make_local((2,), al.u32)
    b_op = al.make_local((2,), al.u32)

    # Global-to-LDS load thread mapping
    # lds_a (64x8): 64 threads, each loads a full row of 8 bf16 elements
    load_a = tid < 64
    a_lds_row_load = tid
    # lds_b (8x64): 64 threads, each loads 8 contiguous elements within a row
    load_b = (tid >= 64) & (tid < 128)
    b_tid = tid - 64
    b_lds_row_load = b_tid // 8
    b_lds_col_load = (b_tid % 8) * 8

    # MFMA operand read indices for buffer 0 and buffer 1 (same pattern)
    a_read_row = warp_row + (lane_id % 32)
    a_read_u32 = (lane_id // 32) * 2
    b_read_k = lane_id % 8
    b_read_n_u32 = (lane_id // 8) * 2 + warp_col // 2

    # === Prologue: load k=0 into buffer 0 ===
    if load_a:
        gr = block_m + a_lds_row_load
        for c in al.range(8):
            lds_a0[a_lds_row_load, c] = X[gr, 0 + c]
    if load_b:
        gr = 0 + b_lds_row_load
        gc = block_n + b_lds_col_load
        for c in al.range(8):
            lds_b0[b_lds_row_load, b_lds_col_load + c] = W[gr, gc + c]

    al.syncthreads()

    # === Main loop: software-pipelined, K-unrolled by 2 ===
    # Each iteration processes two BLOCK_K-wide slices: k and k+8
    # Double buffer: while computing on one buffer, load next data into the other
    for k in al.range(0, K, 16):
        # --- Sub-1: load k+8 into buf[1], compute MFMA on buf[0] (k) ---
        k1 = k + 8
        if k1 < K:
            if load_a:
                gr = block_m + a_lds_row_load
                for c in al.range(8):
                    lds_a1[a_lds_row_load, c] = X[gr, k1 + c]
            if load_b:
                gr = k1 + b_lds_row_load
                gc = block_n + b_lds_col_load
                for c in al.range(8):
                    lds_b1[b_lds_row_load, b_lds_col_load + c] = W[gr, gc + c]

        # Compute on buffer 0 while buffer 1 loads are in flight
        a_op[0] = lds_a0_u32[a_read_row, a_read_u32 + 0]
        a_op[1] = lds_a0_u32[a_read_row, a_read_u32 + 1]
        b_op[0] = lds_b0_u32[b_read_k, b_read_n_u32 + 0]
        b_op[1] = lds_b0_u32[b_read_k, b_read_n_u32 + 1]
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(a_op, b_op, acc)

        al.syncthreads()

        # --- Sub-2: load k+16 into buf[0], compute MFMA on buf[1] (k+8) ---
        k2 = k + 16
        if k2 < K:
            if load_a:
                gr = block_m + a_lds_row_load
                for c in al.range(8):
                    lds_a0[a_lds_row_load, c] = X[gr, k2 + c]
            if load_b:
                gr = k2 + b_lds_row_load
                gc = block_n + b_lds_col_load
                for c in al.range(8):
                    lds_b0[b_lds_row_load, b_lds_col_load + c] = W[gr, gc + c]

        # Compute on buffer 1 while buffer 0 loads are in flight
        if k1 < K:
            a_op[0] = lds_a1_u32[a_read_row, a_read_u32 + 0]
            a_op[1] = lds_a1_u32[a_read_row, a_read_u32 + 1]
            b_op[0] = lds_b1_u32[b_read_k, b_read_n_u32 + 0]
            b_op[1] = lds_b1_u32[b_read_k, b_read_n_u32 + 1]
            acc = al.amdgpu.mfma_32x32x8_bf16_f32(a_op, b_op, acc)

        al.syncthreads()

    # === Output: bias add, divide, GELU activation (erf-based, matching PyTorch default) ===
    g_col = block_n + warp_col + (lane_id % 32)
    bias_val = al.convert(Bias[g_col], al.f32)
    for i in al.range(16):
        g_row = (
            block_m + warp_row
            + 8 * (i // 4) + 4 * (lane_id // 32) + (i % 4)
        )
        val = acc[i] + bias_val
        val = val / divisor_val
        # Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
        val = al.convert(0.5, al.f32) * val * (
            al.convert(1.0, al.f32)
            + al.erf(val / al.convert(1.4142135623730951, al.f32))
        )
        if g_row < M and g_col < N:
            Y[g_row, g_col] = al.convert(val, al.bf16)


class ModelNew(nn.Module):
    def __init__(self, input_size, output_size, divisor):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.divisor = divisor

    def forward(self, x):
        x_bf16 = x.to(dtype=torch.bfloat16).contiguous()
        w = self.linear.weight.t().to(device=x.device, dtype=torch.bfloat16).contiguous()
        bias = self.linear.bias.to(device=x.device, dtype=torch.bfloat16).contiguous()
        M_val = x_bf16.shape[0]
        K_val = x_bf16.shape[1]
        N_val = w.shape[1]
        y = torch.empty((M_val, N_val), device=x.device, dtype=torch.bfloat16)
        grid_m = (M_val + BLOCK_M - 1) // BLOCK_M
        grid_n = (N_val + BLOCK_N - 1) // BLOCK_N
        divisor_bits = struct.unpack("I", struct.pack("f", float(self.divisor)))[0]
        fused_gemm_gelu_kernel[lambda: ((grid_m, grid_n, 1), (THREADS_PER_BLOCK, 1, 1))](
            x_bf16, w, bias, y, M_val, K_val, N_val, divisor_bits,
        )
        return y
