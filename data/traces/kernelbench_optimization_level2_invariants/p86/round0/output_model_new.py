import struct
import torch
import torch.nn as nn

import avelang
import avelang.language as al

BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 16
THREADS_PER_BLOCK = 256
BF16_BYTES = 2


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

    lds_a = al.make_shared((64, 16), al.bf16)
    lds_b = al.make_shared((16, 64), al.bf16)
    lds_a_u32 = al.view(lds_a, al.u32, al.make_layout((64, 8), (8, 1)))
    lds_b_u32 = al.view(lds_b, al.u32, al.make_layout((16, 32), (32, 1)))

    divisor_val = al.bitcast(divisor_bits, al.f32)

    acc = al.make_local((16,), al.f32)
    for i in al.range(16):
        acc[i] = al.convert(0.0, al.f32)

    a0_op = al.make_local((2,), al.u32)
    a1_op = al.make_local((2,), al.u32)
    b0_op = al.make_local((2,), al.u32)
    b1_op = al.make_local((2,), al.u32)

    load_a = tid < 128
    a_lds_row = tid // 2
    a_lds_col = (tid % 2) * 8
    b_lt = tid - 128
    b_lds_row = b_lt // 8
    b_lds_col = (b_lt % 8) * 8

    a_read_row = warp_row + (lane_id % 32)
    a_read_u32 = (lane_id // 32) * 2
    b_read_k0 = lane_id % 8
    b_read_k1 = (lane_id % 8) + 8
    b_read_n_u32 = (lane_id // 8) * 2 + warp_col // 2

    for k_block in al.range(0, K, 16):
        if load_a:
            gr = block_m + a_lds_row
            gc = k_block + a_lds_col
            for c in al.range(8):
                lds_a[a_lds_row, a_lds_col + c] = X[gr, gc + c]
        if not load_a:
            gr = k_block + b_lds_row
            gc = block_n + b_lds_col
            for c in al.range(8):
                lds_b[b_lds_row, b_lds_col + c] = W[gr, gc + c]

        al.syncthreads()

        a0_op[0] = lds_a_u32[a_read_row, a_read_u32 + 0]
        a0_op[1] = lds_a_u32[a_read_row, a_read_u32 + 1]
        a1_op[0] = lds_a_u32[a_read_row, a_read_u32 + 4]
        a1_op[1] = lds_a_u32[a_read_row, a_read_u32 + 5]

        b0_op[0] = lds_b_u32[b_read_k0, b_read_n_u32 + 0]
        b0_op[1] = lds_b_u32[b_read_k0, b_read_n_u32 + 1]
        b1_op[0] = lds_b_u32[b_read_k1, b_read_n_u32 + 0]
        b1_op[1] = lds_b_u32[b_read_k1, b_read_n_u32 + 1]

        acc = al.amdgpu.mfma_32x32x8_bf16_f32(a0_op, b0_op, acc)
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(a1_op, b1_op, acc)

        al.syncthreads()

    g_col = block_n + warp_col + (lane_id % 32)
    bias_val = al.convert(Bias[g_col], al.f32)
    for i in al.range(16):
        g_row = (
            block_m + warp_row
            + 8 * (i // 4) + 4 * (lane_id // 32) + (i % 4)
        )
        val = acc[i] + bias_val
        val = val / divisor_val
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
