import torch
import torch.nn as nn
import avelang
import avelang.language as al

SQRT_2 = 1.4142135623730951
BATCH = 1024
IN_DIM = 8192
OUT_DIM = 8192
DIV = 10.0
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 32
WARP_M = 32; WARP_N = 32
THREADS = 256
BF16_BYTES = 2

@avelang.jit
def fused_kernel(
    X_ptr: al.Pointer(al.bf16), W_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16), Y_ptr: al.Pointer(al.bf16),
    M: al.u32, N: al.u32, K: al.u32,
):
    X = al.make_tensor(X_ptr, al.bf16, al.make_layout((M, K), (K, 1)))
    W = al.make_tensor(W_ptr, al.bf16, al.make_layout((N, K), (K, 1)))
    bias = al.make_tensor(bias_ptr, al.bf16, al.make_layout((N,), (1,)))
    Y = al.make_tensor(Y_ptr, al.bf16, al.make_layout((M, N), (N, 1)))

    c0 = al.convert(0, al.u32); c2 = al.convert(2, al.u32)
    c4 = al.convert(4, al.u32); c8 = al.convert(8, al.u32)
    c16 = al.convert(16, al.u32); c32 = al.convert(32, al.u32)
    c64 = al.convert(64, al.u32)

    block_m = al.convert(al.block_id(0), al.u32)
    block_n = al.convert(al.block_id(1), al.u32)
    tid = al.convert(al.thread_id(0), al.u32)
    warp_id = tid // c64; warp_row = warp_id // c2; warp_col = warp_id % c2
    lane = tid % c64

    block_m_c = al.convert(BLOCK_M, al.u32); block_n_c = al.convert(BLOCK_N, al.u32)
    warp_m_c = al.convert(WARP_M, al.u32); warp_n_c = al.convert(WARP_N, al.u32)
    row_start = block_m * block_m_c + warp_row * warp_m_c
    col_start = block_n * block_n_c + warp_col * warp_n_c
    block_row_base = block_m * block_m_c
    block_col_base = block_n * block_n_c

    acc = al.make_local((16,), al.f32)
    for i in al.range(16): acc[i] = al.convert(0.0, al.f32)

    A_s = al.make_shared((BLOCK_M, BLOCK_K), al.bf16)
    B_s = al.make_shared((BLOCK_K, BLOCK_N), al.bf16)

    range_X = M * K * al.convert(BF16_BYTES, al.u32)
    rsrc_X = al.amdgpu.make_rsrc(X, range_X)
    range_W = N * K * al.convert(BF16_BYTES, al.u32)
    rsrc_W = al.amdgpu.make_rsrc(W, range_W)

    # Precompute per-thread MFMA indices
    kgrp = lane & al.convert(3, al.u32)     # 0..3
    rgrp = lane >> c2                        # 0..15
    a_row0 = warp_row * warp_m_c + rgrp * c2
    b_col_base = warp_col * warp_n_c

    for k_block in al.range(0, K, BLOCK_K):
        k_block_u32 = al.convert(k_block, al.u32)

        # Load A: X[bm*64:bm*64+64, kb:kb+32] -> As
        t8 = tid * c8
        a_row = t8 // c32; a_col = t8 % c32
        byte_a = ((block_row_base + a_row) * K + k_block_u32 + a_col) * al.convert(BF16_BYTES, al.u32)
        fa = al.amdgpu.raw_buffer_load_x4(rsrc_X, byte_a, 0, 0)
        fa_bf16 = al.view(fa, al.Tensor((8,), al.bf16))
        for i in al.range(8): A_s[a_row, a_col + al.convert(i, al.u32)] = fa_bf16[i]

        # Load B: W[bn*64:bn*64+64, kb:kb+32] -> Bs (transposed: Bs[k,n] = W[n,k])
        b_row_k = t8 // c64; b_col_n = t8 % c64
        byte_b = ((block_col_base + b_col_n) * K + k_block_u32 + b_row_k) * al.convert(BF16_BYTES, al.u32)
        fb = al.amdgpu.raw_buffer_load_x4(rsrc_W, byte_b, 0, 0)
        fb_bf16 = al.view(fb, al.Tensor((8,), al.bf16))
        for i in al.range(8): B_s[b_row_k, b_col_n + al.convert(i, al.u32)] = fb_bf16[i]

        al.syncthreads()

        for ks in al.range(0, BLOCK_K, 8):
            ak0 = al.convert(ks, al.u32) + kgrp * c2
            ak1 = ak0 + al.convert(1, al.u32)
            a_row1 = a_row0 + al.convert(1, al.u32)

            af = al.make_local((4,), al.bf16)
            af[0] = A_s[a_row0, ak0]; af[1] = A_s[a_row0, ak1]
            af[2] = A_s[a_row1, ak0]; af[3] = A_s[a_row1, ak1]
            au = al.view(af, al.Tensor((2,), al.u32))

            bk0 = al.convert(ks, al.u32) + kgrp * c2
            bk1 = bk0 + al.convert(1, al.u32)
            bc0 = ((b_col_base + rgrp * c2) % warp_n_c)
            bc1 = (bc0 + c16) % warp_n_c

            bf = al.make_local((4,), al.bf16)
            bf[0] = B_s[bk0, bc0]; bf[1] = B_s[bk1, bc0]
            bf[2] = B_s[bk0, bc1]; bf[3] = B_s[bk1, bc1]
            bu = al.view(bf, al.Tensor((2,), al.u32))

            acc = al.amdgpu.mfma_32x32x8_bf16_f32(au, bu, acc)

        al.syncthreads()

    SQRT2 = al.convert(SQRT_2, al.f32)
    HALF = al.convert(0.5, al.f32); ONE = al.convert(1.0, al.f32)
    DIVISOR = al.convert(DIV, al.f32)

    for i in al.range(16):
        linear_idx = lane * c16 + al.convert(i, al.u32)
        out_row = row_start + (linear_idx // warp_n_c)
        out_col = col_start + (linear_idx % warp_n_c)
        val = acc[i] + al.convert(bias[out_col], al.f32)
        val = val / DIVISOR
        val = HALF * val * (ONE + al.erf(val / SQRT2))
        Y[out_row, out_col] = al.convert(val, al.bf16)


class ModelNew(nn.Module):
    def __init__(self, input_size, output_size, divisor):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.divisor = divisor
    def forward(self, x):
        weight = self.linear.weight.to(device=x.device, dtype=x.dtype).contiguous()
        bias = self.linear.bias.to(device=x.device, dtype=x.dtype).contiguous()
        y = torch.zeros((BATCH, OUT_DIM), device=x.device, dtype=x.dtype)
        grid_m = (BATCH + BLOCK_M - 1) // BLOCK_M
        grid_n = (OUT_DIM + BLOCK_N - 1) // BLOCK_N
        fused_kernel[lambda: ((grid_m, grid_n, 1), (THREADS, 1, 1))](
            x.contiguous(), weight, bias, y, BATCH, OUT_DIM, IN_DIM,
        )
        return y
