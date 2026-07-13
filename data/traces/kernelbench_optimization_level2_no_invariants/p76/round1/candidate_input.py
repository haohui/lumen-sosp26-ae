import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 32
WARP_M = 32
WARP_N = 32
NUM_WARPS = 4
THREADS_PER_WARP = 64
THREADS_PER_BLOCK = NUM_WARPS * THREADS_PER_WARP
BF16_BYTES = 2


@avelang.jit
def fused_kernel(
    X_ptr: al.Pointer(al.bf16),
    W_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    Y_ptr: al.Pointer(al.bf16),
    M: al.u32,
    N: al.u32,
    K: al.u32,
):
    X = al.make_tensor(X_ptr, al.bf16, al.make_layout((M, K), (K, 1)))
    W = al.make_tensor(W_ptr, al.bf16, al.make_layout((K, N), (N, 1)))
    bias = al.make_tensor(bias_ptr, al.bf16, al.make_layout((N,), (1,)))
    Y = al.make_tensor(Y_ptr, al.bf16, al.make_layout((M, N), (N, 1)))

    c0 = al.convert(0, al.u32)
    c2 = al.convert(2, al.u32)
    c4 = al.convert(4, al.u32)
    c8 = al.convert(8, al.u32)
    c16 = al.convert(16, al.u32)
    c24 = al.convert(24, al.u32)
    c32 = al.convert(32, al.u32)
    c64 = al.convert(64, al.u32)

    block_m = al.convert(al.block_id(0), al.u32)
    block_n = al.convert(al.block_id(1), al.u32)

    tid = al.convert(al.thread_id(0), al.u32)
    warp_id = tid // c64
    warp_row = warp_id // c2
    warp_col = warp_id % c2
    lane_id = tid % c64

    warp_m_c = al.convert(WARP_M, al.u32)
    warp_n_c = al.convert(WARP_N, al.u32)
    block_m_c = al.convert(BLOCK_M, al.u32)
    block_n_c = al.convert(BLOCK_N, al.u32)

    row_start = block_m * block_m_c + warp_row * warp_m_c
    col_start = block_n * block_n_c + warp_col * warp_n_c

    acc = al.make_local((16,), al.f32)
    for i in al.range(16):
        acc[i] = al.convert(0.0, al.f32)

    A_s = al.make_shared((BLOCK_M, BLOCK_K), al.bf16)
    B_s = al.make_shared((BLOCK_K, BLOCK_N), al.bf16)

    range_X = M * K * al.convert(BF16_BYTES, al.u32)
    rsrc_X = al.amdgpu.make_rsrc(X, range_X)
    range_W = K * N * al.convert(BF16_BYTES, al.u32)
    rsrc_W = al.amdgpu.make_rsrc(W, range_W)

    block_row_base = block_m * block_m_c
    block_col_base = block_n * block_n_c

    for k_block in al.range(0, K, BLOCK_K):
        k_block_u32 = al.convert(k_block, al.u32)

        t8 = tid * c8
        a_row = t8 // c32
        a_col = t8 % c32
        byte_off_a = (
            (block_row_base + a_row) * K + k_block_u32 + a_col
        ) * al.convert(BF16_BYTES, al.u32)
        frag_a = al.amdgpu.raw_buffer_load_x4(rsrc_X, byte_off_a, 0, 0)
        frag_a_bf16 = al.view(frag_a, al.Tensor((8,), al.bf16))
        for i in al.range(8):
            A_s[a_row, a_col + al.convert(i, al.u32)] = frag_a_bf16[i]

        b_row = t8 // c64
        b_col = t8 % c64
        byte_off_b = (
            (k_block_u32 + b_row) * N + block_col_base + b_col
        ) * al.convert(BF16_BYTES, al.u32)
        frag_b = al.amdgpu.raw_buffer_load_x4(rsrc_W, byte_off_b, 0, 0)
        frag_b_bf16 = al.view(frag_b, al.Tensor((8,), al.bf16))
        for i in al.range(8):
            B_s[b_row, b_col + al.convert(i, al.u32)] = frag_b_bf16[i]

        al.syncthreads()

        warp_m_off = warp_row * warp_m_c
        warp_n_off = warp_col * warp_n_c
        t4 = lane_id * c4

        # MFMA call 0: K offset 0
        a_row0 = warp_m_off + (t4 // c8)
        a_col0 = t4 % c8
        a_bf16_0 = al.make_local((4,), al.bf16)
        for i in al.range(4):
            a_bf16_0[i] = A_s[a_row0, a_col0 + al.convert(i, al.u32)]
        a_u32_0 = al.view(a_bf16_0, al.Tensor((2,), al.u32))

        b_row0 = t4 // c32
        b_col0 = warp_n_off + (t4 % c32)
        b_bf16_0 = al.make_local((4,), al.bf16)
        for i in al.range(4):
            b_bf16_0[i] = B_s[b_row0, b_col0 + al.convert(i, al.u32)]
        b_u32_0 = al.view(b_bf16_0, al.Tensor((2,), al.u32))

        acc = al.amdgpu.mfma_32x32x8_bf16_f32(a_u32_0, b_u32_0, acc)

        # MFMA call 1: K offset 8
        a_bf16_1 = al.make_local((4,), al.bf16)
        for i in al.range(4):
            a_bf16_1[i] = A_s[a_row0, c8 + a_col0 + al.convert(i, al.u32)]
        a_u32_1 = al.view(a_bf16_1, al.Tensor((2,), al.u32))

        b_bf16_1 = al.make_local((4,), al.bf16)
        for i in al.range(4):
            b_bf16_1[i] = B_s[c8 + b_row0, b_col0 + al.convert(i, al.u32)]
        b_u32_1 = al.view(b_bf16_1, al.Tensor((2,), al.u32))

        acc = al.amdgpu.mfma_32x32x8_bf16_f32(a_u32_1, b_u32_1, acc)

        # MFMA call 2: K offset 16
        a_bf16_2 = al.make_local((4,), al.bf16)
        for i in al.range(4):
            a_bf16_2[i] = A_s[a_row0, c16 + a_col0 + al.convert(i, al.u32)]
        a_u32_2 = al.view(a_bf16_2, al.Tensor((2,), al.u32))

        b_bf16_2 = al.make_local((4,), al.bf16)
        for i in al.range(4):
            b_bf16_2[i] = B_s[c16 + b_row0, b_col0 + al.convert(i, al.u32)]
        b_u32_2 = al.view(b_bf16_2, al.Tensor((2,), al.u32))

        acc = al.amdgpu.mfma_32x32x8_bf16_f32(a_u32_2, b_u32_2, acc)

        # MFMA call 3: K offset 24
        a_bf16_3 = al.make_local((4,), al.bf16)
        for i in al.range(4):
            a_bf16_3[i] = A_s[a_row0, c24 + a_col0 + al.convert(i, al.u32)]
        a_u32_3 = al.view(a_bf16_3, al.Tensor((2,), al.u32))

        b_bf16_3 = al.make_local((4,), al.bf16)
        for i in al.range(4):
            b_bf16_3[i] = B_s[c24 + b_row0, b_col0 + al.convert(i, al.u32)]
        b_u32_3 = al.view(b_bf16_3, al.Tensor((2,), al.u32))

        acc = al.amdgpu.mfma_32x32x8_bf16_f32(a_u32_3, b_u32_3, acc)

        al.syncthreads()

    f32_zero = al.convert(0.0, al.f32)

    for i in al.range(16):
        linear_idx = lane_id * c16 + al.convert(i, al.u32)
        out_row = row_start + (linear_idx // warp_n_c)
        out_col = col_start + (linear_idx % warp_n_c)

        val = acc[i]
        bias_val = al.convert(bias[out_col], al.f32)
        val = val + bias_val
        if val < f32_zero:
            val = f32_zero
        Y[out_row, out_col] = al.convert(val, al.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias_shape):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        Bv = x.shape[0]
        Kv = x.shape[1]
        Nv = self.bias.shape[0]

        w_t = self.gemm.weight.t().to(device=x.device, dtype=x.dtype).contiguous()
        eb = self.bias.to(device=x.device, dtype=x.dtype).contiguous()
        y = torch.empty((Bv, Nv), device=x.device, dtype=x.dtype)

        gm = (Bv + BLOCK_M - 1) // BLOCK_M
        gn = (Nv + BLOCK_N - 1) // BLOCK_N

        fused_kernel[lambda: ((gm, gn, 1), (THREADS_PER_BLOCK, 1, 1))](
            x.contiguous(), w_t, eb, y, Bv, Nv, Kv
        )
        return y
