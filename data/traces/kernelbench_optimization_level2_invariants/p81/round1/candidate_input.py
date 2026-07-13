import torch
import torch.nn as nn
import avelang
import avelang.language as al

BM = al.constexpr(64)
BN = al.constexpr(64)
BK = al.constexpr(16)
WM = al.constexpr(32)
WN = al.constexpr(32)


@avelang.jit
def fused_gemm_kernel(
    X_ptr: al.Pointer(al.bf16),
    W_ptr: al.Pointer(al.bf16),
    B_ptr: al.Pointer(al.bf16),
    Y_ptr: al.Pointer(al.bf16),
    M: al.i32,
    N: al.i32,
    K: al.i32,
):
    one = al.convert(1, al.i32)
    two = al.convert(2, al.i32)
    zero = al.convert(0, al.i32)
    four = al.convert(4, al.i32)
    eight = al.convert(8, al.i32)
    sixteen = al.convert(16, al.i32)
    thirty2 = al.convert(32, al.i32)
    bf16_two = al.convert(2, al.i32)
    num_u32_a = al.convert(8, al.i32)
    num_u32_b = al.convert(32, al.i32)

    x_layout = al.make_layout((M, K), (K, one))
    X = al.make_tensor(X_ptr, al.bf16, x_layout)
    w_layout = al.make_layout((K, N), (N, one))
    W = al.make_tensor(W_ptr, al.bf16, w_layout)
    b_layout = al.make_layout((N,), (one,))
    B = al.make_tensor(B_ptr, al.bf16, b_layout)
    y_layout = al.make_layout((M, N), (N, one))
    Y = al.make_tensor(Y_ptr, al.bf16, y_layout)

    block_m = al.block_id(0)
    block_n = al.block_id(1)
    tid = al.thread_id(0)
    lane = tid & al.convert(63, al.i32)
    wave_id = tid >> al.convert(6, al.i32)
    wave_row = wave_id >> one
    wave_col = wave_id & one

    tile_m = block_m * BM
    tile_n = block_n * BN
    wr_off = wave_row * WM
    wc_off = wave_col * WN
    wc_u32 = wave_col * sixteen
    _128 = al.convert(128, al.i32)
    _7 = al.convert(7, al.i32)
    _31 = al.convert(31, al.i32)
    _3 = al.convert(3, al.i32)
    _5 = al.convert(5, al.i32)

    A_shared = al.make_shared((BM, BK), al.bf16)
    B_shared = al.make_shared((BK, BN), al.bf16)

    acc = al.make_local((16,), al.f32)
    for i in al.range(16):
        acc[i] = al.convert(0.0, al.f32)

    x_range = M * K * bf16_two
    w_range = K * N * bf16_two
    X_rsrc = al.amdgpu.make_rsrc(X, x_range)
    W_rsrc = al.amdgpu.make_rsrc(W, w_range)

    for kk in al.range(zero, K, BK):
        if tid < _128:
            a_row = tid >> one
            a_co = (tid & one) << _3
            a_off = ((tile_m + a_row) * K + kk + a_co) * bf16_two
            a_ch = al.amdgpu.raw_buffer_load_x4(X_rsrc, a_off, 0, 0)
            a_v = al.view(a_ch, al.Tensor((8,), al.bf16))
            for c in al.range(8):
                A_shared[a_row, a_co + c] = a_v[c]

        if tid < _128:
            b_row = tid >> _3
            b_co = (tid & _7) << _3
            b_off = ((kk + b_row) * N + tile_n + b_co) * bf16_two
            b_ch = al.amdgpu.raw_buffer_load_x4(W_rsrc, b_off, 0, 0)
            b_v = al.view(b_ch, al.Tensor((8,), al.bf16))
            for c in al.range(8):
                B_shared[b_row, b_co + c] = b_v[c]

        al.syncthreads()

        A_u32 = al.view(A_shared, al.u32, al.make_layout((BM, num_u32_a), (num_u32_a, one)))
        B_u32 = al.view(B_shared, al.u32, al.make_layout((BK, num_u32_b), (num_u32_b, one)))

        ar = wr_off + (lane & _31)
        ac0 = (lane >> _5) << one
        a0 = al.make_local((2,), al.u32)
        a0[0] = A_u32[ar, ac0]
        a0[1] = A_u32[ar, ac0 + one]

        bj = lane & _7
        bc0 = wc_u32 + ((lane >> _3) << one)
        b0 = al.make_local((2,), al.u32)
        b0[0] = B_u32[bj, bc0]
        b0[1] = B_u32[bj, bc0 + one]

        acc = al.amdgpu.mfma_f32_32x32x8_bf16(a0, b0, acc)

        ac1 = ac0 + four
        a1 = al.make_local((2,), al.u32)
        a1[0] = A_u32[ar, ac1]
        a1[1] = A_u32[ar, ac1 + one]

        bj1 = bj + eight
        b1 = al.make_local((2,), al.u32)
        b1[0] = B_u32[bj1, bc0]
        b1[1] = B_u32[bj1, bc0 + one]

        acc = al.amdgpu.mfma_f32_32x32x8_bf16(a1, b1, acc)

        al.syncthreads()

    oc = tile_n + wc_off + (lane & _31)
    orb = tile_m + wr_off
    bv = al.convert(B[oc], al.f32)
    n1 = al.convert(-1.0, al.f32)
    p1 = al.convert(1.0, al.f32)
    hf = al.convert(0.5, al.f32)

    for i in al.range(16):
        rr = orb + eight * (i >> two) + (lane >> _5) * four + (i & _3)
        x = acc[i] + bv
        x = x * (p1 / (p1 + al.exp(-x)))
        x = x * hf
        if x < n1:
            x = n1
        if x > p1:
            x = p1
        x = al.tanh(x)
        if x < n1:
            x = n1
        if x > p1:
            x = p1
        Y[rr, oc] = al.convert(x, al.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self._cached_w_ptr = None
        self._cached_w_t = None

    def forward(self, x):
        x = x.contiguous()
        mv = x.shape[0]
        kv = x.shape[1]
        nv = self.gemm.out_features
        wp = self.gemm.weight.data_ptr()
        if self._cached_w_ptr != wp:
            self._cached_w_t = self.gemm.weight.t().to(device=x.device, dtype=torch.bfloat16).contiguous()
            self._cached_w_ptr = wp
        wt = self._cached_w_t
        bs = self.gemm.bias.to(device=x.device, dtype=torch.bfloat16).contiguous()
        y = torch.empty((mv, nv), device=x.device, dtype=torch.bfloat16)
        gm = (mv + 63) // 64
        gn = (nv + 63) // 64
        fused_gemm_kernel[lambda: ((gm, gn, 1), (256, 1, 1))](x, wt, bs, y, mv, nv, kv)
        return y
