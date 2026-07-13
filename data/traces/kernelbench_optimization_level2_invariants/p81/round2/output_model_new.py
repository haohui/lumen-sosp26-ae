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
    zero_f = al.convert(0.0, al.f32)
    four = al.convert(4, al.i32)
    eight = al.convert(8, al.i32)
    _32 = al.convert(32, al.i32)
    _64 = al.convert(64, al.i32)
    _31 = al.convert(31, al.i32)
    _3 = al.convert(3, al.i32)
    _5 = al.convert(5, al.i32)
    _16_u32 = al.convert(16, al.i32)
    _127 = al.convert(127, al.i32)
    _63 = al.convert(63, al.i32)
    _16 = al.convert(16, al.i32)
    _4 = al.convert(4, al.i32)

    x_layout = al.make_layout((M, K), (K, one))
    X = al.make_tensor(X_ptr, al.bf16, x_layout)
    # W is (N, K) = (out_features, in_features) — uses 1D layout for correct K-consecutive loading
    w_1d = al.make_tensor(W_ptr, al.bf16, al.make_layout((N * K,), (one,)))
    b_layout = al.make_layout((N,), (one,))
    Bb = al.make_tensor(B_ptr, al.bf16, b_layout)
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

    A_sh = al.make_shared((BM, BK), al.bf16)
    B_sh_nm = al.make_shared((BN, BK), al.bf16)

    acc = al.make_local((16,), al.f32)
    for i in al.range(16):
        acc[i] = zero_f

    x_range = M * K * two
    w_range = N * K * two

    X_rsrc = al.amdgpu.make_rsrc(X, x_range)
    W_rsrc = al.amdgpu.make_rsrc(w_1d, w_range)

    num_u32_a = al.convert(8, al.i32)
    num_u32_b = al.convert(8, al.i32)

    ar = wr_off + (lane & _31)
    ac0 = (lane >> _5) << one
    bn = wc_off + (lane & _31)
    bk0 = (lane >> _5) << one

    a_row_l = tid & _63
    b_slot = tid & _127
    b_n_row = b_slot >> one
    b_k_half = b_slot & one

    for kk in al.range(zero, K, BK):
        # Load A: M rows, K cols
        a_off1 = ((tile_m + a_row_l) * K + kk) * two
        a_ch1 = al.amdgpu.raw_buffer_load_x4(X_rsrc, a_off1, 0, 0)
        a_v1 = al.view(a_ch1, al.Tensor((8,), al.bf16))
        for c in al.range(8):
            A_sh[a_row_l, c] = a_v1[c]

        a_off2 = ((tile_m + a_row_l) * K + kk + eight) * two
        a_ch2 = al.amdgpu.raw_buffer_load_x4(X_rsrc, a_off2, 0, 0)
        a_v2 = al.view(a_ch2, al.Tensor((8,), al.bf16))
        for c in al.range(8):
            A_sh[a_row_l, eight + c] = a_v2[c]

        # Load B: (N, K) 1D layout, offset = n_pos * K + k_off
        b_off = ((tile_n + b_n_row) * K + kk + b_k_half * eight) * two
        b_ch = al.amdgpu.raw_buffer_load_x4(W_rsrc, b_off, 0, 0)
        b_v = al.view(b_ch, al.Tensor((8,), al.bf16))
        for c in al.range(8):
            B_sh_nm[b_n_row, b_k_half * eight + c] = b_v[c]

        al.syncthreads()

        vA = al.view(A_sh, al.u32, al.make_layout((BM, num_u32_a), (num_u32_a, one)))
        vB = al.view(B_sh_nm, al.u32, al.make_layout((BN, num_u32_b), (num_u32_b, one)))
        a0 = al.make_local((2,), al.u32)
        b0 = al.make_local((2,), al.u32)

        a0[0] = vA[ar, ac0]
        a0[1] = vA[ar, ac0 + one]
        b0[0] = vB[bn, bk0]
        b0[1] = vB[bn, bk0 + one]
        acc = al.amdgpu.mfma_f32_32x32x8_bf16(a0, b0, acc)

        a0[0] = vA[ar, _4 + ac0]
        a0[1] = vA[ar, _4 + ac0 + one]
        b0[0] = vB[bn, _4 + bk0]
        b0[1] = vB[bn, _4 + bk0 + one]
        acc = al.amdgpu.mfma_f32_32x32x8_bf16(a0, b0, acc)

        al.syncthreads()

    al.syncthreads()

    oc = tile_n + wc_off + (lane & _31)
    orb = tile_m + wr_off
    n1 = al.convert(-1.0, al.f32)
    p1 = al.convert(1.0, al.f32)
    hf = al.convert(0.5, al.f32)

    bv = al.convert(Bb[oc], al.f32)

    for i in al.range(16):
        rr = orb + eight * (i >> two) + (lane >> _5) * four + (i & _3)
        x = acc[i] + bv
        neg_x = zero_f - x
        x = x * (p1 / (p1 + al.exp(neg_x)))
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
        self._cached_w = None

    def forward(self, x):
        x = x.to(dtype=torch.bfloat16).contiguous()
        mv = x.shape[0]
        kv = x.shape[1]
        nv = self.gemm.out_features
        wp = self.gemm.weight.data_ptr()
        if self._cached_w_ptr != wp:
            self._cached_w = (
                self.gemm.weight
                .to(device=x.device, dtype=torch.bfloat16)
                .contiguous()
            )
            self._cached_w_ptr = wp
        w = self._cached_w
        bs = self.gemm.bias.to(device=x.device, dtype=torch.bfloat16).contiguous()
        y = torch.empty((mv, nv), device=x.device, dtype=torch.bfloat16)
        gm = (mv + 63) // 64
        gn = (nv + 63) // 64
        fused_gemm_kernel[lambda: ((gm, gn, 1), (256, 1, 1))](
            x, w, bs, y, mv, nv, kv
        )
        return y
