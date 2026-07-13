import torch
import torch.nn as nn
import avelang
import avelang.language as al

BM = al.constexpr(64)
BN = al.constexpr(64)
BK = al.constexpr(16)
BK_H = al.constexpr(8)
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
    thirty2 = al.convert(32, al.i32)
    _64 = al.convert(64, al.i32)
    _7 = al.convert(7, al.i32)
    _31 = al.convert(31, al.i32)
    _3 = al.convert(3, al.i32)
    _5 = al.convert(5, al.i32)
    _16_u32 = al.convert(16, al.i32)

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
    wc_u32 = wave_col * _16_u32

    # Double-buffered shared memory at half BK size.
    # Each buffer holds BK_H=8 K-elements. A full BK=16 tile uses two passes
    # through the same buffer, overwriting between passes.  This halves the LDS
    # working set, enabling double buffering at the same total LDS as the
    # original single-buffer full-size design.
    A_sh0 = al.make_shared((BM, BK_H), al.bf16)
    B_sh0 = al.make_shared((BK_H, BN), al.bf16)
    A_sh1 = al.make_shared((BM, BK_H), al.bf16)
    B_sh1 = al.make_shared((BK_H, BN), al.bf16)

    acc = al.make_local((16,), al.f32)
    for i in al.range(16):
        acc[i] = zero_f

    x_range = M * K * two
    w_range = K * N * two

    num_u32_a = al.convert(4, al.i32)
    num_u32_b = al.convert(32, al.i32)

    # Precompute MFMA indices (same for all passes, only the buffer changes)
    ar = wr_off + (lane & _31)
    ac0 = (lane >> _5) << one
    bj = lane & _7
    bc0 = wc_u32 + ((lane >> _3) << one)

    # Resource descriptors
    X_rsrc = al.amdgpu.make_rsrc(X, x_range)
    W_rsrc = al.amdgpu.make_rsrc(W, w_range)

    # ================================================================
    # Prefetch tile 0 -> buf0
    #   Pass 1: load K=[0:8]
    #   Pass 2: load K=[8:16]; syncthreads before loads ensures MFMA
    #           pass 1 reads are done before pass 2 writes to buf0.
    # ================================================================
    if tid < _64:
        a_row = tid
        a_off = ((tile_m + a_row) * K + zero) * two
        a_ch = al.amdgpu.raw_buffer_load_x4(X_rsrc, a_off, 0, 0)
        a_v = al.view(a_ch, al.Tensor((8,), al.bf16))
        for c in al.range(8):
            A_sh0[a_row, c] = a_v[c]
    if tid < _64:
        b_row = tid >> _3
        b_co = (tid & _7) << _3
        b_off = ((zero + b_row) * N + tile_n + b_co) * two
        b_ch = al.amdgpu.raw_buffer_load_x4(W_rsrc, b_off, 0, 0)
        b_v = al.view(b_ch, al.Tensor((8,), al.bf16))
        for c in al.range(8):
            B_sh0[b_row, b_co + c] = b_v[c]
    al.syncthreads()

    vA = al.view(A_sh0, al.u32, al.make_layout((BM, num_u32_a), (num_u32_a, one)))
    vB = al.view(B_sh0, al.u32, al.make_layout((BK_H, num_u32_b), (num_u32_b, one)))
    a0 = al.make_local((2,), al.u32)
    b0 = al.make_local((2,), al.u32)
    a0[0] = vA[ar, ac0]
    a0[1] = vA[ar, ac0 + one]
    b0[0] = vB[bj, bc0]
    b0[1] = vB[bj, bc0 + one]
    acc = al.amdgpu.mfma_f32_32x32x8_bf16(a0, b0, acc)
    al.syncthreads()

    if tid < _64:
        a_row = tid
        a_off = ((tile_m + a_row) * K + eight) * two
        a_ch = al.amdgpu.raw_buffer_load_x4(X_rsrc, a_off, 0, 0)
        a_v = al.view(a_ch, al.Tensor((8,), al.bf16))
        for c in al.range(8):
            A_sh0[a_row, c] = a_v[c]
    if tid < _64:
        b_row = tid >> _3
        b_co = (tid & _7) << _3
        b_off = ((eight + b_row) * N + tile_n + b_co) * two
        b_ch = al.amdgpu.raw_buffer_load_x4(W_rsrc, b_off, 0, 0)
        b_v = al.view(b_ch, al.Tensor((8,), al.bf16))
        for c in al.range(8):
            B_sh0[b_row, b_co + c] = b_v[c]
    al.syncthreads()

    vA2 = al.view(A_sh0, al.u32, al.make_layout((BM, num_u32_a), (num_u32_a, one)))
    vB2 = al.view(B_sh0, al.u32, al.make_layout((BK_H, num_u32_b), (num_u32_b, one)))
    a0[0] = vA2[ar, ac0]
    a0[1] = vA2[ar, ac0 + one]
    b0[0] = vB2[bj, bc0]
    b0[1] = vB2[bj, bc0 + one]
    acc = al.amdgpu.mfma_f32_32x32x8_bf16(a0, b0, acc)

    # ================================================================
    # K-loop unrolled by 2.
    #   kk tile    -> buf1 (odd tile index: 1, 3, 5, ...)
    #   kk+BK tile -> buf0 (even tile index: 2, 4, 6, ...)
    # Processes two BK=16 tiles per loop iteration, alternating buffers
    # in a ping-pong pattern.  Each tile is split into two 8-element
    # passes through the same buffer, reducing LDS working-set size.
    # ================================================================
    for kk in al.range(BK, K, BK + BK):
        kk2 = kk + BK

        # ---------- Tile kk -> buf1, pass 1 ----------
        if tid < _64:
            a_row = tid
            a_off = ((tile_m + a_row) * K + kk) * two
            a_ch = al.amdgpu.raw_buffer_load_x4(X_rsrc, a_off, 0, 0)
            a_v = al.view(a_ch, al.Tensor((8,), al.bf16))
            for c in al.range(8):
                A_sh1[a_row, c] = a_v[c]
        if tid < _64:
            b_row = tid >> _3
            b_co = (tid & _7) << _3
            b_off = ((kk + b_row) * N + tile_n + b_co) * two
            b_ch = al.amdgpu.raw_buffer_load_x4(W_rsrc, b_off, 0, 0)
            b_v = al.view(b_ch, al.Tensor((8,), al.bf16))
            for c in al.range(8):
                B_sh1[b_row, b_co + c] = b_v[c]
        al.syncthreads()

        vA1a = al.view(A_sh1, al.u32, al.make_layout((BM, num_u32_a), (num_u32_a, one)))
        vB1a = al.view(B_sh1, al.u32, al.make_layout((BK_H, num_u32_b), (num_u32_b, one)))
        a0[0] = vA1a[ar, ac0]
        a0[1] = vA1a[ar, ac0 + one]
        b0[0] = vB1a[bj, bc0]
        b0[1] = vB1a[bj, bc0 + one]
        acc = al.amdgpu.mfma_f32_32x32x8_bf16(a0, b0, acc)
        al.syncthreads()

        # ---------- Tile kk -> buf1, pass 2 ----------
        k_off = kk + eight
        if tid < _64:
            a_row = tid
            a_off = ((tile_m + a_row) * K + k_off) * two
            a_ch = al.amdgpu.raw_buffer_load_x4(X_rsrc, a_off, 0, 0)
            a_v = al.view(a_ch, al.Tensor((8,), al.bf16))
            for c in al.range(8):
                A_sh1[a_row, c] = a_v[c]
        if tid < _64:
            b_row = tid >> _3
            b_co = (tid & _7) << _3
            b_off = ((k_off + b_row) * N + tile_n + b_co) * two
            b_ch = al.amdgpu.raw_buffer_load_x4(W_rsrc, b_off, 0, 0)
            b_v = al.view(b_ch, al.Tensor((8,), al.bf16))
            for c in al.range(8):
                B_sh1[b_row, b_co + c] = b_v[c]
        al.syncthreads()

        vA1b = al.view(A_sh1, al.u32, al.make_layout((BM, num_u32_a), (num_u32_a, one)))
        vB1b = al.view(B_sh1, al.u32, al.make_layout((BK_H, num_u32_b), (num_u32_b, one)))
        a0[0] = vA1b[ar, ac0]
        a0[1] = vA1b[ar, ac0 + one]
        b0[0] = vB1b[bj, bc0]
        b0[1] = vB1b[bj, bc0 + one]
        acc = al.amdgpu.mfma_f32_32x32x8_bf16(a0, b0, acc)

        # ---------- Tile kk+BK -> buf0  (only if kk2 < K) ----------
        if kk2 < K:
            al.syncthreads()

            if tid < _64:
                a_row = tid
                a_off = ((tile_m + a_row) * K + kk2) * two
                a_ch = al.amdgpu.raw_buffer_load_x4(X_rsrc, a_off, 0, 0)
                a_v = al.view(a_ch, al.Tensor((8,), al.bf16))
                for c in al.range(8):
                    A_sh0[a_row, c] = a_v[c]
            if tid < _64:
                b_row = tid >> _3
                b_co = (tid & _7) << _3
                b_off = ((kk2 + b_row) * N + tile_n + b_co) * two
                b_ch = al.amdgpu.raw_buffer_load_x4(W_rsrc, b_off, 0, 0)
                b_v = al.view(b_ch, al.Tensor((8,), al.bf16))
                for c in al.range(8):
                    B_sh0[b_row, b_co + c] = b_v[c]
            al.syncthreads()

            vA0a = al.view(A_sh0, al.u32, al.make_layout((BM, num_u32_a), (num_u32_a, one)))
            vB0a = al.view(B_sh0, al.u32, al.make_layout((BK_H, num_u32_b), (num_u32_b, one)))
            a0[0] = vA0a[ar, ac0]
            a0[1] = vA0a[ar, ac0 + one]
            b0[0] = vB0a[bj, bc0]
            b0[1] = vB0a[bj, bc0 + one]
            acc = al.amdgpu.mfma_f32_32x32x8_bf16(a0, b0, acc)
            al.syncthreads()

            k_off2 = kk2 + eight
            if tid < _64:
                a_row = tid
                a_off = ((tile_m + a_row) * K + k_off2) * two
                a_ch = al.amdgpu.raw_buffer_load_x4(X_rsrc, a_off, 0, 0)
                a_v = al.view(a_ch, al.Tensor((8,), al.bf16))
                for c in al.range(8):
                    A_sh0[a_row, c] = a_v[c]
            if tid < _64:
                b_row = tid >> _3
                b_co = (tid & _7) << _3
                b_off = ((k_off2 + b_row) * N + tile_n + b_co) * two
                b_ch = al.amdgpu.raw_buffer_load_x4(W_rsrc, b_off, 0, 0)
                b_v = al.view(b_ch, al.Tensor((8,), al.bf16))
                for c in al.range(8):
                    B_sh0[b_row, b_co + c] = b_v[c]
            al.syncthreads()

            vA0b = al.view(A_sh0, al.u32, al.make_layout((BM, num_u32_a), (num_u32_a, one)))
            vB0b = al.view(B_sh0, al.u32, al.make_layout((BK_H, num_u32_b), (num_u32_b, one)))
            a0[0] = vA0b[ar, ac0]
            a0[1] = vA0b[ar, ac0 + one]
            b0[0] = vB0b[bj, bc0]
            b0[1] = vB0b[bj, bc0 + one]
            acc = al.amdgpu.mfma_f32_32x32x8_bf16(a0, b0, acc)

    # ================================================================
    # Epilogue: swish, /2, clamp[-1,1], tanh, clamp[-1,1], store
    # ================================================================
    al.syncthreads()

    oc = tile_n + wc_off + (lane & _31)
    orb = tile_m + wr_off
    n1 = al.convert(-1.0, al.f32)
    p1 = al.convert(1.0, al.f32)
    hf = al.convert(0.5, al.f32)

    bv = al.convert(B[oc], al.f32)

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
        self._cached_w_t = None

    def forward(self, x):
        x = x.contiguous()
        mv = x.shape[0]
        kv = x.shape[1]
        nv = self.gemm.out_features
        wp = self.gemm.weight.data_ptr()
        if self._cached_w_ptr != wp:
            self._cached_w_t = (
                self.gemm.weight.t()
                .to(device=x.device, dtype=torch.bfloat16)
                .contiguous()
            )
            self._cached_w_ptr = wp
        wt = self._cached_w_t
        bs = self.gemm.bias.to(device=x.device, dtype=torch.bfloat16).contiguous()
        y = torch.empty((mv, nv), device=x.device, dtype=torch.bfloat16)
        gm = (mv + 63) // 64
        gn = (nv + 63) // 64
        fused_gemm_kernel[lambda: ((gm, gn, 1), (256, 1, 1))](
            x, wt, bs, y, mv, nv, kv
        )
        return y
