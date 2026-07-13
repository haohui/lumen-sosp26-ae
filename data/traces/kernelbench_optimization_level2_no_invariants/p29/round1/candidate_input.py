import torch
import torch.nn as nn
import avelang
import avelang.language as al

BM = 64
BN = 64
BK = 32
WAVE_SIZE = 64
NUM_WAVES = 4
MFMA_M = 32
MFMA_N = 32
MFMA_K = 8
BF16_BYTES = 2


@avelang.jit
def fused_kernel(
    X_ptr: al.Pointer(al.bf16),
    W_ptr: al.Pointer(al.bf16),
    Bias_ptr: al.Pointer(al.bf16),
    Y_ptr: al.Pointer(al.bf16),
    batch: al.i32,
    in_dim: al.i32,
    out_dim: al.i32,
):
    tid = al.thread_id(0)
    bm = al.block_id(0)
    bn = al.block_id(1)
    ms = bm * BM
    ns = bn * BN

    X = al.make_tensor(X_ptr, al.bf16,
                       al.make_layout((batch, in_dim), (in_dim, 1)))
    W = al.make_tensor(W_ptr, al.bf16,
                       al.make_layout((in_dim, out_dim), (in_dim, 1)))
    Bias = al.make_tensor(Bias_ptr, al.bf16,
                          al.make_layout((out_dim,), (1,)))
    Y = al.make_tensor(Y_ptr, al.bf16,
                       al.make_layout((batch, out_dim), (out_dim, 1)))

    rsrc_X = al.amdgpu.make_rsrc(X, batch * in_dim * BF16_BYTES)
    rsrc_W = al.amdgpu.make_rsrc(W, in_dim * out_dim * BF16_BYTES)

    As = al.make_shared((BM, BK), al.bf16)
    Bs = al.make_shared((BK, BN), al.bf16)
    As_u32 = al.view(As, al.u32,
                     al.make_layout((BM, BK // 2), (BK // 2, 1)))
    Bs_u32 = al.view(Bs, al.u32,
                     al.make_layout((BK, BN // 2), (BN // 2, 1)))

    acc = al.make_local((16,), al.f32)
    for i in al.range(16):
        acc[i] = al.convert(0.0, al.f32)

    wid = tid // WAVE_SIZE
    wm = wid // 2
    wn = wid & 1
    lane = tid % WAVE_SIZE

    nk = in_dim // BK
    for kb in al.range(nk):
        k0 = kb * BK
        art = tid // 4
        act = (tid % 4) * 8
        ab = ((ms + art) * in_dim + k0 + act) * BF16_BYTES
        ar = al.amdgpu.raw_buffer_load_x4(rsrc_X, ab, 0, 0)
        au = al.view(ar, al.Tensor((4,), al.u32))
        ac = act >> 1
        As_u32[art, ac + 0] = au[0]
        As_u32[art, ac + 1] = au[1]
        As_u32[art, ac + 2] = au[2]
        As_u32[art, ac + 3] = au[3]

        brt = tid // 8
        bct = (tid % 8) * 8
        bb = ((ns + bct) * out_dim + k0 + brt) * BF16_BYTES
        br = al.amdgpu.raw_buffer_load_x4(rsrc_W, bb, 0, 0)
        bu = al.view(br, al.Tensor((4,), al.u32))
        bc = bct >> 1
        Bs_u32[brt, bc + 0] = bu[0]
        Bs_u32[brt, bc + 1] = bu[1]
        Bs_u32[brt, bc + 2] = bu[2]
        Bs_u32[brt, bc + 3] = bu[3]

        al.syncthreads()

        for ks in al.range(0, BK, MFMA_K):
            off = lane << 2
            ar_ = off // MFMA_K
            ac_ = off % MFMA_K
            a_frag = al.make_local((2,), al.u32)
            a_frag[0] = As_u32[wm * MFMA_M + ar_, (ks + ac_) >> 1]
            a_frag[1] = As_u32[wm * MFMA_M + ar_, (ks + ac_ + 2) >> 1]

            br_ = off // MFMA_N
            bc_ = off % MFMA_N
            b_frag = al.make_local((2,), al.u32)
            b_frag[0] = Bs_u32[ks + br_, (wn * MFMA_N + bc_) >> 1]
            b_frag[1] = Bs_u32[ks + br_, (wn * MFMA_N + bc_ + 2) >> 1]

            acc = al.amdgpu.mfma_32x32x8_bf16_f32(a_frag, b_frag, acc)

        al.syncthreads()

    for i in al.range(16):
        ri = i >> 2
        ci = i & 3
        gr = ms + wm * MFMA_M + ((lane >> 3) << 2) + ri
        gc = ns + wn * MFMA_N + ((lane & 7) << 2) + ci
        if gr < batch and gc < out_dim:
            x = acc[i] + al.convert(Bias[gc], al.f32)
            x = x * al.tanh(al.log(al.convert(1.0, al.f32) + al.exp(x)))
            x = x * al.tanh(al.log(al.convert(1.0, al.f32) + al.exp(x)))
            Y[gr, gc] = al.convert(x, al.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        Bv, Kv = x.shape
        Nv = self.linear.out_features
        w_t = self.linear.weight.t().to(device=x.device,
                                        dtype=torch.bfloat16).contiguous()
        bias = self.linear.bias.to(device=x.device, dtype=torch.bfloat16)
        xb = x.to(dtype=torch.bfloat16).contiguous()
        y = torch.empty((Bv, Nv), device=x.device, dtype=torch.bfloat16)
        grid = ((Bv + BM - 1) // BM, (Nv + BN - 1) // BN, 1)
        block = (NUM_WAVES * WAVE_SIZE, 1, 1)
        fused_kernel[lambda: (grid, block)](xb, w_t, bias, y, Bv, Kv, Nv)
        return y
