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
                       al.make_layout((in_dim, out_dim), (out_dim, 1)))
    Bias = al.make_tensor(Bias_ptr, al.bf16,
                          al.make_layout((out_dim,), (1,)))
    Y = al.make_tensor(Y_ptr, al.bf16,
                       al.make_layout((batch, out_dim), (out_dim, 1)))

    rsrc_X = al.amdgpu.make_rsrc(X, batch * in_dim * BF16_BYTES)
    rsrc_W = al.amdgpu.make_rsrc(W, in_dim * out_dim * BF16_BYTES)
    rsrc_Y = al.amdgpu.make_rsrc(Y, batch * out_dim * BF16_BYTES)

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

    off = lane << 2
    ar_ = off // MFMA_K
    ac_ = off % MFMA_K
    br_ = off // MFMA_N
    bc_ = off % MFMA_N

    art_a = tid // 4
    act_a = (tid % 4) * 8
    ac_a = act_a >> 1
    brt_b = tid // 8
    bct_b = (tid % 8) * 8
    bc_b = bct_b >> 1

    nk = in_dim // BK

    for k_tile in al.range(nk):
        ks = k_tile * BK
        ab = ((ms + art_a) * in_dim + ks + act_a) * BF16_BYTES
        ar = al.amdgpu.raw_buffer_load_x4(rsrc_X, ab, 0, 0)
        au = al.view(ar, al.Tensor((4,), al.u32))
        As_u32[art_a, ac_a + 0] = au[0]
        As_u32[art_a, ac_a + 1] = au[1]
        As_u32[art_a, ac_a + 2] = au[2]
        As_u32[art_a, ac_a + 3] = au[3]

        bb = ((ks + brt_b) * out_dim + ns + bct_b) * BF16_BYTES
        br = al.amdgpu.raw_buffer_load_x4(rsrc_W, bb, 0, 0)
        bu = al.view(br, al.Tensor((4,), al.u32))
        Bs_u32[brt_b, bc_b + 0] = bu[0]
        Bs_u32[brt_b, bc_b + 1] = bu[1]
        Bs_u32[brt_b, bc_b + 2] = bu[2]
        Bs_u32[brt_b, bc_b + 3] = bu[3]

        al.syncthreads()

        af0 = al.make_local((2,), al.u32)
        af0[0] = As_u32[wm * MFMA_M + ar_, (0 + ac_) >> 1]
        af0[1] = As_u32[wm * MFMA_M + ar_, (0 + ac_ + 2) >> 1]
        bf0 = al.make_local((2,), al.u32)
        bf0[0] = Bs_u32[0 + br_, (wn * MFMA_N + bc_) >> 1]
        bf0[1] = Bs_u32[0 + br_, (wn * MFMA_N + bc_ + 2) >> 1]
        af8 = al.make_local((2,), al.u32)
        af8[0] = As_u32[wm * MFMA_M + ar_, (8 + ac_) >> 1]
        af8[1] = As_u32[wm * MFMA_M + ar_, (8 + ac_ + 2) >> 1]
        bf8 = al.make_local((2,), al.u32)
        bf8[0] = Bs_u32[8 + br_, (wn * MFMA_N + bc_) >> 1]
        bf8[1] = Bs_u32[8 + br_, (wn * MFMA_N + bc_ + 2) >> 1]
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(af0, bf0, acc)
        af16 = al.make_local((2,), al.u32)
        af16[0] = As_u32[wm * MFMA_M + ar_, (16 + ac_) >> 1]
        af16[1] = As_u32[wm * MFMA_M + ar_, (16 + ac_ + 2) >> 1]
        bf16 = al.make_local((2,), al.u32)
        bf16[0] = Bs_u32[16 + br_, (wn * MFMA_N + bc_) >> 1]
        bf16[1] = Bs_u32[16 + br_, (wn * MFMA_N + bc_ + 2) >> 1]
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(af8, bf8, acc)
        af24 = al.make_local((2,), al.u32)
        af24[0] = As_u32[wm * MFMA_M + ar_, (24 + ac_) >> 1]
        af24[1] = As_u32[wm * MFMA_M + ar_, (24 + ac_ + 2) >> 1]
        bf24 = al.make_local((2,), al.u32)
        bf24[0] = Bs_u32[24 + br_, (wn * MFMA_N + bc_) >> 1]
        bf24[1] = Bs_u32[24 + br_, (wn * MFMA_N + bc_ + 2) >> 1]
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(af16, bf16, acc)
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(af24, bf24, acc)

        al.syncthreads()

    for ri in al.range(4):
        gr = ms + wm * MFMA_M + ((lane >> 3) << 2) + ri
        gc_base = ns + wn * MFMA_N + ((lane & 7) << 2)

        out_bf16 = al.make_local((4,), al.bf16)

        v = acc[ri * 4 + 0] + al.convert(Bias[gc_base + 0], al.f32)
        v = v * al.tanh(al.log(al.convert(1.0, al.f32) + al.exp(v)))
        v = v * al.tanh(al.log(al.convert(1.0, al.f32) + al.exp(v)))
        out_bf16[0] = al.convert(v, al.bf16)

        v = acc[ri * 4 + 1] + al.convert(Bias[gc_base + 1], al.f32)
        v = v * al.tanh(al.log(al.convert(1.0, al.f32) + al.exp(v)))
        v = v * al.tanh(al.log(al.convert(1.0, al.f32) + al.exp(v)))
        out_bf16[1] = al.convert(v, al.bf16)

        v = acc[ri * 4 + 2] + al.convert(Bias[gc_base + 2], al.f32)
        v = v * al.tanh(al.log(al.convert(1.0, al.f32) + al.exp(v)))
        v = v * al.tanh(al.log(al.convert(1.0, al.f32) + al.exp(v)))
        out_bf16[2] = al.convert(v, al.bf16)

        v = acc[ri * 4 + 3] + al.convert(Bias[gc_base + 3], al.f32)
        v = v * al.tanh(al.log(al.convert(1.0, al.f32) + al.exp(v)))
        v = v * al.tanh(al.log(al.convert(1.0, al.f32) + al.exp(v)))
        out_bf16[3] = al.convert(v, al.bf16)

        out_u32 = al.view(out_bf16, al.u32, al.make_layout((2,), (1,)))
        byte_off = (gr * out_dim + gc_base) * BF16_BYTES
        al.amdgpu.raw_buffer_store_x2(out_u32, rsrc_Y, byte_off, 0, 0)


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
