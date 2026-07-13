import torch
import torch.nn as nn
import avelang
import avelang.language as al

batch_size = 128
in_channels = 8
out_channels = 64
height = 384
width = 384
kernel_size = 3
pool_kernel_size = 4

BLOCK_SIZE: al.constexpr = 256
TILE_H: al.constexpr = 16
TILE_W: al.constexpr = 16
IC_CONST: al.constexpr = 8
KH_CONST: al.constexpr = 3
KW_CONST: al.constexpr = 3
SHM_H: al.constexpr = 18
SHM_W: al.constexpr = 18
SHM_TOTAL: al.constexpr = 2592


@avelang.jit
def conv2d_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    OC: al.i32,
    IH: al.i32,
    IW: al.i32,
    OH: al.i32,
    OW: al.i32,
):
    tx = al.thread_id(0)
    ty = al.thread_id(1)
    tile_ow = al.block_id(0)
    tile_oh = al.block_id(1)
    batch_oc = al.block_id(2)

    batch_idx = batch_oc // OC
    oc = batch_oc - batch_idx * OC

    oh = tile_oh * TILE_H + ty
    ow = tile_ow * TILE_W + tx

    if oh >= OH:
        return
    if ow >= OW:
        return

    x = al.make_tensor(
        x_ptr, al.bf16,
        al.make_layout(
            (B, IC_CONST, IH, IW),
            (IC_CONST * IH * IW, IH * IW, IW, 1),
        ),
    )
    w = al.make_tensor(
        w_ptr, al.bf16,
        al.make_layout(
            (OC, IC_CONST, KH_CONST, KW_CONST),
            (IC_CONST * KH_CONST * KW_CONST, KH_CONST * KW_CONST, KW_CONST, 1),
        ),
    )
    b = al.make_tensor(b_ptr, al.bf16, al.make_layout((OC,), (1,)))
    out = al.make_tensor(
        out_ptr, al.bf16,
        al.make_layout((B, OC, OH, OW), (OC * OH * OW, OH * OW, OW, 1)),
    )

    in_shm = al.make_shared((IC_CONST, SHM_H, SHM_W), al.bf16)
    lid = ty * TILE_W + tx
    oh_base = tile_oh * TILE_H
    ow_base = tile_ow * TILE_W

    for pos in al.range(lid, SHM_TOTAL, BLOCK_SIZE):
        ic_idx = pos // (SHM_H * SHM_W)
        sp = pos - ic_idx * (SHM_H * SHM_W)
        lh = sp // SHM_W
        lw = sp - lh * SHM_W
        ih_val = oh_base + lh
        iw_val = ow_base + lw
        if ih_val < IH and iw_val < IW:
            in_shm[ic_idx, lh, lw] = x[batch_idx, ic_idx, ih_val, iw_val]

    al.syncthreads()

    bias_val = al.convert(b[oc], al.f32)
    accum = al.convert(0.0, al.f32)
    for ic in al.range(IC_CONST):
        for kh in al.range(KH_CONST):
            for kw in al.range(KW_CONST):
                x_val = al.convert(in_shm[ic, ty + kh, tx + kw], al.f32)
                w_val = al.convert(w[oc, ic, kh, kw], al.f32)
                accum = accum + x_val * w_val

    accum = accum + bias_val
    out[batch_idx, oc, oh, ow] = al.convert(accum, al.bf16)


@avelang.jit
def avgpool_sigmoid_kernel(
    in_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    OC: al.i32,
    OH: al.i32,
    OW: al.i32,
    PH: al.i32,
    PW: al.i32,
    PKH: al.i32,
    PKW: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    batch_idx = bid // OC
    oc = bid - batch_idx * OC

    inp = al.make_tensor(
        in_ptr, al.bf16,
        al.make_layout((B, OC, OH, OW), (OC * OH * OW, OH * OW, OW, 1)),
    )
    out = al.make_tensor(
        out_ptr, al.bf16,
        al.make_layout((B, OC, PH, PW), (OC * PH * PW, PH * PW, PW, 1)),
    )

    total = PH * PW
    pool_area_f32 = al.convert(PKH * PKW, al.f32)
    one = al.convert(1.0, al.f32)

    for idx in al.range(tid, total, BLOCK_SIZE):
        ph = idx // PW
        pw = idx - ph * PW

        psum = al.convert(0.0, al.f32)
        for kh in al.range(PKH):
            for kw in al.range(PKW):
                oh_val = ph * PKH + kh
                ow_val = pw * PKW + kw
                psum = psum + al.convert(inp[batch_idx, oc, oh_val, ow_val], al.f32)

        pavg = psum / pool_area_f32
        neg = al.convert(-1.0, al.f32) * pavg
        sigval = one / (one + al.exp(neg))
        out[batch_idx, oc, ph, pw] = al.convert(sigval, al.bf16)


@avelang.jit
def sum_reduce_kernel(
    in_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    total_elements: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    inp = al.make_tensor(
        in_ptr, al.bf16, al.make_layout((B * total_elements,), (1,)),
    )
    out = al.make_tensor(out_ptr, al.bf16, al.make_layout((B,), (1,)))

    smem = al.make_shared((BLOCK_SIZE,), al.f32)

    local_sum = al.convert(0.0, al.f32)
    base = bid * total_elements

    for i in al.range(tid, total_elements, BLOCK_SIZE):
        local_sum = local_sum + al.convert(inp[base + i], al.f32)

    smem[tid] = local_sum
    al.syncthreads()

    if tid < 128:
        smem[tid] = smem[tid] + smem[tid + 128]
    al.syncthreads()
    if tid < 64:
        smem[tid] = smem[tid] + smem[tid + 64]
    al.syncthreads()
    if tid < 32:
        smem[tid] = smem[tid] + smem[tid + 32]
    al.syncthreads()
    if tid < 16:
        smem[tid] = smem[tid] + smem[tid + 16]
    al.syncthreads()
    if tid < 8:
        smem[tid] = smem[tid] + smem[tid + 8]
    al.syncthreads()
    if tid < 4:
        smem[tid] = smem[tid] + smem[tid + 4]
    al.syncthreads()
    if tid < 2:
        smem[tid] = smem[tid] + smem[tid + 2]
    al.syncthreads()
    if tid < 1:
        smem[tid] = smem[tid] + smem[tid + 1]

    if tid == 0:
        out[bid] = al.convert(smem[0], al.bf16)


def _to_bf16_cuda(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_conv_pool_sigmoid_sum(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    x_bf16 = _to_bf16_cuda(x)
    w_bf16 = _to_bf16_cuda(weight)
    b_bf16 = _to_bf16_cuda(bias)

    B_val, IC_val, IH_val, IW_val = x_bf16.shape
    OC_val, _, KH_val, KW_val = w_bf16.shape
    OH_val = IH_val - KH_val + 1
    OW_val = IW_val - KW_val + 1

    OH_TILES_val = (OH_val + TILE_H - 1) // TILE_H
    OW_TILES_val = (OW_val + TILE_W - 1) // TILE_W

    conv_out = torch.empty(
        (B_val, OC_val, OH_val, OW_val),
        dtype=torch.bfloat16,
        device=x_bf16.device,
    )
    conv2d_kernel[lambda: (
        (OW_TILES_val, OH_TILES_val, B_val * OC_val),
        (TILE_W, TILE_H, 1),
    )](
        x_bf16, w_bf16, b_bf16, conv_out,
        B_val, OC_val, IH_val, IW_val, OH_val, OW_val,
    )

    PKH_val = pool_kernel_size
    PKW_val = pool_kernel_size
    PH_val = OH_val // PKH_val
    PW_val = OW_val // PKW_val

    pool_out = torch.empty(
        (B_val, OC_val, PH_val, PW_val),
        dtype=torch.bfloat16,
        device=x_bf16.device,
    )
    avgpool_sigmoid_kernel[lambda: ((B_val * OC_val, 1, 1), (BLOCK_SIZE, 1, 1))](
        conv_out, pool_out,
        B_val, OC_val, OH_val, OW_val, PH_val, PW_val, PKH_val, PKW_val,
    )

    total_elts = OC_val * PH_val * PW_val
    sum_out = torch.empty((B_val,), dtype=torch.bfloat16, device=x_bf16.device)
    sum_reduce_kernel[lambda: ((B_val, 1, 1), (BLOCK_SIZE, 1, 1))](
        pool_out, sum_out, B_val, total_elts,
    )

    return sum_out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / (fan_in**0.5) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return avelang_conv_pool_sigmoid_sum(x, self.weight, self.bias)


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, pool_kernel_size]
