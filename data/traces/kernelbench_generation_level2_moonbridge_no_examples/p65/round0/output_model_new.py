import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_OC = 16
TILE_H = 8
TILE_W = 8
BLOCK_SIZE = 256


@avelang.jit
def conv2d_bf16_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.f32),
    output_ptr: al.Pointer(al.f32),
    N: al.i32,
    IC: al.i32,
    OC: al.i32,
    H: al.i32,
    W: al.i32,
    OH: al.i32,
    OW: al.i32,
    OC_TILES: al.i32,
    OH_TILES: al.i32,
    OW_TILES: al.i32,
):
    flat_idx = al.block_id(0)
    b = flat_idx // OC_TILES
    oc_tile = flat_idx % OC_TILES
    oh_tile = al.block_id(1)
    ow_tile = al.block_id(2)

    oc_start = oc_tile * BLOCK_OC
    oh_start = oh_tile * TILE_H
    ow_start = ow_tile * TILE_W

    tid = al.thread_id(0)
    total_elems = BLOCK_OC * TILE_H * TILE_W

    input_layout = al.make_layout((N, IC, H, W), (IC * H * W, H * W, W, 1))
    input_t = al.make_tensor(input_ptr, al.bf16, input_layout)

    weight_layout = al.make_layout((OC, IC, 3, 3), (IC * 9, 9, 3, 1))
    weight_t = al.make_tensor(weight_ptr, al.bf16, weight_layout)

    bias_layout = al.make_layout((OC,), (1,))
    bias_t = al.make_tensor(bias_ptr, al.f32, bias_layout)

    output_layout = al.make_layout((N, OC, OH, OW), (OC * OH * OW, OH * OW, OW, 1))
    output_t = al.make_tensor(output_ptr, al.f32, output_layout)

    for elem_idx in al.range(tid, total_elems, BLOCK_SIZE):
        oc_loc = elem_idx // (TILE_H * TILE_W)
        spat_loc = elem_idx % (TILE_H * TILE_W)
        oh_loc = spat_loc // TILE_W
        ow_loc = spat_loc % TILE_W

        oc = oc_start + oc_loc
        oh = oh_start + oh_loc
        ow = ow_start + ow_loc

        if oc < OC and oh < OH and ow < OW:
            accum = al.convert(0.0, al.f32)
            for ic in al.range(IC):
                for kh in al.range(3):
                    for kw in al.range(3):
                        ival = al.convert(input_t[b, ic, oh + kh, ow + kw], al.f32)
                        wval = al.convert(weight_t[oc, ic, kh, kw], al.f32)
                        accum = accum + ival * wval
            accum = accum + bias_t[oc]
            output_t[b, oc, oh, ow] = accum


@avelang.jit
def avgpool_sigmoid_sum_kernel(
    conv_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.f32),
    N: al.i32,
    OC: al.i32,
    OH: al.i32,
    OW: al.i32,
    POOL: al.i32,
    PH: al.i32,
    PW: al.i32,
):
    b = al.block_id(0)
    tid = al.thread_id(0)

    conv_layout = al.make_layout((N, OC, OH, OW), (OC * OH * OW, OH * OW, OW, 1))
    conv_t = al.make_tensor(conv_ptr, al.f32, conv_layout)

    out_layout = al.make_layout((N,), (1,))
    out_t = al.make_tensor(out_ptr, al.f32, out_layout)

    smem = al.make_shared((BLOCK_SIZE,), al.f32)
    smem[tid] = al.convert(0.0, al.f32)
    al.syncthreads()

    total = OC * PH * PW
    pool_area_f = al.convert(POOL * POOL, al.f32)
    one = al.convert(1.0, al.f32)
    zero = al.convert(0.0, al.f32)

    for idx in al.range(tid, total, BLOCK_SIZE):
        oc = idx // (PH * PW)
        spat = idx % (PH * PW)
        ph = spat // PW
        pw = spat % PW

        pool_sum = al.convert(0.0, al.f32)
        for dh in al.range(POOL):
            for dw in al.range(POOL):
                ih = ph * POOL + dh
                iw = pw * POOL + dw
                if ih < OH and iw < OW:
                    pool_sum = pool_sum + conv_t[b, oc, ih, iw]

        pool_avg = pool_sum / pool_area_f
        neg_avg = zero - pool_avg
        sig_val = one / (one + al.exp(neg_avg))
        smem[tid] = smem[tid] + sig_val

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
    al.syncthreads()

    if tid == 0:
        out_t[b] = smem[0]


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super().__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.weight = nn.Parameter(conv.weight.data.clone())
        self.bias = nn.Parameter(conv.bias.data.clone())
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x):
        original_dtype = x.dtype
        x_bf16 = x.to(torch.bfloat16).contiguous()
        w_bf16 = self.weight.to(torch.bfloat16).contiguous()
        bias_f32 = self.bias.to(torch.float32).contiguous()

        N, IC, H, W = x_bf16.shape
        OC = w_bf16.shape[0]
        OH = H - 3 + 1
        OW = W - 3 + 1

        OC_TILES = (OC + BLOCK_OC - 1) // BLOCK_OC
        OH_TILES = (OH + TILE_H - 1) // TILE_H
        OW_TILES = (OW + TILE_W - 1) // TILE_W

        conv_out = torch.empty(N, OC, OH, OW, dtype=torch.float32, device=x.device)

        conv2d_bf16_kernel[
            lambda: ((N * OC_TILES, OH_TILES, OW_TILES), (BLOCK_SIZE, 1, 1))
        ](
            x_bf16, w_bf16, bias_f32, conv_out,
            N, IC, OC, H, W, OH, OW,
            OC_TILES, OH_TILES, OW_TILES,
        )

        POOL = self.pool_kernel_size
        PH = (OH - POOL) // POOL + 1
        PW = (OW - POOL) // POOL + 1

        output = torch.empty(N, dtype=torch.float32, device=x.device)

        avgpool_sigmoid_sum_kernel[
            lambda: ((N, 1, 1), (BLOCK_SIZE, 1, 1))
        ](
            conv_out, output,
            N, OC, OH, OW, POOL, PH, PW,
        )

        return output.to(original_dtype)
