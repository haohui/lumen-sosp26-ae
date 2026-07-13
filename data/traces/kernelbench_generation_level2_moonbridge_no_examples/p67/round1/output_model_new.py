import torch
import torch.nn as nn
import avelang
import avelang.language as al

TILE_H = 8
TILE_W = 8
OC_GROUPS = 4
OC_PER_GROUP = 16


@avelang.jit
def conv2d_gelu_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    N: al.i32,
    IC: al.i32,
    H_in: al.i32,
    W_in: al.i32,
    OC: al.i32,
    KH: al.i32,
    KW: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    grid_h: al.i32,
    grid_w: al.i32,
):
    n = al.block_id(0) // grid_h
    tile_h = al.block_id(0) % grid_h
    tile_w = al.block_id(1)

    tx = al.thread_id(0)
    ty = al.thread_id(1)
    tz = al.thread_id(2)

    h = tile_h * al.convert(8, al.i32) + ty
    w = tile_w * al.convert(8, al.i32) + tx

    is0 = IC * H_in * W_in
    is1 = H_in * W_in
    is2 = W_in
    il = al.make_layout((N, IC, H_in, W_in), (is0, is1, is2, 1))
    inp = al.make_tensor(input_ptr, al.bf16, il)

    ws0 = IC * KH * KW
    ws1 = KH * KW
    ws2 = KW
    wl = al.make_layout((OC, IC, KH, KW), (ws0, ws1, ws2, 1))
    wt = al.make_tensor(weight_ptr, al.bf16, wl)

    bl = al.make_layout((OC,), (1,))
    bias = al.make_tensor(bias_ptr, al.bf16, bl)

    os0 = OC * H_out * W_out
    os1 = H_out * W_out
    os2 = W_out
    ol = al.make_layout((N, OC, H_out, W_out), (os0, os1, os2, 1))
    out = al.make_tensor(output_ptr, al.bf16, ol)

    if h >= H_out or w >= W_out:
        return

    oc_base = tz * al.convert(16, al.i32)
    oc0 = oc_base
    oc1 = oc_base + al.convert(1, al.i32)
    oc2 = oc_base + al.convert(2, al.i32)
    oc3 = oc_base + al.convert(3, al.i32)
    oc4 = oc_base + al.convert(4, al.i32)
    oc5 = oc_base + al.convert(5, al.i32)
    oc6 = oc_base + al.convert(6, al.i32)
    oc7 = oc_base + al.convert(7, al.i32)
    oc8 = oc_base + al.convert(8, al.i32)
    oc9 = oc_base + al.convert(9, al.i32)
    ocA = oc_base + al.convert(10, al.i32)
    ocB = oc_base + al.convert(11, al.i32)
    ocC = oc_base + al.convert(12, al.i32)
    ocD = oc_base + al.convert(13, al.i32)
    ocE = oc_base + al.convert(14, al.i32)
    ocF = oc_base + al.convert(15, al.i32)

    a0 = al.convert(bias[oc0], al.f32)
    a1 = al.convert(bias[oc1], al.f32)
    a2 = al.convert(bias[oc2], al.f32)
    a3 = al.convert(bias[oc3], al.f32)
    a4 = al.convert(bias[oc4], al.f32)
    a5 = al.convert(bias[oc5], al.f32)
    a6 = al.convert(bias[oc6], al.f32)
    a7 = al.convert(bias[oc7], al.f32)
    a8 = al.convert(bias[oc8], al.f32)
    a9 = al.convert(bias[oc9], al.f32)
    aA = al.convert(bias[ocA], al.f32)
    aB = al.convert(bias[ocB], al.f32)
    aC = al.convert(bias[ocC], al.f32)
    aD = al.convert(bias[ocD], al.f32)
    aE = al.convert(bias[ocE], al.f32)
    aF = al.convert(bias[ocF], al.f32)

    for ic in al.range(IC):
        for kh in al.range(KH):
            for kw in al.range(KW):
                in_val = al.convert(inp[n, ic, h + kh, w + kw], al.f32)
                a0 = a0 + in_val * al.convert(wt[oc0, ic, kh, kw], al.f32)
                a1 = a1 + in_val * al.convert(wt[oc1, ic, kh, kw], al.f32)
                a2 = a2 + in_val * al.convert(wt[oc2, ic, kh, kw], al.f32)
                a3 = a3 + in_val * al.convert(wt[oc3, ic, kh, kw], al.f32)
                a4 = a4 + in_val * al.convert(wt[oc4, ic, kh, kw], al.f32)
                a5 = a5 + in_val * al.convert(wt[oc5, ic, kh, kw], al.f32)
                a6 = a6 + in_val * al.convert(wt[oc6, ic, kh, kw], al.f32)
                a7 = a7 + in_val * al.convert(wt[oc7, ic, kh, kw], al.f32)
                a8 = a8 + in_val * al.convert(wt[oc8, ic, kh, kw], al.f32)
                a9 = a9 + in_val * al.convert(wt[oc9, ic, kh, kw], al.f32)
                aA = aA + in_val * al.convert(wt[ocA, ic, kh, kw], al.f32)
                aB = aB + in_val * al.convert(wt[ocB, ic, kh, kw], al.f32)
                aC = aC + in_val * al.convert(wt[ocC, ic, kh, kw], al.f32)
                aD = aD + in_val * al.convert(wt[ocD, ic, kh, kw], al.f32)
                aE = aE + in_val * al.convert(wt[ocE, ic, kh, kw], al.f32)
                aF = aF + in_val * al.convert(wt[ocF, ic, kh, kw], al.f32)

    sqrt_2_pi = al.convert(0.79788456, al.f32)
    coeff = al.convert(0.044715, al.f32)
    half = al.convert(0.5, al.f32)
    one_f = al.convert(1.0, al.f32)

    x3 = a0 * a0 * a0
    inner = sqrt_2_pi * (a0 + coeff * x3)
    out[n, oc0, h, w] = al.convert(half * a0 * (one_f + al.tanh(inner)), al.bf16)
    x3 = a1 * a1 * a1
    inner = sqrt_2_pi * (a1 + coeff * x3)
    out[n, oc1, h, w] = al.convert(half * a1 * (one_f + al.tanh(inner)), al.bf16)
    x3 = a2 * a2 * a2
    inner = sqrt_2_pi * (a2 + coeff * x3)
    out[n, oc2, h, w] = al.convert(half * a2 * (one_f + al.tanh(inner)), al.bf16)
    x3 = a3 * a3 * a3
    inner = sqrt_2_pi * (a3 + coeff * x3)
    out[n, oc3, h, w] = al.convert(half * a3 * (one_f + al.tanh(inner)), al.bf16)
    x3 = a4 * a4 * a4
    inner = sqrt_2_pi * (a4 + coeff * x3)
    out[n, oc4, h, w] = al.convert(half * a4 * (one_f + al.tanh(inner)), al.bf16)
    x3 = a5 * a5 * a5
    inner = sqrt_2_pi * (a5 + coeff * x3)
    out[n, oc5, h, w] = al.convert(half * a5 * (one_f + al.tanh(inner)), al.bf16)
    x3 = a6 * a6 * a6
    inner = sqrt_2_pi * (a6 + coeff * x3)
    out[n, oc6, h, w] = al.convert(half * a6 * (one_f + al.tanh(inner)), al.bf16)
    x3 = a7 * a7 * a7
    inner = sqrt_2_pi * (a7 + coeff * x3)
    out[n, oc7, h, w] = al.convert(half * a7 * (one_f + al.tanh(inner)), al.bf16)
    x3 = a8 * a8 * a8
    inner = sqrt_2_pi * (a8 + coeff * x3)
    out[n, oc8, h, w] = al.convert(half * a8 * (one_f + al.tanh(inner)), al.bf16)
    x3 = a9 * a9 * a9
    inner = sqrt_2_pi * (a9 + coeff * x3)
    out[n, oc9, h, w] = al.convert(half * a9 * (one_f + al.tanh(inner)), al.bf16)
    x3 = aA * aA * aA
    inner = sqrt_2_pi * (aA + coeff * x3)
    out[n, ocA, h, w] = al.convert(half * aA * (one_f + al.tanh(inner)), al.bf16)
    x3 = aB * aB * aB
    inner = sqrt_2_pi * (aB + coeff * x3)
    out[n, ocB, h, w] = al.convert(half * aB * (one_f + al.tanh(inner)), al.bf16)
    x3 = aC * aC * aC
    inner = sqrt_2_pi * (aC + coeff * x3)
    out[n, ocC, h, w] = al.convert(half * aC * (one_f + al.tanh(inner)), al.bf16)
    x3 = aD * aD * aD
    inner = sqrt_2_pi * (aD + coeff * x3)
    out[n, ocD, h, w] = al.convert(half * aD * (one_f + al.tanh(inner)), al.bf16)
    x3 = aE * aE * aE
    inner = sqrt_2_pi * (aE + coeff * x3)
    out[n, ocE, h, w] = al.convert(half * aE * (one_f + al.tanh(inner)), al.bf16)
    x3 = aF * aF * aF
    inner = sqrt_2_pi * (aF + coeff * x3)
    out[n, ocF, h, w] = al.convert(half * aF * (one_f + al.tanh(inner)), al.bf16)


@avelang.jit
def global_avg_pool_kernel(
    input_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    N: al.i32,
    OC: al.i32,
    H: al.i32,
    W: al.i32,
):
    n = al.block_id(0)
    oc = al.block_id(1)

    is0 = OC * H * W
    is1 = H * W
    is2 = W
    il = al.make_layout((N, OC, H, W), (is0, is1, is2, 1))
    inp = al.make_tensor(input_ptr, al.bf16, il)

    os0 = OC
    ol = al.make_layout((N, OC), (os0, 1))
    out = al.make_tensor(output_ptr, al.bf16, ol)

    sum_val = al.convert(0.0, al.f32)
    for h in al.range(H):
        for w in al.range(W):
            sum_val = sum_val + al.convert(inp[n, oc, h, w], al.f32)

    count = al.convert(H, al.f32) * al.convert(W, al.f32)
    avg = sum_val / count
    out[n, oc] = al.convert(avg, al.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        N = x.shape[0]
        IC = x.shape[1]
        H_in = x.shape[2]
        W_in = x.shape[3]

        OC = self.conv.out_channels
        KH = self.conv.kernel_size[0]
        KW = self.conv.kernel_size[1]
        H_out = H_in - KH + 1
        W_out = W_in - KW + 1

        grid_h = (H_out + TILE_H - 1) // TILE_H
        grid_w = (W_out + TILE_W - 1) // TILE_W

        w = self.conv.weight.to(dtype=torch.bfloat16, device=x.device).contiguous()
        b = self.conv.bias.to(dtype=torch.bfloat16, device=x.device).contiguous()
        x_bf16 = x.to(dtype=torch.bfloat16).contiguous()

        conv_out = torch.empty(N, OC, H_out, W_out,
                               dtype=torch.bfloat16, device=x.device)

        conv2d_gelu_kernel[lambda: ((N * grid_h, grid_w, 1), (TILE_H, TILE_W, OC_GROUPS))](
            x_bf16, w, b, conv_out,
            N, IC, H_in, W_in, OC, KH, KW, H_out, W_out, grid_h, grid_w,
        )

        pooled_out = torch.empty(N, OC,
                                 dtype=torch.bfloat16, device=x.device)

        global_avg_pool_kernel[lambda: ((N, OC, 1), (1, 1, 1))](
            conv_out, pooled_out,
            N, OC, H_out, W_out,
        )

        return pooled_out


batch_size = 128
in_channels = 8
out_channels = 64
height, width = 256, 256
kernel_size = 3


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
