import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def conv_hardswish_relu_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    N: al.i32,
    IC: al.i32,
    OC: al.i32,
    H: al.i32,
    W: al.i32,
    KH: al.i32,
    KW: al.i32,
    H_OUT: al.i32,
    W_OUT: al.i32,
    BLOCK_SIZE: al.i32,
):
    x_strides = (IC * H * W, H * W, W, 1)
    x = al.make_tensor(x_ptr, al.bf16, al.make_layout((N, IC, H, W), x_strides))

    w_strides = (IC * KH * KW, KH * KW, KW, 1)
    w = al.make_tensor(w_ptr, al.bf16, al.make_layout((OC, IC, KH, KW), w_strides))

    b = al.make_tensor(b_ptr, al.bf16, al.make_layout((OC,), (1,)))

    out_strides = (OC * H_OUT * W_OUT, H_OUT * W_OUT, W_OUT, 1)
    out = al.make_tensor(out_ptr, al.bf16, al.make_layout((N, OC, H_OUT, W_OUT), out_strides))

    n = al.block_id(0)
    oc = al.block_id(1)
    spatial_tile = al.block_id(2)
    tid = al.thread_id(0)

    hw = spatial_tile * BLOCK_SIZE + tid

    if hw < H_OUT * W_OUT:
        h = hw // W_OUT
        w_idx = hw % W_OUT

        acc = al.convert(b[oc], al.f32)

        for ic in al.range(IC):
            for kh in al.range(KH):
                for kw in al.range(KW):
                    x_val = al.convert(x[n, ic, h + kh, w_idx + kw], al.f32)
                    w_val = al.convert(w[oc, ic, kh, kw], al.f32)
                    acc = acc + x_val * w_val

        # HardSwish: x * clamp(x + 3, 0, 6) / 6
        three = al.convert(3.0, al.f32)
        six = al.convert(6.0, al.f32)
        zero_f = al.convert(0.0, al.f32)

        tmp = acc + three
        if tmp < zero_f:
            tmp = zero_f
        if tmp > six:
            tmp = six

        hs = acc * tmp / six

        # ReLU: max(0, x)
        if hs < zero_f:
            hs = zero_f

        out[n, oc, h, w_idx] = al.convert(hs, al.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.conv.weight
        bias = self.conv.bias

        N, IC, H, W = x.shape
        OC, _, KH, KW = weight.shape

        H_OUT = H - KH + 1
        W_OUT = W - KW + 1

        out = torch.empty(N, OC, H_OUT, W_OUT, dtype=x.dtype, device=x.device)

        BLOCK_SIZE = 256
        spatial_tiles = (H_OUT * W_OUT + BLOCK_SIZE - 1) // BLOCK_SIZE

        conv_hardswish_relu_kernel[lambda: ((N, OC, spatial_tiles), (BLOCK_SIZE, 1, 1))](
            x, weight, bias, out,
            N, IC, OC, H, W, KH, KW, H_OUT, W_OUT, BLOCK_SIZE,
        )

        return out
