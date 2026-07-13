import torch
import torch.nn as nn
import avelang
import avelang.language as al

TILE_H = 16
TILE_W = 16


@avelang.jit
def fused_conv_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    m_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    IC: al.i32,
    OC: al.i32,
    H: al.i32,
    W: al.i32,
    KH: al.i32,
    KW: al.i32,
    H_OUT: al.i32,
    W_OUT: al.i32,
    TILE_H: al.constexpr,
    TILE_W: al.constexpr,
):
    # Create tensor views from raw pointers
    in_layout = al.make_layout((B * IC * H * W,), (1,))
    x = al.make_tensor(x_ptr, al.bf16, in_layout)

    w_layout = al.make_layout((OC * IC * KH * KW,), (1,))
    w = al.make_tensor(w_ptr, al.bf16, w_layout)

    bias_layout = al.make_layout((OC,), (1,))
    bias = al.make_tensor(bias_ptr, al.bf16, bias_layout)

    m_layout = al.make_layout((OC,), (1,))
    m = al.make_tensor(m_ptr, al.bf16, m_layout)

    out_layout = al.make_layout((B * OC * H_OUT * W_OUT,), (1,))
    out = al.make_tensor(out_ptr, al.bf16, out_layout)

    tx = al.thread_id(0)
    ty = al.thread_id(1)
    bx = al.block_id(0)
    by = al.block_id(1)
    b = al.block_id(2)

    h_out = by * TILE_H + ty
    w_out = bx * TILE_W + tx

    if h_out < H_OUT and w_out < W_OUT:
        in_base = b * IC * H * W
        out_base = b * OC * H_OUT * W_OUT

        oc = al.convert(0, al.i32)
        for _oc in al.range(OC):
            acc = al.convert(0.0, al.f32)
            w_base = oc * IC * KH * KW

            for ic in al.range(IC):
                in_ch_base = in_base + ic * H * W
                w_ch_base = w_base + ic * KH * KW

                for kh in al.range(KH):
                    h_in = h_out + kh
                    in_row_base = in_ch_base + h_in * W
                    w_row_base = w_ch_base + kh * KW

                    for kw in al.range(KW):
                        w_in_idx = w_out + kw
                        in_idx = in_row_base + w_in_idx
                        w_idx = w_row_base + kw

                        in_val = al.convert(x[in_idx], al.f32)
                        w_val = al.convert(w[w_idx], al.f32)
                        acc = acc + in_val * w_val

            # Add bias
            acc = acc + al.convert(bias[oc], al.f32)

            # Multiply by per-channel multiplier
            m_val = al.convert(m[oc], al.f32)
            acc = acc * m_val

            # LeakyReLU with negative_slope = 0.01
            zero = al.convert(0.0, al.f32)
            neg_slope = al.convert(0.01, al.f32)
            if not (acc > zero):
                acc = acc * neg_slope

            # GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
            rsqrt2 = al.convert(0.7071067811865475, al.f32)
            half = al.convert(0.5, al.f32)
            one = al.convert(1.0, al.f32)
            erf_arg = acc * rsqrt2
            gelu_val = half * acc * (one + al.erf(erf_arg))

            # Write output
            out_idx = out_base + oc * H_OUT * W_OUT + h_out * W_OUT + w_out
            out[out_idx] = al.convert(gelu_val, al.bf16)

            oc = oc + al.convert(1, al.i32)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))

    def forward(self, x):
        w = self.conv.weight.data
        bias = self.conv.bias.data
        m = self.multiplier.data

        B, IC, H, W = x.shape
        OC = w.shape[0]
        KH, KW = self.conv.kernel_size
        H_OUT = H - KH + 1
        W_OUT = W - KW + 1

        # Convert to BF16
        x_bf16 = x.to(torch.bfloat16).contiguous()
        w_bf16 = w.to(torch.bfloat16).contiguous()
        m_bf16 = m.to(torch.bfloat16).contiguous()
        bias_bf16 = bias.contiguous()

        # Allocate output in BF16
        out_bf16 = torch.empty(
            B, OC, H_OUT, W_OUT, dtype=torch.bfloat16, device=x.device
        )

        num_h_tiles = (H_OUT + TILE_H - 1) // TILE_H
        num_w_tiles = (W_OUT + TILE_W - 1) // TILE_W
        grid = (num_w_tiles, num_h_tiles, B)
        block = (TILE_W, TILE_H, 1)

        fused_conv_kernel[lambda: (grid, block)](
            x_bf16,
            w_bf16,
            bias_bf16,
            m_bf16,
            out_bf16,
            B,
            IC,
            OC,
            H,
            W,
            KH,
            KW,
            H_OUT,
            W_OUT,
            TILE_H,
            TILE_W,
        )

        return out_bf16


batch_size = 64
in_channels = 64
out_channels = 64
height, width = 256, 256
kernel_size = 3
multiplier_shape = (out_channels, 1, 1)


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, multiplier_shape]
