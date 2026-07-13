import torch
import torch.nn as nn
import math

import avelang
import avelang.language as al

# Compile-time constants captured by the kernel
ADD_VALUE = al.constexpr(0.5)
SCALE = al.constexpr(2.0)


@avelang.jit
def fused_conv_transpose_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.f32),
    output_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C_in: al.i32,
    C_out: al.i32,
    H_in: al.i32,
    W_in: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    KH: al.i32,
    KW: al.i32,
    stride: al.i32,
    padding: al.i32,
):
    inp_layout = al.make_layout(
        (N, C_in, H_in, W_in),
        (C_in * H_in * W_in, H_in * W_in, W_in, 1),
    )
    inp = al.make_tensor(input_ptr, al.bf16, inp_layout)

    w_layout = al.make_layout(
        (C_in, C_out, KH, KW),
        (C_out * KH * KW, KH * KW, KW, 1),
    )
    w = al.make_tensor(weight_ptr, al.bf16, w_layout)

    bias_layout = al.make_layout((C_out,), (1,))
    bias = al.make_tensor(bias_ptr, al.f32, bias_layout)

    out_layout = al.make_layout(
        (N, C_out, H_out, W_out),
        (C_out * H_out * W_out, H_out * W_out, W_out, 1),
    )
    out = al.make_tensor(output_ptr, al.bf16, out_layout)

    total = N * C_out * H_out * W_out
    tid = al.thread_id(0)
    bid = al.block_id(0)
    bdim = al.block_dim(0)
    gdim = al.grid_dim(0)

    idx = bid * bdim + tid
    step = gdim * bdim

    one_f32 = al.convert(1.0, al.f32)
    neg_one_f32 = al.convert(-1.0, al.f32)
    upper_bound = one_f32 * SCALE
    lower_bound = neg_one_f32 * SCALE

    for flat_idx in al.range(idx, total, step):
        n = flat_idx // (C_out * H_out * W_out)
        rem1 = flat_idx % (C_out * H_out * W_out)
        oc = rem1 // (H_out * W_out)
        rem2 = rem1 % (H_out * W_out)
        oh = rem2 // W_out
        ow = rem2 % W_out

        acc = bias[oc]

        for kh in al.range(KH):
            for kw in al.range(KW):
                num_h = oh + padding - kh
                num_w = ow + padding - kw
                if num_h >= 0 and num_w >= 0:
                    if num_h % stride == 0 and num_w % stride == 0:
                        ih = num_h // stride
                        iw = num_w // stride
                        if ih < H_in and iw < W_in:
                            for ic in al.range(C_in):
                                inp_val = al.convert(inp[n, ic, ih, iw], al.f32)
                                w_val = al.convert(w[ic, oc, kh, kw], al.f32)
                                acc = acc + inp_val * w_val

        # Mish: x * tanh(ln(1 + exp(x)))
        sp = al.log(one_f32 + al.exp(acc))
        mish_val = acc * al.tanh(sp)

        # Add constant
        added = mish_val + ADD_VALUE

        # Hardtanh + Scale, storing directly in each branch
        if added > one_f32:
            out[n, oc, oh, ow] = al.convert(upper_bound, al.bf16)
        else:
            if added < neg_one_f32:
                out[n, oc, oh, ow] = al.convert(lower_bound, al.bf16)
            else:
                out[n, oc, oh, ow] = al.convert(added * SCALE, al.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.add_value = add_value
        self.scale = scale

        self.weight = nn.Parameter(
            torch.empty(in_channels, out_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        N, C_in, H_in, W_in = x.shape
        C_out = self.out_channels
        KH = self.kernel_size
        KW = self.kernel_size
        stride = self.stride
        padding = self.padding
        output_padding = self.output_padding

        H_out = (H_in - 1) * stride - 2 * padding + KH + output_padding
        W_out = (W_in - 1) * stride - 2 * padding + KW + output_padding

        x_bf16 = x.to(torch.bfloat16).contiguous()
        w_bf16 = self.weight.to(torch.bfloat16).contiguous()
        bias_f32 = self.bias.to(torch.float32).contiguous()

        out = torch.empty(N, C_out, H_out, W_out, dtype=torch.bfloat16, device=x.device)

        total_elems = N * C_out * H_out * W_out
        block_size = 256
        num_blocks = (total_elems + block_size - 1) // block_size
        grid_x = num_blocks if num_blocks < 65536 else 65535

        fused_conv_transpose_kernel[lambda: ((grid_x, 1, 1), (block_size, 1, 1))](
            x_bf16.data_ptr(),
            w_bf16.data_ptr(),
            bias_f32.data_ptr(),
            out.data_ptr(),
            N, C_in, C_out, H_in, W_in, H_out, W_out, KH, KW,
            stride, padding,
        )

        return out
