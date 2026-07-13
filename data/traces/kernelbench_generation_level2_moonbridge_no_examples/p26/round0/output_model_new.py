import torch
import torch.nn as nn
import avelang
import avelang.language as al

batch_size = 128
in_channels = 32
out_channels = 64
D, H, W = 16, 16, 16
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1, 1, 1)


def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W), torch.rand(batch_size, out_channels, D * stride, H * stride, W * stride)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]


@avelang.jit
def conv_transpose3d_kernel(
    x_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    add_input_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    IC: al.i32,
    OC: al.i32,
    D_IN: al.i32,
    H_IN: al.i32,
    W_IN: al.i32,
    D_OUT: al.i32,
    H_OUT: al.i32,
    W_OUT: al.i32,
    K: al.i32,
    PAD: al.i32,
    STRIDE: al.i32,
):
    x_strides = (IC * D_IN * H_IN * W_IN, D_IN * H_IN * W_IN, H_IN * W_IN, W_IN, 1)
    x_layout = al.make_layout((B, IC, D_IN, H_IN, W_IN), x_strides)
    x = al.make_tensor(x_ptr, al.bf16, x_layout)

    out_strides = (OC * D_OUT * H_OUT * W_OUT, D_OUT * H_OUT * W_OUT, H_OUT * W_OUT, W_OUT, 1)
    out_layout = al.make_layout((B, OC, D_OUT, H_OUT, W_OUT), out_strides)
    out = al.make_tensor(out_ptr, al.bf16, out_layout)
    add_input = al.make_tensor(add_input_ptr, al.bf16, out_layout)

    w_strides = (OC * K * K * K, K * K * K, K * K, K, 1)
    weight_layout = al.make_layout((IC, OC, K, K, K), w_strides)
    weight = al.make_tensor(weight_ptr, al.bf16, weight_layout)

    bias_layout = al.make_layout((OC,), (1,))
    bias = al.make_tensor(bias_ptr, al.bf16, bias_layout)

    BLOCK_SIZE = 256
    total_spatial = D_OUT * H_OUT * W_OUT

    batch_oc = al.block_id(0)
    spatial_block = al.block_id(1)
    tid = al.thread_id(0)

    n = batch_oc // OC
    oc = batch_oc % OC

    spatial_idx = spatial_block * BLOCK_SIZE + tid

    if spatial_idx < total_spatial:
        oh_w = H_OUT * W_OUT
        d = spatial_idx // oh_w
        hw_rem = spatial_idx % oh_w
        h = hw_rem // W_OUT
        w = hw_rem % W_OUT

        acc = al.convert(0.0, al.f32)

        for ic in al.range(IC):
            for kd in al.range(K):
                d_num = d + PAD - kd
                if d_num >= 0:
                    if d_num % STRIDE == 0:
                        din = d_num // STRIDE
                        if din < D_IN:
                            for kh in al.range(K):
                                h_num = h + PAD - kh
                                if h_num >= 0:
                                    if h_num % STRIDE == 0:
                                        hin = h_num // STRIDE
                                        if hin < H_IN:
                                            for kw in al.range(K):
                                                w_num = w + PAD - kw
                                                if w_num >= 0:
                                                    if w_num % STRIDE == 0:
                                                        win = w_num // STRIDE
                                                        if win < W_IN:
                                                            x_val = al.convert(x[n, ic, din, hin, win], al.f32)
                                                            w_val = al.convert(weight[ic, oc, kd, kh, kw], al.f32)
                                                            acc = acc + x_val * w_val

        bias_val = al.convert(bias[oc], al.f32)
        acc = acc + bias_val

        add_val = al.convert(add_input[n, oc, d, h, w], al.f32)
        acc = acc + add_val

        # HardSwish: acc * relu6(acc + 3) / 6
        # relu6(y) = min(max(y,0),6) via abs ops (no float comparisons):
        #   max(y, 0) = (y + |y|) * 0.5
        #   min(z, 6) = (z + 6 - |6 - z|) * 0.5
        three = al.convert(3.0, al.f32)
        six = al.convert(6.0, al.f32)
        half = al.convert(0.5, al.f32)

        y = acc + three
        max_val = (y + al.abs(y)) * half
        relu6_val = (max_val + six - al.abs(six - max_val)) * half
        result = acc * acc * relu6_val / six

        out[n, oc, d, h, w] = al.convert(result, al.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding,
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, add_input):
        orig_dtype = x.dtype

        x_bf16 = x.to(torch.bfloat16).contiguous()
        add_input_bf16 = add_input.to(torch.bfloat16).contiguous()
        weight_bf16 = self.conv_transpose.weight.detach().to(torch.bfloat16).contiguous()
        conv_bias_bf16 = self.conv_transpose.bias.detach().to(torch.bfloat16).contiguous()

        B, IC, D_IN, H_IN, W_IN = x.shape
        _, OC, D_OUT, H_OUT, W_OUT = add_input.shape
        K = self.conv_transpose.kernel_size[0]
        PAD = self.conv_transpose.padding[0]
        STRIDE = self.conv_transpose.stride[0]

        out_bf16 = torch.empty(B, OC, D_OUT, H_OUT, W_OUT, dtype=torch.bfloat16, device=x.device)

        BLOCK_SIZE = 256
        total_spatial = D_OUT * H_OUT * W_OUT
        num_spatial_blocks = (total_spatial + BLOCK_SIZE - 1) // BLOCK_SIZE

        conv_transpose3d_kernel[lambda: ((B * OC, num_spatial_blocks, 1), (BLOCK_SIZE, 1, 1))](
            x_bf16, weight_bf16, conv_bias_bf16, add_input_bf16, out_bf16,
            B, IC, OC, D_IN, H_IN, W_IN, D_OUT, H_OUT, W_OUT,
            K, PAD, STRIDE,
        )

        return out_bf16.to(orig_dtype)
