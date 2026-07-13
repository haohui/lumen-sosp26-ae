import torch
import torch.nn as nn
import avelang
import avelang.language as al

_BH = al.constexpr(16)
_BW = al.constexpr(16)


@avelang.jit
def conv_transpose2d_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C_in: al.i32,
    C_out: al.i32,
    H_in: al.i32,
    W_in: al.i32,
    K_H: al.i32,
    K_W: al.i32,
    stride_h: al.i32,
    stride_w: al.i32,
    pad_h: al.i32,
    pad_w: al.i32,
    dil_h: al.i32,
    dil_w: al.i32,
    groups: al.i32,
    H_out: al.i32,
    W_out: al.i32,
):
    C_in_per_group = C_in // groups
    C_out_per_group = C_out // groups

    in_layout = al.make_layout(
        (N, C_in, H_in, W_in),
        (C_in * H_in * W_in, H_in * W_in, W_in, 1),
    )
    inp = al.make_tensor(input_ptr, al.bf16, in_layout)

    wt_layout = al.make_layout(
        (C_in, C_out_per_group, K_H, K_W),
        (C_out_per_group * K_H * K_W, K_H * K_W, K_W, 1),
    )
    wt = al.make_tensor(weight_ptr, al.bf16, wt_layout)

    out_layout = al.make_layout(
        (N, C_out, H_out, W_out),
        (C_out * H_out * W_out, H_out * W_out, W_out, 1),
    )
    out = al.make_tensor(output_ptr, al.bf16, out_layout)

    tid_ow = al.thread_id(0)
    tid_oh = al.thread_id(1)

    ow = al.block_id(0) * _BW + tid_ow
    oh = al.block_id(1) * _BH + tid_oh
    flat_nc = al.block_id(2)
    n = flat_nc // C_out
    oc = flat_nc % C_out

    if ow < W_out:
        if oh < H_out:
            g = oc // C_out_per_group
            ic_start = g * C_in_per_group
            ic_end = ic_start + C_in_per_group
            oc_in_group = oc % C_out_per_group

            acc = al.convert(0.0, al.f32)

            for k_h in al.range(K_H):
                coord_h = oh + pad_h - k_h * dil_h
                if coord_h >= 0:
                    if (coord_h % stride_h) == 0:
                        ih = coord_h // stride_h
                        if ih < H_in:
                            for k_w in al.range(K_W):
                                coord_w = ow + pad_w - k_w * dil_w
                                if coord_w >= 0:
                                    if (coord_w % stride_w) == 0:
                                        iw = coord_w // stride_w
                                        if iw < W_in:
                                            for ic in al.range(ic_start, ic_end):
                                                in_val = al.convert(inp[n, ic, ih, iw], al.f32)
                                                wt_val = al.convert(wt[ic, oc_in_group, k_h, k_w], al.f32)
                                                acc = acc + in_val * wt_val

            out[n, oc, oh, ow] = al.convert(acc, al.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1),
                 padding=(0, 0), dilation=(1, 1), groups=1, bias=False):
        super(ModelNew, self).__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias,
        )

    def forward(self, x):
        return _conv_transpose2d_forward(
            x, self.conv_transpose2d.weight, self.conv_transpose2d.bias,
            self.conv_transpose2d.stride, self.conv_transpose2d.padding,
            self.conv_transpose2d.dilation, self.conv_transpose2d.groups,
        )


def _conv_transpose2d_forward(x, weight, bias, stride, padding, dilation, groups):
    N, C_in, H_in, W_in = x.shape
    K_H, K_W = weight.shape[2], weight.shape[3]
    C_out = weight.shape[1] * groups

    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation

    H_out = (H_in - 1) * stride_h - 2 * pad_h + dil_h * (K_H - 1) + 1
    W_out = (W_in - 1) * stride_w - 2 * pad_w + dil_w * (K_W - 1) + 1

    x_bf16 = x.to(torch.bfloat16).contiguous()
    w_bf16 = weight.to(torch.bfloat16).contiguous()
    out = torch.empty(N, C_out, H_out, W_out, dtype=torch.bfloat16, device=x.device)

    grid_h = (H_out + 16 - 1) // 16
    grid_w = (W_out + 16 - 1) // 16
    grid_nc = N * C_out

    conv_transpose2d_kernel[lambda: ((grid_w, grid_h, grid_nc), (16, 16, 1))](
        x_bf16, w_bf16, out,
        N, C_in, C_out, H_in, W_in, K_H, K_W,
        stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
        groups, H_out, W_out,
    )

    result = out.to(x.dtype)
    if bias is not None:
        result = result + bias.view(1, -1, 1, 1).to(x.dtype)

    return result
