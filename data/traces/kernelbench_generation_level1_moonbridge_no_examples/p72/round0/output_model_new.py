import math
import torch
import torch.nn as nn

import avelang
import avelang.language as al


@avelang.jit
def conv_transpose3d_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    B: al.i32,
    IC: al.i32,
    OC: al.i32,
    D_in: al.i32,
    H_in: al.i32,
    W_in: al.i32,
    D_out: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    Kd: al.i32,
    Kh: al.i32,
    Kw: al.i32,
    stride_d: al.i32,
    stride_h: al.i32,
    stride_w: al.i32,
    pad_d: al.i32,
    pad_h: al.i32,
    pad_w: al.i32,
    groups: al.i32,
    has_bias: al.i32,
):
    total_spatial = D_out * H_out * W_out
    total_per_oc = OC * total_spatial
    total_outputs = B * total_per_oc

    idx = al.block_id(0) * al.block_dim(0) + al.thread_id(0)
    if idx >= total_outputs:
        return

    b = idx // total_per_oc
    r1 = idx - b * total_per_oc
    oc = r1 // total_spatial
    spatial_idx = r1 - oc * total_spatial

    hw = H_out * W_out
    d_out = spatial_idx // hw
    r2 = spatial_idx - d_out * hw
    h_out = r2 // W_out
    w_out = r2 - h_out * W_out

    in_cpg = IC // groups
    out_cpg = OC // groups
    g = oc // out_cpg
    oc_local = oc - g * out_cpg
    ic_start = g * in_cpg

    in_layout = al.make_layout(
        (B, IC, D_in, H_in, W_in),
        (IC * D_in * H_in * W_in, D_in * H_in * W_in, H_in * W_in, W_in, 1),
    )
    input_t = al.make_tensor(input_ptr, al.bf16, in_layout)

    w_layout = al.make_layout(
        (IC, out_cpg, Kd, Kh, Kw),
        (out_cpg * Kd * Kh * Kw, Kd * Kh * Kw, Kh * Kw, Kw, 1),
    )
    weight_t = al.make_tensor(weight_ptr, al.bf16, w_layout)

    acc = al.convert(0.0, al.f32)

    for kd in al.range(Kd):
        dv = d_out + pad_d - kd
        if dv >= 0:
            d_in = dv // stride_d
            if d_in * stride_d == dv:
                if d_in < D_in:
                    for kh in al.range(Kh):
                        hv = h_out + pad_h - kh
                        if hv >= 0:
                            h_in = hv // stride_h
                            if h_in * stride_h == hv:
                                if h_in < H_in:
                                    for kw in al.range(Kw):
                                        wv = w_out + pad_w - kw
                                        if wv >= 0:
                                            w_in = wv // stride_w
                                            if w_in * stride_w == wv:
                                                if w_in < W_in:
                                                    for ic_idx in al.range(in_cpg):
                                                        ic = ic_start + ic_idx
                                                        inp = al.convert(
                                                            input_t[b, ic, d_in, h_in, w_in], al.f32
                                                        )
                                                        wt = al.convert(
                                                            weight_t[ic, oc_local, kd, kh, kw], al.f32
                                                        )
                                                        acc = acc + inp * wt

    if has_bias != 0:
        bias_layout = al.make_layout((OC,), (1,))
        bias_t = al.make_tensor(bias_ptr, al.bf16, bias_layout)
        bval = al.convert(bias_t[oc], al.f32)
        acc = acc + bval

    out_layout = al.make_layout(
        (B, OC, D_out, H_out, W_out),
        (OC * D_out * H_out * W_out, D_out * H_out * W_out, H_out * W_out, W_out, 1),
    )
    output_t = al.make_tensor(output_ptr, al.bf16, out_layout)
    output_t[b, oc, d_out, h_out, w_out] = al.convert(acc, al.bf16)


def _run_conv_transpose3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: tuple,
    padding: tuple,
    output_padding: tuple,
    groups: int,
) -> torch.Tensor:
    B, IC, D_in, H_in, W_in = x.shape
    Kd, Kh, Kw = weight.shape[2], weight.shape[3], weight.shape[4]
    OC = weight.shape[1] * groups

    stride_d, stride_h, stride_w = stride
    pad_d, pad_h, pad_w = padding
    out_pad_d, out_pad_h, out_pad_w = output_padding

    D_out = (D_in - 1) * stride_d - 2 * pad_d + Kd + out_pad_d
    H_out = (H_in - 1) * stride_h - 2 * pad_h + Kh + out_pad_h
    W_out = (W_in - 1) * stride_w - 2 * pad_w + Kw + out_pad_w

    x = x.contiguous()
    weight = weight.contiguous()

    output = torch.empty(
        B, OC, D_out, H_out, W_out, dtype=x.dtype, device=x.device
    )

    has_bias_val = 1 if bias is not None else 0
    if bias is not None:
        bias = bias.contiguous()
        bias_ptr = bias.data_ptr()
    else:
        bias_ptr = torch.empty(1, dtype=x.dtype, device=x.device).data_ptr()

    total_outputs = B * OC * D_out * H_out * W_out
    BLOCK_SIZE = 256
    grid_x = (total_outputs + BLOCK_SIZE - 1) // BLOCK_SIZE

    conv_transpose3d_kernel[lambda: ((grid_x, 1, 1), (BLOCK_SIZE, 1, 1))](
        x.data_ptr(),
        weight.data_ptr(),
        bias_ptr,
        output.data_ptr(),
        B,
        IC,
        OC,
        D_in,
        H_in,
        W_in,
        D_out,
        H_out,
        W_out,
        Kd,
        Kh,
        Kw,
        stride_d,
        stride_h,
        stride_w,
        pad_d,
        pad_h,
        pad_w,
        groups,
        has_bias_val,
    )

    return output


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple = (1, 1, 1),
        padding: tuple = (0, 0, 0),
        output_padding: tuple = (0, 0, 0),
        groups: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.has_bias = bias

        self.weight = nn.Parameter(
            torch.empty(in_channels, out_channels // groups, *kernel_size)
        )
        if bias:
            self.bias_param = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias_param", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias_param is not None:
            fan_in = self.weight.size(0)
            for k in self.weight.shape[2:]:
                fan_in *= k
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_param, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _run_conv_transpose3d(
            x,
            self.weight,
            self.bias_param,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
        )
