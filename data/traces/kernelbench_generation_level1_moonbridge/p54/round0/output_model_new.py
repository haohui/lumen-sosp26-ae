import torch
import torch.nn as nn
import avelang
import avelang.language as al

THREADS = 256


@avelang.jit
def conv3d_bf16_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    IC: al.i32,
    OC: al.i32,
    KD: al.i32,
    KH: al.i32,
    KW: al.i32,
    D: al.i32,
    H: al.i32,
    W: al.i32,
    OD: al.i32,
    OH: al.i32,
    OW: al.i32,
    pad_d: al.i32,
    pad_h: al.i32,
    pad_w: al.i32,
    stride_d: al.i32,
    stride_h: al.i32,
    stride_w: al.i32,
    dilation_d: al.i32,
    dilation_h: al.i32,
    dilation_w: al.i32,
    groups: al.i32,
):
    tid = al.thread_id(0)
    block_spatial = al.block_id(0)
    block_oc = al.block_id(1)
    block_b = al.block_id(2)

    IC_per_group = IC // groups
    K_DIM = IC_per_group * KD * KH * KW
    total_spatial = OD * OH * OW

    x_size = B * IC * D * H * W
    w_size = OC * K_DIM
    out_size = B * OC * OD * OH * OW

    x_flat = al.make_tensor(x_ptr, al.bf16, al.make_layout((x_size,), (1,)))
    w_flat = al.make_tensor(w_ptr, al.bf16, al.make_layout((w_size,), (1,)))
    out_flat = al.make_tensor(out_ptr, al.bf16, al.make_layout((out_size,), (1,)))

    pos = block_spatial * THREADS + tid

    if pos < total_spatial:
        oc = block_oc
        b = block_b

        ow = pos % OW
        oh = (pos // OW) % OH
        od = pos // (OW * OH)

        group_id = oc // (OC // groups)
        ic_start = group_id * IC_per_group
        ic_end = ic_start + IC_per_group

        in_d_base = od * stride_d - pad_d
        in_h_base = oh * stride_h - pad_h
        in_w_base = ow * stride_w - pad_w

        acc = al.convert(0.0, al.f32)

        x_batch_base = b * IC * D * H * W
        x_DHW = D * H * W
        x_HW = H * W

        w_oc_base = oc * K_DIM

        d_ok = 1
        if in_d_base < 0:
            d_ok = 0
        if (in_d_base + (KD - 1) * dilation_d) >= D:
            d_ok = 0
        h_ok = 1
        if in_h_base < 0:
            h_ok = 0
        if (in_h_base + (KH - 1) * dilation_h) >= H:
            h_ok = 0
        w_ok = 1
        if in_w_base < 0:
            w_ok = 0
        if (in_w_base + (KW - 1) * dilation_w) >= W:
            w_ok = 0

        interior = d_ok * h_ok * w_ok

        if interior != 0:
            for ic in al.range(ic_start, ic_end):
                x_ic_base = x_batch_base + ic * x_DHW
                w_ic_base = w_oc_base + (ic - ic_start) * KD * KH * KW

                for kd in al.range(KD):
                    id_val = in_d_base + kd * dilation_d
                    x_d_base = x_ic_base + id_val * x_HW
                    w_d_base = w_ic_base + kd * KH * KW

                    for kh in al.range(KH):
                        ih_val = in_h_base + kh * dilation_h
                        x_h_base = x_d_base + ih_val * W
                        w_h_base = w_d_base + kh * KW

                        for kw in al.range(KW):
                            iw_val = in_w_base + kw * dilation_w
                            x_idx = x_h_base + iw_val
                            w_idx = w_h_base + kw

                            x_val = al.convert(x_flat[x_idx], al.f32)
                            w_val = al.convert(w_flat[w_idx], al.f32)
                            acc = acc + x_val * w_val
        else:
            for ic in al.range(ic_start, ic_end):
                x_ic_base = x_batch_base + ic * x_DHW
                w_ic_base = w_oc_base + (ic - ic_start) * KD * KH * KW

                for kd in al.range(KD):
                    id_val = in_d_base + kd * dilation_d
                    if id_val >= 0:
                        if id_val < D:
                            x_d_base = x_ic_base + id_val * x_HW
                            w_d_base = w_ic_base + kd * KH * KW

                            for kh in al.range(KH):
                                ih_val = in_h_base + kh * dilation_h
                                if ih_val >= 0:
                                    if ih_val < H:
                                        x_h_base = x_d_base + ih_val * W
                                        w_h_base = w_d_base + kh * KW

                                        for kw in al.range(KW):
                                            iw_val = in_w_base + kw * dilation_w
                                            if iw_val >= 0:
                                                if iw_val < W:
                                                    x_idx = x_h_base + iw_val
                                                    w_idx = w_h_base + kw

                                                    x_val = al.convert(x_flat[x_idx], al.f32)
                                                    w_val = al.convert(w_flat[w_idx], al.f32)
                                                    acc = acc + x_val * w_val

        out_idx = b * OC * OD * OH * OW + oc * OD * OH * OW + od * OH * OW + oh * OW + ow
        out_flat[out_idx] = al.convert(acc, al.bf16)


def avelang_conv3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = x.contiguous().cuda().to(torch.bfloat16)
    w_bf16 = weight.contiguous().cuda().to(torch.bfloat16)

    B, IC, D, H, W = x_bf16.shape
    OC, IC_per_group, KD, KH, KW = w_bf16.shape

    if isinstance(stride, int):
        stride_d, stride_h, stride_w = stride, stride, stride
    else:
        stride_d, stride_h, stride_w = stride

    if isinstance(padding, int):
        pad_d, pad_h, pad_w = padding, padding, padding
    else:
        pad_d, pad_h, pad_w = padding

    if isinstance(dilation, int):
        dilation_d, dilation_h, dilation_w = dilation, dilation, dilation
    else:
        dilation_d, dilation_h, dilation_w = dilation

    OD = (D + 2 * pad_d - dilation_d * (KD - 1) - 1) // stride_d + 1
    OH = (H + 2 * pad_h - dilation_h * (KH - 1) - 1) // stride_h + 1
    OW = (W + 2 * pad_w - dilation_w * (KW - 1) - 1) // stride_w + 1

    total_spatial = OD * OH * OW
    num_spatial_blocks = (total_spatial + THREADS - 1) // THREADS

    out = torch.empty((B, OC, OD, OH, OW), device=x_bf16.device, dtype=torch.bfloat16)

    conv3d_bf16_kernel[lambda: ((num_spatial_blocks, OC, B), (THREADS, 1, 1))](
        x_bf16,
        w_bf16,
        out,
        B,
        IC,
        OC,
        KD,
        KH,
        KW,
        D,
        H,
        W,
        OD,
        OH,
        OW,
        pad_d,
        pad_h,
        pad_w,
        stride_d,
        stride_h,
        stride_w,
        dilation_d,
        dilation_h,
        dilation_w,
        groups,
    )
    return out


class ModelNew(nn.Module):
    """
    Performs a standard 3D convolution operation with square input and square kernel,
    accelerated with an AveLang BF16 GPU kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.has_bias = bias

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in = self.in_channels * self.kernel_size * self.kernel_size * self.kernel_size
            bound = 1 / (fan_in**0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = avelang_conv3d(
            x,
            self.weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

        if self.has_bias and self.bias is not None:
            bf16_bias = self.bias.contiguous().cuda().to(torch.bfloat16)
            result = result + bf16_bias.view(1, -1, 1, 1, 1)

        return result


# Test code
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 3
depth = 64
width = 64
height = 64


def get_inputs():
    x = torch.rand(batch_size, in_channels, depth, width, height)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
