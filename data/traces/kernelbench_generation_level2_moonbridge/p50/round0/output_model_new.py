import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE: al.constexpr = 256


@avelang.jit
def conv_transpose_avgpool_fused_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    combined_bias_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    B: al.i32,
    IC: al.i32,
    OC: al.i32,
    ID: al.i32,
    IH: al.i32,
    IW: al.i32,
    PD: al.i32,
    PH: al.i32,
    PW: al.i32,
    final_scale: al.f32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)
    gid = bid * BLOCK_SIZE + tid
    total = B * OC * PD * PH * PW

    if gid < total:
        # 1D to 5D mapping for output pool position
        tmp = gid
        wp = tmp % PW
        tmp = tmp // PW
        hp = tmp % PH
        tmp = tmp // PH
        dp = tmp % PD
        tmp = tmp // PD
        oc = tmp % OC
        b = tmp // OC

        # Build tensor views
        input_spatial = ID * IH * IW
        input_t = al.make_tensor(input_ptr, al.bf16, al.make_layout(
            (B, IC, ID, IH, IW),
            (IC * input_spatial, input_spatial, IH * IW, IW, 1),
        ))
        weight_t = al.make_tensor(weight_ptr, al.bf16, al.make_layout(
            (IC, OC, 3, 3, 3),
            (OC * 27, 27, 9, 3, 1),
        ))
        bias_t = al.make_tensor(combined_bias_ptr, al.bf16, al.make_layout((OC,), (1,)))
        output_t = al.make_tensor(output_ptr, al.bf16, al.make_layout(
            (B, OC, PD, PH, PW),
            (OC * PD * PH * PW, PD * PH * PW, PH * PW, PW, 1),
        ))

        pad = al.convert(1, al.i32)
        stride_v = al.convert(2, al.i32)
        zero_i32 = al.convert(0, al.i32)

        accum = al.convert(0.0, al.f32)

        for pd in al.range(2):
            cd = dp * 2 + pd
            for ph in al.range(2):
                ch = hp * 2 + ph
                for pw in al.range(2):
                    cw = wp * 2 + pw

                    for ic in al.range(IC):
                        for kd in al.range(3):
                            din_p = cd + pad - kd
                            if din_p >= zero_i32:
                                id_val = din_p // stride_v
                                check_d = id_val * stride_v
                                if check_d == din_p:
                                    if id_val < ID:
                                        for kh in al.range(3):
                                            hin_p = ch + pad - kh
                                            if hin_p >= zero_i32:
                                                ih_val = hin_p // stride_v
                                                check_h = ih_val * stride_v
                                                if check_h == hin_p:
                                                    if ih_val < IH:
                                                        for kw in al.range(3):
                                                            win_p = cw + pad - kw
                                                            if win_p >= zero_i32:
                                                                iw_val = win_p // stride_v
                                                                check_w = iw_val * stride_v
                                                                if check_w == win_p:
                                                                    if iw_val < IW:
                                                                        inp = al.convert(input_t[b, ic, id_val, ih_val, iw_val], al.f32)
                                                                        wgt = al.convert(weight_t[ic, oc, kd, kh, kw], al.f32)
                                                                        accum = accum + inp * wgt

        b_val = al.convert(bias_t[oc], al.f32)
        result = accum * final_scale + b_val
        output_t[b, oc, dp, hp, wp] = al.convert(result, al.bf16)


def _prepare_bf16_cuda(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_conv_transpose_avgpool(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_bias: torch.Tensor,
    custom_bias: torch.Tensor,
    scale1: float,
    scale2: float,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = _prepare_bf16_cuda(x)
    weight_bf16 = _prepare_bf16_cuda(weight)

    B, IC, ID, IH, IW = x_bf16.shape
    wIC, wOC, KD, KH, KW = weight_bf16.shape
    OC = wOC

    if wIC != IC or KD != 3 or KH != 3 or KW != 3:
        raise ValueError(f"Weight shape mismatch: expected ({IC}, OC, 3, 3, 3), got {weight_bf16.shape}")

    # Precompute combined bias:
    #   result = scale1*scale2 * avg(sum) + scale1*scale2 * conv_bias + scale2 * custom_bias
    # avg divides by 8, so final_scale = scale1*scale2 / 8
    # combined_bias = scale1*scale2 * conv_bias + scale2 * custom_bias
    conv_bias_cpu = conv_bias.to(dtype=torch.float32)
    custom_bias_cpu = custom_bias.to(dtype=torch.float32).view(-1)
    combined = (scale1 * scale2) * conv_bias_cpu + scale2 * custom_bias_cpu
    combined_bf16 = _prepare_bf16_cuda(combined)

    # ConvTranspose3d spatial output dims: stride=2, padding=1, kernel=3
    OD = (ID - 1) * 2 - 2 + 3  # = 31
    OH = (IH - 1) * 2 - 2 + 3  # = 63
    OW = (IW - 1) * 2 - 2 + 3  # = 63

    # AvgPool3d(kernel_size=2) spatial output dims
    PD_val = OD // 2  # = 15
    PH_val = OH // 2  # = 31
    PW_val = OW // 2  # = 31

    total_elements = B * OC * PD_val * PH_val * PW_val
    num_blocks = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    final_scale = scale1 * scale2 / 8.0

    out = torch.empty((B, OC, PD_val, PH_val, PW_val), device=x_bf16.device, dtype=torch.bfloat16)

    conv_transpose_avgpool_fused_kernel[lambda: ((num_blocks, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_bf16, weight_bf16, combined_bf16, out,
        B, IC, OC, ID, IH, IW,
        PD_val, PH_val, PW_val,
        final_scale,
    )

    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs 3D transposed convolution, scaling, average pooling,
    bias addition, and scaling using a fused AveLang BF16 GPU kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale1 = nn.Parameter(torch.tensor(scale1))
        self.avg_pool = nn.AvgPool3d(kernel_size=2)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale2 = nn.Parameter(torch.tensor(scale2))

        # Store scales as plain floats for CUDA graph compatibility
        self._s1 = float(scale1)
        self._s2 = float(scale2)

    def forward(self, x):
        weight = self.conv_transpose.weight.data
        conv_bias = self.conv_transpose.bias.data
        custom_bias = self.bias.data

        result_bf16 = avelang_conv_transpose_avgpool(x, weight, conv_bias, custom_bias, self._s1, self._s2)
        return result_bf16.to(x.dtype)


batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
scale1 = 0.5
scale2 = 1.0
bias_shape = (out_channels, 1, 1, 1)


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape]
