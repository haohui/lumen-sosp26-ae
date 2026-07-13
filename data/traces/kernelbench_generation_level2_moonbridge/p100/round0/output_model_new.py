import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def _conv_transpose3d_fused_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    N: al.i32,
    IC: al.i32,
    OC: al.i32,
    ID: al.i32,
    IH: al.i32,
    IW: al.i32,
    OD: al.i32,
    OH: al.i32,
    OW: al.i32,
    KD: al.i32,
    KH: al.i32,
    KW: al.i32,
    stride: al.i32,
    padding: al.i32,
    min_value: al.f32,
    divisor: al.f32,
    spatial_total: al.i32,
):
    oc = al.thread_id(0)
    bid = al.block_id(0)

    n_val = bid // spatial_total
    spatial_idx = bid % spatial_total

    if oc < OC:
        tmp = spatial_idx
        ow_val = tmp % OW
        tmp = tmp // OW
        oh_val = tmp % OH
        tmp = tmp // OH
        od_val = tmp % OD

        in_sN = IC * ID * IH * IW
        in_sIC = ID * IH * IW
        in_sID = IH * IW
        in_sIH = IW

        w_sIC = OC * KD * KH * KW
        w_sOC = KD * KH * KW
        w_sKD = KH * KW
        w_sKH = KW

        out_sN = OC * OD * OH * OW
        out_sOC = OD * OH * OW
        out_sOD = OH * OW
        out_sOH = OW

        in_size = N * IC * ID * IH * IW
        w_size = IC * OC * KD * KH * KW
        out_size = N * OC * OD * OH * OW

        in_1d = al.make_layout((in_size,), (1,))
        w_1d = al.make_layout((w_size,), (1,))
        out_1d = al.make_layout((out_size,), (1,))
        bias_1d = al.make_layout((OC,), (1,))

        input_t = al.make_tensor(input_ptr, al.bf16, in_1d)
        weight_t = al.make_tensor(weight_ptr, al.bf16, w_1d)
        output_t = al.make_tensor(output_ptr, al.bf16, out_1d)
        bias_t = al.make_tensor(bias_ptr, al.bf16, bias_1d)

        out_base = n_val * out_sN + oc * out_sOC + od_val * out_sOD + oh_val * out_sOH + ow_val
        in_n_base = n_val * in_sN
        w_oc_base = oc * w_sOC

        d_parity = od_val % stride
        h_parity = oh_val % stride
        w_parity = ow_val % stride

        acc = al.convert(0.0, al.f32)

        for ic in al.range(IC):
            in_ic_base = in_n_base + ic * in_sIC
            w_ic_base = ic * w_sIC + w_oc_base

            if d_parity == 0:
                kd0 = al.convert(1, al.i32)
                id0 = od_val // stride
                in_d0 = id0 * in_sID
                w_d0 = kd0 * w_sKD

                if h_parity == 0:
                    kh0 = al.convert(1, al.i32)
                    ih0 = oh_val // stride
                    in_h0 = ih0 * in_sIH
                    w_h0 = kh0 * w_sKH

                    if w_parity == 0:
                        kw0 = al.convert(1, al.i32)
                        iw0 = ow_val // stride
                        in_idx = in_ic_base + in_d0 + in_h0 + iw0
                        wt_idx = w_ic_base + w_d0 + w_h0 + kw0
                        acc = acc + al.convert(input_t[in_idx], al.f32) * al.convert(weight_t[wt_idx], al.f32)
                    else:
                        kw0 = al.convert(0, al.i32)
                        iw0 = (ow_val + padding) // stride
                        kw1 = al.convert(2, al.i32)
                        iw1 = (ow_val + padding - kw1) // stride
                        in_idx = in_ic_base + in_d0 + in_h0 + iw0
                        acc = acc + al.convert(input_t[in_idx], al.f32) * al.convert(weight_t[w_ic_base + w_d0 + w_h0 + kw0], al.f32)
                        in_idx = in_ic_base + in_d0 + in_h0 + iw1
                        acc = acc + al.convert(input_t[in_idx], al.f32) * al.convert(weight_t[w_ic_base + w_d0 + w_h0 + kw1], al.f32)
                else:
                    kh0 = al.convert(0, al.i32)
                    ih0 = (oh_val + padding) // stride
                    kh1 = al.convert(2, al.i32)
                    ih1 = (oh_val + padding - kh1) // stride
                    in_h0v = ih0 * in_sIH
                    in_h1v = ih1 * in_sIH
                    w_h0v = kh0 * w_sKH
                    w_h1v = kh1 * w_sKH

                    if w_parity == 0:
                        kw0 = al.convert(1, al.i32)
                        iw0 = ow_val // stride
                        in_idx = in_ic_base + in_d0 + in_h0v + iw0
                        acc = acc + al.convert(input_t[in_idx], al.f32) * al.convert(weight_t[w_ic_base + w_d0 + w_h0v + kw0], al.f32)
                        in_idx = in_ic_base + in_d0 + in_h1v + iw0
                        acc = acc + al.convert(input_t[in_idx], al.f32) * al.convert(weight_t[w_ic_base + w_d0 + w_h1v + kw0], al.f32)
                    else:
                        kw0 = al.convert(0, al.i32)
                        iw0 = (ow_val + padding) // stride
                        kw1 = al.convert(2, al.i32)
                        iw1 = (ow_val + padding - kw1) // stride
                        in_idx = in_ic_base + in_d0 + in_h0v + iw0
                        acc = acc + al.convert(input_t[in_idx], al.f32) * al.convert(weight_t[w_ic_base + w_d0 + w_h0v + kw0], al.f32)
                        in_idx = in_ic_base + in_d0 + in_h0v + iw1
                        acc = acc + al.convert(input_t[in_idx], al.f32) * al.convert(weight_t[w_ic_base + w_d0 + w_h0v + kw1], al.f32)
                        in_idx = in_ic_base + in_d0 + in_h1v + iw0
                        acc = acc + al.convert(input_t[in_idx], al.f32) * al.convert(weight_t[w_ic_base + w_d0 + w_h1v + kw0], al.f32)
                        in_idx = in_ic_base + in_d0 + in_h1v + iw1
                        acc = acc + al.convert(input_t[in_idx], al.f32) * al.convert(weight_t[w_ic_base + w_d0 + w_h1v + kw1], al.f32)
            else:
                kd0 = al.convert(0, al.i32)
                id0 = (od_val + padding) // stride
                kd1 = al.convert(2, al.i32)
                id1 = (od_val + padding - kd1) // stride
                in_d0v = id0 * in_sID
                in_d1v = id1 * in_sID
                w_d0v = kd0 * w_sKD
                w_d1v = kd1 * w_sKD

                if h_parity == 0:
                    kh0 = al.convert(1, al.i32)
                    ih0 = oh_val // stride
                    in_h0v = ih0 * in_sIH
                    w_h0v = kh0 * w_sKH

                    if w_parity == 0:
                        kw0 = al.convert(1, al.i32)
                        iw0 = ow_val // stride
                        in_idx = in_ic_base + in_d0v + in_h0v + iw0
                        acc = acc + al.convert(input_t[in_idx], al.f32) * al.convert(weight_t[w_ic_base + w_d0v + w_h0v + kw0], al.f32)
                        in_idx = in_ic_base + in_d1v + in_h0v + iw0
                        acc = acc + al.convert(input_t[in_idx], al.f32) * al.convert(weight_t[w_ic_base + w_d1v + w_h0v + kw0], al.f32)
                    else:
                        kw0 = al.convert(0, al.i32)
                        iw0 = (ow_val + padding) // stride
                        kw1 = al.convert(2, al.i32)
                        iw1 = (ow_val + padding - kw1) // stride
                        in_idx = in_ic_base + in_d0v + in_h0v + iw0
                        acc = acc + al.convert(input_t[in_idx], al.f32) * al.convert(weight_t[w_ic_base + w_d0v + w_h0v + kw0], al.f32)
                        in_idx = in_ic_base + in_d0v + in_h0v + iw1
                        acc = acc + al.convert(input_t[in_idx], al.f32) * al.convert(weight_t[w_ic_base + w_d0v + w_h0v + kw1], al.f32)
                        in_idx = in_ic_base + in_d1v + in_h0v + iw0
                        acc = acc + al.convert(input_t[in_idx], al.f32) * al.convert(weight_t[w_ic_base + w_d1v + w_h0v + kw0], al.f32)
                        in_idx = in_ic_base + in_d1v + in_h0v + iw1
                        acc = acc + al.convert(input_t[in_idx], al.f32) * al.convert(weight_t[w_ic_base + w_d1v + w_h0v + kw1], al.f32)
                else:
                    kh0 = al.convert(0, al.i32)
                    ih0 = (oh_val + padding) // stride
                    kh1 = al.convert(2, al.i32)
                    ih1 = (oh_val + padding - kh1) // stride
                    in_h0v = ih0 * in_sIH
                    in_h1v = ih1 * in_sIH
                    w_h0v = kh0 * w_sKH
                    w_h1v = kh1 * w_sKH

                    if w_parity == 0:
                        kw0 = al.convert(1, al.i32)
                        iw0 = ow_val // stride
                        in_idx = in_ic_base + in_d0v + in_h0v + iw0
                        acc = acc + al.convert(input_t[in_idx], al.f32) * al.convert(weight_t[w_ic_base + w_d0v + w_h0v + kw0], al.f32)
                        in_idx = in_ic_base + in_d0v + in_h1v + iw0
                        acc = acc + al.convert(input_t[in_idx], al.f32) * al.convert(weight_t[w_ic_base + w_d0v + w_h1v + kw0], al.f32)
                        in_idx = in_ic_base + in_d1v + in_h0v + iw0
                        acc = acc + al.convert(input_t[in_idx], al.f32) * al.convert(weight_t[w_ic_base + w_d1v + w_h0v + kw0], al.f32)
                        in_idx = in_ic_base + in_d1v + in_h1v + iw0
                        acc = acc + al.convert(input_t[in_idx], al.f32) * al.convert(weight_t[w_ic_base + w_d1v + w_h1v + kw0], al.f32)
                    else:
                        kw0 = al.convert(0, al.i32)
                        iw0 = (ow_val + padding) // stride
                        kw1 = al.convert(2, al.i32)
                        iw1 = (ow_val + padding - kw1) // stride
                        in_idx = in_ic_base + in_d0v + in_h0v + iw0
                        acc = acc + al.convert(input_t[in_idx], al.f32) * al.convert(weight_t[w_ic_base + w_d0v + w_h0v + kw0], al.f32)
                        in_idx = in_ic_base + in_d0v + in_h0v + iw1
                        acc = acc + al.convert(input_t[in_idx], al.f32) * al.convert(weight_t[w_ic_base + w_d0v + w_h0v + kw1], al.f32)
                        in_idx = in_ic_base + in_d0v + in_h1v + iw0
                        acc = acc + al.convert(input_t[in_idx], al.f32) * al.convert(weight_t[w_ic_base + w_d0v + w_h1v + kw0], al.f32)
                        in_idx = in_ic_base + in_d0v + in_h1v + iw1
                        acc = acc + al.convert(input_t[in_idx], al.f32) * al.convert(weight_t[w_ic_base + w_d0v + w_h1v + kw1], al.f32)
                        in_idx = in_ic_base + in_d1v + in_h0v + iw0
                        acc = acc + al.convert(input_t[in_idx], al.f32) * al.convert(weight_t[w_ic_base + w_d1v + w_h0v + kw0], al.f32)
                        in_idx = in_ic_base + in_d1v + in_h0v + iw1
                        acc = acc + al.convert(input_t[in_idx], al.f32) * al.convert(weight_t[w_ic_base + w_d1v + w_h0v + kw1], al.f32)
                        in_idx = in_ic_base + in_d1v + in_h1v + iw0
                        acc = acc + al.convert(input_t[in_idx], al.f32) * al.convert(weight_t[w_ic_base + w_d1v + w_h1v + kw0], al.f32)
                        in_idx = in_ic_base + in_d1v + in_h1v + iw1
                        acc = acc + al.convert(input_t[in_idx], al.f32) * al.convert(weight_t[w_ic_base + w_d1v + w_h1v + kw1], al.f32)

        acc = acc + al.convert(bias_t[oc], al.f32)
        if acc < min_value:
            acc = min_value
        acc = acc / divisor
        output_t[out_base] = al.convert(acc, al.bf16)


def _prepare_bf16_cuda_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_conv_transpose3d_fused(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: int,
    padding: int,
    min_value: float,
    divisor: float,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = _prepare_bf16_cuda_contiguous(x)
    weight_bf16 = _prepare_bf16_cuda_contiguous(weight)
    if bias is not None:
        bias_bf16 = _prepare_bf16_cuda_contiguous(bias)
    else:
        bias_bf16 = torch.zeros(weight_bf16.shape[1], device=weight_bf16.device, dtype=torch.bfloat16)

    N, IC, ID, IH, IW = x_bf16.shape
    wIC, OC, KD, KH, KW = weight_bf16.shape

    OD = stride * (ID - 1) + KD - 2 * padding
    OH = stride * (IH - 1) + KH - 2 * padding
    OW = stride * (IW - 1) + KW - 2 * padding

    spatial_total = OD * OH * OW
    total_blocks = spatial_total * N
    output = torch.empty((N, OC, OD, OH, OW), device=x_bf16.device, dtype=torch.bfloat16)

    _conv_transpose3d_fused_kernel[lambda: ((total_blocks, 1, 1), (OC, 1, 1))](
        x_bf16,
        weight_bf16,
        bias_bf16,
        output,
        N,
        IC,
        OC,
        ID,
        IH,
        IW,
        OD,
        OH,
        OW,
        KD,
        KH,
        KW,
        stride,
        padding,
        min_value,
        divisor,
        spatial_total,
    )

    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.min_value = min_value
        self.divisor = divisor

        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )

    def forward(self, x):
        orig_dtype = x.dtype
        result_bf16 = avelang_conv_transpose3d_fused(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.stride,
            self.padding,
            self.min_value,
            self.divisor,
        )
        return result_bf16.to(orig_dtype)


batch_size = 16
in_channels = 64
out_channels = 128
depth, height, width = 24, 48, 48
kernel_size = 3
stride = 2
padding = 1
min_value = -1.0
divisor = 2.0


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, min_value, divisor]
