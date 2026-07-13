import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def conv_transpose3d_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.f32),
    output_ptr: al.Pointer(al.f32),
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
    stride_val: al.i32,
    padding_val: al.i32,
    in_sN: al.i32,
    in_sC: al.i32,
    in_sD: al.i32,
    in_sH: al.i32,
    wt_sIC: al.i32,
    wt_sOC: al.i32,
    wt_sKD: al.i32,
    wt_sKH: al.i32,
    out_sN: al.i32,
    out_sC: al.i32,
    out_sD: al.i32,
    out_sH: al.i32,
):
    blk0 = al.block_id(0)
    n = blk0 // OD
    od = blk0 - n * OD
    oh = al.block_id(1)
    ow = al.block_id(2)
    oc = al.thread_id(0)

    inp_total = N * IC * ID * IH * IW
    wt_total = IC * OC * KD * KH * KW
    out_total = N * OC * OD * OH * OW

    inp = al.make_tensor(input_ptr, al.bf16, al.make_layout((inp_total,), (1,)))
    wt = al.make_tensor(weight_ptr, al.bf16, al.make_layout((wt_total,), (1,)))
    bias = al.make_tensor(bias_ptr, al.f32, al.make_layout((OC,), (1,)))
    out = al.make_tensor(output_ptr, al.f32, al.make_layout((out_total,), (1,)))

    acc = al.convert(bias[oc], al.f32)

    for ic in al.range(IC):
        for kd in al.range(KD):
            id_v = od + padding_val - kd
            id_div = id_v // stride_val
            if id_v == id_div * stride_val:
                if id_div >= 0:
                    if id_div < ID:
                        for kh in al.range(KH):
                            ih_v = oh + padding_val - kh
                            ih_div = ih_v // stride_val
                            if ih_v == ih_div * stride_val:
                                if ih_div >= 0:
                                    if ih_div < IH:
                                        for kw in al.range(KW):
                                            iw_v = ow + padding_val - kw
                                            iw_div = iw_v // stride_val
                                            if iw_v == iw_div * stride_val:
                                                if iw_div >= 0:
                                                    if iw_div < IW:
                                                        in_idx = n * in_sN + ic * in_sC + id_div * in_sD + ih_div * in_sH + iw_div
                                                        wt_idx = ic * wt_sIC + oc * wt_sOC + kd * wt_sKD + kh * wt_sKH + kw
                                                        inp_val = al.convert(inp[in_idx], al.f32)
                                                        w_val = al.convert(wt[wt_idx], al.f32)
                                                        acc = acc + inp_val * w_val

    out_idx = n * out_sN + oc * out_sC + od * out_sD + oh * out_sH + ow
    out[out_idx] = acc


@avelang.jit
def softmax_sigmoid_kernel(
    input_ptr: al.Pointer(al.f32),
    output_ptr: al.Pointer(al.bf16),
    N: al.i32,
    OC: al.i32,
    OD: al.i32,
    OH: al.i32,
    OW: al.i32,
    in_sN: al.i32,
    in_sC: al.i32,
    in_sD: al.i32,
    in_sH: al.i32,
    out_sN: al.i32,
    out_sC: al.i32,
    out_sD: al.i32,
    out_sH: al.i32,
):
    blk0 = al.block_id(0)
    n = blk0 // OD
    od = blk0 - n * OD
    oh = al.block_id(1)
    ow = al.block_id(2)
    tid = al.thread_id(0)

    inp_total = N * OC * OD * OH * OW
    inp = al.make_tensor(input_ptr, al.f32, al.make_layout((inp_total,), (1,)))

    zero = al.convert(0, al.f32)
    one = al.convert(1, al.f32)
    two = al.convert(2, al.f32)
    log2e = al.convert(1.4426950408889634, al.f32)

    # Load this thread's channel value
    in_idx = n * in_sN + tid * in_sC + od * in_sD + oh * in_sH + ow
    my_val = al.convert(inp[in_idx], al.f32)

    # Pass 1: find max over all channels
    # Use abs trick: max(a,b) = (a + b + abs(a - b)) / 2
    max_val = my_val
    for c in al.range(OC):
        c_idx = n * in_sN + c * in_sC + od * in_sD + oh * in_sH + ow
        c_val = al.convert(inp[c_idx], al.f32)
        diff = c_val - max_val
        abs_diff = al.abs(diff)
        s = c_val + max_val
        s = s + abs_diff
        max_val = s / two

    # Pass 2: compute sum of exp(x - max)
    # Use exp2 with log2(e) scaling: exp(x) = exp2(x * log2(e))
    sum_val = zero
    for c in al.range(OC):
        c_idx = n * in_sN + c * in_sC + od * in_sD + oh * in_sH + ow
        c_val = al.convert(inp[c_idx], al.f32)
        scaled = c_val - max_val
        scaled = scaled * log2e
        exp_val = al.exp2(scaled)
        sum_val = sum_val + exp_val

    # Normalize (softmax)
    my_scaled = my_val - max_val
    my_scaled = my_scaled * log2e
    my_exp = al.exp2(my_scaled)
    softmax_val = my_exp / sum_val

    # Sigmoid: 1 / (1 + exp(-x))
    # sigmoid(x) = 1 / (1 + exp2(-x * log2e))
    neg_x = zero - softmax_val
    neg_scaled = neg_x * log2e
    neg_exp = al.exp2(neg_scaled)
    denom = one + neg_exp
    sigmoid_val = one / denom

    out_total = N * OC * OD * OH * OW
    out = al.make_tensor(output_ptr, al.bf16, al.make_layout((out_total,), (1,)))
    out_idx = n * out_sN + tid * out_sC + od * out_sD + oh * out_sH + ow
    bf16_val = al.convert(sigmoid_val, al.bf16)
    out[out_idx] = bf16_val


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding, bias=bias
        )

    def forward(self, x):
        N, IC, D_in, H_in, W_in = x.shape
        x = x.contiguous()

        weight = self.conv_transpose.weight.detach().contiguous()
        if self.conv_transpose.bias is not None:
            bias = self.conv_transpose.bias.detach().contiguous().to(torch.float32)
        else:
            OC_val = self.conv_transpose.out_channels
            bias = torch.zeros(OC_val, dtype=torch.float32, device=x.device)

        OC_val = weight.shape[1]
        KD = weight.shape[2]
        KH = weight.shape[3]
        KW_val = weight.shape[4]

        stride_val = self.conv_transpose.stride[0]
        padding_val = self.conv_transpose.padding[0]
        output_padding_val = self.conv_transpose.output_padding[0]

        OD = (D_in - 1) * stride_val - 2 * padding_val + KD + output_padding_val
        OH = (H_in - 1) * stride_val - 2 * padding_val + KH + output_padding_val
        OW = (W_in - 1) * stride_val - 2 * padding_val + KW_val + output_padding_val

        in_sN = IC * D_in * H_in * W_in
        in_sC = D_in * H_in * W_in
        in_sD = H_in * W_in
        in_sH = W_in

        wt_sIC = OC_val * KD * KH * KW_val
        wt_sOC = KD * KH * KW_val
        wt_sKD = KH * KW_val
        wt_sKH = KW_val

        out_sN = OC_val * OD * OH * OW
        out_sC = OD * OH * OW
        out_sD = OH * OW
        out_sH = OW

        # Intermediate FP32 output from conv_transpose
        conv_out = torch.empty(N, OC_val, OD, OH, OW, dtype=torch.float32, device=x.device)

        # Grid: (N * OD, OH, OW) keeps all dims under 65536
        grid_x = N * OD
        grid_y = OH
        grid_z = OW
        conv_transpose3d_kernel[lambda: ((grid_x, grid_y, grid_z), (OC_val, 1, 1))](
            x.data_ptr(), weight.data_ptr(), bias.data_ptr(), conv_out.data_ptr(),
            N, IC, OC_val, D_in, H_in, W_in, OD, OH, OW, KD, KH, KW_val,
            stride_val, padding_val,
            in_sN, in_sC, in_sD, in_sH,
            wt_sIC, wt_sOC, wt_sKD, wt_sKH,
            out_sN, out_sC, out_sD, out_sH,
        )

        # Final BF16 output from softmax + sigmoid
        final_out = torch.empty(N, OC_val, OD, OH, OW, dtype=torch.bfloat16, device=x.device)

        softmax_sigmoid_kernel[lambda: ((grid_x, grid_y, grid_z), (OC_val, 1, 1))](
            conv_out.data_ptr(), final_out.data_ptr(),
            N, OC_val, OD, OH, OW,
            out_sN, out_sC, out_sD, out_sH,
            out_sN, out_sC, out_sD, out_sH,
        )

        return final_out
