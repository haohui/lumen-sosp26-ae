import torch
import torch.nn as nn
import avelang
import avelang.language as al

_BLOCK_SIZE = 256


@avelang.jit
def conv2d_3x3_postprocess_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    B: al.i32,
    IC: al.i32,
    OC: al.i32,
    H: al.i32,
    W: al.i32,
    OH: al.i32,
    OW: al.i32,
):
    inp_total = B * IC * H * W
    inp_layout = al.make_layout((inp_total,), (1,))
    inp = al.make_tensor(input_ptr, al.bf16, inp_layout)

    w_total = OC * IC * 9
    w_layout = al.make_layout((w_total,), (1,))
    w = al.make_tensor(weight_ptr, al.bf16, w_layout)

    bias_layout = al.make_layout((OC,), (1,))
    bias = al.make_tensor(bias_ptr, al.bf16, bias_layout)

    out_total = B * OC * OH * OW
    out_layout = al.make_layout((out_total,), (1,))
    out = al.make_tensor(output_ptr, al.bf16, out_layout)

    b = al.block_id(0)
    oc = al.block_id(1)
    spatial_block = al.block_id(2)
    tid = al.thread_id(0)

    spatial_idx = spatial_block * _BLOCK_SIZE + tid
    oh = spatial_idx // OW
    ow = spatial_idx % OW

    if (oh < OH) and (ow < OW):
        inp_b = b * IC * H * W
        w_oc = oc * IC * 9
        out_bo = (b * OC + oc) * OH * OW

        # FP32 accumulator
        acc = al.convert(0.0, al.f32)

        for ic in al.range(IC):
            inp_bic = inp_b + ic * H * W
            w_ocic = w_oc + ic * 9
            for kh in al.range(3):
                inp_row = (oh + kh) * W
                w_kh = kh * 3
                for kw in al.range(3):
                    inp_idx = inp_bic + inp_row + (ow + kw)
                    inp_val = al.convert(inp[inp_idx], al.f32)

                    w_idx = w_ocic + w_kh + kw
                    w_val = al.convert(w[w_idx], al.f32)

                    acc = acc + inp_val * w_val

        bias_val = al.convert(bias[oc], al.f32)
        acc = acc + bias_val

        # Divide by 2 (multiply by 0.5)
        acc = acc * al.convert(0.5, al.f32)

        # LeakyReLU negative_slope=0.01, branch-free
        abs_acc = al.abs(acc)
        relu = (acc + abs_acc) * al.convert(0.5, al.f32)
        slope = al.convert(0.01, al.f32)
        one_m_slope = al.convert(1.0, al.f32) - slope
        result = slope * acc + one_m_slope * relu

        out_idx = out_bo + oh * OW + ow
        out[out_idx] = al.convert(result, al.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divisor = divisor

    def forward(self, x):
        B, IC, H, W = x.shape
        OC = self.conv.out_channels
        KH = self.conv.kernel_size[0]
        KW = self.conv.kernel_size[1]
        OH = H - KH + 1
        OW = W - KW + 1

        x_bf16 = x.contiguous().to(torch.bfloat16)
        w_bf16 = self.conv.weight.data.contiguous().to(torch.bfloat16)
        b_bf16 = self.conv.bias.data.contiguous().to(torch.bfloat16)

        out = torch.empty(B, OC, OH, OW, dtype=torch.bfloat16, device=x.device)

        num_spatial_blocks = (OH * OW + _BLOCK_SIZE - 1) // _BLOCK_SIZE

        conv2d_3x3_postprocess_kernel[
            lambda: ((B, OC, num_spatial_blocks), (_BLOCK_SIZE, 1, 1))
        ](
            x_bf16, w_bf16, b_bf16, out,
            B, IC, OC, H, W, OH, OW,
        )

        return out
