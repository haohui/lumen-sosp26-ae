import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def postprocess_kernel(
    conv_ptr: al.Pointer(al.bf16),
    model_bias_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    total: al.i32,
    OC: al.i32,
    OD: al.i32,
    OH: al.i32,
    OW: al.i32,
    BLOCK_SIZE: al.i32,
):
    one = al.convert(1, al.i32)
    stride_n = OC * OD * OH * OW
    stride_c = OD * OH * OW
    stride_d = OH * OW
    stride_h = OW
    layout = al.make_layout(
        (total // (OC * OD * OH * OW), OC, OD, OH, OW),
        (stride_n, stride_c, stride_d, stride_h, one),
    )
    conv = al.make_tensor(conv_ptr, al.bf16, layout)
    mb = al.make_tensor(model_bias_ptr, al.bf16, al.make_layout((OC,), (one,)))
    out = al.make_tensor(output_ptr, al.bf16, layout)

    tid = al.thread_id(0)
    bid = al.block_id(0)
    idx = bid * BLOCK_SIZE + tid
    if idx < total:
        n = idx // stride_n
        rem = idx % stride_n
        oc = rem // stride_c
        rem = rem % stride_c
        od_idx = rem // stride_d
        rem = rem % stride_d
        oh_idx = rem // stride_h
        ow_idx = rem % stride_h

        c_val = al.convert(conv[n, oc, od_idx, oh_idx, ow_idx], al.f32)
        b_val = al.convert(mb[oc], al.f32)
        x = c_val + b_val
        x = x + c_val
        x = x * c_val
        x = x + c_val
        out[n, oc, od_idx, oh_idx, ow_idx] = al.convert(x, al.bf16)


def avelang_postprocess(
    conv_out: torch.Tensor,
    model_bias: torch.Tensor,
) -> torch.Tensor:
    assert conv_out.is_cuda, "conv_out must be on GPU"
    conv_bf16 = conv_out.to(torch.bfloat16).contiguous()
    mb_bf16 = model_bias.reshape(-1).to(torch.bfloat16).contiguous()
    B, OC, OD, OH, OW = conv_bf16.shape
    total = B * OC * OD * OH * OW

    out_bf16 = torch.empty_like(conv_bf16)

    BLOCK_SIZE = 256
    grid_x = (total + BLOCK_SIZE - 1) // BLOCK_SIZE

    postprocess_kernel[lambda: ((grid_x, 1, 1), (BLOCK_SIZE, 1, 1))](
        conv_bf16, mb_bf16, out_bf16,
        total, OC, OD, OH, OW, BLOCK_SIZE,
    )
    return out_bf16


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        bias_shape,
    ):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv_transpose(x)
        return avelang_postprocess(x, self.bias)
