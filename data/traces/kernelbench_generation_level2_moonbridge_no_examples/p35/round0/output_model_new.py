import torch
import torch.nn as nn
import avelang
import avelang.language as al

_OC_TILE = 16
_OH_TILE = 4
_OW_TILE = 4


@avelang.jit
def fused_conv_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    N: al.i32,
    IC: al.i32,
    H: al.i32,
    W: al.i32,
    OC: al.i32,
    OH: al.i32,
    OW: al.i32,
    OW_blocks: al.i32,
    OC_TILE: al.constexpr = al.constexpr(_OC_TILE),
    OH_TILE: al.constexpr = al.constexpr(_OH_TILE),
    OW_TILE: al.constexpr = al.constexpr(_OW_TILE),
):
    n_idx = al.block_id(0)
    oc_block = al.block_id(1)
    spatial_block = al.block_id(2)

    oh_block = spatial_block // OW_blocks
    ow_block = spatial_block % OW_blocks

    flat_tid = al.thread_id(0)
    oc = oc_block * OC_TILE + flat_tid // (OH_TILE * OW_TILE)
    spatial_tid = flat_tid % (OH_TILE * OW_TILE)
    oh = oh_block * OH_TILE + spatial_tid // OW_TILE
    ow = ow_block * OW_TILE + spatial_tid % OW_TILE

    if oc < OC and oh < OH and ow < OW:
        in_n_stride = IC * H * W
        in_c_stride = H * W
        in_h_stride = W

        w_oc_stride = IC * 9
        w_ic_stride = 9
        w_kh_stride = 3

        out_n_stride = OC * OH * OW
        out_oc_stride = OH * OW
        out_oh_stride = OW

        inp = al.make_tensor(input_ptr, al.bf16, al.make_layout((N, IC, H, W), (in_n_stride, in_c_stride, in_h_stride, 1)))
        wgt = al.make_tensor(weight_ptr, al.bf16, al.make_layout((OC, IC, 3, 3), (w_oc_stride, w_ic_stride, w_kh_stride, 1)))
        out = al.make_tensor(output_ptr, al.bf16, al.make_layout((N, OC, OH, OW), (out_n_stride, out_oc_stride, out_oh_stride, 1)))

        bias_t = al.make_tensor(bias_ptr, al.bf16, al.make_layout((OC,), (1,)))
        base_val = al.convert(bias_t[oc], al.f32)

        h0 = 2 * oh
        w0 = 2 * ow
        h1 = h0 + 1
        w1 = w0 + 1

        acc00 = base_val
        acc01 = base_val
        acc10 = base_val
        acc11 = base_val

        for ic in al.range(IC):
            for kh in al.range(3):
                inp_h0 = h0 + kh
                inp_h1 = h1 + kh
                for kw in al.range(3):
                    inp_bf16_00 = inp[n_idx, ic, inp_h0, w0 + kw]
                    inp_bf16_01 = inp[n_idx, ic, inp_h0, w1 + kw]
                    inp_bf16_10 = inp[n_idx, ic, inp_h1, w0 + kw]
                    inp_bf16_11 = inp[n_idx, ic, inp_h1, w1 + kw]

                    wgt_bf16 = wgt[oc, ic, kh, kw]

                    inp00_f32 = al.convert(inp_bf16_00, al.f32)
                    inp01_f32 = al.convert(inp_bf16_01, al.f32)
                    inp10_f32 = al.convert(inp_bf16_10, al.f32)
                    inp11_f32 = al.convert(inp_bf16_11, al.f32)
                    wgt_f32 = al.convert(wgt_bf16, al.f32)

                    acc00 = acc00 + inp00_f32 * wgt_f32
                    acc01 = acc01 + inp01_f32 * wgt_f32
                    acc10 = acc10 + inp10_f32 * wgt_f32
                    acc11 = acc11 + inp11_f32 * wgt_f32

        hs00_relu = acc00 + 3.0
        if hs00_relu > 6.0:
            hs00_relu = 6.0
        if hs00_relu < 0.0:
            hs00_relu = 0.0
        hs00 = acc00 * hs00_relu / 6.0

        hs01_relu = acc01 + 3.0
        if hs01_relu > 6.0:
            hs01_relu = 6.0
        if hs01_relu < 0.0:
            hs01_relu = 0.0
        hs01 = acc01 * hs01_relu / 6.0

        hs10_relu = acc10 + 3.0
        if hs10_relu > 6.0:
            hs10_relu = 6.0
        if hs10_relu < 0.0:
            hs10_relu = 0.0
        hs10 = acc10 * hs10_relu / 6.0

        hs11_relu = acc11 + 3.0
        if hs11_relu > 6.0:
            hs11_relu = 6.0
        if hs11_relu < 0.0:
            hs11_relu = 0.0
        hs11 = acc11 * hs11_relu / 6.0

        pooled = hs00
        if hs01 > pooled:
            pooled = hs01
        if hs10 > pooled:
            pooled = hs10
        if hs11 > pooled:
            pooled = hs11

        softplus_val = al.log(1.0 + al.exp(pooled))
        exp_2sp = al.exp(2.0 * softplus_val)
        tanh_sp = (exp_2sp - 1.0) / (exp_2sp + 1.0)
        mish_val = pooled * tanh_sp

        out_bf16_val = al.convert(mish_val, al.bf16)
        out[n_idx, oc, oh, ow] = out_bf16_val


def _run_fused_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    subtract_value: float,
    pool_kernel_size: int,
) -> torch.Tensor:
    assert x.is_cuda, "Input must be on CUDA/HIP device"
    assert weight.is_cuda, "Weight must be on CUDA/HIP device"
    assert bias.is_cuda, "Bias must be on CUDA/HIP device"

    N, IC, H, W = x.shape
    OC = weight.shape[0]
    KH, KW = weight.shape[2], weight.shape[3]

    conv_h = H - KH + 1
    conv_w = W - KW + 1

    OH = conv_h // pool_kernel_size
    OW = conv_w // pool_kernel_size

    x_bf16 = x.to(torch.bfloat16).contiguous()
    w_bf16 = weight.to(torch.bfloat16).contiguous()
    b_adjusted = (bias - subtract_value).to(torch.bfloat16).contiguous()

    out_bf16 = torch.empty(N, OC, OH, OW, dtype=torch.bfloat16, device=x.device)

    OC_TILE = _OC_TILE
    OH_TILE = _OH_TILE
    OW_TILE = _OW_TILE

    oc_blocks = (OC + OC_TILE - 1) // OC_TILE
    oh_blocks = (OH + OH_TILE - 1) // OH_TILE
    ow_blocks = (OW + OW_TILE - 1) // OW_TILE
    spatial_blocks = oh_blocks * ow_blocks

    threads_per_block = OC_TILE * OH_TILE * OW_TILE

    fused_conv_kernel[lambda: (
        (N, oc_blocks, spatial_blocks),
        (threads_per_block, 1, 1),
    )](
        x_bf16.data_ptr(),
        w_bf16.data_ptr(),
        b_adjusted.data_ptr(),
        out_bf16.data_ptr(),
        N, IC, H, W, OC, OH, OW,
        ow_blocks,
        OC_TILE, OH_TILE, OW_TILE,
    )

    return out_bf16


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value = subtract_value
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x):
        weight = self.conv.weight.data
        bias = self.conv.bias.data
        return _run_fused_conv(
            x, weight, bias, self.subtract_value, self.pool_kernel_size
        )
