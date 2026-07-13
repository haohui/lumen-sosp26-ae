import math
import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def conv_transpose3d_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    B: al.i32,
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
    TILE_SPATIAL: al.i32,
):
    tid = al.thread_id(0)
    tile_idx = al.block_id(0)
    oc = al.block_id(1)
    b = al.block_id(2)

    spatial_idx = tile_idx * TILE_SPATIAL + tid
    total_spatial = OD * OH * OW

    if spatial_idx < total_spatial:
        ow = spatial_idx % OW
        oh_rem = spatial_idx // OW
        oh = oh_rem % OH
        od = oh_rem // OH

        input_1d = al.make_tensor(input_ptr, al.bf16, al.make_layout((B * IC * ID * IH * IW,), (1,)))
        weight_1d = al.make_tensor(weight_ptr, al.bf16, al.make_layout((IC * OC * KD * KH * KW,), (1,)))
        output_1d = al.make_tensor(output_ptr, al.bf16, al.make_layout((B * OC * OD * OH * OW,), (1,)))

        inp_ic_stride = ID * IH * IW
        inp_id_stride = IH * IW
        inp_ih_stride = IW
        inp_batch_base = b * (IC * ID * IH * IW)

        wt_oc_stride = KD * KH * KW
        wt_ic_stride = OC * KD * KH * KW

        out_base = b * (OC * OD * OH * OW) + oc * (OD * OH * OW) + od * (OH * OW) + oh * OW + ow

        kd_start = al.max(0, od - ID + 1)
        kd_end = al.min(KD, od + 1)
        kh_start = al.max(0, oh - IH + 1)
        kh_end = al.min(KH, oh + 1)
        kw_start = al.max(0, ow - IW + 1)
        kw_end = al.min(KW, ow + 1)

        acc = al.convert(0.0, al.f32)
        wt_oc_base = oc * wt_oc_stride

        for ic in al.range(IC):
            inp_ic_base = inp_batch_base + ic * inp_ic_stride
            wt_ic_base = ic * wt_ic_stride + wt_oc_base

            for kd in al.range(kd_start, kd_end):
                id_idx = od - kd
                inp_d_base = inp_ic_base + id_idx * inp_id_stride
                wt_d_base = wt_ic_base + kd * (KH * KW)

                for kh in al.range(kh_start, kh_end):
                    ih_idx = oh - kh
                    inp_h_base = inp_d_base + ih_idx * inp_ih_stride
                    wt_h_base = wt_d_base + kh * KW

                    for kw in al.range(kw_start, kw_end):
                        iw_idx = ow - kw
                        inp_val = al.convert(input_1d[inp_h_base + iw_idx], al.f32)
                        w_val = al.convert(weight_1d[wt_h_base + kw], al.f32)
                        acc = acc + inp_val * w_val

        output_1d[out_base] = al.convert(acc, al.bf16)


def _compute_output_spatial(input_size, kernel_size, stride, padding, output_padding):
    return (input_size - 1) * stride - 2 * padding + kernel_size + output_padding


def avelang_conv_transpose3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    stride: int,
    padding: int,
    output_padding: int,
) -> torch.Tensor:
    B, IC, ID, IH, IW = x.shape
    OC = weight.shape[1]
    KD, KH, KW = weight.shape[2], weight.shape[3], weight.shape[4]

    OD = _compute_output_spatial(ID, KD, stride, padding, output_padding)
    OH = _compute_output_spatial(IH, KH, stride, padding, output_padding)
    OW = _compute_output_spatial(IW, KW, stride, padding, output_padding)

    x_bf16 = x.to(torch.bfloat16).contiguous()
    w_bf16 = weight.to(torch.bfloat16).contiguous()
    out_bf16 = torch.empty(B, OC, OD, OH, OW, dtype=torch.bfloat16, device=x.device)

    TILE_SPATIAL = 256
    total_spatial = int(OD) * int(OH) * int(OW)
    grid_x = (total_spatial + TILE_SPATIAL - 1) // TILE_SPATIAL
    grid_y = OC
    grid_z = B

    conv_transpose3d_kernel[lambda: ((grid_x, grid_y, grid_z), (256, 1, 1))](
        x_bf16,
        w_bf16,
        out_bf16,
        B,
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
        256,
    )

    return out_bf16


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

        self.weight = nn.Parameter(
            torch.empty(in_channels, out_channels // groups, kernel_size, kernel_size, kernel_size)
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_conv_transpose3d(
            x,
            self.weight,
            self.stride,
            self.padding,
            self.output_padding,
        )
