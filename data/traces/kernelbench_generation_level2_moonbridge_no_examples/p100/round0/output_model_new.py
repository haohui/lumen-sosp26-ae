import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE = 256


@avelang.jit
def ct3d_clamp_div_v2(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.f32),
    output_ptr: al.Pointer(al.bf16),
    total_elems: al.i32,
    B: al.i32, IC: al.i32, OC: al.i32,
    ID: al.i32, IH: al.i32, IW: al.i32,
    OD: al.i32, OH: al.i32, OW: al.i32,
    KD: al.i32, KH: al.i32, KW: al.i32,
    stride: al.i32, padding: al.i32,
    has_bias: al.i32,
    min_value_bits: al.i32, divisor_bits: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)
    block_dim0 = al.block_dim(0)

    gid = bid * block_dim0 + tid
    if gid >= total_elems:
        return

    is_IC = ID * IH * IW
    is_ID = IH * IW
    is_IH = IW

    ws_OC = KD * KH * KW
    ws_KD = KH * KW
    ws_KH = KW

    os_OC = OD * OH * OW
    os_OD = OH * OW
    os_OH = OW

    input_t = al.make_tensor(input_ptr, al.bf16,
        al.make_layout((B, IC, ID, IH, IW), (IC * is_IC, is_IC, is_ID, is_IH, 1)))
    weight_t = al.make_tensor(weight_ptr, al.bf16,
        al.make_layout((IC, OC, KD, KH, KW), (OC * ws_OC, ws_OC, ws_KD, ws_KH, 1)))
    output_t = al.make_tensor(output_ptr, al.bf16,
        al.make_layout((B, OC, OD, OH, OW), (OC * os_OC, os_OC, os_OD, os_OH, 1)))

    min_value = al.bitcast(min_value_bits, al.f32)
    divisor_val = al.bitcast(divisor_bits, al.f32)

    ow = gid % OW
    tmp1 = gid // OW
    oh = tmp1 % OH
    tmp2 = tmp1 // OH
    od = tmp2 % OD
    tmp3 = tmp2 // OD
    oc = tmp3 % OC
    b = tmp3 // OC

    acc = al.convert(0.0, al.f32)

    for kd in al.range(KD):
        id_val = od + padding - kd
        if id_val >= 0:
            id_rem = id_val % stride
            if id_rem == 0:
                id = id_val // stride
                if id < ID:
                    for kh in al.range(KH):
                        ih_val = oh + padding - kh
                        if ih_val >= 0:
                            ih_rem = ih_val % stride
                            if ih_rem == 0:
                                ih = ih_val // stride
                                if ih < IH:
                                    for kw in al.range(KW):
                                        iw_val = ow + padding - kw
                                        if iw_val >= 0:
                                            iw_rem = iw_val % stride
                                            if iw_rem == 0:
                                                iw = iw_val // stride
                                                if iw < IW:
                                                    for ic in al.range(IC):
                                                        w_val = weight_t[ic, oc, kd, kh, kw]
                                                        in_val = input_t[b, ic, id, ih, iw]
                                                        acc = acc + al.convert(w_val, al.f32) * al.convert(in_val, al.f32)

    if has_bias != 0:
        bias_t = al.make_tensor(bias_ptr, al.f32, al.make_layout((OC,), (1,)))
        acc = acc + bias_t[oc]

    if acc < min_value:
        acc = min_value

    acc = acc / divisor_val

    output_t[b, oc, od, oh, ow] = al.convert(acc, al.bf16)


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
        # Match reference initialization: FP32 ConvTranspose3d (no device/dtype override)
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding
        )

    def forward(self, x):
        B, IC, ID, IH, IW = x.shape
        OC = self.out_channels
        KD = self.kernel_size
        KH = self.kernel_size
        KW = self.kernel_size
        stride = self.stride
        padding = self.padding

        OD = (ID - 1) * stride - 2 * padding + KD
        OH = (IH - 1) * stride - 2 * padding + KH
        OW = (IW - 1) * stride - 2 * padding + KW

        total_elems = B * OC * OD * OH * OW

        # Move weights to GPU and convert to BF16 for kernel
        weight = self.conv_transpose.weight.data
        bias = self.conv_transpose.bias
        has_bias = 1 if bias is not None else 0

        device = x.device
        x_bf16 = x.contiguous().to(device=device, dtype=torch.bfloat16)
        weight_bf16 = weight.to(device=device, dtype=torch.bfloat16).contiguous()
        out_bf16 = torch.empty(B, OC, OD, OH, OW,
                               device=device, dtype=torch.bfloat16)

        if bias is not None:
            bias_tensor = bias.to(device=device, dtype=torch.float32).contiguous()
        else:
            bias_tensor = torch.zeros(1, device=device, dtype=torch.float32)

        num_blocks = (total_elems + BLOCK_SIZE - 1) // BLOCK_SIZE

        ct3d_clamp_div_v2[lambda: ((num_blocks, 1, 1), (BLOCK_SIZE, 1, 1))](
            x_bf16, weight_bf16, bias_tensor, out_bf16,
            total_elems,
            B, IC, OC,
            ID, IH, IW,
            OD, OH, OW,
            KD, KH, KW,
            stride, padding,
            has_bias,
            _float32_to_bits(self.min_value), _float32_to_bits(self.divisor),
        )

        return out_bf16.to(x.dtype)


def _float32_to_bits(f: float) -> int:
    import struct
    return struct.unpack('<i', struct.pack('<f', float(f)))[0]
