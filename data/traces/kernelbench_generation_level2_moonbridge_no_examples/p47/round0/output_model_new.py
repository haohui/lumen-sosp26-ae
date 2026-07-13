import torch
import torch.nn as nn
import avelang
import avelang.language as al

_cN = al.constexpr(16)
_cIC = al.constexpr(32)
_cOC = al.constexpr(64)
_cID = al.constexpr(32)
_cIH = al.constexpr(64)
_cIW = al.constexpr(64)
_cOD = al.constexpr(30)
_cOH = al.constexpr(62)
_cOW = al.constexpr(62)
_cKD = al.constexpr(3)
_cKH = al.constexpr(3)
_cKW = al.constexpr(3)


@avelang.jit
def conv3d_mish_tanh_kernel(
    input_t: al.Tensor((_cN, _cIC, _cID, _cIH, _cIW), al.f32),
    weight_t: al.Tensor((_cOC, _cIC, _cKD, _cKH, _cKW), al.f32),
    output_t: al.Tensor((_cN, _cOC, _cOD, _cOH, _cOW), al.bf16),
):
    bid = al.block_id(0)
    bdx = al.block_dim(0)
    tid = al.thread_id(0)
    gid = bid * bdx + tid

    if gid >= 118087680:
        return

    n = gid // 7380480
    r0 = gid % 7380480
    oc = r0 // 115320
    r1 = r0 % 115320
    od = r1 // 3844
    r2 = r1 % 3844
    oh = r2 // 62
    ow = r2 % 62

    acc = al.convert(0.0, al.f32)

    for ic in al.range(32):
        for kd in al.range(3):
            d_pos = od + kd
            for kh in al.range(3):
                h_pos = oh + kh
                for kw in al.range(3):
                    w_pos = ow + kw
                    ival = input_t[n, ic, d_pos, h_pos, w_pos]
                    wval = weight_t[oc, ic, kd, kh, kw]
                    acc = acc + ival * wval

    one = al.convert(1.0, al.f32)
    exp_acc = al.exp(acc)
    softplus = al.log(one + exp_acc)
    mish_val = acc * al.tanh(softplus)
    result = al.tanh(mish_val)
    output_t[n, oc, od, oh, ow] = al.convert(result, al.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        x = x.contiguous().float()
        weight = self.conv.weight.detach().contiguous().float()
        out = torch.empty(16, 64, 30, 62, 62, dtype=torch.bfloat16, device=x.device)
        BLOCK = 256
        grid = (118087680 + BLOCK - 1) // BLOCK
        conv3d_mish_tanh_kernel[lambda: ((grid, 1, 1), (BLOCK, 1, 1))](x, weight, out)
        return out
