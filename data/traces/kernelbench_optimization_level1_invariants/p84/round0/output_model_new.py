import torch
import torch.nn as nn
import avelang
import avelang.language as al

H_OUT = 254
W_OUT = 510
H_OUT_W_OUT = H_OUT * W_OUT
TOTAL_SPATIAL = 64 * H_OUT_W_OUT
BLOCK_M = 128
KH = 3
KW = 3
CH_PER_WARP = 64
CH_PER_MFMA = 32


def _launch():
    grid_x = TOTAL_SPATIAL // BLOCK_M
    return ((grid_x, 1, 1), (256, 1, 1))


@avelang.jit
def dw_conv_mfma_kernel(
    X: al.Tensor((64, 128, 256, 512), al.bf16),
    W: al.Tensor((128, 1, 3, 3), al.bf16),
    Y: al.Tensor((64, 128, 254, 510), al.bf16),
):
    block_id_m = al.block_id(0)
    spatial_start = block_id_m * BLOCK_M

    tid = al.thread_id(0)
    lane = tid % 64
    warp_id = tid // 64
    warp_row = warp_id // 2
    warp_col = warp_id % 2
    lane_col = lane % 32
    lane_group = lane // 32

    ch_warp = warp_col * CH_PER_WARP

    acc_00 = al.make_local((16,), al.f32)
    acc_01 = al.make_local((16,), al.f32)
    acc_10 = al.make_local((16,), al.f32)
    acc_11 = al.make_local((16,), al.f32)
    for ii in al.range(16):
        acc_00[ii] = al.convert(0.0, al.f32)
        acc_01[ii] = al.convert(0.0, al.f32)
        acc_10[ii] = al.convert(0.0, al.f32)
        acc_11[ii] = al.convert(0.0, al.f32)

    for tm in al.range(2):
        m_base = spatial_start + warp_row * 64 + tm * 32
        for tn in al.range(2):
            ch = ch_warp + tn * CH_PER_MFMA + lane_col

            for r_grp in al.range(4):
                row_base = m_base + r_grp * 8 + lane_group * 4

                for r_off in al.range(4):
                    row = row_base + r_off
                    b_idx = row // H_OUT_W_OUT
                    rem = row % H_OUT_W_OUT
                    ho = rem // W_OUT
                    wo = rem % W_OUT

                    acc_idx = r_grp * 4 + r_off

                    acc_val = al.convert(0.0, al.f32)
                    for ki in al.range(KH):
                        for kj in al.range(KW):
                            in_val = al.convert(X[b_idx, ch, ho + ki, wo + kj], al.f32)
                            w_val = al.convert(W[ch, 0, ki, kj], al.f32)
                            acc_val += in_val * w_val

                    if tm == 0 and tn == 0:
                        acc_00[acc_idx] = acc_val
                    elif tm == 0 and tn == 1:
                        acc_01[acc_idx] = acc_val
                    elif tm == 1 and tn == 0:
                        acc_10[acc_idx] = acc_val
                    else:
                        acc_11[acc_idx] = acc_val

    col_00 = ch_warp + 0 * CH_PER_MFMA + lane_col
    col_01 = ch_warp + 1 * CH_PER_MFMA + lane_col

    for acc_idx in al.range(16):
        row_off = 8 * (acc_idx // 4) + 4 * lane_group + (acc_idx % 4)

        row0 = spatial_start + warp_row * 64 + 0 * 32 + row_off
        b0 = row0 // H_OUT_W_OUT
        rem0 = row0 % H_OUT_W_OUT
        ho0 = rem0 // W_OUT
        wo0 = rem0 % W_OUT
        Y[b0, col_00, ho0, wo0] = al.convert(acc_00[acc_idx], al.bf16)
        Y[b0, col_01, ho0, wo0] = al.convert(acc_01[acc_idx], al.bf16)

        row1 = spatial_start + warp_row * 64 + 1 * 32 + row_off
        b1 = row1 // H_OUT_W_OUT
        rem1 = row1 % H_OUT_W_OUT
        ho1 = rem1 // W_OUT
        wo1 = rem1 % W_OUT
        Y[b1, col_00, ho1, wo1] = al.convert(acc_10[acc_idx], al.bf16)
        Y[b1, col_01, ho1, wo1] = al.convert(acc_11[acc_idx], al.bf16)


class ModelNew(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size), stride=stride, padding=padding, groups=in_channels, bias=bias)

    def forward(self, x):
        orig_dtype = x.dtype
        x_bf16 = x.to(dtype=torch.bfloat16).contiguous()
        w_bf16 = self.conv2d.weight.to(device=x.device, dtype=torch.bfloat16).contiguous()
        y_bf16 = torch.empty((64, 128, 254, 510), device=x.device, dtype=torch.bfloat16)
        dw_conv_mfma_kernel[_launch](x_bf16, w_bf16, y_bf16)
        return y_bf16.to(dtype=orig_dtype)
