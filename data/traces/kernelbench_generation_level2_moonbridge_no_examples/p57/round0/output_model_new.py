import torch
import torch.nn as nn
import avelang
import avelang.language as al

# Tile dimensions: small tiles for good wavefront occupancy (1 wavefront per block)
TILE_H = 4
TILE_W = 4
TILE_OC = 4
THREADS = TILE_H * TILE_W * TILE_OC  # 64 = 1 wavefront


@avelang.jit
def conv2d_relu_hardswish_kernel(
    input_t: al.Tensor((128, 8, 128, 128), al.bf16),
    weight_t: al.Tensor((64, 8, 3, 3), al.bf16),
    bias_t: al.Tensor((64,), al.bf16),
    output_t: al.Tensor((128, 64, 126, 126), al.bf16),
    oc_groups: al.i32,
):
    tid = al.thread_id(0)
    n = al.block_id(2) // oc_groups
    block_oc = (al.block_id(2) % oc_groups) * TILE_OC
    block_oh = al.block_id(0) * TILE_H
    block_ow = al.block_id(1) * TILE_W

    toc = tid % TILE_OC
    th = (tid // TILE_OC) % TILE_H
    tw = tid // (TILE_OC * TILE_H)

    oh = block_oh + th
    ow = block_ow + tw
    oc = block_oc + toc

    if oh < 126 and ow < 126 and oc < 64:
        # Accumulate convolution in f32, starting with bias
        acc = al.convert(bias_t[oc], al.f32)

        for c in al.range(8):
            for kh in al.range(3):
                for kw in al.range(3):
                    in_val = al.convert(input_t[n, c, oh + kh, ow + kw], al.f32)
                    w_val = al.convert(weight_t[oc, c, kh, kw], al.f32)
                    acc = acc + in_val * w_val

        # ReLU: max(0, x) = (x + |x|) / 2
        half = al.convert(0.5, al.f32)
        relu_val = (acc + al.abs(acc)) * half

        # HardSwish: x * clamp((x + 3) / 6, 0, 1)
        three = al.convert(3.0, al.f32)
        six = al.convert(6.0, al.f32)
        one_f32 = al.convert(1.0, al.f32)

        y = (relu_val + three) / six
        clamped = (y + one_f32 - al.abs(y - one_f32)) * half
        result = relu_val * clamped

        output_t[n, oc, oh, ow] = al.convert(result, al.bf16)


def avelang_conv2d_relu_hardswish(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    N, C, H, W = x.shape
    OC, C2, KH, KW = weight.shape
    OH = H - KH + 1
    OW = W - KW + 1

    out = torch.empty(N, OC, OH, OW, dtype=torch.bfloat16, device=x.device)

    oc_groups = (OC + TILE_OC - 1) // TILE_OC
    grid_h = (OH + TILE_H - 1) // TILE_H
    grid_w = (OW + TILE_W - 1) // TILE_W
    grid_z = N * oc_groups

    conv2d_relu_hardswish_kernel[lambda: ((grid_h, grid_w, grid_z), (THREADS, 1, 1))](
        x, weight, bias, out, oc_groups,
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        weight = self.conv.weight.data
        bias = self.conv.bias.data

        x_bf16 = x.to(torch.bfloat16).contiguous()
        w_bf16 = weight.to(torch.bfloat16).contiguous()
        b_bf16 = bias.to(torch.bfloat16).contiguous()

        return avelang_conv2d_relu_hardswish(x_bf16, w_bf16, b_bf16)
