import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def hardswish_kernel(
    input_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C: al.i32,
    D: al.i32,
    H: al.i32,
    W: al.i32,
):
    n = al.block_id(0)
    c = al.block_id(1)
    spatial_block = al.block_id(2)
    tid = al.thread_id(0)

    spatial_idx = spatial_block * 256 + tid
    total_spatial = D * H * W

    if spatial_idx < total_spatial:
        hw_total = H * W
        d = spatial_idx // hw_total
        hw = spatial_idx - d * hw_total
        h = hw // W
        w = hw - h * W

        stride_c = D * H * W
        stride_d = H * W
        stride_h = W

        flat_idx = n * (C * D * H * W) + c * stride_c + d * stride_d + h * stride_h + w

        inp = al.make_tensor(input_ptr, al.bf16, al.make_layout((N * C * D * H * W,), (1,)))
        out = al.make_tensor(output_ptr, al.bf16, al.make_layout((N * C * D * H * W,), (1,)))

        x = al.convert(inp[flat_idx], al.f32)
        three = al.convert(3.0, al.f32)
        zero = al.convert(0.0, al.f32)
        six = al.convert(6.0, al.f32)
        inv_six = al.convert(0.16666667163372039794921875, al.f32)

        x_plus_3 = x + three
        clamped = x_plus_3
        if x_plus_3 < zero:
            clamped = zero
        if x_plus_3 > six:
            clamped = six

        result = x * clamped * inv_six
        out[flat_idx] = al.convert(result, al.bf16)


def _avelang_hardswish(x: torch.Tensor) -> torch.Tensor:
    N, C, D, H, W = x.shape
    out = torch.empty_like(x)
    total_spatial = D * H * W
    num_blocks = (total_spatial + 255) // 256
    hardswish_kernel[lambda: ((N, C, num_blocks), (256, 1, 1))](
        x.data_ptr(), out.data_ptr(), N, C, D, H, W)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, eps, bias=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = torch.sigmoid(x) * x
        x = self.group_norm(x)
        x = _avelang_hardswish(x)
        return x
