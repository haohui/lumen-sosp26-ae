import torch
import torch.nn as nn
import avelang
import avelang.language as al

TILE_H: al.constexpr = 16
BLOCK_SIZE: al.constexpr = 256


@avelang.jit
def depthwise_conv_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    batch_size: al.i32,
    in_channels: al.i32,
    height: al.i32,
    width: al.i32,
    kernel_size: al.i32,
    stride: al.i32,
    dilation: al.i32,
    padding: al.i32,
    height_out: al.i32,
    width_out: al.i32,
    has_bias: al.i32,
    num_h_tiles: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    bc_idx = bid // num_h_tiles
    h_tile = bid - bc_idx * num_h_tiles

    batch_idx = bc_idx // in_channels
    channel_idx = bc_idx - batch_idx * in_channels

    h_start = h_tile * TILE_H

    x_layout = al.make_layout(
        (batch_size, in_channels, height, width),
        (in_channels * height * width, height * width, width, 1),
    )
    x = al.make_tensor(x_ptr, al.bf16, x_layout)

    w_layout = al.make_layout(
        (in_channels, 1, kernel_size, 1),
        (kernel_size, kernel_size, 1, 1),
    )
    w = al.make_tensor(w_ptr, al.bf16, w_layout)

    out_layout = al.make_layout(
        (batch_size, in_channels, height_out, width_out),
        (in_channels * height_out * width_out, height_out * width_out, width_out, 1),
    )
    out = al.make_tensor(out_ptr, al.bf16, out_layout)

    b_layout = al.make_layout((in_channels,), (1,))
    bias = al.make_tensor(b_ptr, al.bf16, b_layout)

    for h_off in al.range(TILE_H):
        h_out = h_start + h_off
        if h_out < height_out:
            for w_out in al.range(tid, width_out, BLOCK_SIZE):
                accum = al.convert(0.0, al.f32)
                for kh in al.range(kernel_size):
                    h_in = h_out * stride + kh * dilation - padding
                    w_in = w_out * stride - padding
                    w_val = al.convert(w[channel_idx, 0, kh, 0], al.f32)
                    x_val = al.convert(x[batch_idx, channel_idx, h_in, w_in], al.f32)
                    prod = w_val * x_val
                    accum = accum + prod

                if has_bias != 0:
                    b_val = al.convert(bias[channel_idx], al.f32)
                    accum = accum + b_val

                out[batch_idx, channel_idx, h_out, w_out] = al.convert(accum, al.bf16)


def _ensure_bf16_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    return t.contiguous().to(torch.bfloat16)


def avelang_depthwise_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: int,
    padding: int,
    dilation: int,
) -> torch.Tensor:
    assert x.is_cuda, "Input must be on CUDA/HIP device."

    batch_size, in_channels, height, width = x.shape
    kernel_size = weight.shape[2]

    height_out = (height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    width_out = (width + 2 * padding - dilation * (1 - 1) - 1) // stride + 1

    x_in = _ensure_bf16_contiguous(x)
    w_in = _ensure_bf16_contiguous(weight)
    out = torch.empty(
        (batch_size, in_channels, height_out, width_out),
        dtype=torch.bfloat16,
        device=x.device,
    )

    has_bias = 1 if bias is not None else 0
    b_tensor = _ensure_bf16_contiguous(bias) if bias is not None else torch.zeros(
        (1,), dtype=torch.bfloat16, device=x.device
    )

    num_h_tiles = (height_out + TILE_H - 1) // TILE_H
    num_blocks = batch_size * in_channels * num_h_tiles

    depthwise_conv_kernel[lambda: ((num_blocks, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_in, w_in, b_tensor, out,
        batch_size, in_channels, height, width,
        kernel_size, stride, dilation, padding,
        height_out, width_out, has_bias, num_h_tiles,
    )

    return out


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.has_bias = bias

        self.weight = nn.Parameter(torch.empty(in_channels, 1, kernel_size, 1))
        if bias:
            self.bias = nn.Parameter(torch.empty(in_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            fan_in = self.in_channels * self.kernel_size
            bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        result = avelang_depthwise_conv(
            x, self.weight, self.bias,
            self.stride, self.padding, self.dilation,
        )
        return result.to(orig_dtype)


# Test code
batch_size = 64
in_channels = 8
kernel_size = 3
width = 512
height = 512
stride = 1
padding = 0
dilation = 1


def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, kernel_size, stride, padding, dilation]
