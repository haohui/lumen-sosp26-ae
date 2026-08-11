import torch
import torch.nn as nn
import substrate
import substrate.language as S


BLOCK_SIZE = 256


@substrate.jit
def fused_residual_kernel_bf16(
    y_ptr: S.Pointer(S.bf16),
    bias_ptr: S.Pointer(S.bf16),
    out_ptr: S.Pointer(S.bf16),
    n: S.u32,
    channels: S.u32,
    spatial_size: S.u32,
):
    """
    Fused kernel computing: result = 2*y^2 + (bias + 1)*y
    where bias is broadcast over (channels, 1, 1, 1).

    y shape: (batch, channels, D, H, W)
    bias shape: (channels, 1, 1, 1)
    """
    idx = S.block_id(0) * S.block_dim(0) + S.thread_id(0)

    if idx < n:
        layout_y = S.make_layout((n,), (1,))
        y = S.make_tensor(y_ptr, S.bf16, layout_y)
        out = S.make_tensor(out_ptr, S.bf16, layout_y)

        bias_layout = S.make_layout((channels,), (1,))
        bias = S.make_tensor(bias_ptr, S.bf16, bias_layout)

        y_val = y[idx]

        channel_idx = (idx // spatial_size) % channels

        bias_val = bias[channel_idx]

        y_f32 = S.convert(y_val, S.f32)
        bias_f32 = S.convert(bias_val, S.f32)

        two = S.convert(2.0, S.f32)
        one = S.convert(1.0, S.f32)

        y_sq = y_f32 * y_f32
        two_y_sq = two * y_sq

        bias_plus_one = bias_f32 + one
        bias_times_y = bias_plus_one * y_f32

        result_f32 = two_y_sq + bias_times_y

        result_bf16 = S.convert(result_f32, S.bf16)
        out[idx] = result_bf16


def substrate_fused_residual(y: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Compute: 2*y^2 + (bias + 1)*y with fused kernel.
    y: (batch, channels, D, H, W)
    bias: (channels, 1, 1, 1)
    """
    if not y.is_cuda:
        raise RuntimeError("Input must be on CUDA/HIP device")

    y = y.contiguous()
    bias = bias.contiguous().squeeze()

    batch, channels, d, h, w = y.shape
    spatial_size = d * h * w
    n = y.numel()

    out = torch.empty_like(y)
    y_flat = y.view(-1)
    out_flat = out.view(-1)

    if n > 0:
        grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE, 1, 1)
        fused_residual_kernel_bf16[lambda: (grid, (BLOCK_SIZE, 1, 1))](
            y_flat, bias, out_flat, n, channels, spatial_size
        )

    return out


class ModelNew(nn.Module):
    """
    Optimized model with Substrate DSL kernel for fused elementwise operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv_transpose(x)
        original_x = x
        return substrate_fused_residual(original_x, self.bias)


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]


batch_size = 16
in_channels = 32
out_channels = 64
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1, 1)
