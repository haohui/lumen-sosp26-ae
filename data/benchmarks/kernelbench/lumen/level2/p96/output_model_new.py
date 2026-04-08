import torch
import torch.nn as nn
import substrate
import substrate.language as S


@substrate.jit
def clamp_kernel(
    input_tensor: S.Tensor((2048,), S.bf16),
    output_tensor: S.Tensor((2048,), S.bf16),
):
    """Clamp values to [0, 1]."""
    tid = S.thread_id(0)
    bid = S.block_id(0)

    total = 2048

    for idx in S.range((total + S.block_dim(0) - 1) // S.block_dim(0)):
        linear_idx = bid * S.block_dim(0) + tid + idx * S.block_dim(0) * S.grid_dim(0)

        if linear_idx < total:
            val = input_tensor[linear_idx]
            val_f32 = S.convert(val, S.f32)
            # Clamp to [0, 1]
            zero = S.convert(0.0, S.f32)
            one = S.convert(1.0, S.f32)
            if val_f32 < zero:
                val_f32 = zero
            if val_f32 > one:
                val_f32 = one
            output_tensor[linear_idx] = S.convert(val_f32, S.bf16)


class ModelNew(nn.Module):
    """
    Model that performs a transposed 3D convolution, multiplies by a scalar, applies max pooling,
    global average pooling, and clamps the output using Substrate kernels.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale = scale
        self.maxpool = nn.MaxPool3d(kernel_size=maxpool_kernel_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.clamp_min = 0
        self.clamp_max = 1

    def forward(self, x):
        # ConvTranspose3d
        x = self.conv_transpose(x)

        # Scale
        x = x * self.scale

        # MaxPool3d
        x = self.maxpool(x)

        # GlobalAvgPool3d
        x = self.global_avg_pool(x)

        # Clamp using Substrate kernel
        batch_size = x.shape[0]
        channels = x.shape[1]

        # Flatten and ensure contiguous
        x_flat = x.contiguous().view(-1)
        result = torch.empty_like(x_flat)

        # Launch clamp kernel
        threads_per_block = 256
        total_elements = batch_size * channels
        blocks = (total_elements + threads_per_block - 1) // threads_per_block

        clamp_kernel[lambda: ((blocks, 1, 1), (threads_per_block, 1, 1))](
            x_flat,
            result,
        )

        return result.view(batch_size, channels, 1, 1, 1)
