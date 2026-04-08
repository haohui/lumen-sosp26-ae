import torch
import torch.nn as nn
import substrate
import substrate.language as S


# Fixed problem shape from target model/get_inputs.
BATCH_SIZE = 64
IN_CHANNELS = 64
OUT_CHANNELS = 64
IN_H = 256
IN_W = 256
K_H = 3
K_W = 3
STRIDE_H = 1
STRIDE_W = 1
PAD_H = 0
PAD_W = 0

OUT_H = (IN_H + 2 * PAD_H - K_H) // STRIDE_H + 1  # 254
OUT_W = (IN_W + 2 * PAD_W - K_W) // STRIDE_W + 1  # 254

# LeakyReLU negative slope
LEAKY_RELU_NEGATIVE_SLOPE = 0.01

# GELU constants
SQRT_2_RCP = 0.7071067811865475  # 1 / sqrt(2)


@substrate.jit
def fused_post_conv_bf16_kernel(
    x_ptr: S.Pointer(S.bf16),
    mult_ptr: S.Pointer(S.bf16),
    y_ptr: S.Pointer(S.bf16),
    n: S.u32,
    out_channels: S.u32,
    out_h: S.u32,
    out_w: S.u32,
):
    """Fused kernel that applies per-channel multiplier, LeakyReLU, and GELU."""
    idx = S.block_id(0) * S.block_dim(0) + S.thread_id(0)

    if idx < n:
        # Layout for 1D flat access (NCHW layout)
        layout = S.make_layout((n,), (1,))
        x = S.make_tensor(x_ptr, S.bf16, layout)
        y = S.make_tensor(y_ptr, S.bf16, layout)

        # Compute channel index for this element (NCHW format)
        spatial_per_channel = out_h * out_w
        spatial_per_batch = out_channels * spatial_per_channel
        batch_idx = idx // spatial_per_batch
        remaining = idx % spatial_per_batch
        channel_idx = remaining // spatial_per_channel
        spatial_idx = remaining % spatial_per_channel

        # Load multiplier for this channel
        mult_layout = S.make_layout((out_channels,), (1,))
        mult = S.make_tensor(mult_ptr, S.bf16, mult_layout)

        # Load input value
        xv = S.convert(x[idx], S.f32)
        mv = S.convert(mult[channel_idx], S.f32)

        # Step 1: Multiply by per-channel multiplier
        xv = xv * mv

        # Step 2: LeakyReLU
        zero = S.convert(0.0, S.f32)
        neg_slope = S.convert(LEAKY_RELU_NEGATIVE_SLOPE, S.f32)

        if xv < zero:
            xv = xv * neg_slope

        # Step 3: GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
        sqrt2_rcp = S.convert(SQRT_2_RCP, S.f32)
        half = S.convert(0.5, S.f32)
        one = S.convert(1.0, S.f32)

        erf_arg = xv * sqrt2_rcp
        erf_val = S.erf(erf_arg)
        gelu_out = xv * half * (one + erf_val)

        y[idx] = S.convert(gelu_out, S.bf16)


def _launch_fused_post_conv(conv_out: torch.Tensor, multiplier: torch.Tensor) -> torch.Tensor:
    """Apply per-channel multiplier, LeakyReLU, and GELU using Substrate kernel."""
    n = conv_out.numel()
    y = torch.empty_like(conv_out)

    block_size = 256
    grid = ((n + block_size - 1) // block_size, 1, 1)

    fused_post_conv_bf16_kernel[lambda: (grid, (block_size, 1, 1))](
        conv_out, multiplier, y, n, OUT_CHANNELS, OUT_H, OUT_W
    )

    return y


class ModelNew(nn.Module):
    """
    Model that performs convolution, per-channel multiplication, LeakyReLU, and GELU
    using optimized Substrate DSL kernel for fused post-processing operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super(ModelNew, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))

    def forward(self, x):
        if x.shape != (BATCH_SIZE, IN_CHANNELS, IN_H, IN_W):
            raise NotImplementedError(
                f"ModelNew currently supports input shape {(BATCH_SIZE, IN_CHANNELS, IN_H, IN_W)}, got {tuple(x.shape)}"
            )

        original_device = x.device
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/HIP device is required for Substrate kernels.")

        if not x.is_cuda:
            x = x.cuda()

        # Move parameters to device
        self.conv = self.conv.to(device=x.device)
        self.multiplier = self.multiplier.to(device=x.device)

        # Ensure contiguous tensors
        x = x.contiguous()

        # Convert to BF16
        x = x.to(torch.bfloat16)
        self.conv = self.conv.to(torch.bfloat16)
        mult = self.multiplier.to(torch.bfloat16)

        # Step 1: Convolution
        conv_out = self.conv(x)

        # Step 2-4: Fused post-conv operations using Substrate kernel
        out = _launch_fused_post_conv(conv_out, mult)

        if original_device.type != "cuda":
            out = out.to(original_device)
        return out


batch_size = 64
in_channels = 64
out_channels = 64
height, width = 256, 256
kernel_size = 3
multiplier_shape = (out_channels, 1, 1)


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, multiplier_shape]
