import torch
import torch.nn as nn
import substrate
import substrate.language as S


THREADS = 256

# Fixed problem shape from target model/get_inputs.
BATCH_SIZE = 128
IN_CHANNELS = 32
OUT_CHANNELS = 64
IN_D, IN_H, IN_W = 16, 16, 16
OUT_D, OUT_H, OUT_W = 32, 32, 32
KERNEL_SIZE = 3
STRIDE = 2
PADDING = 1
OUTPUT_PADDING = 1

OUT_ELEMS = BATCH_SIZE * OUT_CHANNELS * OUT_D * OUT_H * OUT_W  # 8912896


@substrate.jit
def fused_add_hardswish_bf16_kernel(
    conv_out_ptr: S.Pointer(S.bf16),
    add_input_ptr: S.Pointer(S.bf16),
    out_ptr: S.Pointer(S.bf16),
    n: S.u32,
):
    """
    Fused kernel: out = (conv_out + add_input) * hardswish(conv_out + add_input)
    where hardswish(x) = x * relu6(x + 3) / 6
    """
    idx = S.block_id(0) * S.block_dim(0) + S.thread_id(0)

    layout = S.make_layout((n,), (1,))
    conv_out = S.make_tensor(conv_out_ptr, S.bf16, layout)
    add_input = S.make_tensor(add_input_ptr, S.bf16, layout)
    out = S.make_tensor(out_ptr, S.bf16, layout)

    # Constants
    three = S.convert(3.0, S.f32)
    six = S.convert(6.0, S.f32)
    zero = S.convert(0.0, S.f32)
    sixth = S.convert(0.16666667, S.f32)

    if idx < n:
        # Load values and convert to f32 for computation
        v_conv = S.convert(conv_out[idx], S.f32)
        v_add = S.convert(add_input[idx], S.f32)

        # Sum
        x = v_conv + v_add

        # HardSwish: hardswish(x) = x * relu6(x + 3) / 6
        # relu6(x + 3) = min(max(x + 3, 0), 6)
        x_plus_3 = x + three
        relu6_val = x_plus_3
        if relu6_val < zero:
            relu6_val = zero
        if relu6_val > six:
            relu6_val = six

        # hardswish(x) = x * relu6(x + 3) / 6
        hardswish_val = x * relu6_val * sixth

        # Final output: x * hardswish(x)
        result = x * hardswish_val

        out[idx] = S.convert(result, S.bf16)


@substrate.jit
def fused_add_hardswish_f32_kernel(
    conv_out_ptr: S.Pointer(S.f32),
    add_input_ptr: S.Pointer(S.f32),
    out_ptr: S.Pointer(S.f32),
    n: S.u32,
):
    """
    Fused kernel: out = (conv_out + add_input) * hardswish(conv_out + add_input)
    where hardswish(x) = x * relu6(x + 3) / 6
    """
    idx = S.block_id(0) * S.block_dim(0) + S.thread_id(0)

    layout = S.make_layout((n,), (1,))
    conv_out = S.make_tensor(conv_out_ptr, S.f32, layout)
    add_input = S.make_tensor(add_input_ptr, S.f32, layout)
    out = S.make_tensor(out_ptr, S.f32, layout)

    # Constants
    three = S.convert(3.0, S.f32)
    six = S.convert(6.0, S.f32)
    zero = S.convert(0.0, S.f32)
    sixth = S.convert(0.16666667, S.f32)

    if idx < n:
        # Sum
        x = conv_out[idx] + add_input[idx]

        # relu6(x + 3)
        x_plus_3 = x + three
        relu6_val = x_plus_3
        if relu6_val < zero:
            relu6_val = zero
        if relu6_val > six:
            relu6_val = six

        # hardswish(x) = x * relu6(x + 3) / 6
        hardswish_val = x * relu6_val * sixth

        # Final output: x * hardswish(x)
        result = x * hardswish_val

        out[idx] = result


def _launch_fused_bf16(conv_out: torch.Tensor, add_input: torch.Tensor) -> torch.Tensor:
    """Launch fused add + hardswish kernel for BF16."""
    out = torch.empty_like(conv_out)
    n = conv_out.numel()

    if n > 0:
        conv_flat = conv_out.view(-1)
        add_flat = add_input.view(-1)
        out_flat = out.view(-1)
        grid = ((n + THREADS - 1) // THREADS, 1, 1)
        fused_add_hardswish_bf16_kernel[lambda: (grid, (THREADS, 1, 1))](
            conv_flat, add_flat, out_flat, n
        )

    return out


def _launch_fused_f32(conv_out: torch.Tensor, add_input: torch.Tensor) -> torch.Tensor:
    """Launch fused add + hardswish kernel for F32."""
    out = torch.empty_like(conv_out)
    n = conv_out.numel()

    if n > 0:
        conv_flat = conv_out.view(-1)
        add_flat = add_input.view(-1)
        out_flat = out.view(-1)
        grid = ((n + THREADS - 1) // THREADS, 1, 1)
        fused_add_hardswish_f32_kernel[lambda: (grid, (THREADS, 1, 1))](
            conv_flat, add_flat, out_flat, n
        )

    return out


def substrate_fused_add_hardswish(conv_out: torch.Tensor, add_input: torch.Tensor) -> torch.Tensor:
    """
    Fused operation: out = (conv_out + add_input) * hardswish(conv_out + add_input)
    """
    if conv_out.shape != add_input.shape:
        raise ValueError(f"Shape mismatch: conv_out {conv_out.shape}, add_input {add_input.shape}")

    original_device = conv_out.device
    moved = False
    if not conv_out.is_cuda:
        conv_out = conv_out.cuda()
        add_input = add_input.cuda()
        moved = True

    conv_contig = conv_out.contiguous()
    add_contig = add_input.contiguous()

    if conv_contig.dtype == torch.bfloat16:
        result = _launch_fused_bf16(conv_contig, add_contig)
    elif conv_contig.dtype == torch.float32:
        result = _launch_fused_f32(conv_contig, add_contig)
    else:
        raise TypeError(f"Unsupported dtype for fused kernel: {conv_contig.dtype}")

    if moved:
        result = result.to(original_device)

    return result


class ModelNew(nn.Module):
    """
    Optimized model that performs a 3D transposed convolution, adds an input tensor,
    and applies HardSwish activation using Substrate GPU kernels.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x: torch.Tensor, add_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).
            add_input (torch.Tensor): Input tensor to be added after transposed convolution.
        Returns:
            torch.Tensor: Output tensor after HardSwish activation.
        """
        # Transposed convolution
        x = self.conv_transpose(x)

        # Fused add + hardswish using Substrate kernel
        x = substrate_fused_add_hardswish(x, add_input)

        return x


batch_size = 128
in_channels = 32
out_channels = 64
D, H, W = 16, 16, 16
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1, 1, 1)


def get_inputs():
    return [
        torch.rand(batch_size, in_channels, D, H, W),
        torch.rand(batch_size, out_channels, D * stride, H * stride, W * stride)
    ]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]
