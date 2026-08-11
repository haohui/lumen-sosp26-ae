import torch
import torch.nn as nn
import substrate
import substrate.language as S

# Model configuration
batch_size = 16
in_channels = 32
out_channels = 64
D, H, W = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1

# Output dimensions after ConvTranspose3d
# D_out = (D - 1) * stride - 2 * padding + kernel_size + output_padding
D_out = (D - 1) * stride - 2 * padding + kernel_size + output_padding
H_out = (H - 1) * stride - 2 * padding + kernel_size + output_padding
W_out = (W - 1) * stride - 2 * padding + kernel_size + output_padding

BLOCK_SIZE = 256
LOG2E = 1.4426950408889634  # 1 / ln(2)

# Fixed dimensions for the kernel
TOTAL_SPATIAL = batch_size * D_out * H_out * W_out
TOTAL_ELEMENTS = batch_size * out_channels * D_out * H_out * W_out


@substrate.jit
def softmax_sigmoid_bf16_kernel(
    x: S.Tensor((TOTAL_ELEMENTS,), S.bf16),
    y: S.Tensor((TOTAL_ELEMENTS,), S.bf16),
):
    """
    Fused softmax + sigmoid kernel.
    Each thread handles one spatial position across all channels.
    Shape: (N, C, D, H, W) flattened to (N*C*D*H*W,)
    """
    idx = S.block_id(0) * S.block_dim(0) + S.thread_id(0)

    if idx < TOTAL_SPATIAL:
        # Compute spatial position indices
        dhw = D_out * H_out * W_out
        hw = H_out * W_out

        n_idx = idx // dhw
        remainder = idx - n_idx * dhw
        d_idx = remainder // hw
        remainder = remainder - d_idx * hw
        h_idx = remainder // W_out
        w_idx = remainder - h_idx * W_out

        # Base offset for this spatial position (at channel 0)
        channel_stride = D_out * H_out * W_out
        base_offset = n_idx * (out_channels * channel_stride) + d_idx * hw + h_idx * W_out + w_idx

        # Find max across channels for numerical stability
        max_val = S.convert(-1e38, S.f32)
        for c in S.range(out_channels):
            offset = base_offset + c * channel_stride
            v = S.convert(x[offset], S.f32)
            if v > max_val:
                max_val = v

        # Compute exp(x - max) and sum
        exp_sum = S.convert(0.0, S.f32)
        log2e = S.convert(LOG2E, S.f32)

        # First pass: compute exp sum
        for c in S.range(out_channels):
            offset = base_offset + c * channel_stride
            v = S.convert(x[offset], S.f32)
            exp_val = S.exp2((v - max_val) * log2e)
            exp_sum = exp_sum + exp_val

        # Compute reciprocal of sum
        one_f32 = S.convert(1.0, S.f32)
        zero_f32 = S.convert(0.0, S.f32)
        inv_sum = one_f32 / exp_sum

        # Second pass: compute softmax and apply sigmoid
        for c in S.range(out_channels):
            offset = base_offset + c * channel_stride
            v = S.convert(x[offset], S.f32)
            softmax_val = S.exp2((v - max_val) * log2e) * inv_sum

            # Apply sigmoid: sigmoid(x) = 1 / (1 + exp(-x))
            neg_softmax = zero_f32 - softmax_val
            exp_neg = S.exp2(neg_softmax * log2e)
            sigmoid_val = one_f32 / (one_f32 + exp_neg)

            y[offset] = S.convert(sigmoid_val, S.bf16)


@substrate.jit
def softmax_sigmoid_f32_kernel(
    x: S.Tensor((TOTAL_ELEMENTS,), S.f32),
    y: S.Tensor((TOTAL_ELEMENTS,), S.f32),
):
    """
    Fused softmax + sigmoid kernel for float32.
    """
    idx = S.block_id(0) * S.block_dim(0) + S.thread_id(0)

    if idx < TOTAL_SPATIAL:
        dhw = D_out * H_out * W_out
        hw = H_out * W_out

        n_idx = idx // dhw
        remainder = idx - n_idx * dhw
        d_idx = remainder // hw
        remainder = remainder - d_idx * hw
        h_idx = remainder // W_out
        w_idx = remainder - h_idx * W_out

        channel_stride = D_out * H_out * W_out
        base_offset = n_idx * (out_channels * channel_stride) + d_idx * hw + h_idx * W_out + w_idx

        # Find max across channels
        max_val = S.convert(-1e38, S.f32)
        for c in S.range(out_channels):
            offset = base_offset + c * channel_stride
            v = x[offset]
            if v > max_val:
                max_val = v

        # Compute exp sum
        exp_sum = S.convert(0.0, S.f32)
        log2e = S.convert(LOG2E, S.f32)

        for c in S.range(out_channels):
            offset = base_offset + c * channel_stride
            v = x[offset]
            exp_val = S.exp2((v - max_val) * log2e)
            exp_sum = exp_sum + exp_val

        one_f32 = S.convert(1.0, S.f32)
        zero_f32 = S.convert(0.0, S.f32)
        inv_sum = one_f32 / exp_sum

        # Compute softmax + sigmoid
        for c in S.range(out_channels):
            offset = base_offset + c * channel_stride
            v = x[offset]
            softmax_val = S.exp2((v - max_val) * log2e) * inv_sum

            neg_softmax = zero_f32 - softmax_val
            exp_neg = S.exp2(neg_softmax * log2e)
            sigmoid_val = one_f32 / (one_f32 + exp_neg)

            y[offset] = sigmoid_val


def substrate_softmax_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    Apply softmax on channel dimension (dim=1) followed by sigmoid.
    """
    if x.dim() != 5:
        raise ValueError(f"Expected 5D tensor, got {x.dim()}D")

    n, c, d, h, w = x.shape

    original_device = x.device
    moved = False
    if not x.is_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/HIP device required for Substrate kernels")
        x = x.cuda()
        moved = True

    x_contig = x.contiguous()
    y = torch.empty_like(x_contig)

    n_spatial = n * d * h * w
    grid = ((n_spatial + BLOCK_SIZE - 1) // BLOCK_SIZE, 1, 1)

    if x_contig.dtype == torch.bfloat16:
        x_flat = x_contig.view(-1)
        y_flat = y.view(-1)
        softmax_sigmoid_bf16_kernel[lambda: (grid, (BLOCK_SIZE, 1, 1))](x_flat, y_flat)
    elif x_contig.dtype == torch.float32:
        x_flat = x_contig.view(-1)
        y_flat = y.view(-1)
        softmax_sigmoid_f32_kernel[lambda: (grid, (BLOCK_SIZE, 1, 1))](x_flat, y_flat)
    else:
        raise TypeError(f"Unsupported dtype: {x_contig.dtype}")

    if moved:
        return y.to(original_device)
    return y


class ModelNew(nn.Module):
    """
    Optimized model with fused softmax + sigmoid using Substrate GPU kernels.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_transpose(x)
        x = substrate_softmax_sigmoid(x)
        return x


def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding]
