import torch
import torch.nn as nn
import substrate
import substrate.language as S

# Constants for log2(e) and its inverse
LOG2_E = 1.4426950408889634  # log2(e)
INV_LOG2_E = 0.6931471805599453  # 1/log2(e) = ln(2)

# ============================================================================
# ReLU Kernel (elementwise)
# ============================================================================

@substrate.jit
def relu_kernel(
    input: S.Tensor((4 * 32 * 32 * 128 * 128,), S.bf16),
    output: S.Tensor((4 * 32 * 32 * 128 * 128,), S.bf16),
    n: S.u32,
):
    tid = S.thread_id(0)
    bid = S.block_id(0)
    block_size = S.block_dim(0)
    idx = bid * block_size + tid

    if idx < n:
        val = input[idx]
        zero = S.convert(0.0, S.bf16)
        if val < zero:
            output[idx] = zero
        else:
            output[idx] = val


def substrate_relu(x: torch.Tensor) -> torch.Tensor:
    """Apply ReLU using Substrate kernel."""
    assert x.is_cuda, "Tensor must be on CUDA/HIP device."
    assert x.dtype == torch.bfloat16, "Tensor must be bfloat16."

    out = torch.empty_like(x)
    n = x.numel()
    block_size = 256
    grid_size = (n + block_size - 1) // block_size

    relu_kernel[lambda: ((grid_size, 1, 1), (block_size, 1, 1))](x.view(-1), out.view(-1), n)
    return out


# ============================================================================
# MaxPool3D Kernel (2x2x2 pooling with stride 2)
# ============================================================================

IN_SIZE = 4 * 64 * 32 * 128 * 128
OUT_SIZE = 4 * 64 * 16 * 64 * 64

@substrate.jit
def maxpool3d_kernel(
    input: S.Tensor((IN_SIZE,), S.bf16),
    output: S.Tensor((OUT_SIZE,), S.bf16),
    batch_size: S.u32,
    channels: S.u32,
    in_depth: S.u32,
    in_height: S.u32,
    in_width: S.u32,
    out_depth: S.u32,
    out_height: S.u32,
    out_width: S.u32,
):
    """MaxPool3D with kernel_size=2, stride=2."""
    tid = S.thread_id(0)
    bid = S.block_id(0)
    block_size = S.block_dim(0)

    total_out = batch_size * channels * out_depth * out_height * out_width

    idx = bid * block_size + tid

    if idx < total_out:
        out_w = idx % out_width
        tmp = idx // out_width
        out_h = tmp % out_height
        tmp = tmp // out_height
        out_d = tmp % out_depth
        tmp = tmp // out_depth
        c = tmp % channels
        b = tmp // channels

        in_d = out_d * 2
        in_h = out_h * 2
        in_w = out_w * 2

        in_idx = (b * channels * in_depth * in_height * in_width +
                  c * in_depth * in_height * in_width +
                  in_d * in_height * in_width +
                  in_h * in_width +
                  in_w)

        v0 = input[in_idx]
        v1 = input[in_idx + 1]
        v2 = input[in_idx + in_width]
        v3 = input[in_idx + in_width + 1]
        v4 = input[in_idx + in_height * in_width]
        v5 = input[in_idx + in_height * in_width + 1]
        v6 = input[in_idx + in_height * in_width + in_width]
        v7 = input[in_idx + in_height * in_width + in_width + 1]

        m01 = v0 if v0 > v1 else v1
        m23 = v2 if v2 > v3 else v3
        m45 = v4 if v4 > v5 else v5
        m67 = v6 if v6 > v7 else v7
        m0123 = m01 if m01 > m23 else m23
        m4567 = m45 if m45 > m67 else m67
        m = m0123 if m0123 > m4567 else m4567

        output[idx] = m


def substrate_maxpool3d(x: torch.Tensor) -> torch.Tensor:
    """Apply MaxPool3D with kernel_size=2, stride=2 using Substrate kernel."""
    assert x.is_cuda, "Tensor must be on CUDA/HIP device."
    assert x.dtype == torch.bfloat16, "Tensor must be bfloat16."
    assert x.ndim == 5, "Input must be 5D (batch, channel, depth, height, width)"

    batch_size, channels, in_depth, in_height, in_width = x.shape

    out_depth = in_depth // 2
    out_height = in_height // 2
    out_width = in_width // 2

    out = torch.empty((batch_size, channels, out_depth, out_height, out_width),
                      dtype=torch.bfloat16, device=x.device)

    total_out = out.numel()
    block_size = 256
    grid_size = (total_out + block_size - 1) // block_size

    maxpool3d_kernel[lambda: ((grid_size, 1, 1), (block_size, 1, 1))](
        x.view(-1), out.view(-1), batch_size, channels,
        in_depth, in_height, in_width,
        out_depth, out_height, out_width
    )
    return out


# ============================================================================
# LogSumExp Kernel (reduce over channel dimension)
# ============================================================================

LSE_IN_SIZE = 4 * 64 * 16 * 64 * 64
LSE_OUT_SIZE = 4 * 16 * 64 * 64

@substrate.jit
def logsumexp_kernel(
    input: S.Tensor((LSE_IN_SIZE,), S.bf16),
    output: S.Tensor((LSE_OUT_SIZE,), S.bf16),
    batch_size: S.u32,
    channels: S.u32,
    depth: S.u32,
    height: S.u32,
    width: S.u32,
):
    """Compute logsumexp over channel dimension."""
    tid = S.thread_id(0)
    bid = S.block_id(0)
    block_size = S.block_dim(0)

    total_out = batch_size * depth * height * width
    idx = bid * block_size + tid

    if idx < total_out:
        w_coord = idx % width
        tmp = idx // width
        h_coord = tmp % height
        tmp = tmp // height
        d_coord = tmp % depth
        b_coord = tmp // depth

        base_idx = (b_coord * channels * depth * height * width +
                    d_coord * height * width +
                    h_coord * width +
                    w_coord)

        channel_stride = depth * height * width
        max_val_f32 = S.convert(input[base_idx], S.f32)

        for c_iter in S.range(1, channels):
            val_f32 = S.convert(input[base_idx + c_iter * channel_stride], S.f32)
            max_val_f32 = val_f32 if val_f32 > max_val_f32 else max_val_f32

        sum_exp = S.convert(0.0, S.f32)
        log2_e = S.convert(LOG2_E, S.f32)

        for c_iter in S.range(channels):
            val_f32 = S.convert(input[base_idx + c_iter * channel_stride], S.f32)
            diff = val_f32 - max_val_f32
            exp_val = S.exp2(diff * log2_e)
            sum_exp = sum_exp + exp_val

        inv_log2_e = S.convert(INV_LOG2_E, S.f32)
        log_sum = S.log2(sum_exp) * inv_log2_e
        result = max_val_f32 + log_sum

        output[idx] = S.convert(result, S.bf16)


def substrate_logsumexp(x: torch.Tensor, dim: int = 1, keepdim: bool = True) -> torch.Tensor:
    """Compute logsumexp over specified dimension using Substrate kernel."""
    assert x.is_cuda, "Tensor must be on CUDA/HIP device."
    assert x.dtype == torch.bfloat16, "Tensor must be bfloat16."
    assert x.ndim == 5, "Input must be 5D"
    assert dim == 1, "Only reduction over channel dimension (dim=1) is supported"

    batch_size, channels, depth, height, width = x.shape

    if keepdim:
        out_shape = (batch_size, 1, depth, height, width)
    else:
        out_shape = (batch_size, depth, height, width)

    out = torch.empty(out_shape, dtype=torch.bfloat16, device=x.device)

    total_out = batch_size * depth * height * width
    block_size = 256
    grid_size = (total_out + block_size - 1) // block_size

    logsumexp_kernel[lambda: ((grid_size, 1, 1), (block_size, 1, 1))](
        x.view(-1), out.view(-1), batch_size, channels, depth, height, width
    )
    return out


# ============================================================================
# Model
# ============================================================================

class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, max pooling, log sum exp, and ReLU activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = x.to(torch.bfloat16)
        x = self.conv(x)
        x = substrate_maxpool3d(x)
        x = substrate_logsumexp(x, dim=1, keepdim=True)
        x = substrate_relu(x)
        return x


# ============================================================================
# Input generation functions
# ============================================================================

batch_size = 4
in_channels = 32
out_channels = 64
depth, height, width = 32, 128, 128
kernel_size = 3
stride = 1
padding = 1


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]
