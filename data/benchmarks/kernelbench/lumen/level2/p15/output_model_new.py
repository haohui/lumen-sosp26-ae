import torch
import torch.nn as nn
import substrate
import substrate.language as S

# Problem shape from model
BATCH_SIZE = 16
IN_CHANNELS = 16
OUT_CHANNELS = 32
IN_DEPTH = 16
IN_HEIGHT = 32
IN_WIDTH = 32
KERNEL_SIZE = 3
STRIDE = 2
PADDING = 1

# Output spatial dimensions after ConvTranspose3d
# output_size = (input_size - 1) * stride + kernel_size - 2 * padding
OUT_DEPTH = (IN_DEPTH - 1) * STRIDE + KERNEL_SIZE - 2 * PADDING  # 31
OUT_HEIGHT = (IN_HEIGHT - 1) * STRIDE + KERNEL_SIZE - 2 * PADDING  # 63
OUT_WIDTH = (IN_WIDTH - 1) * STRIDE + KERNEL_SIZE - 2 * PADDING  # 63

SPATIAL_SIZE = OUT_DEPTH * OUT_HEIGHT * OUT_WIDTH  # 31 * 63 * 63 = 123039
INV_SPATIAL = 1.0 / SPATIAL_SIZE

BLOCK_SIZE = 256
GRID_BATCH_CHANNEL = BATCH_SIZE * OUT_CHANNELS  # 16 * 32 = 512


@substrate.jit
def sum_spatial_bf16_kernel(
    x: S.Tensor((BATCH_SIZE, OUT_CHANNELS, OUT_DEPTH, OUT_HEIGHT, OUT_WIDTH), S.bf16),
    out: S.Tensor((BATCH_SIZE, OUT_CHANNELS, 1, 1, 1), S.bf16),
):
    """
    Sum over spatial dimensions (D, H, W) for each (batch, channel) pair.
    Each block computes one (batch, channel) sum.
    """
    tid = S.thread_id(0)
    bid = S.block_id(0)

    n = bid // OUT_CHANNELS
    c = bid % OUT_CHANNELS

    # Parallel reduction over spatial elements
    acc = S.convert(0.0, S.f32)
    for idx in S.range(SPATIAL_SIZE):
        pos = idx * BLOCK_SIZE + tid
        if pos < SPATIAL_SIZE:
            d = pos // (OUT_HEIGHT * OUT_WIDTH)
            rem = pos % (OUT_HEIGHT * OUT_WIDTH)
            h = rem // OUT_WIDTH
            w = rem % OUT_WIDTH
            val = S.convert(x[n, c, d, h, w], S.f32)
            acc = acc + val

    out[n, c, 0, 0, 0] = S.convert(acc, S.bf16)


@substrate.jit
def subtract_mean_bf16_kernel(
    x: S.Tensor((BATCH_SIZE, OUT_CHANNELS, OUT_DEPTH, OUT_HEIGHT, OUT_WIDTH), S.bf16),
    mean: S.Tensor((BATCH_SIZE, OUT_CHANNELS, 1, 1, 1), S.bf16),
    out: S.Tensor((BATCH_SIZE, OUT_CHANNELS, OUT_DEPTH, OUT_HEIGHT, OUT_WIDTH), S.bf16),
):
    """
    Subtract mean from each spatial position.
    Each thread handles multiple spatial positions.
    """
    tid = S.thread_id(0)
    bid = S.block_id(0)

    n = bid // OUT_CHANNELS
    c = bid % OUT_CHANNELS

    m_val = mean[n, c, 0, 0, 0]

    for idx in S.range(SPATIAL_SIZE):
        pos = idx * BLOCK_SIZE + tid
        if pos < SPATIAL_SIZE:
            d = pos // (OUT_HEIGHT * OUT_WIDTH)
            rem = pos % (OUT_HEIGHT * OUT_WIDTH)
            h = rem // OUT_WIDTH
            w = rem % OUT_WIDTH
            out[n, c, d, h, w] = x[n, c, d, h, w] - m_val


@substrate.jit
def fused_mean_subtract_bf16_kernel(
    x: S.Tensor((BATCH_SIZE, OUT_CHANNELS, OUT_DEPTH, OUT_HEIGHT, OUT_WIDTH), S.bf16),
    out: S.Tensor((BATCH_SIZE, OUT_CHANNELS, OUT_DEPTH, OUT_HEIGHT, OUT_WIDTH), S.bf16),
    sum_out: S.Tensor((BATCH_SIZE, OUT_CHANNELS, 1, 1, 1), S.bf16),
):
    """
    Fused kernel: compute sum, then subtract mean.
    Uses shared memory for the sum.
    """
    tid = S.thread_id(0)
    bid = S.block_id(0)

    n = bid // OUT_CHANNELS
    c = bid % OUT_CHANNELS

    # Shared memory for partial sums (one per thread)
    s_partial = S.make_shared((BLOCK_SIZE,), S.f32)

    # Initialize partial sum
    s_partial[tid] = S.convert(0.0, S.f32)
    S.syncthreads()

    # Each thread accumulates its portion of spatial elements
    local_sum = S.convert(0.0, S.f32)
    for idx in S.range(SPATIAL_SIZE):
        pos = idx * BLOCK_SIZE + tid
        if pos < SPATIAL_SIZE:
            d = pos // (OUT_HEIGHT * OUT_WIDTH)
            rem = pos % (OUT_HEIGHT * OUT_WIDTH)
            h = rem // OUT_WIDTH
            w = rem % OUT_WIDTH
            local_sum = local_sum + S.convert(x[n, c, d, h, w], S.f32)

    s_partial[tid] = local_sum
    S.syncthreads()

    # Reduce within block (sequential reduction for simplicity)
    if tid == 0:
        total = S.convert(0.0, S.f32)
        for t in S.range(BLOCK_SIZE):
            total = total + s_partial[t]
        # Store sum for verification
        sum_out[n, c, 0, 0, 0] = S.convert(total, S.bf16)

        # Compute mean
        mean_val = total * S.convert(INV_SPATIAL, S.f32)

        # Store mean in shared memory for all threads to use
        s_partial[0] = mean_val

    S.syncthreads()

    # Get the computed mean
    mean_val = s_partial[0]

    # Subtract mean from each element
    for idx in S.range(SPATIAL_SIZE):
        pos = idx * BLOCK_SIZE + tid
        if pos < SPATIAL_SIZE:
            d = pos // (OUT_HEIGHT * OUT_WIDTH)
            rem = pos % (OUT_HEIGHT * OUT_WIDTH)
            h = rem // OUT_WIDTH
            w = rem % OUT_WIDTH
            val = S.convert(x[n, c, d, h, w], S.f32)
            out[n, c, d, h, w] = S.convert(val - mean_val, S.bf16)


def substrate_mean_subtract_spatial(x: torch.Tensor) -> torch.Tensor:
    """
    Compute mean over spatial dimensions (2, 3, 4) and subtract from input.
    Input shape: (BATCH_SIZE, OUT_CHANNELS, OUT_DEPTH, OUT_HEIGHT, OUT_WIDTH)
    """
    if not x.is_cuda:
        raise RuntimeError("Input must be on CUDA/HIP device")

    if x.shape != (BATCH_SIZE, OUT_CHANNELS, OUT_DEPTH, OUT_HEIGHT, OUT_WIDTH):
        raise ValueError(f"Expected shape {(BATCH_SIZE, OUT_CHANNELS, OUT_DEPTH, OUT_HEIGHT, OUT_WIDTH)}, got {x.shape}")

    x_cont = x.contiguous()

    out = torch.empty_like(x_cont)
    sum_out = torch.empty((BATCH_SIZE, OUT_CHANNELS, 1, 1, 1), device=x.device, dtype=x.dtype)

    # Launch fused kernel
    grid = (GRID_BATCH_CHANNEL, 1, 1)
    fused_mean_subtract_bf16_kernel[lambda: (grid, (BLOCK_SIZE, 1, 1))](x_cont, out, sum_out)

    return out


class ModelNew(nn.Module):
    """
    Optimized implementation with Substrate kernel for mean subtraction.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias
        )
        self.batch_norm = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        # ConvTranspose3d
        x = self.conv_transpose(x)

        # BatchNorm3d
        x = self.batch_norm(x)

        # Mean subtraction over spatial dimensions using Substrate kernel
        if x.shape[1:] == (OUT_CHANNELS, OUT_DEPTH, OUT_HEIGHT, OUT_WIDTH):
            x = substrate_mean_subtract_spatial(x)
        else:
            # Fallback for unexpected shapes
            x = x - torch.mean(x, dim=(2, 3, 4), keepdim=True)

        return x


batch_size = BATCH_SIZE
in_channels = IN_CHANNELS
out_channels = OUT_CHANNELS
depth, height, width = IN_DEPTH, IN_HEIGHT, IN_WIDTH
kernel_size = KERNEL_SIZE
stride = STRIDE
padding = PADDING


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]
