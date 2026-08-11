import torch
import torch.nn as nn
import substrate
import substrate.language as S


# Fixed problem configuration
BATCH_SIZE = 128
IN_CHANNELS = 3
OUT_CHANNELS = 16
IN_D, IN_H, IN_W = 16, 32, 32
KERNEL_SIZE = 3
STRIDE = 2
PADDING = 1
OUTPUT_PADDING = 1

# ConvTranspose3d output shape
# D_out = (D_in - 1) * stride - 2*padding + kernel_size + output_padding
CONV_D_OUT = (IN_D - 1) * STRIDE - 2 * PADDING + KERNEL_SIZE + OUTPUT_PADDING  # 32
CONV_H_OUT = (IN_H - 1) * STRIDE - 2 * PADDING + KERNEL_SIZE + OUTPUT_PADDING  # 64
CONV_W_OUT = (IN_W - 1) * STRIDE - 2 * PADDING + KERNEL_SIZE + OUTPUT_PADDING  # 64

# MaxPool3d config
POOL_KERNEL = 2
POOL_STRIDE = 2
POOL_PADDING = 0

# MaxPool3d output shape
POOL_D_OUT = (CONV_D_OUT + 2 * POOL_PADDING - POOL_KERNEL) // POOL_STRIDE + 1  # 16
POOL_H_OUT = (CONV_H_OUT + 2 * POOL_PADDING - POOL_KERNEL) // POOL_STRIDE + 1  # 32
POOL_W_OUT = (CONV_W_OUT + 2 * POOL_PADDING - POOL_KERNEL) // POOL_STRIDE + 1  # 32

BLOCK_THREADS = 256
NEG_INF_F32 = -3.402823466e38


# ============================================================================
# MaxPool3d Kernel
# ============================================================================

@substrate.jit
def maxpool3d_bf16_kernel(
    x: S.Pointer(S.bf16),
    out: S.Pointer(S.bf16),
    batch: S.u32,
    channels: S.u32,
    in_d: S.u32,
    in_h: S.u32,
    in_w: S.u32,
    out_d: S.u32,
    out_h: S.u32,
    out_w: S.u32,
):
    """MaxPool3d kernel for NCDHW tensor."""
    tid = S.block_id(0) * S.block_dim(0) + S.thread_id(0)
    total = batch * channels * out_d * out_h * out_w

    x_layout = S.make_layout((batch, channels, in_d, in_h, in_w), (channels * in_d * in_h * in_w, in_d * in_h * in_w, in_h * in_w, in_w, 1))
    out_layout = S.make_layout((batch, channels, out_d, out_h, out_w), (channels * out_d * out_h * out_w, out_d * out_h * out_w, out_h * out_w, out_w, 1))
    gx = S.make_tensor(x, S.bf16, x_layout)
    gout = S.make_tensor(out, S.bf16, out_layout)

    if tid < total:
        ow = tid % out_w
        rest = tid // out_w
        oh = rest % out_h
        rest = rest // out_h
        od = rest % out_d
        rest = rest // out_d
        c = rest % channels
        n = rest // channels

        maxv = S.convert(NEG_INF_F32, S.f32)

        for kd in S.range(POOL_KERNEL):
            for kh in S.range(POOL_KERNEL):
                for kw in S.range(POOL_KERNEL):
                    id_pos = od * POOL_STRIDE + kd
                    ih_pos = oh * POOL_STRIDE + kh
                    iw_pos = ow * POOL_STRIDE + kw

                    if id_pos < in_d and ih_pos < in_h and iw_pos < in_w:
                        v = S.convert(gx[n, c, id_pos, ih_pos, iw_pos], S.f32)
                        if v > maxv:
                            maxv = v

        gout[n, c, od, oh, ow] = S.convert(maxv, S.bf16)


# ============================================================================
# Softmax Kernel (online softmax for numerical stability)
# ============================================================================

@substrate.jit
def softmax_bf16_kernel(
    x: S.Pointer(S.bf16),
    out: S.Pointer(S.bf16),
    batch: S.u32,
    channels: S.u32,
    spatial: S.u32,
):
    """Softmax over channel dimension (dim=1)."""
    tid = S.block_id(0) * S.block_dim(0) + S.thread_id(0)
    total = batch * spatial

    x_layout = S.make_layout((batch, channels, spatial), (channels * spatial, spatial, 1))
    out_layout = S.make_layout((batch, channels, spatial), (channels * spatial, spatial, 1))
    gx = S.make_tensor(x, S.bf16, x_layout)
    gout = S.make_tensor(out, S.bf16, out_layout)

    if tid < total:
        n = tid // spatial
        s = tid % spatial

        # Find max for numerical stability
        maxv = S.convert(NEG_INF_F32, S.f32)
        for c in S.range(channels):
            v = S.convert(gx[n, c, s], S.f32)
            if v > maxv:
                maxv = v

        # Compute exp and sum
        sum_exp = S.convert(0.0, S.f32)
        for c in S.range(channels):
            v = S.convert(gx[n, c, s], S.f32)
            exp_v = S.exp2((v - maxv) * S.convert(1.4426950408889634, S.f32))  # log2(e)
            sum_exp = sum_exp + exp_v

        # Normalize and write output
        inv_sum = S.convert(1.0, S.f32) / sum_exp
        for c in S.range(channels):
            v = S.convert(gx[n, c, s], S.f32)
            exp_v = S.exp2((v - maxv) * S.convert(1.4426950408889634, S.f32))
            gout[n, c, s] = S.convert(exp_v * inv_sum, S.bf16)


# ============================================================================
# Subtract Kernel (broadcast per channel)
# ============================================================================

@substrate.jit
def subtract_bf16_kernel(
    x: S.Pointer(S.bf16),
    bias: S.Pointer(S.bf16),
    out: S.Pointer(S.bf16),
    batch: S.u32,
    channels: S.u32,
    spatial: S.u32,
):
    """Subtract bias from each channel."""
    tid = S.block_id(0) * S.block_dim(0) + S.thread_id(0)
    total = batch * channels * spatial

    x_layout = S.make_layout((batch, channels, spatial), (channels * spatial, spatial, 1))
    out_layout = S.make_layout((batch, channels, spatial), (channels * spatial, spatial, 1))
    bias_layout = S.make_layout((channels,), (1,))
    gx = S.make_tensor(x, S.bf16, x_layout)
    gout = S.make_tensor(out, S.bf16, out_layout)
    gbias = S.make_tensor(bias, S.bf16, bias_layout)

    if tid < total:
        s = tid % spatial
        rest = tid // spatial
        c = rest % channels
        n = rest // channels

        v = gx[n, c, s]
        b = gbias[c]
        gout[n, c, s] = v - b


# ============================================================================
# Swish Kernel (sigmoid(x) * x)
# ============================================================================

LOG2E = 1.4426950408889634


@substrate.jit
def swish_bf16_kernel(
    x: S.Pointer(S.bf16),
    out: S.Pointer(S.bf16),
    n: S.u32,
):
    """Swish activation: sigmoid(x) * x."""
    tid = S.block_id(0) * S.block_dim(0) + S.thread_id(0)

    layout = S.make_layout((n,), (1,))
    gx = S.make_tensor(x, S.bf16, layout)
    gout = S.make_tensor(out, S.bf16, layout)

    if tid < n:
        v = S.convert(gx[tid], S.f32)
        # sigmoid(x) = 1 / (1 + exp(-x))
        # Use exp2 for efficiency: exp(-x) = exp2(-x * log2(e))
        neg_v = S.convert(0.0, S.f32) - v
        exp_neg_v = S.exp2(neg_v * S.convert(LOG2E, S.f32))
        sigmoid_v = S.convert(1.0, S.f32) / (S.convert(1.0, S.f32) + exp_neg_v)
        swish_v = sigmoid_v * v
        gout[tid] = S.convert(swish_v, S.bf16)


# ============================================================================
# Max over channels Kernel
# ============================================================================

@substrate.jit
def max_over_channels_bf16_kernel(
    x: S.Pointer(S.bf16),
    out: S.Pointer(S.bf16),
    batch: S.u32,
    channels: S.u32,
    spatial: S.u32,
):
    """Max reduction over channel dimension (dim=1)."""
    tid = S.block_id(0) * S.block_dim(0) + S.thread_id(0)
    total = batch * spatial

    x_layout = S.make_layout((batch, channels, spatial), (channels * spatial, spatial, 1))
    out_layout = S.make_layout((batch, spatial), (spatial, 1))
    gx = S.make_tensor(x, S.bf16, x_layout)
    gout = S.make_tensor(out, S.bf16, out_layout)

    if tid < total:
        n = tid // spatial
        s = tid % spatial

        maxv = S.convert(NEG_INF_F32, S.f32)
        for c in S.range(channels):
            v = S.convert(gx[n, c, s], S.f32)
            if v > maxv:
                maxv = v

        gout[n, s] = S.convert(maxv, S.bf16)


# ============================================================================
# Host wrappers
# ============================================================================

def substrate_maxpool3d(x: torch.Tensor) -> torch.Tensor:
    """MaxPool3d with kernel=2, stride=2, padding=0."""
    assert x.dim() == 5, f"Expected 5D input, got {x.dim()}D"
    n, c, d, h, w = x.shape
    out_d = (d + 2 * POOL_PADDING - POOL_KERNEL) // POOL_STRIDE + 1
    out_h = (h + 2 * POOL_PADDING - POOL_KERNEL) // POOL_STRIDE + 1
    out_w = (w + 2 * POOL_PADDING - POOL_KERNEL) // POOL_STRIDE + 1

    out = torch.empty((n, c, out_d, out_h, out_w), device=x.device, dtype=x.dtype)
    total = n * c * out_d * out_h * out_w
    grid = ((total + BLOCK_THREADS - 1) // BLOCK_THREADS, 1, 1)

    maxpool3d_bf16_kernel[lambda: (grid, (BLOCK_THREADS, 1, 1))](
        x, out, n, c, d, h, w, out_d, out_h, out_w
    )
    return out


def substrate_softmax_channel(x: torch.Tensor) -> torch.Tensor:
    """Softmax over channel dimension (dim=1)."""
    assert x.dim() == 5, f"Expected 5D input, got {x.dim()}D"
    n, c, d, h, w = x.shape
    spatial = d * h * w

    x_flat = x.reshape(n, c, spatial)
    out = torch.empty_like(x_flat)

    total = n * spatial
    grid = ((total + BLOCK_THREADS - 1) // BLOCK_THREADS, 1, 1)

    softmax_bf16_kernel[lambda: (grid, (BLOCK_THREADS, 1, 1))](
        x_flat, out, n, c, spatial
    )
    return out.reshape(n, c, d, h, w)


def substrate_subtract_channel(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Subtract bias from each channel."""
    assert x.dim() == 5, f"Expected 5D input, got {x.dim()}D"
    n, c, d, h, w = x.shape
    spatial = d * h * w

    x_flat = x.reshape(n, c, spatial)
    out = torch.empty_like(x_flat)

    total = n * c * spatial
    grid = ((total + BLOCK_THREADS - 1) // BLOCK_THREADS, 1, 1)

    subtract_bf16_kernel[lambda: (grid, (BLOCK_THREADS, 1, 1))](
        x_flat, bias, out, n, c, spatial
    )
    return out.reshape(n, c, d, h, w)


def substrate_swish(x: torch.Tensor) -> torch.Tensor:
    """Swish activation: sigmoid(x) * x."""
    out = torch.empty_like(x)
    n = x.numel()
    grid = ((n + BLOCK_THREADS - 1) // BLOCK_THREADS, 1, 1)

    swish_bf16_kernel[lambda: (grid, (BLOCK_THREADS, 1, 1))](x, out, n)
    return out


def substrate_max_channel(x: torch.Tensor) -> torch.Tensor:
    """Max reduction over channel dimension (dim=1)."""
    assert x.dim() == 5, f"Expected 5D input, got {x.dim()}D"
    n, c, d, h, w = x.shape
    spatial = d * h * w

    x_flat = x.reshape(n, c, spatial)
    out = torch.empty((n, spatial), device=x.device, dtype=x.dtype)

    total = n * spatial
    grid = ((total + BLOCK_THREADS - 1) // BLOCK_THREADS, 1, 1)

    max_over_channels_bf16_kernel[lambda: (grid, (BLOCK_THREADS, 1, 1))](
        x_flat, out, n, c, spatial
    )
    return out.reshape(n, d, h, w)


# ============================================================================
# Model
# ============================================================================

class ModelNew(nn.Module):
    """
    Optimized Substrate implementation of:
        - ConvTranspose3d
        - MaxPool3d
        - Softmax
        - Subtract
        - Swish
        - Max
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding,
                 pool_kernel_size, pool_stride, pool_padding):
        super(ModelNew, self).__init__()
        # Use PyTorch for ConvTranspose3d (complex operation without substrate example)
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.subtract = nn.Parameter(torch.randn(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_device = x.device
        if not x.is_cuda:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA/HIP device is required for Substrate kernels.")
            x = x.cuda()

        # Convert to bfloat16 for optimized kernels
        orig_dtype = x.dtype
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)

        x = x.contiguous()

        # ConvTranspose3d (using PyTorch - complex operation)
        x = self.conv_transpose(x)
        x = x.to(torch.bfloat16).contiguous()  # Ensure bf16 after conv

        # MaxPool3d (Substrate kernel)
        x = substrate_maxpool3d(x)

        # Softmax (Substrate kernel)
        x = substrate_softmax_channel(x)

        # Subtract (Substrate kernel)
        bias = self.subtract.to(x.device).to(torch.bfloat16)
        x = substrate_subtract_channel(x, bias)

        # Swish (Substrate kernel)
        x = substrate_swish(x)

        # Max over channels (Substrate kernel)
        x = substrate_max_channel(x)

        # Restore original dtype if needed
        if orig_dtype != torch.bfloat16:
            x = x.to(orig_dtype)

        if orig_device.type != "cuda":
            x = x.to(orig_device)

        return x


batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
pool_kernel_size = 2
pool_stride = 2
pool_padding = 0


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding,
            pool_kernel_size, pool_stride, pool_padding]
