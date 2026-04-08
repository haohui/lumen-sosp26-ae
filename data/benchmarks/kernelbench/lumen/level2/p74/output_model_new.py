import torch
import torch.nn as nn
import substrate
import substrate.language as S


# Fixed problem configuration
BATCH_SIZE = 16
IN_CHANNELS = 16
OUT_CHANNELS = 32
IN_D = 16
IN_H = 32
IN_W = 32

KERNEL_SIZE = 3
STRIDE = 2
PADDING = 1
OUTPUT_PADDING = 1

# ConvTranspose3d output shapes
OUT_D = (IN_D - 1) * STRIDE - 2 * PADDING + KERNEL_SIZE + OUTPUT_PADDING  # 32
OUT_H = (IN_H - 1) * STRIDE - 2 * PADDING + KERNEL_SIZE + OUTPUT_PADDING  # 64
OUT_W = (IN_W - 1) * STRIDE - 2 * PADDING + KERNEL_SIZE + OUTPUT_PADDING  # 64

# MaxPool3d output shapes
POOL_D = OUT_D // 2  # 16
POOL_H = OUT_H // 2  # 32
POOL_W = OUT_W // 2  # 32

THREADS = 256

# LeakyReLU negative slope
NEGATIVE_SLOPE = 0.2


# ============================================================================
# LeakyReLU Kernel (BF16)
# ============================================================================

@substrate.jit
def leaky_relu_bf16_kernel(
    x_ptr: S.Pointer(S.bf16),
    y_ptr: S.Pointer(S.bf16),
    n: S.u32,
):
    """LeakyReLU activation: max(x, negative_slope * x)"""
    idx = S.block_id(0) * S.block_dim(0) + S.thread_id(0)
    layout = S.make_layout((n,), (1,))
    x = S.make_tensor(x_ptr, S.bf16, layout)
    y = S.make_tensor(y_ptr, S.bf16, layout)

    if idx < n:
        xv = S.convert(x[idx], S.f32)
        zero = S.convert(0.0, S.f32)
        slope = S.convert(NEGATIVE_SLOPE, S.f32)

        if xv >= zero:
            y[idx] = x[idx]
        else:
            result = xv * slope
            y[idx] = S.convert(result, S.bf16)


@substrate.jit
def leaky_relu_f32_kernel(
    x_ptr: S.Pointer(S.f32),
    y_ptr: S.Pointer(S.f32),
    n: S.u32,
):
    """LeakyReLU activation: max(x, negative_slope * x)"""
    idx = S.block_id(0) * S.block_dim(0) + S.thread_id(0)
    layout = S.make_layout((n,), (1,))
    x = S.make_tensor(x_ptr, S.f32, layout)
    y = S.make_tensor(y_ptr, S.f32, layout)

    if idx < n:
        xv = x[idx]
        zero = S.convert(0.0, S.f32)
        slope = S.convert(NEGATIVE_SLOPE, S.f32)

        if xv >= zero:
            y[idx] = xv
        else:
            y[idx] = xv * slope


# ============================================================================
# Channel-wise Broadcast Multiplication Kernel
# ============================================================================

@substrate.jit
def channel_scale_bf16_kernel(
    x_ptr: S.Pointer(S.bf16),
    mult_ptr: S.Pointer(S.bf16),
    y_ptr: S.Pointer(S.bf16),
    n: S.u32,
    c: S.u32,
    d: S.u32,
    h: S.u32,
    w: S.u32,
):
    """Multiply x by channel-wise broadcast multiplier."""
    idx = S.block_id(0) * S.block_dim(0) + S.thread_id(0)

    if idx < n:
        # Compute indices
        spatial = d * h * w
        chw = c * spatial
        hw = h * w

        flat_idx = idx
        batch = flat_idx // chw
        flat_idx = flat_idx - batch * chw
        channel = flat_idx // spatial
        spatial_idx = flat_idx - channel * spatial

        d_idx = spatial_idx // hw
        hw_idx = spatial_idx - d_idx * hw
        h_idx = hw_idx // w
        w_idx = hw_idx - h_idx * w

        # Index into contiguous tensor
        full_idx = ((batch * c + channel) * d + d_idx) * h * w + h_idx * w + w_idx

        layout = S.make_layout((n,), (1,))
        x = S.make_tensor(x_ptr, S.bf16, layout)
        mult = S.make_tensor(mult_ptr, S.bf16, layout)
        y = S.make_tensor(y_ptr, S.bf16, layout)

        xv = x[full_idx]
        mv = mult[channel]
        y[full_idx] = xv * mv


@substrate.jit
def channel_scale_f32_kernel(
    x_ptr: S.Pointer(S.f32),
    mult_ptr: S.Pointer(S.f32),
    y_ptr: S.Pointer(S.f32),
    n: S.u32,
    c: S.u32,
    d: S.u32,
    h: S.u32,
    w: S.u32,
):
    """Multiply x by channel-wise broadcast multiplier."""
    idx = S.block_id(0) * S.block_dim(0) + S.thread_id(0)

    if idx < n:
        # Compute indices
        spatial = d * h * w
        chw = c * spatial
        hw = h * w

        flat_idx = idx
        batch = flat_idx // chw
        flat_idx = flat_idx - batch * chw
        channel = flat_idx // spatial
        spatial_idx = flat_idx - channel * spatial

        d_idx = spatial_idx // hw
        hw_idx = spatial_idx - d_idx * hw
        h_idx = hw_idx // w
        w_idx = hw_idx - h_idx * w

        # Index into contiguous tensor
        full_idx = ((batch * c + channel) * d + d_idx) * h * w + h_idx * w + w_idx

        layout = S.make_layout((n,), (1,))
        x = S.make_tensor(x_ptr, S.f32, layout)
        mult = S.make_tensor(mult_ptr, S.f32, layout)
        y = S.make_tensor(y_ptr, S.f32, layout)

        xv = x[full_idx]
        mv = mult[channel]
        y[full_idx] = xv * mv


# ============================================================================
# MaxPool3d Kernel - Simple 1D block per output element
# ============================================================================

NEG_INF_F32 = -3.402823466e38


@substrate.jit
def maxpool3d_bf16_kernel(
    x_ptr: S.Pointer(S.bf16),
    out_ptr: S.Pointer(S.bf16),
    n_out: S.u32,
    in_d: S.u32,
    in_h: S.u32,
    in_w: S.u32,
    out_d: S.u32,
    out_h: S.u32,
    out_w: S.u32,
    channels: S.u32,
):
    """MaxPool3d with kernel_size=2, stride=2. One thread per output element."""
    idx = S.block_id(0) * S.block_dim(0) + S.thread_id(0)

    if idx < n_out:
        # Decode output indices
        hw_out = out_h * out_w
        dhw_out = out_d * hw_out
        cdhw_out = channels * dhw_out

        batch = idx // cdhw_out
        rem1 = idx - batch * cdhw_out
        channel = rem1 // dhw_out
        rem2 = rem1 - channel * dhw_out
        od = rem2 // hw_out
        rem3 = rem2 - od * hw_out
        oh = rem3 // out_w
        ow = rem3 - oh * out_w

        # Input indices (stride 2)
        id_start = od * 2
        ih_start = oh * 2
        iw_start = ow * 2

        # Compute flat input index
        hw_in = in_h * in_w
        dhw_in = in_d * hw_in
        cdhw_in = channels * dhw_in
        base_in = batch * cdhw_in + channel * dhw_in

        layout_in = S.make_layout((batch * cdhw_in + channels * dhw_in,), (1,))
        x = S.make_tensor(x_ptr, S.bf16, layout_in)

        # Find max over 2x2x2 window
        m = S.convert(NEG_INF_F32, S.f32)
        for kz in S.range(2):
            iz = id_start + kz
            if iz < in_d:
                for ky in S.range(2):
                    ih = ih_start + ky
                    if ih < in_h:
                        for kx in S.range(2):
                            iw = iw_start + kx
                            if iw < in_w:
                                in_idx = base_in + iz * hw_in + ih * in_w + iw
                                val = S.convert(x[in_idx], S.f32)
                                if val > m:
                                    m = val

        layout_out = S.make_layout((n_out,), (1,))
        out = S.make_tensor(out_ptr, S.bf16, layout_out)
        out[idx] = S.convert(m, S.bf16)


@substrate.jit
def maxpool3d_f32_kernel(
    x_ptr: S.Pointer(S.f32),
    out_ptr: S.Pointer(S.f32),
    n_out: S.u32,
    in_d: S.u32,
    in_h: S.u32,
    in_w: S.u32,
    out_d: S.u32,
    out_h: S.u32,
    out_w: S.u32,
    channels: S.u32,
):
    """MaxPool3d with kernel_size=2, stride=2. One thread per output element."""
    idx = S.block_id(0) * S.block_dim(0) + S.thread_id(0)

    if idx < n_out:
        # Decode output indices
        hw_out = out_h * out_w
        dhw_out = out_d * hw_out
        cdhw_out = channels * dhw_out

        batch = idx // cdhw_out
        rem1 = idx - batch * cdhw_out
        channel = rem1 // dhw_out
        rem2 = rem1 - channel * dhw_out
        od = rem2 // hw_out
        rem3 = rem2 - od * hw_out
        oh = rem3 // out_w
        ow = rem3 - oh * out_w

        # Input indices (stride 2)
        id_start = od * 2
        ih_start = oh * 2
        iw_start = ow * 2

        # Compute flat input index
        hw_in = in_h * in_w
        dhw_in = in_d * hw_in
        cdhw_in = channels * dhw_in
        base_in = batch * cdhw_in + channel * dhw_in

        layout_in = S.make_layout((batch * cdhw_in + channels * dhw_in,), (1,))
        x = S.make_tensor(x_ptr, S.f32, layout_in)

        # Find max over 2x2x2 window
        m = S.convert(NEG_INF_F32, S.f32)
        for kz in S.range(2):
            iz = id_start + kz
            if iz < in_d:
                for ky in S.range(2):
                    ih = ih_start + ky
                    if ih < in_h:
                        for kx in S.range(2):
                            iw = iw_start + kx
                            if iw < in_w:
                                in_idx = base_in + iz * hw_in + ih * in_w + iw
                                val = x[in_idx]
                                if val > m:
                                    m = val

        layout_out = S.make_layout((n_out,), (1,))
        out = S.make_tensor(out_ptr, S.f32, layout_out)
        out[idx] = m


# ============================================================================
# Host wrappers
# ============================================================================

def substrate_leaky_relu(x: torch.Tensor) -> torch.Tensor:
    """LeakyReLU using Substrate kernel."""
    if x.numel() == 0:
        return torch.empty_like(x)

    orig_device = x.device
    if not x.is_cuda:
        x = x.cuda()

    x_contig = x.contiguous()
    out = torch.empty_like(x_contig)
    n = x_contig.numel()

    grid = (n + THREADS - 1) // THREADS

    if x_contig.dtype == torch.bfloat16:
        leaky_relu_bf16_kernel[lambda: ((grid, 1, 1), (THREADS, 1, 1))](
            x_contig.view(-1), out.view(-1), n
        )
    elif x_contig.dtype == torch.float32:
        leaky_relu_f32_kernel[lambda: ((grid, 1, 1), (THREADS, 1, 1))](
            x_contig.view(-1), out.view(-1), n
        )
    else:
        raise TypeError(f"Unsupported dtype for LeakyReLU: {x_contig.dtype}")

    if orig_device.type != "cuda":
        out = out.to(orig_device)
    return out


def substrate_channel_scale(x: torch.Tensor, mult: torch.Tensor) -> torch.Tensor:
    """Channel-wise broadcast multiplication using Substrate kernel."""
    if x.numel() == 0:
        return torch.empty_like(x)

    orig_device = x.device
    if not x.is_cuda:
        x = x.cuda()
        mult = mult.cuda()

    x_contig = x.contiguous()
    mult_contig = mult.contiguous().view(-1)
    out = torch.empty_like(x_contig)

    n = x_contig.numel()
    c, d, h, w = x_contig.shape[1], x_contig.shape[2], x_contig.shape[3], x_contig.shape[4]

    grid = (n + THREADS - 1) // THREADS

    if x_contig.dtype == torch.bfloat16:
        channel_scale_bf16_kernel[lambda: ((grid, 1, 1), (THREADS, 1, 1))](
            x_contig.view(-1), mult_contig, out.view(-1), n, c, d, h, w
        )
    elif x_contig.dtype == torch.float32:
        channel_scale_f32_kernel[lambda: ((grid, 1, 1), (THREADS, 1, 1))](
            x_contig.view(-1), mult_contig, out.view(-1), n, c, d, h, w
        )
    else:
        raise TypeError(f"Unsupported dtype for channel_scale: {x_contig.dtype}")

    if orig_device.type != "cuda":
        out = out.to(orig_device)
    return out


def substrate_maxpool3d(x: torch.Tensor) -> torch.Tensor:
    """MaxPool3d using Substrate kernel."""
    if x.numel() == 0:
        return torch.empty_like(x)

    orig_device = x.device
    if not x.is_cuda:
        x = x.cuda()

    x_contig = x.contiguous()

    batch, channels, in_d, in_h, in_w = x_contig.shape
    out_d = in_d // 2
    out_h = in_h // 2
    out_w = in_w // 2

    out = torch.empty((batch, channels, out_d, out_h, out_w),
                       device=x_contig.device, dtype=x_contig.dtype)

    n_out = batch * channels * out_d * out_h * out_w
    grid = (n_out + THREADS - 1) // THREADS

    if x_contig.dtype == torch.bfloat16:
        maxpool3d_bf16_kernel[lambda: ((grid, 1, 1), (THREADS, 1, 1))](
            x_contig.view(-1), out.view(-1), n_out, in_d, in_h, in_w, out_d, out_h, out_w, channels
        )
    elif x_contig.dtype == torch.float32:
        maxpool3d_f32_kernel[lambda: ((grid, 1, 1), (THREADS, 1, 1))](
            x_contig.view(-1), out.view(-1), n_out, in_d, in_h, in_w, out_d, out_h, out_w, channels
        )
    else:
        raise TypeError(f"Unsupported dtype for MaxPool3d: {x_contig.dtype}")

    if orig_device.type != "cuda":
        out = out.to(orig_device)
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs 3D transposed convolution, LeakyReLU,
    channel-wise multiplication, LeakyReLU, and MaxPool3d.
    Uses Substrate GPU kernels for activation, scaling, and pooling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Move to GPU if needed
        orig_device = x.device
        if not x.is_cuda:
            x = x.cuda()

        # Step 1: ConvTranspose3d (using PyTorch's optimized implementation)
        x = self.conv_transpose(x)

        # Step 2: LeakyReLU (Substrate kernel)
        x = substrate_leaky_relu(x)

        # Step 3: Channel-wise multiplication (Substrate kernel)
        mult = self.multiplier.to(device=x.device, dtype=x.dtype)
        x = substrate_channel_scale(x, mult)

        # Step 4: LeakyReLU (Substrate kernel)
        x = substrate_leaky_relu(x)

        # Step 5: MaxPool3d (Substrate kernel)
        x = substrate_maxpool3d(x)

        if orig_device.type != "cuda":
            x = x.to(orig_device)
        return x


batch_size = 16
in_channels = 16
out_channels = 32
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
multiplier_shape = (out_channels, 1, 1, 1)


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier_shape]
