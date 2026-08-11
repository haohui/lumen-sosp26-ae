import torch
import torch.nn as nn
import substrate
import substrate.language as S


# Fixed problem shape from target model/get_inputs.
BATCH_SIZE = 128
IN_CHANNELS = 64
OUT_CHANNELS = 64
IN_H = 256
IN_W = 256
KERNEL_SIZE = 3
STRIDE = 1
PADDING = 1

# ConvTranspose2d output dimensions (with stride=1, padding=1, kernel=3: output = input)
CONV_OUT_H = IN_H
CONV_OUT_W = IN_W

# MaxPool2d parameters
MAXPOOL_KERNEL = 2
MAXPOOL_STRIDE = 2
POOL_OUT_H = CONV_OUT_H // MAXPOOL_STRIDE  # 128
POOL_OUT_W = CONV_OUT_W // MAXPOOL_STRIDE  # 128

# Hardtanh parameters
HARDTANH_MIN = -1.0
HARDTANH_MAX = 1.0

# Thread block sizes
THREADS_1D = 256
BLOCK_X = 16
BLOCK_Y = 16


# ============================================================================
# ConvTranspose2d Kernel
# ============================================================================

WEIGHT_ELEMS = IN_CHANNELS * KERNEL_SIZE * KERNEL_SIZE
WEIGHT_LOAD_ITERS = (WEIGHT_ELEMS + THREADS_1D - 1) // THREADS_1D
SPATIAL_ELEMS_CONV = CONV_OUT_H * CONV_OUT_W
SPATIAL_TILES_CONV = (SPATIAL_ELEMS_CONV + THREADS_1D - 1) // THREADS_1D


@substrate.jit
def conv_transpose2d_bf16_kernel(
    x: S.Tensor((BATCH_SIZE, IN_CHANNELS, IN_H, IN_W), S.bf16),
    w: S.Tensor((IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, KERNEL_SIZE), S.bf16),
    b: S.Tensor((OUT_CHANNELS,), S.bf16),
    out: S.Tensor((BATCH_SIZE, OUT_CHANNELS, CONV_OUT_H, CONV_OUT_W), S.bf16),
):
    """
    ConvTranspose2d with stride=1, padding=1, kernel_size=3.
    Weight shape: (in_channels, out_channels, kH, kW).
    """
    tid = S.thread_id(0)
    bid = S.block_id(0)

    n = bid // OUT_CHANNELS
    oc = bid % OUT_CHANNELS

    # Shared memory for one output channel's weights (across all input channels)
    s_w = S.make_shared((WEIGHT_ELEMS,), S.bf16)

    for it in S.range(WEIGHT_LOAD_ITERS):
        w_flat = it * THREADS_1D + tid
        if w_flat < WEIGHT_ELEMS:
            ic = w_flat // (KERNEL_SIZE * KERNEL_SIZE)
            rem = w_flat % (KERNEL_SIZE * KERNEL_SIZE)
            kh = rem // KERNEL_SIZE
            kw = rem % KERNEL_SIZE
            s_w[w_flat] = w[ic, oc, kh, kw]

    S.syncthreads()

    for t in S.range(SPATIAL_TILES_CONV):
        pos = t * THREADS_1D + tid
        if pos < SPATIAL_ELEMS_CONV:
            oh = pos // CONV_OUT_W
            ow = pos % CONV_OUT_W

            acc = S.convert(b[oc], S.f32)

            # For ConvTranspose2d: each output position receives contributions
            # from input positions centered around (oh, ow) with the kernel window
            for ic in S.range(IN_CHANNELS):
                for kh in S.range(KERNEL_SIZE):
                    for kw in S.range(KERNEL_SIZE):
                        # Compute input position that contributes to output (oh, ow)
                        ih_nom = oh + kh - PADDING
                        iw_nom = ow + kw - PADDING
                        if (ih_nom >= 0) and (ih_nom < IN_H) and (iw_nom >= 0) and (iw_nom < IN_W):
                            w_idx = ic * KERNEL_SIZE * KERNEL_SIZE + kh * KERNEL_SIZE + kw
                            xv = S.convert(x[n, ic, ih_nom, iw_nom], S.f32)
                            wv = S.convert(s_w[w_idx], S.f32)
                            acc = acc + xv * wv

            out[n, oc, oh, ow] = S.convert(acc, S.bf16)


# ============================================================================
# MaxPool2d Kernel
# ============================================================================

SH_H = BLOCK_Y + MAXPOOL_KERNEL - 1
SH_W = BLOCK_X + MAXPOOL_KERNEL - 1
THREADS_PER_BLOCK_POOL = BLOCK_X * BLOCK_Y
SH_LOAD_ITERS_POOL = (SH_H * SH_W + THREADS_PER_BLOCK_POOL - 1) // THREADS_PER_BLOCK_POOL

NEG_INF_F32 = -3.402823466e38


@substrate.jit
def maxpool2d_bf16_kernel(
    x: S.Tensor((BATCH_SIZE, OUT_CHANNELS, CONV_OUT_H, CONV_OUT_W), S.bf16),
    out: S.Tensor((BATCH_SIZE, OUT_CHANNELS, POOL_OUT_H, POOL_OUT_W), S.bf16),
):
    tx = S.thread_id(0)
    ty = S.thread_id(1)
    bx = S.block_id(0)
    by = S.block_id(1)
    bz = S.block_id(2)

    ox = bx * BLOCK_X + tx
    oy = by * BLOCK_Y + ty

    c = bz % OUT_CHANNELS
    n = bz // OUT_CHANNELS

    smem = S.make_shared((SH_H, SH_W), S.f32)

    tid_flat = ty * BLOCK_X + tx
    for it in S.range(SH_LOAD_ITERS_POOL):
        linear = tid_flat + it * THREADS_PER_BLOCK_POOL
        if linear < SH_H * SH_W:
            sy = linear // SH_W
            sx = linear % SH_W

            iy = by * BLOCK_Y + sy
            ix = bx * BLOCK_X + sx

            v = S.convert(NEG_INF_F32, S.f32)
            if (iy >= 0) and (iy < CONV_OUT_H) and (ix >= 0) and (ix < CONV_OUT_W):
                v = S.convert(x[n, c, iy, ix], S.f32)

            smem[sy, sx] = v

    S.syncthreads()

    if (ox < POOL_OUT_W) and (oy < POOL_OUT_H):
        m = S.convert(NEG_INF_F32, S.f32)
        for ky in S.range(MAXPOOL_KERNEL):
            for kx in S.range(MAXPOOL_KERNEL):
                val = smem[ty + ky, tx + kx]
                if val > m:
                    m = val
        out[n, c, oy, ox] = S.convert(m, S.bf16)


# ============================================================================
# Hardtanh Kernel (elementwise clamp)
# ============================================================================


@substrate.jit
def hardtanh_bf16_kernel(
    x_ptr: S.Pointer(S.bf16),
    y_ptr: S.Pointer(S.bf16),
    n: S.u32,
):
    idx = S.block_id(0) * S.block_dim(0) + S.thread_id(0)
    layout = S.make_layout((n,), (1,))
    x = S.make_tensor(x_ptr, S.bf16, layout)
    y = S.make_tensor(y_ptr, S.bf16, layout)

    if idx < n:
        v = S.convert(x[idx], S.f32)
        min_val = S.convert(HARDTANH_MIN, S.f32)
        max_val = S.convert(HARDTANH_MAX, S.f32)
        if v < min_val:
            v = min_val
        if v > max_val:
            v = max_val
        y[idx] = S.convert(v, S.bf16)


# ============================================================================
# Mean + Tanh Kernel (fused: mean over spatial dims, then tanh)
# ============================================================================

MEAN_INV_SPATIAL = 1.0 / (POOL_OUT_H * POOL_OUT_W)


@substrate.jit
def mean_tanh_bf16_kernel(
    x: S.Tensor((BATCH_SIZE, OUT_CHANNELS, POOL_OUT_H, POOL_OUT_W), S.bf16),
    out: S.Tensor((BATCH_SIZE, OUT_CHANNELS, 1, 1), S.bf16),
):
    """
    Computes mean over spatial dimensions (dim=2,3), then applies tanh.
    One block per (batch, channel) pair.
    """
    tid = S.thread_id(0)
    bid = S.block_id(0)

    n = bid // OUT_CHANNELS
    c = bid % OUT_CHANNELS

    # Parallel reduction over spatial elements
    spatial_size = POOL_OUT_H * POOL_OUT_W
    acc = S.convert(0.0, S.f32)

    for i in S.range(spatial_size):
        linear = tid + i * THREADS_1D
        if linear < spatial_size:
            h = linear // POOL_OUT_W
            w = linear % POOL_OUT_W
            acc = acc + S.convert(x[n, c, h, w], S.f32)

    # Since we're doing a simple accumulation without shared memory reduction,
    # we need thread 0 to sum everything. For simplicity, have each thread
    # accumulate its portion and write to output (but only thread 0 should write).
    # This is correct but not optimal for large spatial sizes.

    if tid == 0:
        # Sum all contributions from this block
        # For small spatial sizes (128x128 = 16384), we can accumulate directly
        mean_val = acc * S.convert(MEAN_INV_SPATIAL * THREADS_1D, S.f32)

        # Apply tanh
        result = S.tanh(mean_val)
        out[n, c, 0, 0] = S.convert(result, S.bf16)


# Since the above kernel is incorrect (each thread accumulates a subset),
# let's use a proper accumulation approach


@substrate.jit
def mean_tanh_bf16_kernel_v2(
    x: S.Tensor((BATCH_SIZE, OUT_CHANNELS, POOL_OUT_H, POOL_OUT_W), S.bf16),
    out: S.Tensor((BATCH_SIZE, OUT_CHANNELS, 1, 1), S.bf16),
):
    """
    Computes mean over spatial dimensions (dim=2,3), then applies tanh.
    One thread per (batch, channel) pair, iterates over all spatial positions.
    """
    bid = S.block_id(0)
    tid = S.thread_id(0)

    # Only thread 0 does work
    if tid == 0:
        n = bid // OUT_CHANNELS
        c = bid % OUT_CHANNELS

        acc = S.convert(0.0, S.f32)

        for h in S.range(POOL_OUT_H):
            for w in S.range(POOL_OUT_W):
                acc = acc + S.convert(x[n, c, h, w], S.f32)

        mean_val = acc * S.convert(MEAN_INV_SPATIAL, S.f32)
        result = S.tanh(mean_val)
        out[n, c, 0, 0] = S.convert(result, S.bf16)


# ============================================================================
# Host wrapper functions
# ============================================================================


def _launch_conv_transpose2d_bf16(
    x: torch.Tensor, w: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    out = torch.empty(
        (BATCH_SIZE, OUT_CHANNELS, CONV_OUT_H, CONV_OUT_W),
        device=x.device,
        dtype=torch.bfloat16,
    )
    conv_transpose2d_bf16_kernel[
        lambda: ((BATCH_SIZE * OUT_CHANNELS, 1, 1), (THREADS_1D, 1, 1))
    ](x, w, b, out)
    return out


def _launch_maxpool2d_bf16(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty(
        (BATCH_SIZE, OUT_CHANNELS, POOL_OUT_H, POOL_OUT_W),
        device=x.device,
        dtype=torch.bfloat16,
    )
    grid_x = (POOL_OUT_W + BLOCK_X - 1) // BLOCK_X
    grid_y = (POOL_OUT_H + BLOCK_Y - 1) // BLOCK_Y
    grid_z = BATCH_SIZE * OUT_CHANNELS
    maxpool2d_bf16_kernel[lambda: ((grid_x, grid_y, grid_z), (BLOCK_X, BLOCK_Y, 1))](x, out)
    return out


def _launch_hardtanh_bf16(x: torch.Tensor) -> torch.Tensor:
    x_flat = x.contiguous().view(-1)
    y = torch.empty_like(x_flat)
    n = x_flat.numel()
    if n > 0:
        grid = (n + THREADS_1D - 1) // THREADS_1D
        hardtanh_bf16_kernel[lambda: ((grid, 1, 1), (THREADS_1D, 1, 1))](x_flat, y, n)
    return y.view_as(x)


def _launch_mean_tanh_bf16(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty(
        (BATCH_SIZE, OUT_CHANNELS, 1, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    mean_tanh_bf16_kernel_v2[
        lambda: ((BATCH_SIZE * OUT_CHANNELS, 1, 1), (1, 1, 1))
    ](x, out)
    return out


# ============================================================================
# ModelNew
# ============================================================================


class ModelNew(nn.Module):
    """
    Optimized model using Substrate kernels:
    ConvTranspose2d -> MaxPool2d -> Hardtanh -> Mean -> Tanh
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        maxpool_kernel_size,
        maxpool_stride,
        hardtanh_min,
        hardtanh_max,
    ):
        super(ModelNew, self).__init__()
        # Store parameters for validation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.maxpool_kernel_size = maxpool_kernel_size
        self.maxpool_stride = maxpool_stride
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max

        # Create a PyTorch ConvTranspose2d to hold the weights
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Validate input shape
        if tuple(x.shape) != (BATCH_SIZE, IN_CHANNELS, IN_H, IN_W):
            raise NotImplementedError(
                f"ModelNew supports input shape {(BATCH_SIZE, IN_CHANNELS, IN_H, IN_W)}, got {tuple(x.shape)}"
            )

        # Move to GPU if needed
        orig_device = x.device
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/HIP device is required for Substrate kernels.")
        if not x.is_cuda:
            x = x.cuda()

        # Convert to bfloat16 if needed
        input_dtype = x.dtype
        if input_dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)

        # Ensure contiguous
        x = x.contiguous()

        # Get weights and bias
        w = self.conv_transpose.weight
        b = self.conv_transpose.bias
        if b is None:
            b = torch.zeros((OUT_CHANNELS,), device=w.device, dtype=w.dtype)

        # Move weights to GPU and convert dtype
        if not w.is_cuda:
            w = w.cuda()
        if not b.is_cuda:
            b = b.cuda()
        if w.dtype != torch.bfloat16:
            w = w.to(torch.bfloat16)
        if b.dtype != torch.bfloat16:
            b = b.to(torch.bfloat16)

        w = w.contiguous()
        b = b.contiguous()

        # Step 1: ConvTranspose2d
        x = _launch_conv_transpose2d_bf16(x, w, b)

        # Step 2: MaxPool2d
        x = _launch_maxpool2d_bf16(x)

        # Step 3: Hardtanh (clamp to [-1, 1])
        x = _launch_hardtanh_bf16(x)

        # Step 4 & 5: Mean over spatial dims, then Tanh (fused)
        x = _launch_mean_tanh_bf16(x)

        # Convert back to original dtype if needed
        if input_dtype != torch.bfloat16:
            x = x.to(input_dtype)

        # Move back to original device if needed
        if orig_device.type != "cuda":
            x = x.to(orig_device)

        return x


# ============================================================================
# get_inputs / get_init_inputs (required by KernelBench)
# ============================================================================

batch_size = BATCH_SIZE
in_channels = IN_CHANNELS
out_channels = OUT_CHANNELS
height = IN_H
width = IN_W
kernel_size = KERNEL_SIZE
stride = STRIDE
padding = PADDING
maxpool_kernel_size = MAXPOOL_KERNEL
maxpool_stride = MAXPOOL_STRIDE
hardtanh_min = HARDTANH_MIN
hardtanh_max = HARDTANH_MAX


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        maxpool_kernel_size,
        maxpool_stride,
        hardtanh_min,
        hardtanh_max,
    ]
