import torch
import torch.nn as nn
import substrate
import substrate.language as S


# Fixed problem shape from target model/get_inputs.
BATCH_SIZE = 64
IN_CHANNELS = 3
OUT_CHANNELS = 16
IN_D, IN_H, IN_W = 32, 32, 32
KERNEL_SIZE = 3
STRIDE = 2
PADDING = 1

# ConvTranspose3d output shape: (input - 1) * stride - 2 * padding + kernel_size
OUT_D = (IN_D - 1) * STRIDE - 2 * PADDING + KERNEL_SIZE  # 63
OUT_H = (IN_H - 1) * STRIDE - 2 * PADDING + KERNEL_SIZE  # 63
OUT_W = (IN_W - 1) * STRIDE - 2 * PADDING + KERNEL_SIZE  # 63

# After AvgPool3d (kernel_size=2, stride=2)
POOL1_D, POOL1_H, POOL1_W = OUT_D // 2, OUT_H // 2, OUT_W // 2  # 31
POOL2_D, POOL2_H, POOL2_W = POOL1_D // 2, POOL1_H // 2, POOL1_W // 2  # 15

THREADS_PER_BLOCK = 256


# =============================================================================
# ConvTranspose3d Kernel
# =============================================================================

WEIGHT_ELEMS = IN_CHANNELS * KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE  # 81
WEIGHT_LOAD_ITERS = (WEIGHT_ELEMS + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
SPATIAL_ELEMS_CONV = OUT_D * OUT_H * OUT_W  # 250047
SPATIAL_TILES_CONV = (SPATIAL_ELEMS_CONV + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK


@substrate.jit
def conv_transpose3d_bf16_kernel(
    x: S.Tensor((BATCH_SIZE, IN_CHANNELS, IN_D, IN_H, IN_W), S.bf16),
    w: S.Tensor((IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, KERNEL_SIZE, KERNEL_SIZE), S.bf16),
    b: S.Tensor((OUT_CHANNELS,), S.bf16),
    out: S.Tensor((BATCH_SIZE, OUT_CHANNELS, OUT_D, OUT_H, OUT_W), S.bf16),
):
    tid = S.thread_id(0)
    bid = S.block_id(0)

    n = bid // OUT_CHANNELS
    oc = bid % OUT_CHANNELS

    s_w = S.make_shared((WEIGHT_ELEMS,), S.bf16)

    for it in S.range(WEIGHT_LOAD_ITERS):
        w_flat = it * THREADS_PER_BLOCK + tid
        if w_flat < WEIGHT_ELEMS:
            ic = w_flat // (KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE)
            rem = w_flat % (KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE)
            kd = rem // (KERNEL_SIZE * KERNEL_SIZE)
            rem2 = rem % (KERNEL_SIZE * KERNEL_SIZE)
            kh = rem2 // KERNEL_SIZE
            kw = rem2 % KERNEL_SIZE
            s_w[w_flat] = w[ic, oc, kd, kh, kw]

    S.syncthreads()

    for t in S.range(SPATIAL_TILES_CONV):
        pos = t * THREADS_PER_BLOCK + tid
        if pos < SPATIAL_ELEMS_CONV:
            od = pos // (OUT_H * OUT_W)
            rem = pos % (OUT_H * OUT_W)
            oh = rem // OUT_W
            ow = rem % OUT_W

            acc = S.convert(b[oc], S.f32)

            for ic in S.range(IN_CHANNELS):
                for kd in S.range(KERNEL_SIZE):
                    for kh in S.range(KERNEL_SIZE):
                        for kw in S.range(KERNEL_SIZE):
                            id_nom = od + PADDING - kd
                            ih_nom = oh + PADDING - kh
                            iw_nom = ow + PADDING - kw

                            id_valid = (id_nom >= 0) and (id_nom < IN_D * STRIDE) and ((id_nom % STRIDE) == 0)
                            ih_valid = (ih_nom >= 0) and (ih_nom < IN_H * STRIDE) and ((ih_nom % STRIDE) == 0)
                            iw_valid = (iw_nom >= 0) and (iw_nom < IN_W * STRIDE) and ((iw_nom % STRIDE) == 0)

                            if id_valid and ih_valid and iw_valid:
                                id_val = id_nom // STRIDE
                                ih_val = ih_nom // STRIDE
                                iw_val = iw_nom // STRIDE
                                wf = ic * (KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE) + kd * (KERNEL_SIZE * KERNEL_SIZE) + kh * KERNEL_SIZE + kw
                                xv = S.convert(x[n, ic, id_val, ih_val, iw_val], S.f32)
                                wv = S.convert(s_w[wf], S.f32)
                                acc = acc + xv * wv

            out[n, oc, od, oh, ow] = S.convert(acc, S.bf16)


# =============================================================================
# BatchNorm3d Kernels (Training Mode)
# =============================================================================

# For training mode BatchNorm, we need to compute batch statistics
# and then normalize. This is done in two passes:
# 1. Compute mean and variance for each channel
# 2. Apply normalization

BN_SPATIAL_ELEMS = OUT_D * OUT_H * OUT_W  # 250047
BN_SPATIAL_TILES = (BN_SPATIAL_ELEMS + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
BN_TOTAL_ELEMS = BATCH_SIZE * BN_SPATIAL_ELEMS  # 64 * 250047 = 16,003,008
BN_REDUCE_TILES = (BN_TOTAL_ELEMS + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK


@substrate.jit
def batchnorm3d_mean_kernel(
    x: S.Tensor((BATCH_SIZE, OUT_CHANNELS, OUT_D, OUT_H, OUT_W), S.bf16),
    mean: S.Tensor((OUT_CHANNELS,), S.f32),
):
    """Compute mean for each channel across batch and spatial dimensions."""
    tid = S.thread_id(0)
    c = S.block_id(0)

    acc = S.convert(0.0, S.f32)
    for t in S.range(BN_REDUCE_TILES):
        pos = t * THREADS_PER_BLOCK + tid
        if pos < BN_TOTAL_ELEMS:
            n = pos // BN_SPATIAL_ELEMS
            rem = pos % BN_SPATIAL_ELEMS
            d = rem // (OUT_H * OUT_W)
            rem2 = rem % (OUT_H * OUT_W)
            h = rem2 // OUT_W
            w = rem2 % OUT_W

            acc = acc + S.convert(x[n, c, d, h, w], S.f32)

    # Store partial sum in shared memory
    s_partial = S.make_shared((THREADS_PER_BLOCK,), S.f32)
    s_partial[tid] = acc
    S.syncthreads()

    # Parallel reduction
    half = S.convert(THREADS_PER_BLOCK // 2, S.i32)
    for s in S.range(8):  # log2(256) = 8
        if tid < half:
            s_partial[tid] = s_partial[tid] + s_partial[tid + half]
        S.syncthreads()
        half = half // 2
        if half == 0:
            break

    if tid == 0:
        inv_count = S.convert(1.0, S.f32) / S.convert(BN_TOTAL_ELEMS, S.f32)
        mean[c] = s_partial[0] * inv_count


@substrate.jit
def batchnorm3d_var_kernel(
    x: S.Tensor((BATCH_SIZE, OUT_CHANNELS, OUT_D, OUT_H, OUT_W), S.bf16),
    mean: S.Tensor((OUT_CHANNELS,), S.f32),
    var: S.Tensor((OUT_CHANNELS,), S.f32),
):
    """Compute variance for each channel."""
    tid = S.thread_id(0)
    c = S.block_id(0)

    m = mean[c]
    acc = S.convert(0.0, S.f32)

    for t in S.range(BN_REDUCE_TILES):
        pos = t * THREADS_PER_BLOCK + tid
        if pos < BN_TOTAL_ELEMS:
            n = pos // BN_SPATIAL_ELEMS
            rem = pos % BN_SPATIAL_ELEMS
            d = rem // (OUT_H * OUT_W)
            rem2 = rem % (OUT_H * OUT_W)
            h = rem2 // OUT_W
            w = rem2 % OUT_W

            diff = S.convert(x[n, c, d, h, w], S.f32) - m
            acc = acc + diff * diff

    s_partial = S.make_shared((THREADS_PER_BLOCK,), S.f32)
    s_partial[tid] = acc
    S.syncthreads()

    half = S.convert(THREADS_PER_BLOCK // 2, S.i32)
    for s in S.range(8):
        if tid < half:
            s_partial[tid] = s_partial[tid] + s_partial[tid + half]
        S.syncthreads()
        half = half // 2
        if half == 0:
            break

    if tid == 0:
        inv_count = S.convert(1.0, S.f32) / S.convert(BN_TOTAL_ELEMS, S.f32)
        var[c] = s_partial[0] * inv_count


@substrate.jit
def batchnorm3d_apply_kernel(
    x: S.Tensor((BATCH_SIZE, OUT_CHANNELS, OUT_D, OUT_H, OUT_W), S.bf16),
    mean: S.Tensor((OUT_CHANNELS,), S.f32),
    var: S.Tensor((OUT_CHANNELS,), S.f32),
    gamma: S.Tensor((OUT_CHANNELS,), S.bf16),
    beta: S.Tensor((OUT_CHANNELS,), S.bf16),
    out: S.Tensor((BATCH_SIZE, OUT_CHANNELS, OUT_D, OUT_H, OUT_W), S.bf16),
):
    """Apply batch normalization."""
    tid = S.thread_id(0)
    bid = S.block_id(0)

    n = bid // OUT_CHANNELS
    c = bid % OUT_CHANNELS

    m = mean[c]
    v = var[c]
    g = S.convert(gamma[c], S.f32)
    be = S.convert(beta[c], S.f32)

    eps = S.convert(1e-5, S.f32)
    inv_std = S.convert(1.0, S.f32) / S.sqrt(v + eps)
    scale = g * inv_std
    bias = be - m * scale

    for t in S.range(BN_SPATIAL_TILES):
        pos = t * THREADS_PER_BLOCK + tid
        if pos < BN_SPATIAL_ELEMS:
            d = pos // (OUT_H * OUT_W)
            rem = pos % (OUT_H * OUT_W)
            h = rem // OUT_W
            w = rem % OUT_W

            xv = S.convert(x[n, c, d, h, w], S.f32)
            outv = xv * scale + bias
            out[n, c, d, h, w] = S.convert(outv, S.bf16)


# =============================================================================
# AvgPool3d Kernels (static shapes)
# =============================================================================

POOL1_SPATIAL_ELEMS = POOL1_D * POOL1_H * POOL1_W  # 29791
POOL1_SPATIAL_TILES = (POOL1_SPATIAL_ELEMS + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

POOL2_SPATIAL_ELEMS = POOL2_D * POOL2_H * POOL2_W  # 3375
POOL2_SPATIAL_TILES = (POOL2_SPATIAL_ELEMS + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK


@substrate.jit
def avgpool3d_pool1_bf16_kernel(
    x: S.Tensor((BATCH_SIZE, OUT_CHANNELS, OUT_D, OUT_H, OUT_W), S.bf16),
    out: S.Tensor((BATCH_SIZE, OUT_CHANNELS, POOL1_D, POOL1_H, POOL1_W), S.bf16),
):
    tid = S.thread_id(0)
    bid = S.block_id(0)

    n = bid // OUT_CHANNELS
    c = bid % OUT_CHANNELS

    for t in S.range(POOL1_SPATIAL_TILES):
        pos = t * THREADS_PER_BLOCK + tid
        if pos < POOL1_SPATIAL_ELEMS:
            od = pos // (POOL1_H * POOL1_W)
            rem = pos % (POOL1_H * POOL1_W)
            oh = rem // POOL1_W
            ow = rem % POOL1_W

            id_start = od * 2
            ih_start = oh * 2
            iw_start = ow * 2

            acc = S.convert(0.0, S.f32)
            cnt = S.convert(0, S.i32)

            for kd in S.range(2):
                for kh in S.range(2):
                    for kw in S.range(2):
                        id_val = id_start + kd
                        ih_val = ih_start + kh
                        iw_val = iw_start + kw
                        if (id_val < OUT_D) and (ih_val < OUT_H) and (iw_val < OUT_W):
                            acc = acc + S.convert(x[n, c, id_val, ih_val, iw_val], S.f32)
                            cnt = cnt + 1

            avg = acc / S.convert(cnt, S.f32)
            out[n, c, od, oh, ow] = S.convert(avg, S.bf16)


@substrate.jit
def avgpool3d_pool2_bf16_kernel(
    x: S.Tensor((BATCH_SIZE, OUT_CHANNELS, POOL1_D, POOL1_H, POOL1_W), S.bf16),
    out: S.Tensor((BATCH_SIZE, OUT_CHANNELS, POOL2_D, POOL2_H, POOL2_W), S.bf16),
):
    tid = S.thread_id(0)
    bid = S.block_id(0)

    n = bid // OUT_CHANNELS
    c = bid % OUT_CHANNELS

    for t in S.range(POOL2_SPATIAL_TILES):
        pos = t * THREADS_PER_BLOCK + tid
        if pos < POOL2_SPATIAL_ELEMS:
            od = pos // (POOL2_H * POOL2_W)
            rem = pos % (POOL2_H * POOL2_W)
            oh = rem // POOL2_W
            ow = rem % POOL2_W

            id_start = od * 2
            ih_start = oh * 2
            iw_start = ow * 2

            acc = S.convert(0.0, S.f32)
            cnt = S.convert(0, S.i32)

            for kd in S.range(2):
                for kh in S.range(2):
                    for kw in S.range(2):
                        id_val = id_start + kd
                        ih_val = ih_start + kh
                        iw_val = iw_start + kw
                        if (id_val < POOL1_D) and (ih_val < POOL1_H) and (iw_val < POOL1_W):
                            acc = acc + S.convert(x[n, c, id_val, ih_val, iw_val], S.f32)
                            cnt = cnt + 1

            avg = acc / S.convert(cnt, S.f32)
            out[n, c, od, oh, ow] = S.convert(avg, S.bf16)


# =============================================================================
# Host wrappers
# =============================================================================

def launch_conv_transpose3d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    out = torch.empty((BATCH_SIZE, OUT_CHANNELS, OUT_D, OUT_H, OUT_W),
                       device=x.device, dtype=torch.bfloat16)
    grid = BATCH_SIZE * OUT_CHANNELS
    conv_transpose3d_bf16_kernel[
        lambda: ((grid, 1, 1), (THREADS_PER_BLOCK, 1, 1))
    ](x, weight, bias, out)
    return out


def launch_batchnorm3d(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
                       running_mean: torch.Tensor, running_var: torch.Tensor,
                       training: bool = True) -> torch.Tensor:
    """Launch BatchNorm3d kernel. Training mode computes batch statistics."""
    out = torch.empty_like(x)

    if training:
        # Training mode: compute batch statistics
        mean = torch.empty((OUT_CHANNELS,), device=x.device, dtype=torch.float32)
        var = torch.empty((OUT_CHANNELS,), device=x.device, dtype=torch.float32)

        batchnorm3d_mean_kernel[
            lambda: ((OUT_CHANNELS, 1, 1), (THREADS_PER_BLOCK, 1, 1))
        ](x, mean)

        batchnorm3d_var_kernel[
            lambda: ((OUT_CHANNELS, 1, 1), (THREADS_PER_BLOCK, 1, 1))
        ](x, mean, var)

        batchnorm3d_apply_kernel[
            lambda: ((BATCH_SIZE * OUT_CHANNELS, 1, 1), (THREADS_PER_BLOCK, 1, 1))
        ](x, mean, var, gamma, beta, out)
    else:
        # Eval mode: use running statistics
        grid = BATCH_SIZE * OUT_CHANNELS
        # For eval mode, we still use the apply kernel but with running stats
        batchnorm3d_apply_kernel[
            lambda: ((grid, 1, 1), (THREADS_PER_BLOCK, 1, 1))
        ](x, running_mean.float(), running_var.float(), gamma, beta, out)

    return out


def launch_avgpool_pool1(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty((BATCH_SIZE, OUT_CHANNELS, POOL1_D, POOL1_H, POOL1_W),
                       device=x.device, dtype=torch.bfloat16)
    grid = BATCH_SIZE * OUT_CHANNELS
    avgpool3d_pool1_bf16_kernel[
        lambda: ((grid, 1, 1), (THREADS_PER_BLOCK, 1, 1))
    ](x, out)
    return out


def launch_avgpool_pool2(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty((BATCH_SIZE, OUT_CHANNELS, POOL2_D, POOL2_H, POOL2_W),
                       device=x.device, dtype=torch.bfloat16)
    grid = BATCH_SIZE * OUT_CHANNELS
    avgpool3d_pool2_bf16_kernel[
        lambda: ((grid, 1, 1), (THREADS_PER_BLOCK, 1, 1))
    ](x, out)
    return out


class ModelNew(nn.Module):
    """
    Substrate-accelerated model for ConvTranspose3d + BatchNorm3d + AvgPool3d x2.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()

        # Use PyTorch layers for weight initialization
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.batch_norm = nn.BatchNorm3d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if tuple(x.shape) != (BATCH_SIZE, IN_CHANNELS, IN_D, IN_H, IN_W):
            raise NotImplementedError(
                f"ModelNew currently supports input shape {(BATCH_SIZE, IN_CHANNELS, IN_D, IN_H, IN_W)}, "
                f"got {tuple(x.shape)}"
            )

        orig_device = x.device
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/HIP device is required for Substrate kernels.")

        if not x.is_cuda:
            x = x.cuda()

        x = x.to(torch.bfloat16)

        weight = self.conv_transpose.weight.to(x.device).to(torch.bfloat16).contiguous()
        bias = self.conv_transpose.bias
        if bias is None:
            bias = torch.zeros((OUT_CHANNELS,), device=x.device, dtype=torch.bfloat16)
        else:
            bias = bias.to(x.device).to(torch.bfloat16).contiguous()

        gamma = self.batch_norm.weight.to(x.device).to(torch.bfloat16).contiguous()
        beta = self.batch_norm.bias.to(x.device).to(torch.bfloat16).contiguous()
        running_mean = self.batch_norm.running_mean.to(x.device).to(torch.bfloat16).contiguous()
        running_var = self.batch_norm.running_var.to(x.device).to(torch.bfloat16).contiguous()

        x = x.contiguous()

        # ConvTranspose3d
        x = launch_conv_transpose3d(x, weight, bias)

        # BatchNorm3d (use training mode if model is in training mode)
        x = launch_batchnorm3d(x, gamma, beta, running_mean, running_var, training=self.training)

        # AvgPool3d x2
        x = launch_avgpool_pool1(x)
        x = launch_avgpool_pool2(x)

        if orig_device.type != "cuda":
            x = x.to(orig_device)

        return x


# =============================================================================
# Entry points
# =============================================================================

batch_size = BATCH_SIZE
in_channels = IN_CHANNELS
out_channels = OUT_CHANNELS
depth, height, width = IN_D, IN_H, IN_W
kernel_size = KERNEL_SIZE
stride = STRIDE
padding = PADDING
bias_shape = (out_channels, 1, 1, 1)


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, bias_shape]
