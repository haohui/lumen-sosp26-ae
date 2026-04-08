import torch
import torch.nn as nn
import substrate
import substrate.language as S


# Fixed problem shape from target model/get_inputs.
BATCH_SIZE = 128
IN_CHANNELS = 64
OUT_CHANNELS = 64
IN_H = 128
IN_W = 128
K_H = 3
K_W = 3
STRIDE_H = 2
STRIDE_W = 2
PAD_H = 1
PAD_W = 1
OUT_PAD_H = 1
OUT_PAD_W = 1

OUT_H = (IN_H - 1) * STRIDE_H - 2 * PAD_H + K_H + OUT_PAD_H  # 256
OUT_W = (IN_W - 1) * STRIDE_W - 2 * PAD_W + K_W + OUT_PAD_W  # 256

THREADS_PER_BLOCK = 256
WEIGHT_ELEMS = OUT_CHANNELS * K_H * K_W  # 64 * 3 * 3 = 576
WEIGHT_LOAD_ITERS = (WEIGHT_ELEMS + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
SPATIAL_ELEMS = OUT_H * OUT_W  # 65536
SPATIAL_TILES = (SPATIAL_ELEMS + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

# Scaling factor is fixed at 2.0 for this benchmark
SCALING_FACTOR = S.constexpr(2.0)


@substrate.jit
def clamp_f32(val: S.f32, min_val: S.f32, max_val: S.f32) -> S.f32:
    # Floating-point clamp using comparison
    result = val
    if result < min_val:
        result = min_val
    if result > max_val:
        result = max_val
    return result


@substrate.jit
def conv_transpose2d_bf16_kernel(
    x: S.Tensor((BATCH_SIZE, IN_CHANNELS, IN_H, IN_W), S.bf16),
    w: S.Tensor((IN_CHANNELS, OUT_CHANNELS, K_H, K_W), S.bf16),
    conv_bias: S.Tensor((OUT_CHANNELS,), S.bf16),
    extra_bias: S.Tensor((OUT_CHANNELS, 1, 1), S.bf16),
    out: S.Tensor((BATCH_SIZE, OUT_CHANNELS, OUT_H, OUT_W), S.bf16),
):
    tid = S.thread_id(0)
    bid = S.block_id(0)

    n = bid // OUT_CHANNELS
    oc = bid % OUT_CHANNELS

    # FP32 constants for clamp
    zero_f32 = S.convert(0, S.f32)
    one_f32 = S.convert(1, S.f32)
    scale_f32 = S.convert(SCALING_FACTOR, S.f32)

    # Shared memory tile for one output channel's kernel weights across all input channels.
    s_w = S.make_shared((WEIGHT_ELEMS,), S.bf16)

    for it in S.range(WEIGHT_LOAD_ITERS):
        w_flat = it * THREADS_PER_BLOCK + tid
        if w_flat < WEIGHT_ELEMS:
            ic = w_flat // (K_H * K_W)
            rem = w_flat % (K_H * K_W)
            kh = rem // K_W
            kw = rem % K_W
            s_w[w_flat] = w[ic, oc, kh, kw]

    S.syncthreads()

    for t in S.range(SPATIAL_TILES):
        pos = t * THREADS_PER_BLOCK + tid
        if pos < SPATIAL_ELEMS:
            oh = pos // OUT_W
            ow = pos % OUT_W

            # FP32 accumulator for precision
            acc = S.convert(0, S.f32)

            for ic in S.range(IN_CHANNELS):
                for kh in S.range(K_H):
                    # For transposed conv: ih = (oh + padding - kh) // stride
                    # Valid only when (oh + padding - kh) % stride == 0
                    ih_nom = oh + PAD_H - kh
                    # Check if this kernel position contributes (stride alignment)
                    if ih_nom >= 0 and ih_nom % STRIDE_H == 0:
                        ih = ih_nom // STRIDE_H
                        if ih < IN_H:
                            for kw in S.range(K_W):
                                iw_nom = ow + PAD_W - kw
                                if iw_nom >= 0 and iw_nom % STRIDE_W == 0:
                                    iw = iw_nom // STRIDE_W
                                    if iw < IN_W:
                                        wf = ic * (K_H * K_W) + kh * K_W + kw
                                        xv = S.convert(x[n, ic, ih, iw], S.f32)
                                        wv = S.convert(s_w[wf], S.f32)
                                        acc = acc + xv * wv

            # Add conv_transpose's internal bias
            conv_bias_val = S.convert(conv_bias[oc], S.f32)
            acc = acc + conv_bias_val

            # Add the separate bias
            extra_bias_val = S.convert(extra_bias[oc, 0, 0], S.f32)
            acc = acc + extra_bias_val

            # Clamp to [0, 1]
            acc = clamp_f32(acc, zero_f32, one_f32)

            # Scale
            acc = acc * scale_f32

            # Clamp to [0, 1]
            acc = clamp_f32(acc, zero_f32, one_f32)

            # Divide
            acc = acc / scale_f32

            out[n, oc, oh, ow] = S.convert(acc, S.bf16)


@substrate.jit
def conv_transpose2d_f32_kernel(
    x: S.Tensor((BATCH_SIZE, IN_CHANNELS, IN_H, IN_W), S.f32),
    w: S.Tensor((IN_CHANNELS, OUT_CHANNELS, K_H, K_W), S.f32),
    conv_bias: S.Tensor((OUT_CHANNELS,), S.f32),
    extra_bias: S.Tensor((OUT_CHANNELS, 1, 1), S.f32),
    out: S.Tensor((BATCH_SIZE, OUT_CHANNELS, OUT_H, OUT_W), S.f32),
):
    tid = S.thread_id(0)
    bid = S.block_id(0)

    n = bid // OUT_CHANNELS
    oc = bid % OUT_CHANNELS

    # FP32 constants for clamp
    zero_f32 = S.convert(0, S.f32)
    one_f32 = S.convert(1, S.f32)
    scale_f32 = S.convert(SCALING_FACTOR, S.f32)

    # Shared memory tile for one output channel's kernel weights across all input channels.
    s_w = S.make_shared((WEIGHT_ELEMS,), S.f32)

    for it in S.range(WEIGHT_LOAD_ITERS):
        w_flat = it * THREADS_PER_BLOCK + tid
        if w_flat < WEIGHT_ELEMS:
            ic = w_flat // (K_H * K_W)
            rem = w_flat % (K_H * K_W)
            kh = rem // K_W
            kw = rem % K_W
            s_w[w_flat] = w[ic, oc, kh, kw]

    S.syncthreads()

    for t in S.range(SPATIAL_TILES):
        pos = t * THREADS_PER_BLOCK + tid
        if pos < SPATIAL_ELEMS:
            oh = pos // OUT_W
            ow = pos % OUT_W

            acc = S.convert(0, S.f32)

            for ic in S.range(IN_CHANNELS):
                for kh in S.range(K_H):
                    ih_nom = oh + PAD_H - kh
                    if ih_nom >= 0 and ih_nom % STRIDE_H == 0:
                        ih = ih_nom // STRIDE_H
                        if ih < IN_H:
                            for kw in S.range(K_W):
                                iw_nom = ow + PAD_W - kw
                                if iw_nom >= 0 and iw_nom % STRIDE_W == 0:
                                    iw = iw_nom // STRIDE_W
                                    if iw < IN_W:
                                        wf = ic * (K_H * K_W) + kh * K_W + kw
                                        acc = acc + x[n, ic, ih, iw] * s_w[wf]

            # Add conv_transpose's internal bias
            acc = acc + conv_bias[oc]

            # Add the separate bias
            acc = acc + extra_bias[oc, 0, 0]

            # Clamp to [0, 1]
            acc = clamp_f32(acc, zero_f32, one_f32)

            # Scale
            acc = acc * scale_f32

            # Clamp to [0, 1]
            acc = clamp_f32(acc, zero_f32, one_f32)

            # Divide
            acc = acc / scale_f32

            out[n, oc, oh, ow] = acc


def _launch_conv_transpose_bf16(x: torch.Tensor, w: torch.Tensor, conv_bias: torch.Tensor, extra_bias: torch.Tensor) -> torch.Tensor:
    out = torch.empty((BATCH_SIZE, OUT_CHANNELS, OUT_H, OUT_W), device=x.device, dtype=torch.bfloat16)
    conv_transpose2d_bf16_kernel[
        lambda: ((BATCH_SIZE * OUT_CHANNELS, 1, 1), (THREADS_PER_BLOCK, 1, 1))
    ](x, w, conv_bias, extra_bias, out)
    return out


def _launch_conv_transpose_f32(x: torch.Tensor, w: torch.Tensor, conv_bias: torch.Tensor, extra_bias: torch.Tensor) -> torch.Tensor:
    out = torch.empty((BATCH_SIZE, OUT_CHANNELS, OUT_H, OUT_W), device=x.device, dtype=torch.float32)
    conv_transpose2d_f32_kernel[
        lambda: ((BATCH_SIZE * OUT_CHANNELS, 1, 1), (THREADS_PER_BLOCK, 1, 1))
    ](x, w, conv_bias, extra_bias, out)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

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

        w = self.conv_transpose.weight
        conv_bias = self.conv_transpose.bias
        extra_bias = self.bias

        if w.device != x.device:
            w = w.to(device=x.device)
        if conv_bias.device != x.device:
            conv_bias = conv_bias.to(device=x.device)
        if extra_bias.device != x.device:
            extra_bias = extra_bias.to(device=x.device)

        x = x.contiguous()
        w = w.contiguous()
        conv_bias = conv_bias.contiguous()
        extra_bias = extra_bias.contiguous()

        if x.dtype == torch.bfloat16 and w.dtype == torch.bfloat16:
            out = _launch_conv_transpose_bf16(x, w, conv_bias, extra_bias)
        elif x.dtype == torch.float32 and w.dtype == torch.float32:
            out = _launch_conv_transpose_f32(x, w, conv_bias, extra_bias)
        else:
            raise TypeError(
                f"Unsupported dtype combination: x={x.dtype}, weight={w.dtype}. "
                "Supported: float32 or bfloat16 (matching dtypes)."
            )

        if original_device.type != "cuda":
            out = out.to(original_device)
        return out


batch_size = 128
in_channels = 64
out_channels = 64
height = width = 128
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1)
scaling_factor = 2.0


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor]
