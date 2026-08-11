import torch
import torch.nn as nn
import substrate
import substrate.language as S

# Fixed problem shape from target model/get_inputs.
BATCH_SIZE = 16
IN_CHANNELS = 64
OUT_CHANNELS = 128
IN_H = 512
IN_W = 512
K_H = 3
K_W = 3
STRIDE = 1
PADDING = 0

# ConvTranspose2d output size: H_out = (H_in - 1) * stride + kernel_size - 2*padding
OUT_H = (IN_H - 1) * STRIDE + K_H - 2 * PADDING  # 514
OUT_W = (IN_W - 1) * STRIDE + K_W - 2 * PADDING  # 514

THREADS = 256
WEIGHT_ELEMS = IN_CHANNELS * K_H * K_W
WEIGHT_LOAD_ITERS = (WEIGHT_ELEMS + THREADS - 1) // THREADS
SPATIAL_ELEMS = OUT_H * OUT_W
SPATIAL_TILES = (SPATIAL_ELEMS + THREADS - 1) // THREADS

# After global avg pool: (BATCH_SIZE, OUT_CHANNELS, 1, 1)
POOL_SIZE = OUT_H * OUT_W
INV_POOL_SIZE = 1.0 / POOL_SIZE

LOG2E = 1.4426950408889634


@substrate.jit
def conv_transpose2d_bf16_kernel(
    x: S.Tensor((BATCH_SIZE, IN_CHANNELS, IN_H, IN_W), S.bf16),
    w: S.Tensor((IN_CHANNELS, OUT_CHANNELS, K_H, K_W), S.bf16),
    conv_b: S.Tensor((OUT_CHANNELS,), S.bf16),
    out: S.Tensor((BATCH_SIZE, OUT_CHANNELS, OUT_H, OUT_W), S.bf16),
):
    tid = S.thread_id(0)
    bid = S.block_id(0)

    n = bid // OUT_CHANNELS
    oc = bid % OUT_CHANNELS

    s_w = S.make_shared((WEIGHT_ELEMS,), S.bf16)

    for it in S.range(WEIGHT_LOAD_ITERS):
        w_flat = it * THREADS + tid
        if w_flat < WEIGHT_ELEMS:
            ic = w_flat // (K_H * K_W)
            rem = w_flat % (K_H * K_W)
            kh = rem // K_W
            kw = rem % K_W
            s_w[w_flat] = w[ic, oc, kh, kw]

    S.syncthreads()

    for t in S.range(SPATIAL_TILES):
        pos = t * THREADS + tid
        if pos < SPATIAL_ELEMS:
            oh = pos // OUT_W
            ow = pos % OUT_W

            acc = S.convert(0.0, S.f32)

            for kh in S.range(K_H):
                ih = oh - kh
                if ih >= 0 and ih < IN_H:
                    for kw in S.range(K_W):
                        iw = ow - kw
                        if iw >= 0 and iw < IN_W:
                            for ic in S.range(IN_CHANNELS):
                                wf = ic * (K_H * K_W) + kh * K_W + kw
                                xv = S.convert(x[n, ic, ih, iw], S.f32)
                                wv = S.convert(s_w[wf], S.f32)
                                acc = acc + xv * wv

            # Add conv_transpose internal bias
            acc = acc + S.convert(conv_b[oc], S.f32)
            out[n, oc, oh, ow] = S.convert(acc, S.bf16)


@substrate.jit
def global_avg_pool_bf16_kernel(
    x: S.Tensor((BATCH_SIZE, OUT_CHANNELS, OUT_H, OUT_W), S.bf16),
    out: S.Tensor((BATCH_SIZE, OUT_CHANNELS, 1, 1), S.bf16),
):
    bid = S.block_id(0)

    n = bid // OUT_CHANNELS
    oc = bid % OUT_CHANNELS

    acc = S.convert(0.0, S.f32)

    for i in S.range(POOL_SIZE):
        h = i // OUT_W
        w = i % OUT_W
        acc = acc + S.convert(x[n, oc, h, w], S.f32)

    inv = S.convert(INV_POOL_SIZE, S.f32)
    out[n, oc, 0, 0] = S.convert(acc * inv, S.bf16)


@substrate.jit
def add_bias_bf16_kernel(
    x: S.Tensor((BATCH_SIZE, OUT_CHANNELS, 1, 1), S.bf16),
    model_bias: S.Tensor((OUT_CHANNELS,), S.bf16),
    out: S.Tensor((BATCH_SIZE, OUT_CHANNELS, 1, 1), S.bf16),
):
    bid = S.block_id(0)

    n = bid // OUT_CHANNELS
    oc = bid % OUT_CHANNELS

    out[n, oc, 0, 0] = S.convert(
        S.convert(x[n, oc, 0, 0], S.f32) + S.convert(model_bias[oc], S.f32), S.bf16
    )


@substrate.jit
def logsumexp_bf16_kernel(
    x: S.Tensor((BATCH_SIZE, OUT_CHANNELS, 1, 1), S.bf16),
    out: S.Tensor((BATCH_SIZE, 1, 1, 1), S.bf16),
):
    bid = S.block_id(0)
    n = bid

    max_val = S.convert(-3.402823466e38, S.f32)
    for c in S.range(OUT_CHANNELS):
        v = S.convert(x[n, c, 0, 0], S.f32)
        if v > max_val:
            max_val = v

    sum_exp = S.convert(0.0, S.f32)
    log2e = S.convert(LOG2E, S.f32)
    for c in S.range(OUT_CHANNELS):
        v = S.convert(x[n, c, 0, 0], S.f32)
        diff = v - max_val
        sum_exp = sum_exp + S.exp2(diff * log2e)

    log_sum = S.log(sum_exp)
    result = max_val + log_sum
    out[n, 0, 0, 0] = S.convert(result, S.bf16)


@substrate.jit
def sum_spatial_bf16_kernel(
    x: S.Tensor((BATCH_SIZE, 1, 1, 1), S.bf16),
    out: S.Tensor((BATCH_SIZE, 1), S.bf16),
):
    bid = S.block_id(0)
    n = bid

    out[n, 0] = x[n, 0, 0, 0]


@substrate.jit
def scale_bf16_kernel(
    x: S.Tensor((BATCH_SIZE, 1), S.bf16),
    out: S.Tensor((BATCH_SIZE, 1), S.bf16),
):
    tid = S.thread_id(0)
    idx = S.block_id(0) * S.block_dim(0) + tid

    if idx < BATCH_SIZE:
        scale = S.convert(10.0, S.f32)
        v = S.convert(x[idx, 0], S.f32)
        out[idx, 0] = S.convert(v * scale, S.bf16)


def substrate_conv_transpose2d(x: torch.Tensor, weight: torch.Tensor, conv_bias: torch.Tensor) -> torch.Tensor:
    out = torch.empty((BATCH_SIZE, OUT_CHANNELS, OUT_H, OUT_W), device=x.device, dtype=torch.bfloat16)
    conv_transpose2d_bf16_kernel[
        lambda: ((BATCH_SIZE * OUT_CHANNELS, 1, 1), (THREADS, 1, 1))
    ](x, weight, conv_bias, out)
    return out


def substrate_global_avg_pool(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty((BATCH_SIZE, OUT_CHANNELS, 1, 1), device=x.device, dtype=torch.bfloat16)
    global_avg_pool_bf16_kernel[
        lambda: ((BATCH_SIZE * OUT_CHANNELS, 1, 1), (1, 1, 1))
    ](x, out)
    return out


def substrate_add_bias(x: torch.Tensor, model_bias: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    add_bias_bf16_kernel[
        lambda: ((BATCH_SIZE * OUT_CHANNELS, 1, 1), (1, 1, 1))
    ](x, model_bias, out)
    return out


def substrate_logsumexp(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty((BATCH_SIZE, 1, 1, 1), device=x.device, dtype=torch.bfloat16)
    logsumexp_bf16_kernel[
        lambda: ((BATCH_SIZE, 1, 1), (1, 1, 1))
    ](x, out)
    return out


def substrate_sum_spatial(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty((BATCH_SIZE, 1), device=x.device, dtype=torch.bfloat16)
    sum_spatial_bf16_kernel[
        lambda: ((BATCH_SIZE, 1, 1), (1, 1, 1))
    ](x, out)
    return out


def substrate_scale(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    blocks = (BATCH_SIZE + THREADS - 1) // THREADS
    scale_bf16_kernel[
        lambda: ((blocks, 1, 1), (THREADS, 1, 1))
    ](x, out)
    return out


class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, global average pooling, adds a bias,
    applies log-sum-exp, sum, and multiplication using Substrate GPU kernels.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        # Keep original layer to preserve weight initialization
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        if not x.is_cuda:
            x = x.cuda()

        # Get weights from original layer
        weight = self.conv_transpose.weight
        conv_bias = self.conv_transpose.bias
        if conv_bias is None:
            conv_bias = torch.zeros((OUT_CHANNELS,), device=weight.device, dtype=weight.dtype)
        model_bias = self.bias.squeeze()

        # Ensure BF16
        x = x.to(torch.bfloat16)
        weight = weight.to(torch.bfloat16)
        conv_bias = conv_bias.to(torch.bfloat16)
        model_bias = model_bias.to(torch.bfloat16)

        # Check shapes
        if tuple(x.shape) != (BATCH_SIZE, IN_CHANNELS, IN_H, IN_W):
            raise NotImplementedError(
                f"Expected input shape {(BATCH_SIZE, IN_CHANNELS, IN_H, IN_W)}, got {tuple(x.shape)}"
            )

        x = x.contiguous()
        weight = weight.contiguous()
        conv_bias = conv_bias.contiguous()
        model_bias = model_bias.contiguous()

        # ConvTranspose2d (with its internal bias)
        x = substrate_conv_transpose2d(x, weight, conv_bias)

        # Global average pooling
        x = substrate_global_avg_pool(x)

        # Add model bias (separate from conv_transpose bias)
        x = substrate_add_bias(x, model_bias)

        # Log-sum-exp
        x = substrate_logsumexp(x)

        # Sum over spatial dims
        x = substrate_sum_spatial(x)

        # Scale by 10.0
        x = substrate_scale(x)

        return x


batch_size = 16
in_channels = 64
out_channels = 128
height = width = 512
kernel_size = 3
bias_shape = (out_channels, 1, 1)


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]
