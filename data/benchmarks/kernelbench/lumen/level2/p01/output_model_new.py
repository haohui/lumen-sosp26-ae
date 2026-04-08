import torch
import torch.nn as nn
import substrate
import substrate.language as S


# Fixed problem shape from target model/get_inputs.
BATCH_SIZE = 128
IN_CHANNELS = 64
OUT_CHANNELS = 128
IN_H = 128
IN_W = 128
K_H = 3
K_W = 3
STRIDE_H = 1
STRIDE_W = 1
PAD_H = 0
PAD_W = 0

OUT_H = (IN_H + 2 * PAD_H - K_H) // STRIDE_H + 1  # 126
OUT_W = (IN_W + 2 * PAD_W - K_W) // STRIDE_W + 1  # 126

THREADS_PER_BLOCK = 256
WEIGHT_ELEMS = IN_CHANNELS * K_H * K_W  # 576
WEIGHT_LOAD_ITERS = (WEIGHT_ELEMS + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
SPATIAL_ELEMS = OUT_H * OUT_W  # 15876
SPATIAL_TILES = (SPATIAL_ELEMS + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK


@substrate.jit
def conv2d_relu_bias_bf16_kernel(
    x: S.Tensor((BATCH_SIZE, IN_CHANNELS, IN_H, IN_W), S.bf16),
    w: S.Tensor((OUT_CHANNELS, IN_CHANNELS, K_H, K_W), S.bf16),
    conv_bias: S.Tensor((OUT_CHANNELS,), S.bf16),
    add_bias: S.Tensor((OUT_CHANNELS, 1, 1), S.bf16),
    out: S.Tensor((BATCH_SIZE, OUT_CHANNELS, OUT_H, OUT_W), S.bf16),
):
    tid = S.thread_id(0)
    bid = S.block_id(0)

    n = bid // OUT_CHANNELS
    oc = bid % OUT_CHANNELS

    # Shared memory tile for one output channel's kernel weights.
    s_w = S.make_shared((WEIGHT_ELEMS,), S.bf16)

    for it in S.range(WEIGHT_LOAD_ITERS):
        w_flat = it * THREADS_PER_BLOCK + tid
        if w_flat < WEIGHT_ELEMS:
            ic = w_flat // (K_H * K_W)
            rem = w_flat % (K_H * K_W)
            kh = rem // K_W
            kw = rem % K_W
            s_w[w_flat] = w[oc, ic, kh, kw]

    S.syncthreads()

    for t in S.range(SPATIAL_TILES):
        pos = t * THREADS_PER_BLOCK + tid
        if pos < SPATIAL_ELEMS:
            oh = pos // OUT_W
            ow = pos % OUT_W

            # Start with conv bias, accumulate in f32
            acc = S.convert(conv_bias[oc], S.f32)

            for ic in S.range(IN_CHANNELS):
                for kh in S.range(K_H):
                    ih = oh * STRIDE_H + kh
                    if ih >= PAD_H and ih < IN_H + PAD_H:
                        ih_actual = ih - PAD_H
                        for kw in S.range(K_W):
                            iw = ow * STRIDE_W + kw
                            if iw >= PAD_W and iw < IN_W + PAD_W:
                                iw_actual = iw - PAD_W
                                wf = ic * (K_H * K_W) + kh * K_W + kw
                                xv = S.convert(x[n, ic, ih_actual, iw_actual], S.f32)
                                wv = S.convert(s_w[wf], S.f32)
                                acc = acc + xv * wv

            # Apply ReLU
            zero = S.convert(0.0, S.f32)
            if acc < zero:
                acc = zero

            # Add the additional bias
            bias_val = S.convert(add_bias[oc, 0, 0], S.f32)
            acc = acc + bias_val

            out[n, oc, oh, ow] = S.convert(acc, S.bf16)


def _launch_conv_relu_bias_bf16(
    x: torch.Tensor, w: torch.Tensor, conv_bias: torch.Tensor, add_bias: torch.Tensor
) -> torch.Tensor:
    out = torch.empty(
        (BATCH_SIZE, OUT_CHANNELS, OUT_H, OUT_W),
        device=x.device,
        dtype=torch.bfloat16,
    )
    conv2d_relu_bias_bf16_kernel[
        lambda: ((BATCH_SIZE * OUT_CHANNELS, 1, 1), (THREADS_PER_BLOCK, 1, 1))
    ](x, w, conv_bias, add_bias, out)
    return out


class ModelNew(nn.Module):
    """
    Simple model that performs a convolution, applies ReLU, and adds a bias term.
    Optimized with Substrate GPU kernels for BF16 precision.
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        original_device = x.device
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/HIP device is required for Substrate kernels.")

        if not x.is_cuda:
            x = x.cuda()

        # Check shape constraints
        if x.shape != (BATCH_SIZE, IN_CHANNELS, IN_H, IN_W):
            raise NotImplementedError(
                f"ModelNew currently supports input shape {(BATCH_SIZE, IN_CHANNELS, IN_H, IN_W)}, got {tuple(x.shape)}"
            )

        w = self.conv.weight
        conv_b = self.conv.bias
        if conv_b is None:
            conv_b = torch.zeros((OUT_CHANNELS,), device=w.device, dtype=w.dtype)
        add_b = self.bias

        # Ensure all tensors are on the same device and contiguous
        if w.device != x.device:
            w = w.to(device=x.device)
        if conv_b.device != x.device:
            conv_b = conv_b.to(device=x.device)
        if add_b.device != x.device:
            add_b = add_b.to(device=x.device)

        x = x.contiguous()
        w = w.contiguous()
        conv_b = conv_b.contiguous()
        add_b = add_b.contiguous()

        # Convert to BF16 for optimized kernel
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)
        if w.dtype != torch.bfloat16:
            w = w.to(torch.bfloat16)
        if conv_b.dtype != torch.bfloat16:
            conv_b = conv_b.to(torch.bfloat16)
        if add_b.dtype != torch.bfloat16:
            add_b = add_b.to(torch.bfloat16)

        out = _launch_conv_relu_bias_bf16(x, w, conv_b, add_b)

        if original_device.type != "cuda":
            out = out.to(original_device)
        return out


batch_size = 128
in_channels = 64
out_channels = 128
height = width = 128
kernel_size = 3
bias_shape = (out_channels, 1, 1)


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]
