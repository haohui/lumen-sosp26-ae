import torch
import torch.nn as nn
import substrate
import substrate.language as S


# Fixed problem shape from target model/get_inputs.
BATCH_SIZE = 64
IN_CHANNELS = 64
OUT_CHANNELS = 128
IN_H = 256
IN_W = 256
K_H = 3
K_W = 3
STRIDE_H = 1
STRIDE_W = 1
PAD_H = 0
PAD_W = 0

OUT_H = (IN_H + 2 * PAD_H - K_H) // STRIDE_H + 1  # 254
OUT_W = (IN_W + 2 * PAD_W - K_W) // STRIDE_W + 1  # 254

THREADS_PER_BLOCK = 256
WEIGHT_ELEMS = IN_CHANNELS * K_H * K_W  # 64 * 9 = 576
WEIGHT_LOAD_ITERS = (WEIGHT_ELEMS + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
SPATIAL_ELEMS = OUT_H * OUT_W  # 64516
SPATIAL_TILES = (SPATIAL_ELEMS + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

SCALE_FACTOR = 2.0


@substrate.jit
def conv2d_3x3_s1_p0_bf16_kernel(
    x: S.Tensor((BATCH_SIZE, IN_CHANNELS, IN_H, IN_W), S.bf16),
    w: S.Tensor((OUT_CHANNELS, IN_CHANNELS, K_H, K_W), S.bf16),
    b: S.Tensor((OUT_CHANNELS,), S.bf16),
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

            acc = S.convert(b[oc], S.f32)

            for ic in S.range(IN_CHANNELS):
                for kh in S.range(K_H):
                    ih = oh * STRIDE_H + kh
                    if ih < IN_H:
                        for kw in S.range(K_W):
                            iw = ow * STRIDE_W + kw
                            if iw < IN_W:
                                wf = ic * (K_H * K_W) + kh * K_W + kw
                                xv = S.convert(x[n, ic, ih, iw], S.f32)
                                wv = S.convert(s_w[wf], S.f32)
                                acc = acc + xv * wv

            out[n, oc, oh, ow] = S.convert(acc, S.bf16)


@substrate.jit
def conv_scale_min_bf16_kernel(
    x: S.Tensor((BATCH_SIZE, IN_CHANNELS, IN_H, IN_W), S.bf16),
    w: S.Tensor((OUT_CHANNELS, IN_CHANNELS, K_H, K_W), S.bf16),
    b: S.Tensor((OUT_CHANNELS,), S.bf16),
    out: S.Tensor((BATCH_SIZE, 1, OUT_H, OUT_W), S.bf16),
):
    """Fused conv + scale + min over channels."""
    tid = S.thread_id(0)

    # Grid position: each block computes one (n, oh, ow) output element
    bid = S.block_id(0)
    n = bid // SPATIAL_ELEMS
    rem = bid % SPATIAL_ELEMS
    oh = rem // OUT_W
    ow = rem % OUT_W

    scale = S.convert(SCALE_FACTOR, S.f32)

    # Initialize with large value for min reduction
    min_val = S.convert(1e30, S.f32)

    # Iterate over output channels
    for oc in S.range(OUT_CHANNELS):
        acc = S.convert(b[oc], S.f32)

        for ic in S.range(IN_CHANNELS):
            for kh in S.range(K_H):
                ih = oh * STRIDE_H + kh
                if ih < IN_H:
                    for kw in S.range(K_W):
                        iw = ow * STRIDE_W + kw
                        if iw < IN_W:
                            w_val = S.convert(w[oc, ic, kh, kw], S.f32)
                            x_val = S.convert(x[n, ic, ih, iw], S.f32)
                            acc = acc + x_val * w_val

        # Apply scale
        scaled = acc * scale

        # Update minimum
        if scaled < min_val:
            min_val = scaled

    out[n, 0, oh, ow] = S.convert(min_val, S.bf16)


@substrate.jit
def min_dim1_bf16_kernel(
    x: S.Tensor((BATCH_SIZE, OUT_CHANNELS, OUT_H, OUT_W), S.bf16),
    out: S.Tensor((BATCH_SIZE, 1, OUT_H, OUT_W), S.bf16),
):
    """Min reduction along dim=1 (channels) with keepdim=True."""
    tid = S.thread_id(0)

    # Grid: each block computes one (n, oh, ow) output element
    bid = S.block_id(0)
    n = bid // SPATIAL_ELEMS
    rem = bid % SPATIAL_ELEMS
    oh = rem // OUT_W
    ow = rem % OUT_W

    # Initialize with large value
    min_val = S.convert(1e30, S.f32)

    for oc in S.range(OUT_CHANNELS):
        val = S.convert(x[n, oc, oh, ow], S.f32)
        if val < min_val:
            min_val = val

    out[n, 0, oh, ow] = S.convert(min_val, S.bf16)


def _launch_conv_bf16(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty((BATCH_SIZE, OUT_CHANNELS, OUT_H, OUT_W), device=x.device, dtype=torch.bfloat16)
    conv2d_3x3_s1_p0_bf16_kernel[
        lambda: ((BATCH_SIZE * OUT_CHANNELS, 1, 1), (THREADS_PER_BLOCK, 1, 1))
    ](x, w, b, out)
    return out


def _launch_min_bf16(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty((BATCH_SIZE, 1, OUT_H, OUT_W), device=x.device, dtype=torch.bfloat16)
    grid_size = BATCH_SIZE * SPATIAL_ELEMS
    min_dim1_bf16_kernel[
        lambda: ((grid_size, 1, 1), (1, 1, 1))
    ](x, out)
    return out


def _launch_conv_scale_min_bf16(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty((BATCH_SIZE, 1, OUT_H, OUT_W), device=x.device, dtype=torch.bfloat16)
    grid_size = BATCH_SIZE * SPATIAL_ELEMS
    conv_scale_min_bf16_kernel[
        lambda: ((grid_size, 1, 1), (1, 1, 1))
    ](x, w, b, out)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor

    def forward(self, x):
        original_device = x.device
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/HIP device is required for Substrate kernels.")

        if not x.is_cuda:
            x = x.cuda()

        # Get weights
        w = self.conv.weight
        b = self.conv.bias
        if b is None:
            b = torch.zeros((OUT_CHANNELS,), device=w.device, dtype=w.dtype)

        if w.device != x.device:
            w = w.to(device=x.device)
        if b.device != x.device:
            b = b.to(device=x.device)

        x = x.contiguous()
        w = w.contiguous()
        b = b.contiguous()

        # Convert to BF16 if needed
        input_dtype = x.dtype
        if input_dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)
            w = w.to(torch.bfloat16)
            b = b.to(torch.bfloat16)

        # Use fused kernel for conv + scale + min
        out = _launch_conv_scale_min_bf16(x, w, b)

        if original_device.type != "cuda":
            out = out.to(original_device)
        return out


batch_size = 64
in_channels = 64
out_channels = 128
height = width = 256
kernel_size = 3
scale_factor = 2.0


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scale_factor]
