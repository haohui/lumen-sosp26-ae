import torch
import torch.nn as nn
import substrate
import substrate.language as S


# Fixed problem shape from target model/get_inputs.
BATCH_SIZE = 128
IN_CHANNELS = 8
OUT_CHANNELS = 64
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
WEIGHT_ELEMS = IN_CHANNELS * K_H * K_W  # 72
WEIGHT_LOAD_ITERS = (WEIGHT_ELEMS + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
SPATIAL_ELEMS = OUT_H * OUT_W  # 64516
SPATIAL_TILES = (SPATIAL_ELEMS + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

# GELU constants: gelu(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
GELU_HALF = 0.5
SQRT2 = 1.4142135623730951
INV_SQRT2 = 1.0 / SQRT2

# Global average pooling constants
POOL_H = OUT_H  # 254
POOL_W = OUT_W  # 254
INV_POOL_HW = 1.0 / (POOL_H * POOL_W)
BLOCK_SIZE_REDUCE = 256


@substrate.jit
def conv2d_3x3_bf16_kernel(
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
                    ih = oh * STRIDE_H + kh - PAD_H
                    if ih >= 0 and ih < IN_H:
                        for kw in S.range(K_W):
                            iw = ow * STRIDE_W + kw - PAD_W
                            if iw >= 0 and iw < IN_W:
                                wf = ic * (K_H * K_W) + kh * K_W + kw
                                xv = S.convert(x[n, ic, ih, iw], S.f32)
                                wv = S.convert(s_w[wf], S.f32)
                                acc = acc + xv * wv

            out[n, oc, oh, ow] = S.convert(acc, S.bf16)


@substrate.jit
def gelu_bf16_kernel(
    x_ptr: S.Pointer(S.bf16),
    y_ptr: S.Pointer(S.bf16),
    n: S.u32,
):
    idx = S.block_id(0) * S.block_dim(0) + S.thread_id(0)
    layout = S.make_layout((n,), (1,))
    x = S.make_tensor(x_ptr, S.bf16, layout)
    y = S.make_tensor(y_ptr, S.bf16, layout)

    if idx < n:
        xv = S.convert(x[idx], S.f32)
        half = S.convert(GELU_HALF, S.f32)
        inv_sqrt2 = S.convert(INV_SQRT2, S.f32)
        one = S.convert(1.0, S.f32)

        erf_arg = xv * inv_sqrt2
        erf_val = S.erf(erf_arg)
        gelu_val = xv * half * (one + erf_val)

        y[idx] = S.convert(gelu_val, S.bf16)


@substrate.jit
def global_avg_pool_bf16_kernel(
    x: S.Tensor((BATCH_SIZE, OUT_CHANNELS, POOL_H, POOL_W), S.bf16),
    out: S.Tensor((BATCH_SIZE, OUT_CHANNELS), S.bf16),
):
    tid = S.thread_id(0)
    bc_idx = S.block_id(0) * S.block_dim(0) + tid

    if bc_idx < BATCH_SIZE * OUT_CHANNELS:
        n = bc_idx // OUT_CHANNELS
        oc = bc_idx % OUT_CHANNELS

        acc = S.convert(0.0, S.f32)
        for ph in S.range(POOL_H):
            for pw in S.range(POOL_W):
                acc = acc + S.convert(x[n, oc, ph, pw], S.f32)

        inv_hw = S.convert(INV_POOL_HW, S.f32)
        out[n, oc] = S.convert(acc * inv_hw, S.bf16)


def _launch_conv2d_bf16(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty((BATCH_SIZE, OUT_CHANNELS, OUT_H, OUT_W), device=x.device, dtype=torch.bfloat16)
    conv2d_3x3_bf16_kernel[
        lambda: ((BATCH_SIZE * OUT_CHANNELS, 1, 1), (THREADS_PER_BLOCK, 1, 1))
    ](x, w, b, out)
    return out


def _launch_gelu_bf16(x: torch.Tensor) -> torch.Tensor:
    x_flat = x.view(-1)
    y = torch.empty_like(x_flat)
    n = x_flat.numel()
    if n > 0:
        grid = ((n + BLOCK_SIZE_REDUCE - 1) // BLOCK_SIZE_REDUCE, 1, 1)
        gelu_bf16_kernel[lambda: (grid, (BLOCK_SIZE_REDUCE, 1, 1))](x_flat, y, n)
    return y.view_as(x)


def _launch_global_avg_pool_bf16(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty((BATCH_SIZE, OUT_CHANNELS), device=x.device, dtype=torch.bfloat16)
    grid = ((BATCH_SIZE * OUT_CHANNELS + BLOCK_SIZE_REDUCE - 1) // BLOCK_SIZE_REDUCE, 1, 1)
    global_avg_pool_bf16_kernel[lambda: (grid, (BLOCK_SIZE_REDUCE, 1, 1))](x, out)
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs convolution, GELU activation, and global average pooling
    using Substrate GPU kernels for AMD MI300X (BF16 precision).
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
        Returns:
            Output tensor of shape (batch_size, out_channels)
        """
        if x.shape != (BATCH_SIZE, IN_CHANNELS, IN_H, IN_W):
            raise NotImplementedError(
                f"ModelNew currently supports input shape {(BATCH_SIZE, IN_CHANNELS, IN_H, IN_W)}, got {tuple(x.shape)}"
            )

        original_device = x.device
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/HIP device is required for Substrate kernels.")

        if not x.is_cuda:
            x = x.cuda()

        # Convert to BF16 if needed
        input_dtype = x.dtype
        if input_dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)

        w = self.conv.weight
        b = self.conv.bias
        if b is None:
            b = torch.zeros((OUT_CHANNELS,), device=w.device, dtype=w.dtype)

        if w.device != x.device:
            w = w.to(device=x.device)
        if b.device != x.device:
            b = b.to(device=x.device)

        # Convert weights to BF16 if needed
        if w.dtype != torch.bfloat16:
            w = w.to(torch.bfloat16)
        if b.dtype != torch.bfloat16:
            b = b.to(torch.bfloat16)

        x = x.contiguous()
        w = w.contiguous()
        b = b.contiguous()

        # Conv2d
        x = _launch_conv2d_bf16(x, w, b)

        # GELU activation
        x = _launch_gelu_bf16(x)

        # Global average pooling
        x = _launch_global_avg_pool_bf16(x)

        # Output is already (batch_size, out_channels) from global_avg_pool
        if original_device.type != "cuda":
            x = x.to(original_device)

        return x


batch_size = 128
in_channels = 8
out_channels = 64
height, width = 256, 256
kernel_size = 3


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
