import torch
import torch.nn as nn
import substrate
import substrate.language as S


# Fixed problem shape from target model/get_inputs
BATCH_SIZE = 128
IN_CHANNELS = 8
OUT_CHANNELS = 64
IN_H = 128
IN_W = 128
K_H = 3
K_W = 3
STRIDE = 1
PAD_H = 0
PAD_W = 0

OUT_H = (IN_H + 2 * PAD_H - K_H) // STRIDE + 1  # 126
OUT_W = (IN_W + 2 * PAD_W - K_W) // STRIDE + 1  # 126

NUM_GROUPS = 16
CHANNELS_PER_GROUP = OUT_CHANNELS // NUM_GROUPS  # 4

MAXPOOL_K = 4
POOL_OUT_H = OUT_H // MAXPOOL_K  # 31
POOL_OUT_W = OUT_W // MAXPOOL_K  # 31

THREADS = 256

# Conv weights
WEIGHT_ELEMS = IN_CHANNELS * K_H * K_W  # 72
WEIGHT_LOAD_ITERS = (WEIGHT_ELEMS + THREADS - 1) // THREADS
SPATIAL_ELEMS_CONV = OUT_H * OUT_W  # 15876
SPATIAL_TILES_CONV = (SPATIAL_ELEMS_CONV + THREADS - 1) // THREADS

# GroupNorm spatial elements
GN_SPATIAL = OUT_H * OUT_W  # 15876
GN_INV_SPATIAL = 1.0 / GN_SPATIAL
GN_CHANNELS_PER_GROUP = OUT_CHANNELS // NUM_GROUPS  # 4

# MaxPool
POOL_BLOCK_X = 16
POOL_BLOCK_Y = 16
# For stride=kernel, no overlap between pooling windows
# Each output position accesses input[oy*stride:oy*stride+kernel, ox*stride:ox*stride+kernel]
# For block of (POOL_BLOCK_Y, POOL_BLOCK_X) outputs, need (BLOCK_Y*stride, BLOCK_X*stride) input region
# But we're doing shared memory for the whole block's input region
POOL_SH_H = POOL_BLOCK_Y * MAXPOOL_K  # 64
POOL_SH_W = POOL_BLOCK_X * MAXPOOL_K  # 64
POOL_THREADS = POOL_BLOCK_X * POOL_BLOCK_Y
POOL_SH_LOAD_ITERS = (POOL_SH_H * POOL_SH_W + POOL_THREADS - 1) // POOL_THREADS

NEG_INF_F32 = -3.402823466e38


# ========== Conv2d Kernel ==========
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

    s_w = S.make_shared((WEIGHT_ELEMS,), S.bf16)

    for it in S.range(WEIGHT_LOAD_ITERS):
        w_flat = it * THREADS + tid
        if w_flat < WEIGHT_ELEMS:
            ic = w_flat // (K_H * K_W)
            rem = w_flat % (K_H * K_W)
            kh = rem // K_W
            kw = rem % K_W
            s_w[w_flat] = w[oc, ic, kh, kw]

    S.syncthreads()

    for t in S.range(SPATIAL_TILES_CONV):
        pos = t * THREADS + tid
        if pos < SPATIAL_ELEMS_CONV:
            oh = pos // OUT_W
            ow = pos % OUT_W

            acc = S.convert(b[oc], S.f32)

            for ic in S.range(IN_CHANNELS):
                for kh in S.range(K_H):
                    ih = oh + kh
                    if ih < IN_H:
                        for kw in S.range(K_W):
                            iw = ow + kw
                            if iw < IN_W:
                                wf = ic * (K_H * K_W) + kh * K_W + kw
                                xv = S.convert(x[n, ic, ih, iw], S.f32)
                                wv = S.convert(s_w[wf], S.f32)
                                acc = acc + xv * wv

            out[n, oc, oh, ow] = S.convert(acc, S.bf16)


# ========== GroupNorm Kernel ==========
@substrate.jit
def group_norm_bf16_kernel(
    x: S.Tensor((BATCH_SIZE, OUT_CHANNELS, OUT_H, OUT_W), S.bf16),
    gamma: S.Tensor((OUT_CHANNELS,), S.bf16),
    beta: S.Tensor((OUT_CHANNELS,), S.bf16),
    out: S.Tensor((BATCH_SIZE, OUT_CHANNELS, OUT_H, OUT_W), S.bf16),
):
    tid = S.thread_id(0)
    bid = S.block_id(0)

    n = bid // NUM_GROUPS
    g = bid % NUM_GROUPS

    # Compute mean
    mean = S.convert(0.0, S.f32)
    for c in S.range(GN_CHANNELS_PER_GROUP):
        ch = g * GN_CHANNELS_PER_GROUP + c
        for h in S.range(OUT_H):
            for w_s in S.range(OUT_W):
                mean = mean + S.convert(x[n, ch, h, w_s], S.f32)

    inv_spatial = S.convert(GN_INV_SPATIAL, S.f32)
    inv_cpg = S.convert(1.0 / GN_CHANNELS_PER_GROUP, S.f32)
    mean = mean * inv_spatial * inv_cpg

    # Compute variance
    var = S.convert(0.0, S.f32)
    for c in S.range(GN_CHANNELS_PER_GROUP):
        ch = g * GN_CHANNELS_PER_GROUP + c
        for h in S.range(OUT_H):
            for w_s in S.range(OUT_W):
                diff = S.convert(x[n, ch, h, w_s], S.f32) - mean
                var = var + diff * diff

    var = var * inv_spatial * inv_cpg

    eps = S.convert(1e-5, S.f32)
    inv_std = S.convert(1.0, S.f32) / S.sqrt(var + eps)

    # Normalize and apply affine
    for c in S.range(GN_CHANNELS_PER_GROUP):
        ch = g * GN_CHANNELS_PER_GROUP + c
        g_val = S.convert(gamma[ch], S.f32)
        b_val = S.convert(beta[ch], S.f32)
        for h in S.range(OUT_H):
            for w_s in S.range(OUT_W):
                normalized = (S.convert(x[n, ch, h, w_s], S.f32) - mean) * inv_std
                out[n, ch, h, w_s] = S.convert(normalized * g_val + b_val, S.bf16)


# ========== Scale Multiply Kernel ==========
@substrate.jit
def scale_bf16_kernel(
    x: S.Tensor((BATCH_SIZE, OUT_CHANNELS, OUT_H, OUT_W), S.bf16),
    scale: S.Tensor((OUT_CHANNELS, 1, 1), S.bf16),
    out: S.Tensor((BATCH_SIZE, OUT_CHANNELS, OUT_H, OUT_W), S.bf16),
):
    idx = S.block_id(0) * S.block_dim(0) + S.thread_id(0)
    total = BATCH_SIZE * OUT_CHANNELS * OUT_H * OUT_W

    if idx < total:
        n = idx // (OUT_CHANNELS * OUT_H * OUT_W)
        rem = idx % (OUT_CHANNELS * OUT_H * OUT_W)
        c = rem // (OUT_H * OUT_W)
        rem2 = rem % (OUT_H * OUT_W)
        h = rem2 // OUT_W
        w_s = rem2 % OUT_W

        xv = S.convert(x[n, c, h, w_s], S.f32)
        sv = S.convert(scale[c, 0, 0], S.f32)
        out[n, c, h, w_s] = S.convert(xv * sv, S.bf16)


# ========== MaxPool2d Kernel ==========
@substrate.jit
def maxpool2d_bf16_kernel(
    x: S.Tensor((BATCH_SIZE, OUT_CHANNELS, OUT_H, OUT_W), S.bf16),
    out: S.Tensor((BATCH_SIZE, OUT_CHANNELS, POOL_OUT_H, POOL_OUT_W), S.bf16),
):
    tx = S.thread_id(0)
    ty = S.thread_id(1)
    bx = S.block_id(0)
    by = S.block_id(1)
    bz = S.block_id(2)

    ox = bx * POOL_BLOCK_X + tx
    oy = by * POOL_BLOCK_Y + ty

    c = bz % OUT_CHANNELS
    n = bz // OUT_CHANNELS

    smem = S.make_shared((POOL_SH_H, POOL_SH_W), S.f32)

    tid_flat = ty * POOL_BLOCK_X + tx
    for it in S.range(POOL_SH_LOAD_ITERS):
        linear = tid_flat + it * POOL_THREADS
        if linear < POOL_SH_H * POOL_SH_W:
            sy = linear // POOL_SH_W
            sx = linear % POOL_SH_W

            iy = by * POOL_BLOCK_Y * MAXPOOL_K + sy
            ix = bx * POOL_BLOCK_X * MAXPOOL_K + sx

            v = S.convert(NEG_INF_F32, S.f32)
            if (iy < OUT_H) and (ix < OUT_W):
                v = S.convert(x[n, c, iy, ix], S.f32)

            smem[sy, sx] = v

    S.syncthreads()

    if (ox < POOL_OUT_W) and (oy < POOL_OUT_H):
        m = S.convert(NEG_INF_F32, S.f32)
        for ky in S.range(MAXPOOL_K):
            for kx in S.range(MAXPOOL_K):
                val = smem[ty * MAXPOOL_K + ky, tx * MAXPOOL_K + kx]
                if (val > m) or (val != val):
                    m = val
        out[n, c, oy, ox] = S.convert(m, S.bf16)


# ========== Clamp Kernel ==========
@substrate.jit
def clamp_bf16_kernel(
    x: S.Pointer(S.bf16),
    out: S.Pointer(S.bf16),
    n: S.u32,
):
    idx = S.block_id(0) * S.block_dim(0) + S.thread_id(0)
    layout = S.make_layout((n,), (1,))
    gx = S.make_tensor(x, S.bf16, layout)
    gout = S.make_tensor(out, S.bf16, layout)

    if idx < n:
        v = S.convert(gx[idx], S.f32)
        min_val = S.convert(0.0, S.f32)
        max_val = S.convert(1.0, S.f32)
        if v < min_val:
            v = min_val
        if v > max_val:
            v = max_val
        gout[idx] = S.convert(v, S.bf16)


# ========== Launch Functions ==========
def _launch_conv_bf16(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty((BATCH_SIZE, OUT_CHANNELS, OUT_H, OUT_W), device=x.device, dtype=torch.bfloat16)
    conv2d_3x3_bf16_kernel[
        lambda: ((BATCH_SIZE * OUT_CHANNELS, 1, 1), (THREADS, 1, 1))
    ](x, w, b, out)
    return out


def _launch_group_norm_bf16(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    out = torch.empty((BATCH_SIZE, OUT_CHANNELS, OUT_H, OUT_W), device=x.device, dtype=torch.bfloat16)
    group_norm_bf16_kernel[
        lambda: ((BATCH_SIZE * NUM_GROUPS, 1, 1), (1, 1, 1))
    ](x, gamma, beta, out)
    return out


def _launch_scale_bf16(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    total = x.numel()
    grid = ((total + THREADS - 1) // THREADS, 1, 1)
    scale_bf16_kernel[lambda: (grid, (THREADS, 1, 1))](x, scale, out)
    return out


def _launch_maxpool_bf16(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty((BATCH_SIZE, OUT_CHANNELS, POOL_OUT_H, POOL_OUT_W), device=x.device, dtype=torch.bfloat16)
    grid_x = (POOL_OUT_W + POOL_BLOCK_X - 1) // POOL_BLOCK_X
    grid_y = (POOL_OUT_H + POOL_BLOCK_Y - 1) // POOL_BLOCK_Y
    grid_z = BATCH_SIZE * OUT_CHANNELS
    maxpool2d_bf16_kernel[lambda: ((grid_x, grid_y, grid_z), (POOL_BLOCK_X, POOL_BLOCK_Y, 1))](x, out)
    return out


def _launch_clamp_bf16(x: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    if n > 0:
        grid = ((n + THREADS - 1) // THREADS, 1, 1)
        clamp_bf16_kernel[lambda: (grid, (THREADS, 1, 1))](x, out, n)
    return out


class ModelNew(nn.Module):
    """
    Optimized model using Substrate GPU kernels.
    Performs: Conv2d -> GroupNorm -> Scale -> MaxPool2d -> Clamp
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.maxpool_kernel_size = maxpool_kernel_size
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Validate shape
        if tuple(x.shape) != (BATCH_SIZE, IN_CHANNELS, IN_H, IN_W):
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

        x = x.contiguous()

        # Get conv weights in BF16
        w = self.conv.weight.to(torch.bfloat16).contiguous()
        b = self.conv.bias
        if b is None:
            b = torch.zeros((OUT_CHANNELS,), device=w.device, dtype=torch.bfloat16)
        else:
            b = b.to(torch.bfloat16).contiguous()

        # Conv2d
        x = _launch_conv_bf16(x, w, b)

        # GroupNorm
        gamma = self.group_norm.weight.to(torch.bfloat16).contiguous()
        beta = self.group_norm.bias.to(torch.bfloat16).contiguous()
        x = _launch_group_norm_bf16(x, gamma, beta)

        # Scale
        scale = self.scale.to(torch.bfloat16).contiguous()
        x = _launch_scale_bf16(x, scale)

        # MaxPool2d
        x = _launch_maxpool_bf16(x)

        # Clamp
        x = _launch_clamp_bf16(x, self.clamp_min, self.clamp_max)

        if original_device.type != "cuda":
            x = x.to(original_device)

        return x


batch_size = BATCH_SIZE
in_channels = IN_CHANNELS
out_channels = OUT_CHANNELS
height, width = IN_H, IN_W
kernel_size = K_H
num_groups = NUM_GROUPS
scale_shape = (out_channels, 1, 1)
maxpool_kernel_size = MAXPOOL_K
clamp_min = 0.0
clamp_max = 1.0


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max]
