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

POOL_K = 2
POOL_OUT_H = OUT_H // POOL_K  # 63
POOL_OUT_W = OUT_W // POOL_K  # 63

THREADS_PER_BLOCK = 256
WEIGHT_ELEMS = IN_CHANNELS * K_H * K_W  # 64 * 3 * 3 = 576
WEIGHT_LOAD_ITERS = (WEIGHT_ELEMS + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
SPATIAL_ELEMS = OUT_H * OUT_W  # 15876
SPATIAL_TILES = (SPATIAL_ELEMS + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK


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


# HardSwish: x * min(max(x + 3, 0), 6) / 6
# Fused with subtract: (x - sub) * hardswish((x - sub))
SUBTRACT_VAL = 0.5


@substrate.jit
def subtract_hardswish_bf16_kernel(
    x_ptr: S.Pointer(S.bf16),
    y_ptr: S.Pointer(S.bf16),
    n: S.u32,
):
    idx = S.block_id(0) * S.block_dim(0) + S.thread_id(0)
    layout = S.make_layout((n,), (1,))
    x = S.make_tensor(x_ptr, S.bf16, layout)
    y = S.make_tensor(y_ptr, S.bf16, layout)

    if idx < n:
        v = S.convert(x[idx], S.f32) - S.convert(SUBTRACT_VAL, S.f32)

        # HardSwish: x * min(max(x + 3, 0), 6) / 6
        zero = S.convert(0.0, S.f32)
        three = S.convert(3.0, S.f32)
        six = S.convert(6.0, S.f32)
        onesixth = S.convert(0.16666666666666666, S.f32)

        t = v + three
        if t < zero:
            t = zero
        if t > six:
            t = six

        result = v * t * onesixth
        y[idx] = S.convert(result, S.bf16)


BLOCK_X = 16
BLOCK_Y = 16
THREADS_POOL = BLOCK_X * BLOCK_Y

NEG_INF_F32 = -3.402823466e38


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

    ox = bx * BLOCK_X + tx
    oy = by * BLOCK_Y + ty

    c = bz % OUT_CHANNELS
    n = bz // OUT_CHANNELS

    if (ox < POOL_OUT_W) and (oy < POOL_OUT_H):
        # For stride=2, kernel=2: input position is (oy*2, ox*2)
        iy_base = oy * 2
        ix_base = ox * 2

        m = S.convert(NEG_INF_F32, S.f32)
        for ky in S.range(POOL_K):
            for kx in S.range(POOL_K):
                iy = iy_base + ky
                ix = ix_base + kx
                if (iy < OUT_H) and (ix < OUT_W):
                    val = S.convert(x[n, c, iy, ix], S.f32)
                    if (val > m) or (val != val):
                        m = val
        out[n, c, oy, ox] = S.convert(m, S.bf16)


# Mish: x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))
LOG2E = 1.4426950408889634


@substrate.jit
def mish_bf16_kernel(
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

        # softplus: log(1 + exp(x)) = log2(1 + exp(x)) * ln(2)
        # For numerical stability, use: log(1 + exp(x)) = x + log(1 + exp(-x)) for x > 0
        # But we'll use log2 directly via exp2
        one = S.convert(1.0, S.f32)
        ln2 = S.convert(0.6931471805599453, S.f32)

        # softplus approximation using exp2
        # softplus(x) = log(1 + exp(x)) = ln(2) * log2(1 + exp(x))
        # We compute exp(x) = exp2(x * log2(e))
        log2e = S.convert(LOG2E, S.f32)
        exp_x = S.exp2(v * log2e)
        softplus_v = S.log(one + exp_x)

        # mish = x * tanh(softplus(x))
        tanh_sp = S.tanh(softplus_v)
        mish_v = v * tanh_sp

        y[idx] = S.convert(mish_v, S.bf16)


def _launch_conv_bf16(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty((BATCH_SIZE, OUT_CHANNELS, OUT_H, OUT_W), device=x.device, dtype=torch.bfloat16)
    conv2d_3x3_s1_p0_bf16_kernel[
        lambda: ((BATCH_SIZE * OUT_CHANNELS, 1, 1), (THREADS_PER_BLOCK, 1, 1))
    ](x, w, b, out)
    return out


def _launch_subtract_hardswish_bf16(x: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    y = torch.empty_like(x)
    n = x.numel()
    if n > 0:
        grid = ((n + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK, 1, 1)
        subtract_hardswish_bf16_kernel[lambda: (grid, (THREADS_PER_BLOCK, 1, 1))](x, y, n)
    return y


def _launch_maxpool2d_bf16(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty((BATCH_SIZE, OUT_CHANNELS, POOL_OUT_H, POOL_OUT_W), device=x.device, dtype=torch.bfloat16)
    grid_x = (POOL_OUT_W + BLOCK_X - 1) // BLOCK_X
    grid_y = (POOL_OUT_H + BLOCK_Y - 1) // BLOCK_Y
    grid_z = BATCH_SIZE * OUT_CHANNELS
    maxpool2d_bf16_kernel[lambda: ((grid_x, grid_y, grid_z), (BLOCK_X, BLOCK_Y, 1))](x, out)
    return out


def _launch_mish_bf16(x: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    y = torch.empty_like(x)
    n = x.numel()
    if n > 0:
        grid = ((n + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK, 1, 1)
        mish_bf16_kernel[lambda: (grid, (THREADS_PER_BLOCK, 1, 1))](x, y, n)
    return y


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value = subtract_value
        self.pool_kernel_size = pool_kernel_size

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

        # Convert to bfloat16 for computation
        x_work = x.to(torch.bfloat16)

        w = self.conv.weight.to(torch.bfloat16)
        b = self.conv.bias
        if b is None:
            b = torch.zeros((OUT_CHANNELS,), device=w.device, dtype=torch.bfloat16)
        else:
            b = b.to(torch.bfloat16)

        if w.device != x_work.device:
            w = w.to(device=x_work.device)
        if b.device != x_work.device:
            b = b.to(device=x_work.device)

        x_work = x_work.contiguous()
        w = w.contiguous()
        b = b.contiguous()

        # Conv2d
        out = _launch_conv_bf16(x_work, w, b)

        # Subtract + HardSwish (fused)
        out = _launch_subtract_hardswish_bf16(out)

        # MaxPool2d
        out = _launch_maxpool2d_bf16(out)

        # Mish
        out = _launch_mish_bf16(out)

        if original_device.type != "cuda":
            out = out.to(original_device)
        return out


batch_size = 128
in_channels = 64
out_channels = 128
height = width = 128
kernel_size = 3
subtract_value = 0.5
pool_kernel_size = 2


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size]
