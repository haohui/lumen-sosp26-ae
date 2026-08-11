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

# Conv output dimensions
CONV_OUT_H = (IN_H + 2 * PAD_H - K_H) // STRIDE_H + 1  # 126
CONV_OUT_W = (IN_W + 2 * PAD_W - K_W) // STRIDE_W + 1  # 126

# Pool dimensions
POOL_K = 2
POOL_STRIDE = 2
POOL_PAD = 0
POOL_OUT_H = (CONV_OUT_H - POOL_K) // POOL_STRIDE + 1  # 63
POOL_OUT_W = (CONV_OUT_W - POOL_K) // POOL_STRIDE + 1  # 63

THREADS_PER_BLOCK = 256
WEIGHT_ELEMS = IN_CHANNELS * K_H * K_W  # 576
WEIGHT_LOAD_ITERS = (WEIGHT_ELEMS + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
CONV_SPATIAL_ELEMS = CONV_OUT_H * CONV_OUT_W  # 15876
CONV_SPATIAL_TILES = (CONV_SPATIAL_ELEMS + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

# Pooling kernel config - use 1D kernel for simplicity
POOL_THREADS = 256
POOL_SPATIAL_ELEMS = POOL_OUT_H * POOL_OUT_W
POOL_SPATIAL_TILES = (POOL_SPATIAL_ELEMS + POOL_THREADS - 1) // POOL_THREADS

# Elementwise kernel config
EW_THREADS = 256


@substrate.jit
def conv2d_3x3_s1_p0_bf16_kernel(
    x: S.Tensor((BATCH_SIZE, IN_CHANNELS, IN_H, IN_W), S.bf16),
    w: S.Tensor((OUT_CHANNELS, IN_CHANNELS, K_H, K_W), S.bf16),
    b: S.Tensor((OUT_CHANNELS,), S.bf16),
    out: S.Tensor((BATCH_SIZE, OUT_CHANNELS, CONV_OUT_H, CONV_OUT_W), S.bf16),
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

    for t in S.range(CONV_SPATIAL_TILES):
        pos = t * THREADS_PER_BLOCK + tid
        if pos < CONV_SPATIAL_ELEMS:
            oh = pos // CONV_OUT_W
            ow = pos % CONV_OUT_W

            acc = S.convert(b[oc], S.f32)

            for ic in S.range(IN_CHANNELS):
                for kh in S.range(K_H):
                    ih = oh * STRIDE_H + kh
                    for kw in S.range(K_W):
                        iw = ow * STRIDE_W + kw
                        wf = ic * (K_H * K_W) + kh * K_W + kw
                        xv = S.convert(x[n, ic, ih, iw], S.f32)
                        wv = S.convert(s_w[wf], S.f32)
                        acc = acc + xv * wv

            out[n, oc, oh, ow] = S.convert(acc, S.bf16)


@substrate.jit
def fused_subtract_tanh_subtract_bf16_kernel(
    x_ptr: S.Pointer(S.bf16),
    y_ptr: S.Pointer(S.bf16),
    sub_vals: S.Tensor((2,), S.bf16),
    n: S.u32,
):
    idx = S.block_id(0) * S.block_dim(0) + S.thread_id(0)
    layout = S.make_layout((n,), (1,))
    x = S.make_tensor(x_ptr, S.bf16, layout)
    y = S.make_tensor(y_ptr, S.bf16, layout)

    if idx < n:
        v = S.convert(x[idx], S.f32)
        sub1_f = S.convert(sub_vals[0], S.f32)
        sub2_f = S.convert(sub_vals[1], S.f32)
        v = v - sub1_f
        v = S.tanh(v)
        v = v - sub2_f
        y[idx] = S.convert(v, S.bf16)


@substrate.jit
def avgpool2d_bf16_kernel(
    x: S.Tensor((BATCH_SIZE, OUT_CHANNELS, CONV_OUT_H, CONV_OUT_W), S.bf16),
    out: S.Tensor((BATCH_SIZE, OUT_CHANNELS, POOL_OUT_H, POOL_OUT_W), S.bf16),
):
    tid = S.thread_id(0)
    bid = S.block_id(0)

    n = bid // OUT_CHANNELS
    c = bid % OUT_CHANNELS

    for t in S.range(POOL_SPATIAL_TILES):
        pos = t * POOL_THREADS + tid
        if pos < POOL_SPATIAL_ELEMS:
            oy = pos // POOL_OUT_W
            ox = pos % POOL_OUT_W

            acc = S.convert(0.0, S.f32)
            for ky in S.range(POOL_K):
                for kx in S.range(POOL_K):
                    iy = oy * POOL_STRIDE + ky
                    ix = ox * POOL_STRIDE + kx
                    acc = acc + S.convert(x[n, c, iy, ix], S.f32)

            avg = acc / S.convert(POOL_K * POOL_K, S.f32)
            out[n, c, oy, ox] = S.convert(avg, S.bf16)


def _launch_conv_bf16(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty((BATCH_SIZE, OUT_CHANNELS, CONV_OUT_H, CONV_OUT_W), device=x.device, dtype=torch.bfloat16)
    conv2d_3x3_s1_p0_bf16_kernel[
        lambda: ((BATCH_SIZE * OUT_CHANNELS, 1, 1), (THREADS_PER_BLOCK, 1, 1))
    ](x, w, b, out)
    return out


def _launch_fused_elementwise(x: torch.Tensor, sub1: float, sub2: float) -> torch.Tensor:
    x = x.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    if n > 0:
        grid = ((n + EW_THREADS - 1) // EW_THREADS, 1, 1)
        # Pack subtraction values into a small tensor
        sub_vals = torch.tensor([sub1, sub2], dtype=torch.bfloat16, device=x.device)
        fused_subtract_tanh_subtract_bf16_kernel[lambda: (grid, (EW_THREADS, 1, 1))](
            x.view(-1), out.view(-1), sub_vals, n
        )
    return out


def _launch_avgpool_bf16(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty((BATCH_SIZE, OUT_CHANNELS, POOL_OUT_H, POOL_OUT_W), device=x.device, dtype=torch.bfloat16)
    avgpool2d_bf16_kernel[
        lambda: ((BATCH_SIZE * OUT_CHANNELS, 1, 1), (POOL_THREADS, 1, 1))
    ](x, out)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.avgpool = nn.AvgPool2d(kernel_size_pool)

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

        # Convert to bfloat16 for optimized computation
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

        x = x.contiguous()
        w = w.contiguous()
        b = b.contiguous()

        # Convert weights to bfloat16
        if w.dtype != torch.bfloat16:
            w = w.to(torch.bfloat16)
        if b.dtype != torch.bfloat16:
            b = b.to(torch.bfloat16)

        # Step 1: Conv2d
        x = _launch_conv_bf16(x, w, b)

        # Step 2-4: Fused subtract, tanh, subtract
        x = _launch_fused_elementwise(x, self.subtract1_value, self.subtract2_value)

        # Step 5: AvgPool2d
        x = _launch_avgpool_bf16(x)

        if original_device.type != "cuda":
            x = x.to(original_device)

        return x


batch_size = 128
in_channels = 64
out_channels = 128
height, width = 128, 128
kernel_size = 3
subtract1_value = 0.5
subtract2_value = 0.2
kernel_size_pool = 2


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool]
