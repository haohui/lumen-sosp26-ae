import torch
import torch.nn as nn
import substrate
import substrate.language as S


# Fixed problem shape from target model/get_inputs.
BATCH_SIZE = 128
IN_CHANNELS = 8
OUT_CHANNELS = 64
IN_H = 384
IN_W = 384
K_H = 3
K_W = 3
STRIDE_H = 1
STRIDE_W = 1
PAD_H = 0
PAD_W = 0

# Conv output shape
CONV_OUT_H = (IN_H + 2 * PAD_H - K_H) // STRIDE_H + 1  # 382
CONV_OUT_W = (IN_W + 2 * PAD_W - K_W) // STRIDE_W + 1  # 382

# Pool parameters
POOL_K = 4
POOL_STRIDE = 4
POOL_PAD = 0

# Pool output shape
POOL_OUT_H = (CONV_OUT_H - POOL_K) // POOL_STRIDE + 1  # 95
POOL_OUT_W = (CONV_OUT_W - POOL_K) // POOL_STRIDE + 1  # 95

THREADS_PER_BLOCK = 256
WEIGHT_ELEMS = IN_CHANNELS * K_H * K_W  # 72
WEIGHT_LOAD_ITERS = (WEIGHT_ELEMS + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
SPATIAL_ELEMS = CONV_OUT_H * CONV_OUT_W  # 145924
SPATIAL_TILES = (SPATIAL_ELEMS + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK


# ----------------- Conv2d Kernel -----------------
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

    for t in S.range(SPATIAL_TILES):
        pos = t * THREADS_PER_BLOCK + tid
        if pos < SPATIAL_ELEMS:
            oh = pos // CONV_OUT_W
            ow = pos % CONV_OUT_W

            acc = S.convert(b[oc], S.f32)

            for ic in S.range(IN_CHANNELS):
                for kh in S.range(K_H):
                    ih = oh * STRIDE_H + kh
                    if ih >= 0 and ih < IN_H:
                        for kw in S.range(K_W):
                            iw = ow * STRIDE_W + kw
                            if iw >= 0 and iw < IN_W:
                                wf = ic * (K_H * K_W) + kh * K_W + kw
                                xv = S.convert(x[n, ic, ih, iw], S.f32)
                                wv = S.convert(s_w[wf], S.f32)
                                acc = acc + xv * wv

            out[n, oc, oh, ow] = S.convert(acc, S.bf16)


# ----------------- AvgPool2d Kernel -----------------
@substrate.jit
def avgpool2d_4x4_s4_bf16_kernel(
    x: S.Tensor((BATCH_SIZE, OUT_CHANNELS, CONV_OUT_H, CONV_OUT_W), S.bf16),
    out: S.Tensor((BATCH_SIZE, OUT_CHANNELS, POOL_OUT_H, POOL_OUT_W), S.bf16),
):
    tid = S.thread_id(0)
    bid = S.block_id(0)

    # Flatten batch and channel
    n = bid // OUT_CHANNELS
    c = bid % OUT_CHANNELS

    # Each thread handles one output element
    num_outputs = POOL_OUT_H * POOL_OUT_W

    for t in S.range((num_outputs + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK):
        pos = t * THREADS_PER_BLOCK + tid
        if pos < num_outputs:
            oh = pos // POOL_OUT_W
            ow = pos % POOL_OUT_W

            # Input window start
            ih_start = oh * POOL_STRIDE
            iw_start = ow * POOL_STRIDE

            acc = S.convert(0.0, S.f32)
            count = S.convert(0, S.i32)

            for kh in S.range(POOL_K):
                ih = ih_start + kh
                if ih < CONV_OUT_H:
                    for kw in S.range(POOL_K):
                        iw = iw_start + kw
                        if iw < CONV_OUT_W:
                            acc = acc + S.convert(x[n, c, ih, iw], S.f32)
                            count = count + 1

            # Average: divide by count
            inv_count = S.convert(1.0, S.f32) / S.convert(count, S.f32)
            out[n, c, oh, ow] = S.convert(acc * inv_count, S.bf16)


# ----------------- Sigmoid Kernel -----------------
LOG2E = 1.4426950408889634


@substrate.jit
def sigmoid_bf16_kernel(
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
        # sigmoid(x) = 1 / (1 + exp(-x)) = 1 / (1 + 2^(-x * log2(e)))
        neg_x = S.convert(0.0, S.f32) - xv
        exp_term = S.exp2(neg_x * S.convert(LOG2E, S.f32))
        one = S.convert(1.0, S.f32)
        result = one / (one + exp_term)
        y[idx] = S.convert(result, S.bf16)


# ----------------- Sum Reduction Kernel -----------------
# Sum over all spatial and channel dimensions for each batch element
ELEMENTS_PER_BATCH = OUT_CHANNELS * POOL_OUT_H * POOL_OUT_W  # 64 * 95 * 95 = 577600


@substrate.jit
def sum_all_bf16_kernel(
    x: S.Tensor((BATCH_SIZE, OUT_CHANNELS, POOL_OUT_H, POOL_OUT_W), S.bf16),
    out: S.Tensor((BATCH_SIZE,), S.bf16),
):
    b = S.block_id(0)
    tid = S.thread_id(0)

    acc = S.convert(0.0, S.f32)

    # Each thread accumulates a partial sum over its assigned elements
    for i in S.range((ELEMENTS_PER_BATCH + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK):
        idx = i * THREADS_PER_BLOCK + tid
        if idx < ELEMENTS_PER_BATCH:
            c = idx // (POOL_OUT_H * POOL_OUT_W)
            rem = idx % (POOL_OUT_H * POOL_OUT_W)
            oh = rem // POOL_OUT_W
            ow = rem % POOL_OUT_W
            acc = acc + S.convert(x[b, c, oh, ow], S.f32)

    # Block-level reduction using shared memory
    smem = S.make_shared((THREADS_PER_BLOCK,), S.f32)
    smem[tid] = acc
    S.syncthreads()

    # Tree reduction
    step = THREADS_PER_BLOCK // 2
    for s in S.range(8):  # log2(256) = 8
        if tid < step:
            smem[tid] = smem[tid] + smem[tid + step]
        step = step // 2
        S.syncthreads()

    if tid == 0:
        out[b] = S.convert(smem[0], S.bf16)


# ----------------- Host wrappers -----------------
def _launch_conv_bf16(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty((BATCH_SIZE, OUT_CHANNELS, CONV_OUT_H, CONV_OUT_W), device=x.device, dtype=torch.bfloat16)
    conv2d_3x3_s1_p0_bf16_kernel[
        lambda: ((BATCH_SIZE * OUT_CHANNELS, 1, 1), (THREADS_PER_BLOCK, 1, 1))
    ](x, w, b, out)
    return out


def _launch_avgpool_bf16(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty((BATCH_SIZE, OUT_CHANNELS, POOL_OUT_H, POOL_OUT_W), device=x.device, dtype=torch.bfloat16)
    avgpool2d_4x4_s4_bf16_kernel[
        lambda: ((BATCH_SIZE * OUT_CHANNELS, 1, 1), (THREADS_PER_BLOCK, 1, 1))
    ](x, out)
    return out


def _launch_sigmoid_bf16(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    grid = (n + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    sigmoid_bf16_kernel[lambda: ((grid, 1, 1), (THREADS_PER_BLOCK, 1, 1))](x, out, n)
    return out


def _launch_sum_all_bf16(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty((BATCH_SIZE,), device=x.device, dtype=torch.bfloat16)
    sum_all_bf16_kernel[lambda: ((BATCH_SIZE, 1, 1), (THREADS_PER_BLOCK, 1, 1))](x, out)
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs convolution, average pooling, sigmoid, and sum
    using Substrate GPU kernels optimized for BF16 on AMD MI300X.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, pool_kernel_size: int):
        super(ModelNew, self).__init__()
        # Store conv weights
        self.conv_weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.conv_bias = nn.Parameter(torch.empty(out_channels))
        # Initialize with same method as nn.Conv2d
        nn.init.kaiming_uniform_(self.conv_weight, a=5**0.5)
        fan_in = in_channels * kernel_size * kernel_size
        bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
        nn.init.uniform_(self.conv_bias, -bound, bound)

        self.pool_kernel_size = pool_kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape != (BATCH_SIZE, IN_CHANNELS, IN_H, IN_W):
            raise NotImplementedError(
                f"ModelNew currently supports input shape {(BATCH_SIZE, IN_CHANNELS, IN_H, IN_W)}, got {tuple(x.shape)}"
            )

        original_device = x.device
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/HIP device is required for Substrate kernels.")

        if not x.is_cuda:
            x = x.cuda()

        # Convert to bf16 if needed
        orig_dtype = x.dtype
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)

        w = self.conv_weight.to(device=x.device, dtype=torch.bfloat16)
        b = self.conv_bias.to(device=x.device, dtype=torch.bfloat16)

        x = x.contiguous()
        w = w.contiguous()
        b = b.contiguous()

        # Conv2d
        x = _launch_conv_bf16(x, w, b)

        # AvgPool2d
        x = _launch_avgpool_bf16(x)

        # Sigmoid
        x = _launch_sigmoid_bf16(x)

        # Sum over dims [1,2,3]
        x = _launch_sum_all_bf16(x)

        # Convert back to original dtype if needed
        if orig_dtype != torch.bfloat16:
            x = x.to(orig_dtype)

        if original_device.type != "cuda":
            x = x.to(original_device)
        return x


batch_size = 128
in_channels = 8
out_channels = 64
height, width = 384, 384
kernel_size = 3
pool_size = 4


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, pool_size]
