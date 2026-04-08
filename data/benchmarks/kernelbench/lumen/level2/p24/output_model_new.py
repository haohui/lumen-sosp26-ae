import torch
import torch.nn as nn
import substrate
import substrate.language as S


# Fixed problem shape from target model/get_inputs.
BATCH_SIZE = 128
IN_CHANNELS = 3
OUT_CHANNELS = 24
IN_D = 24
IN_H = 32
IN_W = 32
KERNEL_SIZE = 3
PADDING = 0
STRIDE = 1

OUT_D = (IN_D - KERNEL_SIZE) // STRIDE + 1  # 22
OUT_H = (IN_H - KERNEL_SIZE) // STRIDE + 1  # 30
OUT_W = (IN_W - KERNEL_SIZE) // STRIDE + 1  # 30

# After min reduction along dim=2 (depth), the depth dimension is squeezed
FINAL_H = OUT_H  # 30
FINAL_W = OUT_W  # 30

THREADS_PER_BLOCK = 256

# Conv3d kernel constants
WEIGHT_ELEMS = IN_CHANNELS * KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE  # 81
WEIGHT_LOAD_ITERS = (WEIGHT_ELEMS + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
CONV_OUT_ELEMS = OUT_D * OUT_H * OUT_W  # 22 * 30 * 30 = 19800
CONV_OUT_TILES = (CONV_OUT_ELEMS + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

# Min reduction constants
MIN_REDUCE_DIM = OUT_D  # 22

# Softmax constants
SOFTMAX_CHANNELS = OUT_CHANNELS  # 24
SOFTMAX_SPATIAL = FINAL_H * FINAL_W  # 30 * 30 = 900

LOG2E = 1.4426950408889634


@substrate.jit
def conv3d_bf16_kernel(
    x: S.Tensor((BATCH_SIZE, IN_CHANNELS, IN_D, IN_H, IN_W), S.bf16),
    w: S.Tensor((OUT_CHANNELS, IN_CHANNELS, KERNEL_SIZE, KERNEL_SIZE, KERNEL_SIZE), S.bf16),
    b: S.Tensor((OUT_CHANNELS,), S.bf16),
    out: S.Tensor((BATCH_SIZE, OUT_CHANNELS, OUT_D, OUT_H, OUT_W), S.bf16),
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
            ic = w_flat // (KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE)
            rem1 = w_flat % (KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE)
            kd = rem1 // (KERNEL_SIZE * KERNEL_SIZE)
            rem2 = rem1 % (KERNEL_SIZE * KERNEL_SIZE)
            kh = rem2 // KERNEL_SIZE
            kw = rem2 % KERNEL_SIZE
            s_w[w_flat] = w[oc, ic, kd, kh, kw]

    S.syncthreads()

    for t in S.range(CONV_OUT_TILES):
        pos = t * THREADS_PER_BLOCK + tid
        if pos < CONV_OUT_ELEMS:
            od = pos // (OUT_H * OUT_W)
            rem = pos % (OUT_H * OUT_W)
            oh = rem // OUT_W
            ow = rem % OUT_W

            acc = S.convert(b[oc], S.f32)

            for ic in S.range(IN_CHANNELS):
                for kd in S.range(KERNEL_SIZE):
                    id_nom = od * STRIDE + kd
                    if id_nom >= PADDING and id_nom < IN_D + PADDING:
                        id = id_nom - PADDING
                        for kh in S.range(KERNEL_SIZE):
                            ih_nom = oh * STRIDE + kh
                            if ih_nom >= PADDING and ih_nom < IN_H + PADDING:
                                ih = ih_nom - PADDING
                                for kw in S.range(KERNEL_SIZE):
                                    iw_nom = ow * STRIDE + kw
                                    if iw_nom >= PADDING and iw_nom < IN_W + PADDING:
                                        iw = iw_nom - PADDING
                                        wf = ic * (KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE) + kd * (KERNEL_SIZE * KERNEL_SIZE) + kh * KERNEL_SIZE + kw
                                        xv = S.convert(x[n, ic, id, ih, iw], S.f32)
                                        wv = S.convert(s_w[wf], S.f32)
                                        acc = acc + xv * wv

            out[n, oc, od, oh, ow] = S.convert(acc, S.bf16)


@substrate.jit
def min_reduce_dim_bf16_kernel(
    x: S.Tensor((BATCH_SIZE, OUT_CHANNELS, OUT_D, OUT_H, OUT_W), S.bf16),
    out: S.Tensor((BATCH_SIZE, OUT_CHANNELS, FINAL_H, FINAL_W), S.bf16),
):
    tid = S.thread_id(0)
    bid = S.block_id(0)

    spatial_pos = bid % (FINAL_H * FINAL_W)
    channel_block = bid // (FINAL_H * FINAL_W)

    n = channel_block // OUT_CHANNELS
    oc = channel_block % OUT_CHANNELS

    oh = spatial_pos // FINAL_W
    ow = spatial_pos % FINAL_W

    # Each thread processes one spatial position across all depth values
    min_val = S.convert(x[n, oc, 0, oh, ow], S.f32)

    for d in S.range(1, MIN_REDUCE_DIM):
        v = S.convert(x[n, oc, d, oh, ow], S.f32)
        if v < min_val:
            min_val = v

    out[n, oc, oh, ow] = S.convert(min_val, S.bf16)


@substrate.jit
def softmax_channels_bf16_kernel(
    x: S.Tensor((BATCH_SIZE, OUT_CHANNELS, FINAL_H, FINAL_W), S.bf16),
    out: S.Tensor((BATCH_SIZE, OUT_CHANNELS, FINAL_H, FINAL_W), S.bf16),
):
    tid = S.thread_id(0)
    bid = S.block_id(0)

    spatial_total = FINAL_H * FINAL_W
    n = bid // spatial_total
    spatial_pos = bid % spatial_total

    oh = spatial_pos // FINAL_W
    ow = spatial_pos % FINAL_W

    # Find max for numerical stability
    max_val = S.convert(x[n, 0, oh, ow], S.f32)
    for c in S.range(1, SOFTMAX_CHANNELS):
        v = S.convert(x[n, c, oh, ow], S.f32)
        if v > max_val:
            max_val = v

    # Compute exp(x - max) and sum
    log2e = S.convert(LOG2E, S.f32)
    sum_exp = S.convert(0.0, S.f32)
    exp_vals = S.make_local((SOFTMAX_CHANNELS,), S.f32)

    for c in S.range(SOFTMAX_CHANNELS):
        v = S.convert(x[n, c, oh, ow], S.f32)
        shifted = v - max_val
        exp_v = S.exp2(shifted * log2e)
        exp_vals[c] = exp_v
        sum_exp = sum_exp + exp_v

    # Normalize and store
    inv_sum = S.convert(1.0, S.f32) / sum_exp
    for c in S.range(SOFTMAX_CHANNELS):
        out[n, c, oh, ow] = S.convert(exp_vals[c] * inv_sum, S.bf16)


def _launch_conv3d_bf16(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty(
        (BATCH_SIZE, OUT_CHANNELS, OUT_D, OUT_H, OUT_W),
        device=x.device,
        dtype=torch.bfloat16
    )
    conv3d_bf16_kernel[
        lambda: ((BATCH_SIZE * OUT_CHANNELS, 1, 1), (THREADS_PER_BLOCK, 1, 1))
    ](x, w, b, out)
    return out


def _launch_min_reduce_bf16(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty(
        (BATCH_SIZE, OUT_CHANNELS, FINAL_H, FINAL_W),
        device=x.device,
        dtype=torch.bfloat16
    )
    grid = BATCH_SIZE * OUT_CHANNELS * FINAL_H * FINAL_W
    min_reduce_dim_bf16_kernel[
        lambda: ((grid, 1, 1), (1, 1, 1))
    ](x, out)
    return out


def _launch_softmax_bf16(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    grid = BATCH_SIZE * FINAL_H * FINAL_W
    softmax_channels_bf16_kernel[
        lambda: ((grid, 1, 1), (1, 1, 1))
    ](x, out)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.dim = dim

    def forward(self, x):
        if tuple(x.shape) != (BATCH_SIZE, IN_CHANNELS, IN_D, IN_H, IN_W):
            raise NotImplementedError(
                f"ModelNew currently supports input shape {(BATCH_SIZE, IN_CHANNELS, IN_D, IN_H, IN_W)}, got {tuple(x.shape)}"
            )

        original_device = x.device
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/HIP device is required for Substrate kernels.")

        if not x.is_cuda:
            x = x.cuda()

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

        # Step 1: Conv3d
        x = _launch_conv3d_bf16(x, w, b)

        # Step 2: Min reduction along depth dimension
        x = _launch_min_reduce_bf16(x)

        # Step 3: Softmax along channel dimension
        x = _launch_softmax_bf16(x)

        if original_device.type != "cuda":
            x = x.to(original_device)
        return x


batch_size = 128
in_channels = 3
out_channels = 24
D, H, W = 24, 32, 32
kernel_size = 3
dim = 2


def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, dim]
