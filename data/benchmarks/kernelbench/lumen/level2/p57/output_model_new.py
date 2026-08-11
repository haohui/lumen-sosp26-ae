import torch
import torch.nn as nn
import substrate
import substrate.language as S


# Fixed problem shape from target model/get_inputs.
BATCH_SIZE = 128
IN_CHANNELS = 8
OUT_CHANNELS = 64
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
WEIGHT_ELEMS = IN_CHANNELS * K_H * K_W  # 72
WEIGHT_LOAD_ITERS = (WEIGHT_ELEMS + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
SPATIAL_ELEMS = OUT_H * OUT_W  # 15876
SPATIAL_TILES = (SPATIAL_ELEMS + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK


@substrate.jit
def conv2d_relu_hardswish_bf16_kernel(
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

            # Fused ReLU: max(0, acc)
            zero = S.convert(0.0, S.f32)
            if acc < zero:
                acc = zero

            # Fused HardSwish: x * clamp((x + 3) / 6, 0, 1)
            # = x * clamp(x/6 + 0.5, 0, 1)
            one = S.convert(1.0, S.f32)
            three = S.convert(3.0, S.f32)
            six = S.convert(6.0, S.f32)

            # t = (x + 3) / 6 = x/6 + 0.5
            t = acc / six + S.convert(0.5, S.f32)

            # clamp(t, 0, 1)
            if t < zero:
                t = zero
            if t > one:
                t = one

            # hardswish_out = x * t
            hardswish_out = acc * t

            out[n, oc, oh, ow] = S.convert(hardswish_out, S.bf16)


@substrate.jit
def conv2d_relu_hardswish_f32_kernel(
    x: S.Tensor((BATCH_SIZE, IN_CHANNELS, IN_H, IN_W), S.f32),
    w: S.Tensor((OUT_CHANNELS, IN_CHANNELS, K_H, K_W), S.f32),
    b: S.Tensor((OUT_CHANNELS,), S.f32),
    out: S.Tensor((BATCH_SIZE, OUT_CHANNELS, OUT_H, OUT_W), S.f32),
):
    tid = S.thread_id(0)
    bid = S.block_id(0)

    n = bid // OUT_CHANNELS
    oc = bid % OUT_CHANNELS

    # Shared memory tile for one output channel's kernel weights.
    s_w = S.make_shared((WEIGHT_ELEMS,), S.f32)

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

            acc = b[oc]

            for ic in S.range(IN_CHANNELS):
                for kh in S.range(K_H):
                    ih = oh * STRIDE_H + kh
                    if ih < IN_H:
                        for kw in S.range(K_W):
                            iw = ow * STRIDE_W + kw
                            if iw < IN_W:
                                wf = ic * (K_H * K_W) + kh * K_W + kw
                                acc = acc + x[n, ic, ih, iw] * s_w[wf]

            # Fused ReLU: max(0, acc)
            zero = S.convert(0.0, S.f32)
            if acc < zero:
                acc = zero

            # Fused HardSwish: x * clamp((x + 3) / 6, 0, 1)
            one = S.convert(1.0, S.f32)
            six = S.convert(6.0, S.f32)

            # t = (x + 3) / 6 = x/6 + 0.5
            t = acc / six + S.convert(0.5, S.f32)

            # clamp(t, 0, 1)
            if t < zero:
                t = zero
            if t > one:
                t = one

            # hardswish_out = x * t
            hardswish_out = acc * t

            out[n, oc, oh, ow] = hardswish_out


def _launch_conv_bf16(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty((BATCH_SIZE, OUT_CHANNELS, OUT_H, OUT_W), device=x.device, dtype=torch.bfloat16)
    conv2d_relu_hardswish_bf16_kernel[
        lambda: ((BATCH_SIZE * OUT_CHANNELS, 1, 1), (THREADS_PER_BLOCK, 1, 1))
    ](x, w, b, out)
    return out


def _launch_conv_f32(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty((BATCH_SIZE, OUT_CHANNELS, OUT_H, OUT_W), device=x.device, dtype=torch.float32)
    conv2d_relu_hardswish_f32_kernel[
        lambda: ((BATCH_SIZE * OUT_CHANNELS, 1, 1), (THREADS_PER_BLOCK, 1, 1))
    ](x, w, b, out)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

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

        if x.dtype == torch.float32 and w.dtype == torch.float32 and b.dtype == torch.float32:
            out = _launch_conv_f32(x, w, b)
        elif x.dtype == torch.bfloat16 and w.dtype == torch.bfloat16 and b.dtype == torch.bfloat16:
            out = _launch_conv_bf16(x, w, b)
        else:
            raise TypeError(
                f"Unsupported dtype combination: x={x.dtype}, weight={w.dtype}, bias={b.dtype}. "
                "Supported: float32 or bfloat16 (matching dtypes)."
            )

        if original_device.type != "cuda":
            out = out.to(original_device)
        return out


batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
