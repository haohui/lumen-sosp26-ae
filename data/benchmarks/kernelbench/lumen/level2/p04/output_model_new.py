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
WEIGHT_ELEMS = IN_CHANNELS * K_H * K_W  # 64 * 3 * 3 = 576
WEIGHT_LOAD_ITERS = (WEIGHT_ELEMS + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
SPATIAL_ELEMS = OUT_H * OUT_W  # 254 * 254 = 64516
SPATIAL_TILES = (SPATIAL_ELEMS + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

# Mish constants
LOG2E = 1.4426950408889634


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
        xv = S.convert(x[idx], S.f32)
        log2e = S.convert(LOG2E, S.f32)
        one = S.convert(1.0, S.f32)

        # softplus(x) = ln(1 + exp(x)) = ln(1 + exp2(x * log2(e)))
        exp_val = S.exp2(xv * log2e)
        softplus_val = S.log(one + exp_val)

        # mish(x) = x * tanh(softplus(x))
        tanh_val = S.tanh(softplus_val)
        mish_val = xv * tanh_val

        y[idx] = S.convert(mish_val, S.bf16)


def _launch_conv_bf16(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty((BATCH_SIZE, OUT_CHANNELS, OUT_H, OUT_W), device=x.device, dtype=torch.bfloat16)
    conv2d_3x3_s1_p0_bf16_kernel[
        lambda: ((BATCH_SIZE * OUT_CHANNELS, 1, 1), (THREADS_PER_BLOCK, 1, 1))
    ](x, w, b, out)
    return out


def substrate_mish(x: torch.Tensor) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for Substrate kernels.")

    moved = False
    if not x.is_cuda:
        x_dev = x.cuda()
        moved = True
    else:
        x_dev = x

    x_contig = x_dev.contiguous()
    out = torch.empty_like(x_contig)

    n = x_contig.numel()
    if n > 0:
        block_size = 256
        grid = (n + block_size - 1) // block_size
        x_flat = x_contig.view(-1)
        y_flat = out.view(-1)

        if x_contig.dtype == torch.bfloat16:
            mish_bf16_kernel[lambda: ((grid, 1, 1), (block_size, 1, 1))](x_flat, y_flat, n)
        else:
            raise TypeError(f"Unsupported dtype for substrate_mish: {x_contig.dtype}")

    if moved:
        return out.cpu()
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

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

        # Convert to BF16 for the kernel
        x_bf16 = x.to(torch.bfloat16)
        w_bf16 = w.to(torch.bfloat16)
        b_bf16 = b.to(torch.bfloat16)

        # Conv2d
        out = _launch_conv_bf16(x_bf16, w_bf16, b_bf16)

        # Mish activation (twice)
        out = substrate_mish(out)
        out = substrate_mish(out)

        if original_device.type != "cuda":
            out = out.to(original_device)
        return out


batch_size = 64
in_channels = 64
out_channels = 128
height = width = 256
kernel_size = 3


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
