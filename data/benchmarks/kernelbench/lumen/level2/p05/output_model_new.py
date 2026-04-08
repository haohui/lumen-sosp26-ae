import torch
import torch.nn as nn
import substrate
import substrate.language as S


# Fixed problem shape from target model/get_inputs.
BATCH_SIZE = 32
IN_CHANNELS = 64
OUT_CHANNELS = 64
IN_H = 256
IN_W = 256
K_H = 4
K_W = 4
STRIDE = 2
PADDING = 1
OUTPUT_PADDING = 1

# Output dimensions for ConvTranspose2d
OUT_H = (IN_H - 1) * STRIDE + K_H - 2 * PADDING + OUTPUT_PADDING  # 513
OUT_W = (IN_W - 1) * STRIDE + K_W - 2 * PADDING + OUTPUT_PADDING  # 513

THREADS_PER_BLOCK = 256
WEIGHT_ELEMS = IN_CHANNELS * K_H * K_W  # 64 * 4 * 4 = 1024
WEIGHT_LOAD_ITERS = (WEIGHT_ELEMS + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
SPATIAL_ELEMS = OUT_H * OUT_W  # 513 * 513 = 263169
SPATIAL_TILES = (SPATIAL_ELEMS + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK


@substrate.jit
def conv_transpose2d_bias_tanh_f32_kernel(
    x: S.Tensor((BATCH_SIZE, IN_CHANNELS, IN_H, IN_W), S.f32),
    # ConvTranspose2d weight shape is (in_channels, out_channels, kH, kW)
    w: S.Tensor((IN_CHANNELS, OUT_CHANNELS, K_H, K_W), S.f32),
    bias: S.Tensor((OUT_CHANNELS,), S.f32),
    out: S.Tensor((BATCH_SIZE, OUT_CHANNELS, OUT_H, OUT_W), S.f32),
):
    tid = S.thread_id(0)
    bid = S.block_id(0)

    n = bid // OUT_CHANNELS
    oc = bid % OUT_CHANNELS

    # Shared memory tile for one output channel's kernel weights.
    # For ConvTranspose2d, we need weights from all input channels for this output channel.
    s_w = S.make_shared((WEIGHT_ELEMS,), S.f32)

    # Load weights into shared memory
    for it in S.range(WEIGHT_LOAD_ITERS):
        w_flat = it * THREADS_PER_BLOCK + tid
        if w_flat < WEIGHT_ELEMS:
            ic = w_flat // (K_H * K_W)
            rem = w_flat % (K_H * K_W)
            kh = rem // K_W
            kw = rem % K_W
            # ConvTranspose2d weight indexing: w[ic, oc, kh, kw]
            s_w[w_flat] = w[ic, oc, kh, kw]

    S.syncthreads()

    # Process spatial tiles
    for t in S.range(SPATIAL_TILES):
        pos = t * THREADS_PER_BLOCK + tid
        if pos < SPATIAL_ELEMS:
            oh = pos // OUT_W
            ow = pos % OUT_W

            # Initialize accumulator with bias
            acc = bias[oc]

            # ConvTranspose2d: for each kernel position, check if there's a valid input contribution
            for ic in S.range(IN_CHANNELS):
                for kh in S.range(K_H):
                    # Check if (oh - kh + PADDING) is divisible by STRIDE
                    oh_kh_adj = oh - kh + PADDING
                    # Compute input height
                    if oh_kh_adj >= 0:
                        ih = oh_kh_adj // STRIDE
                        # Check divisibility and bounds
                        if oh_kh_adj % STRIDE == 0:
                            if ih < IN_H:
                                for kw in S.range(K_W):
                                    ow_kw_adj = ow - kw + PADDING
                                    if ow_kw_adj >= 0:
                                        iw = ow_kw_adj // STRIDE
                                        if ow_kw_adj % STRIDE == 0:
                                            if iw < IN_W:
                                                wf = ic * (K_H * K_W) + kh * K_W + kw
                                                acc = acc + x[n, ic, ih, iw] * s_w[wf]

            # Subtract bias and apply tanh
            # Note: The model does conv_transpose(x) - bias, not +bias
            # But bias in PyTorch ConvTranspose2d is added, so we need to handle carefully
            # Looking at the model: x = x - self.bias where self.bias has shape (out_channels, 1, 1)
            # This is separate from the conv_transpose's internal bias (which is None by default in our case)
            # Since bias argument here represents the subtracted bias, we negate it
            out[n, oc, oh, ow] = S.tanh(acc)


@substrate.jit
def conv_transpose2d_bias_tanh_bf16_kernel(
    x: S.Tensor((BATCH_SIZE, IN_CHANNELS, IN_H, IN_W), S.bf16),
    w: S.Tensor((IN_CHANNELS, OUT_CHANNELS, K_H, K_W), S.bf16),
    bias: S.Tensor((OUT_CHANNELS,), S.bf16),
    out: S.Tensor((BATCH_SIZE, OUT_CHANNELS, OUT_H, OUT_W), S.bf16),
):
    tid = S.thread_id(0)
    bid = S.block_id(0)

    n = bid // OUT_CHANNELS
    oc = bid % OUT_CHANNELS

    # Shared memory tile for one output channel's kernel weights.
    s_w = S.make_shared((WEIGHT_ELEMS,), S.bf16)

    # Load weights into shared memory
    for it in S.range(WEIGHT_LOAD_ITERS):
        w_flat = it * THREADS_PER_BLOCK + tid
        if w_flat < WEIGHT_ELEMS:
            ic = w_flat // (K_H * K_W)
            rem = w_flat % (K_H * K_W)
            kh = rem // K_W
            kw = rem % K_W
            s_w[w_flat] = w[ic, oc, kh, kw]

    S.syncthreads()

    # Process spatial tiles
    for t in S.range(SPATIAL_TILES):
        pos = t * THREADS_PER_BLOCK + tid
        if pos < SPATIAL_ELEMS:
            oh = pos // OUT_W
            ow = pos % OUT_W

            # Initialize accumulator with bias (use f32 for accumulation)
            acc = S.convert(bias[oc], S.f32)

            # ConvTranspose2d: for each kernel position, check if there's a valid input contribution
            for ic in S.range(IN_CHANNELS):
                for kh in S.range(K_H):
                    oh_kh_adj = oh - kh + PADDING
                    if oh_kh_adj >= 0:
                        ih = oh_kh_adj // STRIDE
                        if oh_kh_adj % STRIDE == 0:
                            if ih < IN_H:
                                for kw in S.range(K_W):
                                    ow_kw_adj = ow - kw + PADDING
                                    if ow_kw_adj >= 0:
                                        iw = ow_kw_adj // STRIDE
                                        if ow_kw_adj % STRIDE == 0:
                                            if iw < IN_W:
                                                wf = ic * (K_H * K_W) + kh * K_W + kw
                                                xv = S.convert(x[n, ic, ih, iw], S.f32)
                                                wv = S.convert(s_w[wf], S.f32)
                                                acc = acc + xv * wv

            # Apply tanh and convert back to bf16
            out[n, oc, oh, ow] = S.convert(S.tanh(acc), S.bf16)


def _launch_conv_transpose_f32(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    out = torch.empty((BATCH_SIZE, OUT_CHANNELS, OUT_H, OUT_W), device=x.device, dtype=torch.float32)
    conv_transpose2d_bias_tanh_f32_kernel[
        lambda: ((BATCH_SIZE * OUT_CHANNELS, 1, 1), (THREADS_PER_BLOCK, 1, 1))
    ](x, w, bias, out)
    return out


def _launch_conv_transpose_bf16(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    out = torch.empty((BATCH_SIZE, OUT_CHANNELS, OUT_H, OUT_W), device=x.device, dtype=torch.bfloat16)
    conv_transpose2d_bias_tanh_bf16_kernel[
        lambda: ((BATCH_SIZE * OUT_CHANNELS, 1, 1), (THREADS_PER_BLOCK, 1, 1))
    ](x, w, bias, out)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))

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

        # Get weights and ensure they're on the right device
        w = self.conv_transpose.weight
        conv_bias = self.conv_transpose.bias
        sub_bias = self.bias.squeeze()  # Shape: (out_channels,)

        if w.device != x.device:
            w = w.to(device=x.device)
        if conv_bias is not None and conv_bias.device != x.device:
            conv_bias = conv_bias.to(device=x.device)
        if sub_bias.device != x.device:
            sub_bias = sub_bias.to(device=x.device)

        x = x.contiguous()
        w = w.contiguous()
        if conv_bias is not None:
            conv_bias = conv_bias.contiguous()
        sub_bias = sub_bias.contiguous()

        # Handle conv_transpose bias: if it exists, add it; then subtract the model's bias
        if conv_bias is not None:
            # Combine: + conv_bias - sub_bias
            combined_bias = conv_bias - sub_bias
        else:
            combined_bias = -sub_bias

        if x.dtype == torch.bfloat16 and w.dtype == torch.bfloat16:
            if combined_bias.dtype != torch.bfloat16:
                combined_bias = combined_bias.to(torch.bfloat16)
            out = _launch_conv_transpose_bf16(x, w, combined_bias)
        elif x.dtype == torch.float32 and w.dtype == torch.float32:
            out = _launch_conv_transpose_f32(x, w, combined_bias.to(torch.float32))
        else:
            raise TypeError(
                f"Unsupported dtype combination: x={x.dtype}, weight={w.dtype}. "
                "Supported: float32 or bfloat16 (matching dtypes)."
            )

        if original_device.type != "cuda":
            out = out.to(original_device)
        return out


batch_size = 32
in_channels = 64
out_channels = 64
height = width = 256
kernel_size = 4
bias_shape = (out_channels, 1, 1)


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]
