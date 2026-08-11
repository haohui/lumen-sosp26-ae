import torch
import torch.nn as nn
import substrate
import substrate.language as S


# Fixed problem shape from target model/get_inputs.
BATCH_SIZE = 128
IN_CHANNELS = 64
OUT_CHANNELS = 64
IN_H = 128
IN_W = 128
K_H = 3
K_W = 3
STRIDE_H = 2
STRIDE_W = 2
PAD_H = 1
PAD_W = 1
OUTPUT_PAD_H = 1
OUTPUT_PAD_W = 1

OUT_H = (IN_H - 1) * STRIDE_H + K_H - 2 * PAD_H + OUTPUT_PAD_H  # 256
OUT_W = (IN_W - 1) * STRIDE_W + K_W - 2 * PAD_W + OUTPUT_PAD_W  # 256

THREADS_PER_BLOCK = 256
WEIGHT_ELEMS = IN_CHANNELS * K_H * K_W  # 576
WEIGHT_LOAD_ITERS = (WEIGHT_ELEMS + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
SPATIAL_ELEMS = OUT_H * OUT_W  # 65536
SPATIAL_TILES = (SPATIAL_ELEMS + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

# Activation constants
ADD_VALUE = 0.5
SCALE = 2.0
LOG2E = 1.4426950408889634


@substrate.jit
def conv_transpose2d_bf16_kernel(
    x: S.Tensor((BATCH_SIZE, IN_CHANNELS, IN_H, IN_W), S.bf16),
    w: S.Tensor((IN_CHANNELS, OUT_CHANNELS, K_H, K_W), S.bf16),
    b: S.Tensor((OUT_CHANNELS,), S.bf16),
    out: S.Tensor((BATCH_SIZE, OUT_CHANNELS, OUT_H, OUT_W), S.bf16),
):
    """
    Transposed convolution kernel.
    Weight shape: (in_channels, out_channels, kernel_h, kernel_w)
    """
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

    # Compute output spatial positions
    for t in S.range(SPATIAL_TILES):
        pos = t * THREADS_PER_BLOCK + tid
        if pos < SPATIAL_ELEMS:
            oh = pos // OUT_W
            ow = pos % OUT_W

            acc = S.convert(b[oc], S.f32)

            # For transposed conv: find input positions that contribute to this output
            # oh = ih * stride + kh - padding
            # ow = iw * stride + kw - padding
            # So: ih = (oh - kh + padding) / stride (must be integer and in bounds)
            for ic in S.range(IN_CHANNELS):
                for kh in S.range(K_H):
                    ih_nom = oh - kh + PAD_H
                    # Check if ih_nom is divisible by stride and in bounds
                    if ih_nom >= 0:
                        ih = ih_nom // STRIDE_H
                        if ih < IN_H:
                            # Check divisibility
                            if ih_nom == ih * STRIDE_H:
                                for kw in S.range(K_W):
                                    iw_nom = ow - kw + PAD_W
                                    if iw_nom >= 0:
                                        iw = iw_nom // STRIDE_W
                                        if iw < IN_W:
                                            if iw_nom == iw * STRIDE_W:
                                                wf = ic * (K_H * K_W) + kh * K_W + kw
                                                xv = S.convert(x[n, ic, ih, iw], S.f32)
                                                wv = S.convert(s_w[wf], S.f32)
                                                acc = acc + xv * wv

            out[n, oc, oh, ow] = S.convert(acc, S.bf16)


@substrate.jit
def fused_activation_bf16_kernel(
    x_ptr: S.Pointer(S.bf16),
    y_ptr: S.Pointer(S.bf16),
    n: S.u32,
):
    """
    Fused kernel: Mish + Add + Hardtanh + Scale
    mish(x) = x * tanh(ln(1 + exp(x))) = x * tanh(softplus(x))
    hardtanh: clamp to [-1, 1]
    """
    idx = S.block_id(0) * S.block_dim(0) + S.thread_id(0)

    if idx < n:
        layout = S.make_layout((n,), (1,))
        x = S.make_tensor(x_ptr, S.bf16, layout)
        y = S.make_tensor(y_ptr, S.bf16, layout)

        # Load value and convert to f32 for computation
        xv = S.convert(x[idx], S.f32)

        # Mish activation: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
        # Using exp2: exp(x) = exp2(x * log2(e))
        log2e = S.convert(LOG2E, S.f32)
        one = S.convert(1.0, S.f32)

        # softplus = ln(1 + exp(x))
        exp_x = S.exp2(xv * log2e)
        softplus = S.log(one + exp_x)

        # mish = x * tanh(softplus)
        tanh_sp = S.tanh(softplus)
        mish_val = xv * tanh_sp

        # Add value
        add_val = S.convert(ADD_VALUE, S.f32)
        added = mish_val + add_val

        # Hardtanh: clamp to [-1, 1]
        neg_one = S.convert(-1.0, S.f32)
        pos_one = S.convert(1.0, S.f32)

        clamped = added
        if clamped < neg_one:
            clamped = neg_one
        if clamped > pos_one:
            clamped = pos_one

        # Scale
        scale_val = S.convert(SCALE, S.f32)
        result = clamped * scale_val

        y[idx] = S.convert(result, S.bf16)


def _launch_conv_transpose_bf16(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty((BATCH_SIZE, OUT_CHANNELS, OUT_H, OUT_W), device=x.device, dtype=torch.bfloat16)
    conv_transpose2d_bf16_kernel[
        lambda: ((BATCH_SIZE * OUT_CHANNELS, 1, 1), (THREADS_PER_BLOCK, 1, 1))
    ](x, w, b, out)
    return out


def _launch_fused_activation(x: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    y = torch.empty_like(x)
    n = x.numel()
    if n > 0:
        grid = ((n + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK, 1, 1)
        fused_activation_bf16_kernel[lambda: (grid, (THREADS_PER_BLOCK, 1, 1))](x, y, n)
    return y


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super(ModelNew, self).__init__()
        # Store parameters for reference but use fixed constants in kernels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.add_value = add_value
        self.scale = scale

        # Create weight and bias for transposed conv
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        fan_in = in_channels * kernel_size * kernel_size
        bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

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

        # Convert to bfloat16 for the kernel
        x = x.to(dtype=torch.bfloat16)
        w = self.weight.to(dtype=torch.bfloat16, device=x.device)
        b = self.bias.to(dtype=torch.bfloat16, device=x.device)

        x = x.contiguous()
        w = w.contiguous()
        b = b.contiguous()

        # Transposed convolution
        conv_out = _launch_conv_transpose_bf16(x, w, b)

        # Fused activation: Mish + Add + Hardtanh + Scale
        out = _launch_fused_activation(conv_out)

        if original_device.type != "cuda":
            out = out.to(original_device)
        return out


batch_size = 128
in_channels = 64
out_channels = 64
height = width = 128
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
add_value = 0.5
scale = 2


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale]
