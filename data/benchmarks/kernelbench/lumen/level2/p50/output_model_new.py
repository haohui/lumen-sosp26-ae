import torch
import torch.nn as nn
import substrate
import substrate.language as S


# Problem configuration from target model
BATCH_SIZE = 128
IN_CHANNELS = 3
OUT_CHANNELS = 16
IN_D, IN_H, IN_W = 16, 32, 32
KERNEL_SIZE = 3
STRIDE = 2
PADDING = 1

# ConvTranspose3d output dimensions
# D_out = (D_in - 1) * stride - 2 * padding + kernel_size
OUT_D = (IN_D - 1) * STRIDE - 2 * PADDING + KERNEL_SIZE  # 31
OUT_H = (IN_H - 1) * STRIDE - 2 * PADDING + KERNEL_SIZE  # 63
OUT_W = (IN_W - 1) * STRIDE - 2 * PADDING + KERNEL_SIZE  # 63

# AvgPool3d output dimensions (kernel_size=2, stride=2 by default)
POOL_D = OUT_D // 2  # 15
POOL_H = OUT_H // 2  # 31
POOL_W = OUT_W // 2  # 31

# Kernel launch parameters
THREADS = 256

# Pool normalization: 1/(2*2*2) = 1/8
INVERSE_POOL_KERNEL = 0.125


# ========== ConvTranspose3d Kernel ==========

@substrate.jit
def conv_transpose3d_bf16_kernel(
    x_ptr: S.Pointer(S.bf16),
    w_ptr: S.Pointer(S.bf16),
    b_ptr: S.Pointer(S.bf16),
    out_ptr: S.Pointer(S.bf16),
    n: S.u32,
    oc: S.u32,
    od: S.u32,
    oh: S.u32,
    ow: S.u32,
):
    """Compute one output element of ConvTranspose3d with bias."""
    tid = S.thread_id(0)
    bid = S.block_id(0)

    total_spatial = od * oh * ow
    total_oc_spatial = oc * total_spatial
    total_n = n * total_oc_spatial

    idx = bid * THREADS + tid
    if idx >= total_n:
        return

    n_idx = idx // total_oc_spatial
    rem = idx - n_idx * total_oc_spatial
    oc_idx = rem // total_spatial
    rem = rem - oc_idx * total_spatial
    od_idx = rem // (oh * ow)
    rem = rem - od_idx * oh * ow
    oh_idx = rem // ow
    ow_idx = rem - oh_idx * ow

    x_layout = S.make_layout(
        (n, IN_CHANNELS, IN_D, IN_H, IN_W),
        (IN_CHANNELS * IN_D * IN_H * IN_W, IN_D * IN_H * IN_W, IN_H * IN_W, IN_W, 1)
    )
    w_layout = S.make_layout(
        (IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, KERNEL_SIZE, KERNEL_SIZE),
        (OUT_CHANNELS * KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE,
         KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE, KERNEL_SIZE * KERNEL_SIZE, KERNEL_SIZE, 1)
    )
    out_layout = S.make_layout(
        (n, OUT_CHANNELS, OUT_D, OUT_H, OUT_W),
        (OUT_CHANNELS * OUT_D * OUT_H * OUT_W, OUT_D * OUT_H * OUT_W, OUT_H * OUT_W, OUT_W, 1)
    )
    bias_layout = S.make_layout((OUT_CHANNELS,), (1,))

    x = S.make_tensor(x_ptr, S.bf16, x_layout)
    w = S.make_tensor(w_ptr, S.bf16, w_layout)
    out = S.make_tensor(out_ptr, S.bf16, out_layout)
    bias = S.make_tensor(b_ptr, S.bf16, bias_layout)

    # Start with bias value for this output channel
    acc = S.convert(bias[oc_idx], S.f32)

    for ic in S.range(IN_CHANNELS):
        for kd in S.range(KERNEL_SIZE):
            id_nom = od_idx - kd + PADDING
            if id_nom >= 0:
                if id_nom % STRIDE == 0:
                    id_idx = id_nom // STRIDE
                    if id_idx < IN_D:
                        for kh in S.range(KERNEL_SIZE):
                            ih_nom = oh_idx - kh + PADDING
                            if ih_nom >= 0:
                                if ih_nom % STRIDE == 0:
                                    ih_idx = ih_nom // STRIDE
                                    if ih_idx < IN_H:
                                        for kw in S.range(KERNEL_SIZE):
                                            iw_nom = ow_idx - kw + PADDING
                                            if iw_nom >= 0:
                                                if iw_nom % STRIDE == 0:
                                                    iw_idx = iw_nom // STRIDE
                                                    if iw_idx < IN_W:
                                                        xv = S.convert(x[n_idx, ic, id_idx, ih_idx, iw_idx], S.f32)
                                                        wv = S.convert(w[ic, oc_idx, kd, kh, kw], S.f32)
                                                        acc = acc + xv * wv

    out[n_idx, oc_idx, od_idx, oh_idx, ow_idx] = S.convert(acc, S.bf16)


@substrate.jit
def conv_transpose3d_f32_kernel(
    x_ptr: S.Pointer(S.f32),
    w_ptr: S.Pointer(S.f32),
    b_ptr: S.Pointer(S.f32),
    out_ptr: S.Pointer(S.f32),
    n: S.u32,
    oc: S.u32,
    od: S.u32,
    oh: S.u32,
    ow: S.u32,
):
    """Compute one output element of ConvTranspose3d with bias."""
    tid = S.thread_id(0)
    bid = S.block_id(0)

    total_spatial = od * oh * ow
    total_oc_spatial = oc * total_spatial
    total_n = n * total_oc_spatial

    idx = bid * THREADS + tid
    if idx >= total_n:
        return

    n_idx = idx // total_oc_spatial
    rem = idx - n_idx * total_oc_spatial
    oc_idx = rem // total_spatial
    rem = rem - oc_idx * total_spatial
    od_idx = rem // (oh * ow)
    rem = rem - od_idx * oh * ow
    oh_idx = rem // ow
    ow_idx = rem - oh_idx * ow

    x_layout = S.make_layout(
        (n, IN_CHANNELS, IN_D, IN_H, IN_W),
        (IN_CHANNELS * IN_D * IN_H * IN_W, IN_D * IN_H * IN_W, IN_H * IN_W, IN_W, 1)
    )
    w_layout = S.make_layout(
        (IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, KERNEL_SIZE, KERNEL_SIZE),
        (OUT_CHANNELS * KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE,
         KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE, KERNEL_SIZE * KERNEL_SIZE, KERNEL_SIZE, 1)
    )
    out_layout = S.make_layout(
        (n, OUT_CHANNELS, OUT_D, OUT_H, OUT_W),
        (OUT_CHANNELS * OUT_D * OUT_H * OUT_W, OUT_D * OUT_H * OUT_W, OUT_H * OUT_W, OUT_W, 1)
    )
    bias_layout = S.make_layout((OUT_CHANNELS,), (1,))

    x = S.make_tensor(x_ptr, S.f32, x_layout)
    w = S.make_tensor(w_ptr, S.f32, w_layout)
    out = S.make_tensor(out_ptr, S.f32, out_layout)
    bias = S.make_tensor(b_ptr, S.f32, bias_layout)

    acc = bias[oc_idx]

    for ic in S.range(IN_CHANNELS):
        for kd in S.range(KERNEL_SIZE):
            id_nom = od_idx - kd + PADDING
            if id_nom >= 0:
                if id_nom % STRIDE == 0:
                    id_idx = id_nom // STRIDE
                    if id_idx < IN_D:
                        for kh in S.range(KERNEL_SIZE):
                            ih_nom = oh_idx - kh + PADDING
                            if ih_nom >= 0:
                                if ih_nom % STRIDE == 0:
                                    ih_idx = ih_nom // STRIDE
                                    if ih_idx < IN_H:
                                        for kw in S.range(KERNEL_SIZE):
                                            iw_nom = ow_idx - kw + PADDING
                                            if iw_nom >= 0:
                                                if iw_nom % STRIDE == 0:
                                                    iw_idx = iw_nom // STRIDE
                                                    if iw_idx < IN_W:
                                                        acc = acc + x[n_idx, ic, id_idx, ih_idx, iw_idx] * w[ic, oc_idx, kd, kh, kw]

    out[n_idx, oc_idx, od_idx, oh_idx, ow_idx] = acc


# ========== Scale + AvgPool3d Fused Kernel ==========

@substrate.jit
def scale_avgpool3d_bf16_kernel(
    x_ptr: S.Pointer(S.bf16),
    out_ptr: S.Pointer(S.bf16),
    n: S.u32,
    oc: S.u32,
    pd: S.u32,
    ph: S.u32,
    pw: S.u32,
    scale: S.constexpr,
):
    """Scale followed by AvgPool3d for 3D tensors."""
    tid = S.thread_id(0)
    bid = S.block_id(0)

    total_spatial = pd * ph * pw
    total_oc_spatial = oc * total_spatial
    total_n = n * total_oc_spatial

    idx = bid * THREADS + tid
    if idx >= total_n:
        return

    n_idx = idx // total_oc_spatial
    rem = idx - n_idx * total_oc_spatial
    oc_idx = rem // total_spatial
    rem = rem - oc_idx * total_spatial
    pd_idx = rem // (ph * pw)
    rem = rem - pd_idx * ph * pw
    ph_idx = rem // pw
    pw_idx = rem - ph_idx * pw

    x_layout = S.make_layout(
        (n, oc, OUT_D, OUT_H, OUT_W),
        (oc * OUT_D * OUT_H * OUT_W, OUT_D * OUT_H * OUT_W, OUT_H * OUT_W, OUT_W, 1)
    )
    out_layout = S.make_layout(
        (n, oc, pd, ph, pw),
        (oc * pd * ph * pw, pd * ph * pw, ph * pw, pw, 1)
    )

    x = S.make_tensor(x_ptr, S.bf16, x_layout)
    out = S.make_tensor(out_ptr, S.bf16, out_layout)

    # AvgPool3d with kernel_size=2, stride=2
    # Apply scale first, then average
    acc = S.convert(0.0, S.f32)
    for dd in S.range(2):
        for dh in S.range(2):
            for dw in S.range(2):
                xd = pd_idx * 2 + dd
                xh = ph_idx * 2 + dh
                xw = pw_idx * 2 + dw
                v = S.convert(x[n_idx, oc_idx, xd, xh, xw], S.f32)
                acc = acc + v

    # scale * average = scale * (sum / 8) = (sum * scale) / 8
    scale_f32 = S.convert(scale, S.f32)
    inv_pool = S.convert(INVERSE_POOL_KERNEL, S.f32)
    result = acc * scale_f32 * inv_pool
    out[n_idx, oc_idx, pd_idx, ph_idx, pw_idx] = S.convert(result, S.bf16)


@substrate.jit
def scale_avgpool3d_f32_kernel(
    x_ptr: S.Pointer(S.f32),
    out_ptr: S.Pointer(S.f32),
    n: S.u32,
    oc: S.u32,
    pd: S.u32,
    ph: S.u32,
    pw: S.u32,
    scale: S.constexpr,
):
    """Scale followed by AvgPool3d for 3D tensors."""
    tid = S.thread_id(0)
    bid = S.block_id(0)

    total_spatial = pd * ph * pw
    total_oc_spatial = oc * total_spatial
    total_n = n * total_oc_spatial

    idx = bid * THREADS + tid
    if idx >= total_n:
        return

    n_idx = idx // total_oc_spatial
    rem = idx - n_idx * total_oc_spatial
    oc_idx = rem // total_spatial
    rem = rem - oc_idx * total_spatial
    pd_idx = rem // (ph * pw)
    rem = rem - pd_idx * ph * pw
    ph_idx = rem // pw
    pw_idx = rem - ph_idx * pw

    x_layout = S.make_layout(
        (n, oc, OUT_D, OUT_H, OUT_W),
        (oc * OUT_D * OUT_H * OUT_W, OUT_D * OUT_H * OUT_W, OUT_H * OUT_W, OUT_W, 1)
    )
    out_layout = S.make_layout(
        (n, oc, pd, ph, pw),
        (oc * pd * ph * pw, pd * ph * pw, ph * pw, pw, 1)
    )

    x = S.make_tensor(x_ptr, S.f32, x_layout)
    out = S.make_tensor(out_ptr, S.f32, out_layout)

    acc = S.convert(0.0, S.f32)
    for dd in S.range(2):
        for dh in S.range(2):
            for dw in S.range(2):
                xd = pd_idx * 2 + dd
                xh = ph_idx * 2 + dh
                xw = pw_idx * 2 + dw
                acc = acc + x[n_idx, oc_idx, xd, xh, xw]

    scale_f32 = S.convert(scale, S.f32)
    inv_pool = S.convert(INVERSE_POOL_KERNEL, S.f32)
    result = acc * scale_f32 * inv_pool
    out[n_idx, oc_idx, pd_idx, ph_idx, pw_idx] = result


# ========== Bias + Scale Kernel ==========

@substrate.jit
def bias_scale_bf16_kernel(
    x_ptr: S.Pointer(S.bf16),
    bias_ptr: S.Pointer(S.bf16),
    out_ptr: S.Pointer(S.bf16),
    n: S.u32,
    oc: S.u32,
    pd: S.u32,
    ph: S.u32,
    pw: S.u32,
    scale: S.constexpr,
):
    """Add bias and scale for 3D tensors."""
    tid = S.thread_id(0)
    bid = S.block_id(0)

    total_spatial = pd * ph * pw
    total_oc_spatial = oc * total_spatial
    total_n = n * total_oc_spatial

    idx = bid * THREADS + tid
    if idx >= total_n:
        return

    n_idx = idx // total_oc_spatial
    rem = idx - n_idx * total_oc_spatial
    oc_idx = rem // total_spatial
    rem = rem - oc_idx * total_spatial
    pd_idx = rem // (ph * pw)
    rem = rem - pd_idx * ph * pw
    ph_idx = rem // pw
    pw_idx = rem - ph_idx * pw

    x_layout = S.make_layout(
        (n, oc, pd, ph, pw),
        (oc * pd * ph * pw, pd * ph * pw, ph * pw, pw, 1)
    )
    bias_layout = S.make_layout((oc,), (1,))

    x = S.make_tensor(x_ptr, S.bf16, x_layout)
    out = S.make_tensor(out_ptr, S.bf16, x_layout)
    bias = S.make_tensor(bias_ptr, S.bf16, bias_layout)

    xv = S.convert(x[n_idx, oc_idx, pd_idx, ph_idx, pw_idx], S.f32)
    bv = S.convert(bias[oc_idx], S.f32)
    scale_f32 = S.convert(scale, S.f32)
    result = (xv + bv) * scale_f32
    out[n_idx, oc_idx, pd_idx, ph_idx, pw_idx] = S.convert(result, S.bf16)


@substrate.jit
def bias_scale_f32_kernel(
    x_ptr: S.Pointer(S.f32),
    bias_ptr: S.Pointer(S.f32),
    out_ptr: S.Pointer(S.f32),
    n: S.u32,
    oc: S.u32,
    pd: S.u32,
    ph: S.u32,
    pw: S.u32,
    scale: S.constexpr,
):
    """Add bias and scale for 3D tensors."""
    tid = S.thread_id(0)
    bid = S.block_id(0)

    total_spatial = pd * ph * pw
    total_oc_spatial = oc * total_spatial
    total_n = n * total_oc_spatial

    idx = bid * THREADS + tid
    if idx >= total_n:
        return

    n_idx = idx // total_oc_spatial
    rem = idx - n_idx * total_oc_spatial
    oc_idx = rem // total_spatial
    rem = rem - oc_idx * total_spatial
    pd_idx = rem // (ph * pw)
    rem = rem - pd_idx * ph * pw
    ph_idx = rem // pw
    pw_idx = rem - ph_idx * pw

    x_layout = S.make_layout(
        (n, oc, pd, ph, pw),
        (oc * pd * ph * pw, pd * ph * pw, ph * pw, pw, 1)
    )
    bias_layout = S.make_layout((oc,), (1,))

    x = S.make_tensor(x_ptr, S.f32, x_layout)
    out = S.make_tensor(out_ptr, S.f32, x_layout)
    bias = S.make_tensor(bias_ptr, S.f32, bias_layout)

    xv = x[n_idx, oc_idx, pd_idx, ph_idx, pw_idx]
    bv = bias[oc_idx]
    scale_f32 = S.convert(scale, S.f32)
    result = (xv + bv) * scale_f32
    out[n_idx, oc_idx, pd_idx, ph_idx, pw_idx] = result


# ========== Host Wrappers ==========

def _launch_conv_transpose3d_bf16(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty((BATCH_SIZE, OUT_CHANNELS, OUT_D, OUT_H, OUT_W),
                      device=x.device, dtype=torch.bfloat16)
    total = BATCH_SIZE * OUT_CHANNELS * OUT_D * OUT_H * OUT_W
    blocks = (total + THREADS - 1) // THREADS
    conv_transpose3d_bf16_kernel[
        lambda: ((blocks, 1, 1), (THREADS, 1, 1))
    ](x, w, b, out, BATCH_SIZE, OUT_CHANNELS, OUT_D, OUT_H, OUT_W)
    return out


def _launch_conv_transpose3d_f32(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty((BATCH_SIZE, OUT_CHANNELS, OUT_D, OUT_H, OUT_W),
                      device=x.device, dtype=torch.float32)
    total = BATCH_SIZE * OUT_CHANNELS * OUT_D * OUT_H * OUT_W
    blocks = (total + THREADS - 1) // THREADS
    conv_transpose3d_f32_kernel[
        lambda: ((blocks, 1, 1), (THREADS, 1, 1))
    ](x, w, b, out, BATCH_SIZE, OUT_CHANNELS, OUT_D, OUT_H, OUT_W)
    return out


def _launch_scale_avgpool3d_bf16(x: torch.Tensor, scale: float) -> torch.Tensor:
    out = torch.empty((BATCH_SIZE, OUT_CHANNELS, POOL_D, POOL_H, POOL_W),
                      device=x.device, dtype=torch.bfloat16)
    total = BATCH_SIZE * OUT_CHANNELS * POOL_D * POOL_H * POOL_W
    blocks = (total + THREADS - 1) // THREADS
    scale_avgpool3d_bf16_kernel[
        lambda: ((blocks, 1, 1), (THREADS, 1, 1))
    ](x, out, BATCH_SIZE, OUT_CHANNELS, POOL_D, POOL_H, POOL_W, scale)
    return out


def _launch_scale_avgpool3d_f32(x: torch.Tensor, scale: float) -> torch.Tensor:
    out = torch.empty((BATCH_SIZE, OUT_CHANNELS, POOL_D, POOL_H, POOL_W),
                      device=x.device, dtype=torch.float32)
    total = BATCH_SIZE * OUT_CHANNELS * POOL_D * POOL_H * POOL_W
    blocks = (total + THREADS - 1) // THREADS
    scale_avgpool3d_f32_kernel[
        lambda: ((blocks, 1, 1), (THREADS, 1, 1))
    ](x, out, BATCH_SIZE, OUT_CHANNELS, POOL_D, POOL_H, POOL_W, scale)
    return out


def _launch_bias_scale_bf16(x: torch.Tensor, bias: torch.Tensor, scale: float) -> torch.Tensor:
    out = torch.empty_like(x)
    total = BATCH_SIZE * OUT_CHANNELS * POOL_D * POOL_H * POOL_W
    blocks = (total + THREADS - 1) // THREADS
    bias_scale_bf16_kernel[
        lambda: ((blocks, 1, 1), (THREADS, 1, 1))
    ](x, bias, out, BATCH_SIZE, OUT_CHANNELS, POOL_D, POOL_H, POOL_W, scale)
    return out


def _launch_bias_scale_f32(x: torch.Tensor, bias: torch.Tensor, scale: float) -> torch.Tensor:
    out = torch.empty_like(x)
    total = BATCH_SIZE * OUT_CHANNELS * POOL_D * POOL_H * POOL_W
    blocks = (total + THREADS - 1) // THREADS
    bias_scale_f32_kernel[
        lambda: ((blocks, 1, 1), (THREADS, 1, 1))
    ](x, bias, out, BATCH_SIZE, OUT_CHANNELS, POOL_D, POOL_H, POOL_W, scale)
    return out


class ModelNew(nn.Module):
    """
    Optimized model using Substrate GPU kernels for:
    - ConvTranspose3d (with bias)
    - Scaling
    - AvgPool3d
    - Bias addition
    - Scaling
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape):
        super(ModelNew, self).__init__()
        # Create ConvTranspose3d layer (to match original model parameter structure)
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        # Separate parameters
        self.scale1 = nn.Parameter(torch.tensor(scale1))
        self.avg_pool = nn.AvgPool3d(kernel_size=2)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale2 = nn.Parameter(torch.tensor(scale2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expected_shape = (BATCH_SIZE, IN_CHANNELS, IN_D, IN_H, IN_W)
        if tuple(x.shape) != expected_shape:
            raise NotImplementedError(
                f"ModelNew expects input shape {expected_shape}, got {tuple(x.shape)}"
            )

        orig_device = x.device
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/HIP device is required for Substrate kernels.")

        if not x.is_cuda:
            x = x.cuda()

        # Work in BF16 for efficiency
        input_dtype = x.dtype
        x = x.to(torch.bfloat16)

        # Get conv_transpose parameters
        conv_w = self.conv_transpose.weight.to(device=x.device, dtype=x.dtype).contiguous()
        conv_b = self.conv_transpose.bias.to(device=x.device, dtype=x.dtype).contiguous()
        x = x.contiguous()

        # Step 1: ConvTranspose3d with bias
        x = _launch_conv_transpose3d_bf16(x, conv_w, conv_b)

        # Step 2: Scale + AvgPool3d (fused)
        scale1_val = self.scale1.item()
        x = _launch_scale_avgpool3d_bf16(x, scale1_val)

        # Step 3: Bias + Scale (fused)
        separate_bias = self.bias.to(device=x.device, dtype=x.dtype).squeeze().contiguous()
        scale2_val = self.scale2.item()
        x = _launch_bias_scale_bf16(x, separate_bias, scale2_val)

        # Convert back to original dtype if needed
        if input_dtype != torch.bfloat16:
            x = x.to(input_dtype)

        if orig_device.type != "cuda":
            x = x.to(orig_device)

        return x


# Problem configuration for get_inputs / get_init_inputs
batch_size = BATCH_SIZE
in_channels = IN_CHANNELS
out_channels = OUT_CHANNELS
depth, height, width = IN_D, IN_H, IN_W
kernel_size = KERNEL_SIZE
stride = STRIDE
padding = PADDING
scale1 = 0.5
scale2 = 1.0
bias_shape = (out_channels, 1, 1, 1)


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape]
