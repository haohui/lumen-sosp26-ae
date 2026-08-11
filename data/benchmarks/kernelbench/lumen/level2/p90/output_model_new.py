import torch
import torch.nn as nn
import substrate
import substrate.language as S


# Fixed problem shape from target model/get_inputs.
BATCH_SIZE = 128
IN_CHANNELS = 8
OUT_CHANNELS = 64
IN_D, IN_H, IN_W = 16, 64, 64
K_D, K_H, K_W = 3, 3, 3

# Output spatial dimensions (no padding, stride=1)
OUT_D = IN_D - K_D + 1  # 14
OUT_H = IN_H - K_H + 1  # 62
OUT_W = IN_W - K_W + 1  # 62

# Thread and tile configuration
THREADS = 256
WEIGHT_ELEMS = IN_CHANNELS * K_D * K_H * K_W  # 216
WEIGHT_LOAD_ITERS = (WEIGHT_ELEMS + THREADS - 1) // THREADS  # 1
SPATIAL_ELEMS = OUT_D * OUT_H * OUT_W  # 53816
SPATIAL_TILES = (SPATIAL_ELEMS + THREADS - 1) // THREADS  # 211

# Activation constants
LEAKY_RELU_NEG_SLOPE = 0.2
CLAMP_MIN = -1.0
CLAMP_MAX = 1.0
SQRT_2_OVER_PI = 0.7978845608028654
GELU_COEF = 0.044715


@substrate.jit
def conv3d_fused_bf16_kernel(
    x_ptr: S.Pointer(S.bf16),
    w_ptr: S.Pointer(S.bf16),
    b_ptr: S.Pointer(S.bf16),
    sum_ptr: S.Pointer(S.bf16),
    out_ptr: S.Pointer(S.bf16),
):
    """Fused 3D convolution + LeakyReLU + Add + Clamp + GELU kernel for BF16."""
    tid = S.thread_id(0)
    bid = S.block_id(0)

    n = bid // OUT_CHANNELS
    oc = bid % OUT_CHANNELS

    # Shared memory for one output channel's weights
    s_w = S.make_shared((WEIGHT_ELEMS,), S.bf16)

    # Load weights into shared memory
    for it in S.range(WEIGHT_LOAD_ITERS):
        w_flat = it * THREADS + tid
        if w_flat < WEIGHT_ELEMS:
            ic = w_flat // (K_D * K_H * K_W)
            rem = w_flat % (K_D * K_H * K_W)
            kd = rem // (K_H * K_W)
            rem2 = rem % (K_H * K_W)
            kh = rem2 // K_W
            kw = rem2 % K_W
            # Weight tensor layout: (OUT_CHANNELS, IN_CHANNELS, K_D, K_H, K_W)
            w_idx = oc * WEIGHT_ELEMS + ic * (K_D * K_H * K_W) + kd * (K_H * K_W) + kh * K_W + kw
            w_layout = S.make_layout((OUT_CHANNELS * WEIGHT_ELEMS,), (1,))
            w_tensor = S.make_tensor(w_ptr, S.bf16, w_layout)
            s_w[w_flat] = w_tensor[w_idx]

    S.syncthreads()

    # Create bias tensor accessor
    b_layout = S.make_layout((OUT_CHANNELS,), (1,))
    b_tensor = S.make_tensor(b_ptr, S.bf16, b_layout)

    # Create sum_tensor accessor
    sum_layout = S.make_layout((OUT_CHANNELS,), (1,))
    sum_tensor = S.make_tensor(sum_ptr, S.bf16, sum_layout)

    # Create input tensor accessor (B, IC, D, H, W)
    x_stride_b = IN_CHANNELS * IN_D * IN_H * IN_W
    x_stride_c = IN_D * IN_H * IN_W
    x_stride_d = IN_H * IN_W
    x_stride_h = IN_W
    x_layout = S.make_layout(
        (BATCH_SIZE * IN_CHANNELS * IN_D * IN_H * IN_W,),
        (1,)
    )
    x_tensor = S.make_tensor(x_ptr, S.bf16, x_layout)

    # Create output tensor accessor
    out_layout = S.make_layout(
        (BATCH_SIZE * OUT_CHANNELS * OUT_D * OUT_H * OUT_W,),
        (1,)
    )
    out_tensor = S.make_tensor(out_ptr, S.bf16, out_layout)

    # FP32 constants
    zero_f = S.convert(0.0, S.f32)
    neg_slope_f = S.convert(LEAKY_RELU_NEG_SLOPE, S.f32)
    clamp_min_f = S.convert(CLAMP_MIN, S.f32)
    clamp_max_f = S.convert(CLAMP_MAX, S.f32)
    sqrt_2_over_pi_f = S.convert(SQRT_2_OVER_PI, S.f32)
    gelu_coef_f = S.convert(GELU_COEF, S.f32)
    one_f = S.convert(1.0, S.f32)
    half_f = S.convert(0.5, S.f32)

    # Process spatial tiles
    for t in S.range(SPATIAL_TILES):
        pos = t * THREADS + tid
        if pos < SPATIAL_ELEMS:
            od = pos // (OUT_H * OUT_W)
            rem = pos % (OUT_H * OUT_W)
            oh = rem // OUT_W
            ow = rem % OUT_W

            # Start with bias in FP32
            acc = S.convert(b_tensor[oc], S.f32)

            # 3D Convolution
            for ic in S.range(IN_CHANNELS):
                for kd in S.range(K_D):
                    id_nom = od + kd
                    if id_nom < IN_D:
                        for kh in S.range(K_H):
                            ih_nom = oh + kh
                            if ih_nom < IN_H:
                                for kw in S.range(K_W):
                                    iw_nom = ow + kw
                                    if iw_nom < IN_W:
                                        # Weight index in shared memory
                                        wf = ic * (K_D * K_H * K_W) + kd * (K_H * K_W) + kh * K_W + kw
                                        # Input index
                                        x_idx = n * x_stride_b + ic * x_stride_c + id_nom * x_stride_d + ih_nom * x_stride_h + iw_nom
                                        xv = S.convert(x_tensor[x_idx], S.f32)
                                        wv = S.convert(s_w[wf], S.f32)
                                        acc = acc + xv * wv

            # LeakyReLU: max(x, 0) + negative_slope * min(x, 0)
            if acc < zero_f:
                acc = acc * neg_slope_f

            # Add sum_tensor (broadcast over batch and spatial dims)
            sum_v = S.convert(sum_tensor[oc], S.f32)
            acc = acc + sum_v

            # Clamp to [-1, 1]
            if acc < clamp_min_f:
                acc = clamp_min_f
            if acc > clamp_max_f:
                acc = clamp_max_f

            # GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            x3 = acc * acc * acc
            inner = sqrt_2_over_pi_f * (acc + gelu_coef_f * x3)
            tanh_inner = S.tanh(inner)
            gelu_out = half_f * acc * (one_f + tanh_inner)

            # Write output
            out_idx = n * (OUT_CHANNELS * OUT_D * OUT_H * OUT_W) + oc * (OUT_D * OUT_H * OUT_W) + od * (OUT_H * OUT_W) + oh * OUT_W + ow
            out_tensor[out_idx] = S.convert(gelu_out, S.bf16)


def _launch_conv3d_fused_bf16(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    sum_t: torch.Tensor,
) -> torch.Tensor:
    """Launch the fused convolution + activation kernel."""
    out = torch.empty(
        (BATCH_SIZE, OUT_CHANNELS, OUT_D, OUT_H, OUT_W),
        device=x.device,
        dtype=torch.bfloat16
    )

    # Ensure contiguous and correct dtype
    x = x.contiguous().to(torch.bfloat16)
    w = w.contiguous().to(torch.bfloat16)
    b = b.contiguous().to(torch.bfloat16)
    sum_t = sum_t.contiguous().to(torch.bfloat16)

    grid = (BATCH_SIZE * OUT_CHANNELS, 1, 1)
    block = (THREADS, 1, 1)

    conv3d_fused_bf16_kernel[lambda: (grid, block)](
        x.view(-1),
        w.view(-1),
        b.view(-1),
        sum_t.view(-1),
        out.view(-1)
    )

    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs 3D convolution + LeakyReLU + Add + Clamp + GELU
    using fused Substrate GPU kernels.
    """

    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_device = x.device

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/HIP device is required for Substrate kernels.")

        # Move to GPU if needed
        if not x.is_cuda:
            x = x.cuda()

        # Validate input shape
        if x.shape != (BATCH_SIZE, IN_CHANNELS, IN_D, IN_H, IN_W):
            raise NotImplementedError(
                f"ModelNew currently supports input shape "
                f"{(BATCH_SIZE, IN_CHANNELS, IN_D, IN_H, IN_W)}, got {tuple(x.shape)}"
            )

        # Get weights and bias from conv layer
        w = self.conv.weight
        b = self.conv.bias
        if b is None:
            b = torch.zeros((OUT_CHANNELS,), device=w.device, dtype=w.dtype)

        # Ensure parameters are on the same device as input
        if w.device != x.device:
            w = w.to(device=x.device)
        if b.device != x.device:
            b = b.to(device=x.device)

        sum_t = self.sum_tensor
        if sum_t.device != x.device:
            sum_t = sum_t.to(device=x.device)

        # Launch fused kernel
        out = _launch_conv3d_fused_bf16(x, w, b, sum_t)

        # Move back to original device if needed
        if original_device.type != "cuda":
            out = out.to(original_device)

        return out


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, sum_tensor_shape]


# These are module-level constants for get_inputs/get_init_inputs compatibility
batch_size = BATCH_SIZE
in_channels = IN_CHANNELS
out_channels = OUT_CHANNELS
depth, height, width = IN_D, IN_H, IN_W
kernel_size = K_D
sum_tensor_shape = (out_channels, 1, 1, 1)
