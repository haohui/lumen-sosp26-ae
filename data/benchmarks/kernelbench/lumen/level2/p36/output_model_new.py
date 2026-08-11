import torch
import torch.nn as nn
import substrate
import substrate.language as S


# Fixed problem shape from target model/get_inputs.
BATCH_SIZE = 16
IN_CHANNELS = 64
OUT_CHANNELS = 128
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

# Output dimensions after ConvTranspose2d
# H_out = (H_in - 1) * stride - 2 * padding + kernel_size + output_padding
OUT_H = (IN_H - 1) * STRIDE_H - 2 * PAD_H + K_H + OUTPUT_PAD_H  # 256
OUT_W = (IN_W - 1) * STRIDE_W - 2 * PAD_W + K_W + OUTPUT_PAD_W  # 256

# After min along channel: (BATCH_SIZE, 1, OUT_H, OUT_W)
# After sum along height: (BATCH_SIZE, 1, 1, OUT_W)

BLOCK_SIZE = 256

# GELU constants
SQRT_2_INV = 0.7071067811865475  # 1/sqrt(2)
GELU_HALF = 0.5


# ========== GELU Activation Kernels ==========

@substrate.jit
def gelu_f32_kernel(
    x_ptr: S.Pointer(S.f32),
    y_ptr: S.Pointer(S.f32),
    n: S.u32,
):
    idx = S.block_id(0) * S.block_dim(0) + S.thread_id(0)
    layout = S.make_layout((n,), (1,))
    x = S.make_tensor(x_ptr, S.f32, layout)
    y = S.make_tensor(y_ptr, S.f32, layout)

    if idx < n:
        xv = x[idx]
        half = S.convert(GELU_HALF, S.f32)
        sqrt2_inv = S.convert(SQRT_2_INV, S.f32)
        one = S.convert(1.0, S.f32)

        erf_arg = xv * sqrt2_inv
        erf_val = S.erf(erf_arg)
        y[idx] = xv * half * (one + erf_val)


@substrate.jit
def gelu_bf16_kernel(
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
        half = S.convert(GELU_HALF, S.f32)
        sqrt2_inv = S.convert(SQRT_2_INV, S.f32)
        one = S.convert(1.0, S.f32)

        erf_arg = xv * sqrt2_inv
        erf_val = S.erf(erf_arg)
        result = xv * half * (one + erf_val)
        y[idx] = S.convert(result, S.bf16)


# ========== Min Reduction Kernel (along channel dim) ==========

@substrate.jit
def min_dim1_bf16_kernel(
    x: S.Tensor((BATCH_SIZE, OUT_CHANNELS, OUT_H, OUT_W), S.bf16),
    out: S.Tensor((BATCH_SIZE, 1, OUT_H, OUT_W), S.bf16),
):
    tid = S.thread_id(0)
    bid = S.block_id(0)

    # Each block handles one (n, oh, ow) position
    n = bid // (OUT_H * OUT_W)
    rem = bid % (OUT_H * OUT_W)
    oh = rem // OUT_W
    ow = rem % OUT_W

    # Initialize with large positive value
    min_val = S.convert(1e38, S.f32)

    # Thread-cooperative min across channels
    for c_start in S.range((OUT_CHANNELS + BLOCK_SIZE - 1) // BLOCK_SIZE):
        c = c_start * BLOCK_SIZE + tid
        if c < OUT_CHANNELS:
            v = S.convert(x[n, c, oh, ow], S.f32)
            if v < min_val:
                min_val = v

    # Reduce within block using shared memory
    shm = S.make_shared((BLOCK_SIZE,), S.f32)
    shm[tid] = min_val
    S.syncthreads()

    # Tree reduction
    step = BLOCK_SIZE // 2
    for _s in S.range(8):  # log2(256) = 8
        if step > 0:
            if tid < step:
                other = shm[tid + step]
                if other < shm[tid]:
                    shm[tid] = other
            S.syncthreads()
            step = step // 2

    if tid == 0:
        out[n, 0, oh, ow] = S.convert(shm[0], S.bf16)


@substrate.jit
def min_dim1_f32_kernel(
    x: S.Tensor((BATCH_SIZE, OUT_CHANNELS, OUT_H, OUT_W), S.f32),
    out: S.Tensor((BATCH_SIZE, 1, OUT_H, OUT_W), S.f32),
):
    tid = S.thread_id(0)
    bid = S.block_id(0)

    n = bid // (OUT_H * OUT_W)
    rem = bid % (OUT_H * OUT_W)
    oh = rem // OUT_W
    ow = rem % OUT_W

    min_val = S.convert(1e38, S.f32)

    for c_start in S.range((OUT_CHANNELS + BLOCK_SIZE - 1) // BLOCK_SIZE):
        c = c_start * BLOCK_SIZE + tid
        if c < OUT_CHANNELS:
            v = x[n, c, oh, ow]
            if v < min_val:
                min_val = v

    shm = S.make_shared((BLOCK_SIZE,), S.f32)
    shm[tid] = min_val
    S.syncthreads()

    step = BLOCK_SIZE // 2
    for _s in S.range(8):
        if step > 0:
            if tid < step:
                other = shm[tid + step]
                if other < shm[tid]:
                    shm[tid] = other
            S.syncthreads()
            step = step // 2

    if tid == 0:
        out[n, 0, oh, ow] = shm[0]


# ========== Sum Reduction Kernel (along height dim) ==========

@substrate.jit
def sum_dim2_bf16_kernel(
    x: S.Tensor((BATCH_SIZE, 1, OUT_H, OUT_W), S.bf16),
    out: S.Tensor((BATCH_SIZE, 1, 1, OUT_W), S.bf16),
):
    tid = S.thread_id(0)
    bid = S.block_id(0)

    # Each block handles one (n, ow) position
    n = bid // OUT_W
    ow = bid % OUT_W

    acc = S.convert(0.0, S.f32)

    # Thread-cooperative sum across height
    for h_start in S.range((OUT_H + BLOCK_SIZE - 1) // BLOCK_SIZE):
        h = h_start * BLOCK_SIZE + tid
        if h < OUT_H:
            acc = acc + S.convert(x[n, 0, h, ow], S.f32)

    # Reduce within block
    shm = S.make_shared((BLOCK_SIZE,), S.f32)
    shm[tid] = acc
    S.syncthreads()

    step = BLOCK_SIZE // 2
    for _s in S.range(8):
        if step > 0:
            if tid < step:
                shm[tid] = shm[tid] + shm[tid + step]
            S.syncthreads()
            step = step // 2

    if tid == 0:
        out[n, 0, 0, ow] = S.convert(shm[0], S.bf16)


@substrate.jit
def sum_dim2_f32_kernel(
    x: S.Tensor((BATCH_SIZE, 1, OUT_H, OUT_W), S.f32),
    out: S.Tensor((BATCH_SIZE, 1, 1, OUT_W), S.f32),
):
    tid = S.thread_id(0)
    bid = S.block_id(0)

    n = bid // OUT_W
    ow = bid % OUT_W

    acc = S.convert(0.0, S.f32)

    for h_start in S.range((OUT_H + BLOCK_SIZE - 1) // BLOCK_SIZE):
        h = h_start * BLOCK_SIZE + tid
        if h < OUT_H:
            acc = acc + x[n, 0, h, ow]

    shm = S.make_shared((BLOCK_SIZE,), S.f32)
    shm[tid] = acc
    S.syncthreads()

    step = BLOCK_SIZE // 2
    for _s in S.range(8):
        if step > 0:
            if tid < step:
                shm[tid] = shm[tid] + shm[tid + step]
            S.syncthreads()
            step = step // 2

    if tid == 0:
        out[n, 0, 0, ow] = shm[0]


# ========== Elementwise Add Kernel ==========

@substrate.jit
def add_bias_f32_kernel(
    x_ptr: S.Pointer(S.f32),
    bias_ptr: S.Pointer(S.f32),
    y_ptr: S.Pointer(S.f32),
    n: S.u32,
):
    idx = S.block_id(0) * S.block_dim(0) + S.thread_id(0)
    layout = S.make_layout((n,), (1,))
    x = S.make_tensor(x_ptr, S.f32, layout)
    bias = S.make_tensor(bias_ptr, S.f32, layout)
    y = S.make_tensor(y_ptr, S.f32, layout)

    if idx < n:
        y[idx] = x[idx] + bias[0]


@substrate.jit
def add_bias_bf16_kernel(
    x_ptr: S.Pointer(S.bf16),
    bias_ptr: S.Pointer(S.bf16),
    y_ptr: S.Pointer(S.bf16),
    n: S.u32,
):
    idx = S.block_id(0) * S.block_dim(0) + S.thread_id(0)
    layout = S.make_layout((n,), (1,))
    x = S.make_tensor(x_ptr, S.bf16, layout)
    bias = S.make_tensor(bias_ptr, S.bf16, layout)
    y = S.make_tensor(y_ptr, S.bf16, layout)

    if idx < n:
        xv = S.convert(x[idx], S.f32)
        bv = S.convert(bias[0], S.f32)
        y[idx] = S.convert(xv + bv, S.bf16)


# ========== Host Wrapper Functions ==========

def substrate_min_dim1(x: torch.Tensor) -> torch.Tensor:
    """Min reduction along dimension 1 with keepdim=True."""
    assert x.shape == (BATCH_SIZE, OUT_CHANNELS, OUT_H, OUT_W), \
        f"Expected shape {(BATCH_SIZE, OUT_CHANNELS, OUT_H, OUT_W)}, got {tuple(x.shape)}"

    out = torch.empty((BATCH_SIZE, 1, OUT_H, OUT_W), device=x.device, dtype=x.dtype)
    grid = (BATCH_SIZE * OUT_H * OUT_W, 1, 1)

    if x.dtype == torch.bfloat16:
        min_dim1_bf16_kernel[lambda: (grid, (BLOCK_SIZE, 1, 1))](x, out)
    elif x.dtype == torch.float32:
        min_dim1_f32_kernel[lambda: (grid, (BLOCK_SIZE, 1, 1))](x, out)
    else:
        raise TypeError(f"Unsupported dtype: {x.dtype}")

    return out


def substrate_sum_dim2(x: torch.Tensor) -> torch.Tensor:
    """Sum reduction along dimension 2 with keepdim=True."""
    assert x.shape == (BATCH_SIZE, 1, OUT_H, OUT_W), \
        f"Expected shape {(BATCH_SIZE, 1, OUT_H, OUT_W)}, got {tuple(x.shape)}"

    out = torch.empty((BATCH_SIZE, 1, 1, OUT_W), device=x.device, dtype=x.dtype)
    grid = (BATCH_SIZE * OUT_W, 1, 1)

    if x.dtype == torch.bfloat16:
        sum_dim2_bf16_kernel[lambda: (grid, (BLOCK_SIZE, 1, 1))](x, out)
    elif x.dtype == torch.float32:
        sum_dim2_f32_kernel[lambda: (grid, (BLOCK_SIZE, 1, 1))](x, out)
    else:
        raise TypeError(f"Unsupported dtype: {x.dtype}")

    return out


def substrate_gelu(x: torch.Tensor) -> torch.Tensor:
    """GELU activation."""
    if x.numel() == 0:
        return torch.empty_like(x)

    out = torch.empty_like(x)
    n = x.numel()
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE, 1, 1)

    if x.dtype == torch.bfloat16:
        gelu_bf16_kernel[lambda: (grid, (BLOCK_SIZE, 1, 1))](x, out, n)
    elif x.dtype == torch.float32:
        gelu_f32_kernel[lambda: (grid, (BLOCK_SIZE, 1, 1))](x, out, n)
    else:
        raise TypeError(f"Unsupported dtype: {x.dtype}")

    return out


def substrate_add_bias(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Elementwise add with broadcast bias."""
    out = torch.empty_like(x)
    n = x.numel()
    if n == 0:
        return out

    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE, 1, 1)

    if x.dtype == torch.bfloat16:
        add_bias_bf16_kernel[lambda: (grid, (BLOCK_SIZE, 1, 1))](x, bias, out, n)
    elif x.dtype == torch.float32:
        add_bias_f32_kernel[lambda: (grid, (BLOCK_SIZE, 1, 1))](x, bias, out, n)
    else:
        raise TypeError(f"Unsupported dtype: {x.dtype}")

    return out


class ModelNew(nn.Module):
    """
    Optimized model using Substrate GPU kernels for:
    - Min reduction along channel dimension
    - Sum reduction along height dimension
    - GELU activation
    - Bias addition

    ConvTranspose2d uses PyTorch's optimized implementation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        # Move to GPU if needed
        orig_device = x.device
        if not x.is_cuda:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA/HIP device required for Substrate kernels")
            x = x.cuda()

        # Ensure contiguous
        x = x.contiguous()

        # Step 1: ConvTranspose2d (using PyTorch's optimized implementation)
        x = self.conv_transpose(x)
        x = x.contiguous()

        # Verify expected shape after conv_transpose
        if tuple(x.shape) != (BATCH_SIZE, OUT_CHANNELS, OUT_H, OUT_W):
            raise NotImplementedError(
                f"Expected shape {(BATCH_SIZE, OUT_CHANNELS, OUT_H, OUT_W)}, got {tuple(x.shape)}"
            )

        # Step 2: Min along channel dimension
        x = substrate_min_dim1(x)

        # Step 3: Sum along height dimension
        x = substrate_sum_dim2(x)

        # Step 4: GELU activation
        x = substrate_gelu(x)

        # Step 5: Add bias
        bias = self.bias.to(device=x.device, dtype=x.dtype)
        x = substrate_add_bias(x, bias)

        return x


batch_size = BATCH_SIZE
in_channels = IN_CHANNELS
out_channels = OUT_CHANNELS
height, width = IN_H, IN_W
kernel_size = K_H
stride = STRIDE_H
padding = PAD_H
output_padding = OUTPUT_PAD_H
bias_shape = (1, 1, 1)


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]
