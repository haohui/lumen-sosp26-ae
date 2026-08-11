import torch
import torch.nn as nn
import substrate
import substrate.language as S


# Problem shape constants
BATCH_SIZE = 32
IN_CHANNELS = 32
OUT_CHANNELS = 64
IN_D, IN_H, IN_W = 16, 32, 32
KERNEL_SIZE = 3
STRIDE = 2
PADDING = 1
OUTPUT_PADDING = 1

# Output shapes after ConvTranspose3d
OUT_D = (IN_D - 1) * STRIDE - 2 * PADDING + KERNEL_SIZE + OUTPUT_PADDING  # 32
OUT_H = (IN_H - 1) * STRIDE - 2 * PADDING + KERNEL_SIZE + OUTPUT_PADDING  # 64
OUT_W = (IN_W - 1) * STRIDE - 2 * PADDING + KERNEL_SIZE + OUTPUT_PADDING  # 64

# After AvgPool3d with kernel_size=2
POOL_D, POOL_H, POOL_W = OUT_D // 2, OUT_H // 2, OUT_W // 2  # 16, 32, 32

# Kernel launch parameters
THREADS = 256
BLOCK_POOL = 8
POOL_K = 2  # kernel size for pooling

# GELU constants (as Python floats, converted inside kernel)
SQRT_2_OVER_PI = 0.7978845608028654
GELU_COEFF = 0.044715
EPS = 1e-5


# ============================================================================
# 1. Elementwise add scalar kernel - hardcoded scalar value
# ============================================================================
SUM_WEIGHT = 1.0


@substrate.jit
def add_scalar_bf16_kernel(
    x_ptr: S.Pointer(S.bf16),
    y_ptr: S.Pointer(S.bf16),
    n: S.u32,
):
    """Add scalar to each element."""
    idx = S.block_id(0) * S.block_dim(0) + S.thread_id(0)

    layout = S.make_layout((n,), (1,))
    x = S.make_tensor(x_ptr, S.bf16, layout)
    y = S.make_tensor(y_ptr, S.bf16, layout)

    scalar_val = S.convert(SUM_WEIGHT, S.f32)

    if idx < n:
        xv = S.convert(x[idx], S.f32)
        y[idx] = S.convert(xv + scalar_val, S.bf16)


# ============================================================================
# 2. LayerNorm kernel (normalizes over last dimension)
# ============================================================================
@substrate.jit
def layernorm_bf16_kernel(
    x_ptr: S.Pointer(S.bf16),
    y_ptr: S.Pointer(S.bf16),
    gamma_ptr: S.Pointer(S.bf16),
    beta_ptr: S.Pointer(S.bf16),
    outer_size: S.u32,
    inner_size: S.u32,
):
    outer_idx = S.block_id(0)

    if outer_idx < outer_size:
        # Compute mean
        sum_val = S.convert(0.0, S.f32)
        for i in S.range(inner_size):
            layout = S.make_layout((outer_size, inner_size), (inner_size, 1))
            x = S.make_tensor(x_ptr, S.bf16, layout)
            v = S.convert(x[outer_idx, i], S.f32)
            sum_val = sum_val + v

        inv_inner = S.amdgpu.rcp(S.convert(inner_size, S.f32))
        mean = sum_val * inv_inner

        # Compute variance
        var_sum = S.convert(0.0, S.f32)
        for i in S.range(inner_size):
            layout = S.make_layout((outer_size, inner_size), (inner_size, 1))
            x = S.make_tensor(x_ptr, S.bf16, layout)
            v = S.convert(x[outer_idx, i], S.f32)
            diff = v - mean
            var_sum = var_sum + diff * diff

        variance = var_sum * inv_inner
        eps_val = S.convert(EPS, S.f32)
        inv_std = S.amdgpu.rcp(S.sqrt(variance + eps_val))

        # Normalize and apply affine transform
        gamma_layout = S.make_layout((inner_size,), (1,))
        beta_layout = S.make_layout((inner_size,), (1,))
        gamma = S.make_tensor(gamma_ptr, S.bf16, gamma_layout)
        beta = S.make_tensor(beta_ptr, S.bf16, beta_layout)

        for i in S.range(inner_size):
            x_layout = S.make_layout((outer_size, inner_size), (inner_size, 1))
            y_layout = S.make_layout((outer_size, inner_size), (inner_size, 1))
            x = S.make_tensor(x_ptr, S.bf16, x_layout)
            y = S.make_tensor(y_ptr, S.bf16, y_layout)
            xv = S.convert(x[outer_idx, i], S.f32)
            normalized = (xv - mean) * inv_std
            gv = S.convert(gamma[i], S.f32)
            bv = S.convert(beta[i], S.f32)
            result = normalized * gv + bv
            y[outer_idx, i] = S.convert(result, S.bf16)


# ============================================================================
# 3. AvgPool3d kernel
# ============================================================================
@substrate.jit
def avgpool3d_bf16_kernel(
    x_ptr: S.Pointer(S.bf16),
    y_ptr: S.Pointer(S.bf16),
    in_d: S.u32,
    in_h: S.u32,
    in_w: S.u32,
    out_d: S.u32,
    out_h: S.u32,
    out_w: S.u32,
    batch_size: S.u32,
    channels: S.u32,
):
    tx = S.thread_id(0)
    ty = S.thread_id(1)
    bx = S.block_id(0)
    by = S.block_id(1)
    bz = S.block_id(2)

    od = bz % out_d
    batch_channel = bz // out_d
    oc = batch_channel % channels
    on = batch_channel // channels

    oh = by * BLOCK_POOL + ty
    ow = bx * BLOCK_POOL + tx

    if oh < out_h and ow < out_w:
        # Input layout: (N, C, D, H, W)
        in_stride_n = channels * in_d * in_h * in_w
        in_stride_c = in_d * in_h * in_w
        in_stride_d = in_h * in_w
        in_stride_h = in_w
        in_stride_w = 1

        in_layout = S.make_layout(
            (batch_size, channels, in_d, in_h, in_w),
            (in_stride_n, in_stride_c, in_stride_d, in_stride_h, in_stride_w)
        )
        x = S.make_tensor(x_ptr, S.bf16, in_layout)

        # Compute average over 2x2x2 window
        acc = S.convert(0.0, S.f32)
        count = S.convert(0, S.u32)

        for pd in S.range(POOL_K):
            id = od * POOL_K + pd
            if id < in_d:
                for ph in S.range(POOL_K):
                    ih = oh * POOL_K + ph
                    if ih < in_h:
                        for pw in S.range(POOL_K):
                            iw = ow * POOL_K + pw
                            if iw < in_w:
                                v = S.convert(x[on, oc, id, ih, iw], S.f32)
                                acc = acc + v
                                count = count + 1

        inv_count = S.amdgpu.rcp(S.convert(count, S.f32))
        avg = acc * inv_count

        # Output layout: (N, C, out_d, out_h, out_w)
        out_stride_n = channels * out_d * out_h * out_w
        out_stride_c = out_d * out_h * out_w
        out_stride_d = out_h * out_w
        out_stride_h = out_w
        out_stride_w = 1

        out_layout = S.make_layout(
            (batch_size, channels, out_d, out_h, out_w),
            (out_stride_n, out_stride_c, out_stride_d, out_stride_h, out_stride_w)
        )
        y = S.make_tensor(y_ptr, S.bf16, out_layout)
        y[on, oc, od, oh, ow] = S.convert(avg, S.bf16)


# ============================================================================
# 4. GELU kernel
# ============================================================================
@substrate.jit
def gelu_bf16_kernel(
    x_ptr: S.Pointer(S.bf16),
    y_ptr: S.Pointer(S.bf16),
    n: S.u32,
):
    idx = S.block_id(0) * S.block_dim(0) + S.thread_id(0)
    if idx < n:
        layout = S.make_layout((n,), (1,))
        x = S.make_tensor(x_ptr, S.bf16, layout)
        y = S.make_tensor(y_ptr, S.bf16, layout)

        xv = S.convert(x[idx], S.f32)
        sqrt_2_over_pi = S.convert(SQRT_2_OVER_PI, S.f32)
        coeff = S.convert(GELU_COEFF, S.f32)
        half = S.convert(0.5, S.f32)
        one = S.convert(1.0, S.f32)

        # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        x_cubed = xv * xv * xv
        inner = sqrt_2_over_pi * (xv + coeff * x_cubed)
        tanh_inner = S.tanh(inner)
        gelu_out = half * xv * (one + tanh_inner)

        y[idx] = S.convert(gelu_out, S.bf16)


# ============================================================================
# Host wrapper functions
# ============================================================================
def substrate_add_scalar(x: torch.Tensor, scalar: float) -> torch.Tensor:
    """Add scalar to tensor using Substrate kernel."""
    if not x.is_cuda:
        x = x.cuda()

    x = x.contiguous()
    out = torch.empty_like(x)
    n = x.numel()

    if n > 0:
        grid = ((n + THREADS - 1) // THREADS, 1, 1)
        add_scalar_bf16_kernel[lambda: (grid, (THREADS, 1, 1))](x, out, n)

    return out


def substrate_layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """LayerNorm over last dimension using Substrate kernel."""
    if not x.is_cuda:
        x = x.cuda()
    if not weight.is_cuda:
        weight = weight.cuda()
    if not bias.is_cuda:
        bias = bias.cuda()

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    out = torch.empty_like(x)
    inner_size = x.shape[-1]
    outer_size = x.numel() // inner_size

    if outer_size > 0:
        grid = (outer_size, 1, 1)
        layernorm_bf16_kernel[lambda: (grid, (THREADS, 1, 1))](
            x, out, weight, bias, outer_size, inner_size
        )

    return out


def substrate_avgpool3d(x: torch.Tensor, kernel_size: int = 2) -> torch.Tensor:
    """AvgPool3d using Substrate kernel."""
    if not x.is_cuda:
        x = x.cuda()

    x = x.contiguous()

    n, c, in_d, in_h, in_w = x.shape
    out_d = in_d // kernel_size
    out_h = in_h // kernel_size
    out_w = in_w // kernel_size

    out = torch.empty((n, c, out_d, out_h, out_w), dtype=x.dtype, device=x.device)

    grid_x = (out_w + BLOCK_POOL - 1) // BLOCK_POOL
    grid_y = (out_h + BLOCK_POOL - 1) // BLOCK_POOL
    grid_z = n * c * out_d

    avgpool3d_bf16_kernel[lambda: ((grid_x, grid_y, grid_z), (BLOCK_POOL, BLOCK_POOL, 1))](
        x, out, in_d, in_h, in_w, out_d, out_h, out_w, n, c
    )

    return out


def substrate_gelu(x: torch.Tensor) -> torch.Tensor:
    """GELU activation using Substrate kernel."""
    if not x.is_cuda:
        x = x.cuda()

    x = x.contiguous()
    out = torch.empty_like(x)
    n = x.numel()

    if n > 0:
        grid = ((n + THREADS - 1) // THREADS, 1, 1)
        gelu_bf16_kernel[lambda: (grid, (THREADS, 1, 1))](x, out, n)

    return out


# ============================================================================
# ModelNew
# ============================================================================
class ModelNew(nn.Module):
    """
    Optimized model that performs 3D transposed convolution, scalar addition,
    layer normalization, average pooling, and GELU activation.
    Uses Substrate GPU kernels for element-wise operations, LayerNorm, pooling, and activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 output_padding, sum_weight, norm_shape, pool_kernel_size):
        super(ModelNew, self).__init__()

        # Use PyTorch's optimized ConvTranspose3d (cuDNN backend)
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.sum_weight = nn.Parameter(torch.tensor(sum_weight))
        self.norm = nn.LayerNorm(norm_shape)
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x):
        # Step 1: ConvTranspose3d (use PyTorch cuDNN)
        x = self.conv_transpose(x)

        # Step 2: Add scalar (Substrate kernel)
        x = substrate_add_scalar(x, self.sum_weight.item())

        # Step 3: LayerNorm (Substrate kernel)
        x = substrate_layernorm(x, self.norm.weight, self.norm.bias)

        # Step 4: AvgPool3d (Substrate kernel)
        x = substrate_avgpool3d(x, self.pool_kernel_size[0])

        # Step 5: GELU (Substrate kernel)
        x = substrate_gelu(x)

        return x


batch_size = 32
in_channels = 32
out_channels = 64
depth, height, width = 16, 32, 32
kernel_size = (3, 3, 3)
stride = (2, 2, 2)
padding = (1, 1, 1)
output_padding = (1, 1, 1)
sum_weight = 1.0
norm_shape = (out_channels,)
pool_kernel_size = (2, 2, 2)


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding,
            output_padding, sum_weight, norm_shape, pool_kernel_size]
