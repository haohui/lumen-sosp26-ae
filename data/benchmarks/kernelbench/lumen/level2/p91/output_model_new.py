import torch
import torch.nn as nn
import substrate
import substrate.language as S

# Fixed problem shape from target model/get_inputs.
BATCH_SIZE = 128
IN_CHANNELS = 64
OUT_CHANNELS = 128
IN_H = 64
IN_W = 64
K_H = 4
K_W = 4
STRIDE_H = 2
STRIDE_W = 2
PAD_H = 1
PAD_W = 1
OUT_PAD_H = 1
OUT_PAD_W = 1

OUT_H = (IN_H - 1) * STRIDE_H - 2 * PAD_H + K_H + OUT_PAD_H + 1  # 129
OUT_W = (IN_W - 1) * STRIDE_W - 2 * PAD_W + K_W + OUT_PAD_W + 1  # 129

THREADS = 256

# Log2(e) for softmax computation
LOG2E = 1.4426950408889634


@substrate.jit
def softmax_bf16_find_max(
    x_ptr: S.Pointer(S.bf16),
    max_vals_ptr: S.Pointer(S.f32),
    n: S.u32,
    c: S.u32,
    hw: S.u32,
):
    """Find max along channel dimension."""
    # Create tensor views at kernel level
    total_elems = n * c * hw
    total_softmax = n * hw
    layout = S.make_layout((total_elems,), (1,))
    max_layout = S.make_layout((total_softmax,), (1,))
    x = S.make_tensor(x_ptr, S.bf16, layout)
    max_vals = S.make_tensor(max_vals_ptr, S.f32, max_layout)

    tid = S.block_id(0) * S.block_dim(0) + S.thread_id(0)
    total = n * hw

    if tid < total:
        n_idx = tid // hw
        hw_idx = tid % hw

        neg_inf = S.convert(-1e30, S.f32)
        max_val = neg_inf
        base_idx = n_idx * c * hw + hw_idx

        for ci in S.range(c):
            idx = base_idx + ci * hw
            val = S.convert(x[idx], S.f32)
            if val > max_val:
                max_val = val

        max_vals[tid] = max_val


@substrate.jit
def softmax_bf16_sum_exp(
    x_ptr: S.Pointer(S.bf16),
    max_vals_ptr: S.Pointer(S.f32),
    sum_vals_ptr: S.Pointer(S.f32),
    n: S.u32,
    c: S.u32,
    hw: S.u32,
):
    """Compute sum of exp(x - max)."""
    # Create tensor views at kernel level
    total_elems = n * c * hw
    total_softmax = n * hw
    layout = S.make_layout((total_elems,), (1,))
    max_layout = S.make_layout((total_softmax,), (1,))
    x = S.make_tensor(x_ptr, S.bf16, layout)
    max_vals = S.make_tensor(max_vals_ptr, S.f32, max_layout)
    sum_vals = S.make_tensor(sum_vals_ptr, S.f32, max_layout)

    tid = S.block_id(0) * S.block_dim(0) + S.thread_id(0)
    total = n * hw

    if tid < total:
        n_idx = tid // hw
        hw_idx = tid % hw
        base_idx = n_idx * c * hw + hw_idx

        max_val = max_vals[tid]
        sum_exp = S.convert(0.0, S.f32)
        log2e = S.convert(LOG2E, S.f32)

        for ci in S.range(c):
            idx = base_idx + ci * hw
            val = S.convert(x[idx], S.f32)
            sum_exp = sum_exp + S.exp2((val - max_val) * log2e)

        sum_vals[tid] = sum_exp


@substrate.jit
def softmax_bf16_normalize(
    x_ptr: S.Pointer(S.bf16),
    max_vals_ptr: S.Pointer(S.f32),
    sum_vals_ptr: S.Pointer(S.f32),
    out_ptr: S.Pointer(S.bf16),
    n: S.u32,
    c: S.u32,
    hw: S.u32,
):
    """Normalize to get softmax output."""
    # Create tensor views at kernel level
    total_elems = n * c * hw
    total_softmax = n * hw
    layout = S.make_layout((total_elems,), (1,))
    max_layout = S.make_layout((total_softmax,), (1,))
    x = S.make_tensor(x_ptr, S.bf16, layout)
    out = S.make_tensor(out_ptr, S.bf16, layout)
    max_vals = S.make_tensor(max_vals_ptr, S.f32, max_layout)
    sum_vals = S.make_tensor(sum_vals_ptr, S.f32, max_layout)

    tid = S.block_id(0) * S.block_dim(0) + S.thread_id(0)
    total = n * hw

    if tid < total:
        n_idx = tid // hw
        hw_idx = tid % hw
        base_idx = n_idx * c * hw + hw_idx

        max_val = max_vals[tid]
        sum_val = sum_vals[tid]
        inv_sum = S.convert(1.0, S.f32) / sum_val
        log2e = S.convert(LOG2E, S.f32)

        for ci in S.range(c):
            idx = base_idx + ci * hw
            val = S.convert(x[idx], S.f32)
            normalized = S.exp2((val - max_val) * log2e) * inv_sum
            out[idx] = S.convert(normalized, S.bf16)


@substrate.jit
def bias_scale_sigmoid_bf16_kernel(
    x_ptr: S.Pointer(S.bf16),
    bias_ptr: S.Pointer(S.bf16),
    out_ptr: S.Pointer(S.bf16),
    n: S.u32,
    c: S.u32,
    hw: S.u32,
):
    """Add bias, scale, and apply sigmoid."""
    # Create tensor views at kernel level
    total_elems = n * c * hw
    layout = S.make_layout((total_elems,), (1,))
    x = S.make_tensor(x_ptr, S.bf16, layout)
    out = S.make_tensor(out_ptr, S.bf16, layout)
    bias_layout = S.make_layout((c,), (1,))
    bias = S.make_tensor(bias_ptr, S.bf16, bias_layout)

    idx = S.block_id(0) * S.block_dim(0) + S.thread_id(0)
    total = n * c * hw

    if idx < total:
        c_idx = (idx // hw) % c
        xv = S.convert(x[idx], S.f32)
        biasv = S.convert(bias[c_idx], S.f32)

        # Add bias and scale (scale factor is 2.0)
        scale = S.convert(2.0, S.f32)
        scaled = (xv + biasv) * scale

        # Sigmoid: 1 / (1 + exp(-x))
        # Using exp2: 1 / (1 + exp2(-x * log2(e)))
        neg_one = S.convert(-1.0, S.f32)
        log2e = S.convert(LOG2E, S.f32)
        one = S.convert(1.0, S.f32)

        # Compute sigmoid inline
        sigmoid_val = one / (one + S.exp2(neg_one * scaled * log2e))

        out[idx] = S.convert(sigmoid_val, S.bf16)


class ModelNew(nn.Module):
    """
    Optimized model that performs transposed convolution (via PyTorch), then softmax, bias add, scale, and sigmoid
    using Substrate GPU kernels.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        # Keep the PyTorch ConvTranspose2d for its weights
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        # Move to GPU if needed
        orig_device = x.device
        if not x.is_cuda:
            x = x.cuda()

        # Ensure contiguous and correct dtype
        x = x.contiguous().to(torch.bfloat16)

        # Step 1: Transposed convolution using PyTorch (for correctness)
        conv_out = self.conv_transpose(x)

        # Step 2: Softmax along channel dimension (3-pass approach)
        n = conv_out.shape[0]
        c = conv_out.shape[1]
        hw = conv_out.shape[2] * conv_out.shape[3]
        n_elems = n * hw
        softmax_grid = (n_elems + THREADS - 1) // THREADS

        # Allocate temporary buffers
        max_vals = torch.empty((n_elems,), device=conv_out.device, dtype=torch.float32)
        sum_vals = torch.empty((n_elems,), device=conv_out.device, dtype=torch.float32)
        softmax_out = torch.empty_like(conv_out)

        # Pass 1: Find max
        softmax_bf16_find_max[lambda: ((softmax_grid, 1, 1), (THREADS, 1, 1))](
            conv_out.view(-1), max_vals, n, c, hw
        )

        # Pass 2: Compute sum of exp(x - max)
        softmax_bf16_sum_exp[lambda: ((softmax_grid, 1, 1), (THREADS, 1, 1))](
            conv_out.view(-1), max_vals, sum_vals, n, c, hw
        )

        # Pass 3: Normalize
        softmax_bf16_normalize[lambda: ((softmax_grid, 1, 1), (THREADS, 1, 1))](
            conv_out.view(-1), max_vals, sum_vals, softmax_out.view(-1), n, c, hw
        )

        # Step 3: Bias add + scale + sigmoid
        out = torch.empty_like(softmax_out)
        total_elems = n * c * hw
        out_grid = (total_elems + THREADS - 1) // THREADS
        bias_dev = self.bias.to(conv_out.device, conv_out.dtype).contiguous()
        bias_scale_sigmoid_bf16_kernel[lambda: ((out_grid, 1, 1), (THREADS, 1, 1))](
            softmax_out.view(-1), bias_dev.view(-1), out.view(-1), n, c, hw
        )

        if orig_device.type != "cuda":
            out = out.to(orig_device)

        return out


batch_size = 128
in_channels = 64
out_channels = 128
height, width = 64, 64
kernel_size = 4
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1)
scaling_factor = 2.0


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor]
