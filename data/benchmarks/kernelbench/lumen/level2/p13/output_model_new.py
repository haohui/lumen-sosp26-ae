import torch
import torch.nn as nn
import substrate
import substrate.language as S

# Problem shape from input_model.py
BATCH_SIZE = 16
IN_CHANNELS = 16
OUT_CHANNELS = 64
DEPTH = 32
HEIGHT = 128
WIDTH = 128
KERNEL_SIZE = 3
STRIDE = 1
PADDING = 1
SCALING_FACTOR = 2.0

THREADS = 256

# For softmax: LOG2E constant
LOG2E = 1.4426950408889634


@substrate.jit
def mean_bias_kernel_bf16(
    x: S.Tensor((BATCH_SIZE, OUT_CHANNELS, DEPTH, HEIGHT, WIDTH), S.bf16),
    bias: S.Tensor((OUT_CHANNELS,), S.bf16),
    out: S.Tensor((BATCH_SIZE, OUT_CHANNELS, HEIGHT, WIDTH), S.bf16),
):
    """Mean pool over depth + bias add. Output shape: (n, c, h, w)"""
    tid = S.block_id(0) * S.block_dim(0) + S.thread_id(0)
    total = BATCH_SIZE * OUT_CHANNELS * HEIGHT * WIDTH

    inv_d = S.convert(1.0 / DEPTH, S.f32)

    if tid < total:
        # Compute (n, c, h, w) from linear index
        rem = tid
        out_w = rem % WIDTH
        rem = rem // WIDTH
        out_h = rem % HEIGHT
        rem = rem // HEIGHT
        out_c = rem % OUT_CHANNELS
        out_n = rem // OUT_CHANNELS

        # Accumulate mean over depth
        acc = S.convert(0.0, S.f32)
        for dd in S.range(DEPTH):
            v = S.convert(x[out_n, out_c, dd, out_h, out_w], S.f32)
            acc = acc + v
        mean_val = acc * inv_d

        # Add bias
        bias_val = S.convert(bias[out_c], S.f32)
        result = mean_val + bias_val

        # Store output
        out[out_n, out_c, out_h, out_w] = S.convert(result, S.bf16)


@substrate.jit
def softmax_channels_kernel_bf16(
    x: S.Tensor((BATCH_SIZE, OUT_CHANNELS, HEIGHT, WIDTH), S.bf16),
    out: S.Tensor((BATCH_SIZE, OUT_CHANNELS, HEIGHT, WIDTH), S.bf16),
):
    """Softmax over channels. Each block handles one (n, h, w) location."""
    bid = S.block_id(0)

    total_hw = BATCH_SIZE * HEIGHT * WIDTH
    if bid < total_hw:
        # Decode (n, h, w) from block id
        rem = bid
        loc_w = rem % WIDTH
        rem = rem // WIDTH
        loc_h = rem % HEIGHT
        loc_n = rem // HEIGHT

        # Find max across channels (for numerical stability)
        max_val = S.convert(-1e30, S.f32)
        for cc in S.range(OUT_CHANNELS):
            v = S.convert(x[loc_n, cc, loc_h, loc_w], S.f32)
            if v > max_val:
                max_val = v

        # Compute exp(x - max) and sum
        sum_exp = S.convert(0.0, S.f32)
        for cc in S.range(OUT_CHANNELS):
            v = S.convert(x[loc_n, cc, loc_h, loc_w], S.f32)
            diff = v - max_val
            exp_val = S.exp2(diff * S.convert(LOG2E, S.f32))
            sum_exp = sum_exp + exp_val

        # Normalize and write
        inv_sum = S.convert(1.0, S.f32) / sum_exp
        for cc in S.range(OUT_CHANNELS):
            v = S.convert(x[loc_n, cc, loc_h, loc_w], S.f32)
            diff = v - max_val
            exp_val = S.exp2(diff * S.convert(LOG2E, S.f32))
            out[loc_n, cc, loc_h, loc_w] = S.convert(exp_val * inv_sum, S.bf16)


@substrate.jit
def tanh_scale_kernel_bf16(
    x: S.Tensor((BATCH_SIZE, OUT_CHANNELS, HEIGHT, WIDTH), S.bf16),
    out: S.Tensor((BATCH_SIZE, OUT_CHANNELS, HEIGHT, WIDTH), S.bf16),
):
    """Tanh + scale: out = tanh(x) * scale"""
    tid = S.block_id(0) * S.block_dim(0) + S.thread_id(0)
    total = BATCH_SIZE * OUT_CHANNELS * HEIGHT * WIDTH

    scale = S.convert(SCALING_FACTOR, S.f32)

    if tid < total:
        # Compute indices
        rem = tid
        out_w = rem % WIDTH
        rem = rem // WIDTH
        out_h = rem % HEIGHT
        rem = rem // HEIGHT
        out_c = rem % OUT_CHANNELS
        out_n = rem // OUT_CHANNELS

        v = S.convert(x[out_n, out_c, out_h, out_w], S.f32)
        tanh_v = S.tanh(v)
        result = tanh_v * scale
        out[out_n, out_c, out_h, out_w] = S.convert(result, S.bf16)


class ModelNew(nn.Module):
    """
    Model that performs:
    1. Transposed 3D convolution (using PyTorch)
    2. Mean pooling over depth (Substrate kernel)
    3. Bias addition (fused with mean pooling)
    4. Softmax over channels (Substrate kernel)
    5. Tanh activation (Substrate kernel)
    6. Scaling (fused with tanh)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias_param = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        # Step 1: Transposed 3D convolution (using PyTorch)
        x = self.conv_transpose(x)

        # Ensure contiguous and on GPU
        if not x.is_cuda:
            x = x.cuda()
        x = x.contiguous()

        # Convert to bf16 for kernel operations
        input_dtype = x.dtype
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)

        B, C, D, H, W = x.shape
        assert B == BATCH_SIZE and C == OUT_CHANNELS and D == DEPTH and H == HEIGHT and W == WIDTH

        # Step 2 & 3: Mean pooling over depth + bias add
        mean_bias_out = torch.empty((B, C, H, W), dtype=torch.bfloat16, device=x.device)
        bias_flat = self.bias_param.view(-1).to(torch.bfloat16).contiguous()

        total_mean_out = B * C * H * W
        grid_mean = ((total_mean_out + THREADS - 1) // THREADS, 1, 1)
        mean_bias_kernel_bf16[lambda: (grid_mean, (THREADS, 1, 1))](x, bias_flat, mean_bias_out)

        # Step 4: Softmax over channels
        softmax_out = torch.empty_like(mean_bias_out)
        total_softmax_blocks = B * H * W
        grid_softmax = (total_softmax_blocks, 1, 1)
        softmax_channels_kernel_bf16[lambda: (grid_softmax, (1, 1, 1))](mean_bias_out, softmax_out)

        # Step 5 & 6: Tanh + scale
        out = torch.empty_like(softmax_out)
        grid_tanh = ((total_mean_out + THREADS - 1) // THREADS, 1, 1)

        tanh_scale_kernel_bf16[lambda: (grid_tanh, (THREADS, 1, 1))](softmax_out, out)

        # Reshape back to (B, C, 1, H, W) to match expected output
        out = out.view(B, C, 1, H, W)

        # Convert back to original dtype if needed
        if input_dtype != torch.bfloat16:
            out = out.to(input_dtype)

        return out


batch_size = 16
in_channels = 16
out_channels = 64
depth = 32
height = 128
width = 128
kernel_size = 3
stride = 1
padding = 1
scaling_factor = 2.0


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, scaling_factor]
