import torch
import torch.nn as nn
import substrate
import substrate.language as S


# Fixed problem shape from target model/get_inputs.
BATCH_SIZE = 128
IN_CHANNELS = 3
OUT_CHANNELS = 16
IN_D = 16
IN_H = 32
IN_W = 32
K_D = 3
K_H = 3
K_W = 3
STRIDE = 2
PADDING = 1

# Output size calculation for ConvTranspose3d
# output = (input - 1) * stride - 2 * padding + kernel_size
OUT_D = (IN_D - 1) * STRIDE - 2 * PADDING + K_D  # 31
OUT_H = (IN_H - 1) * STRIDE - 2 * PADDING + K_H  # 63
OUT_W = (IN_W - 1) * STRIDE - 2 * PADDING + K_W  # 63
OUT_SPATIAL = OUT_D * OUT_H * OUT_W

THREADS = 256
SPATIAL_TILES = (OUT_SPATIAL + THREADS - 1) // THREADS

# LogSumExp constants
LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453


@substrate.jit
def conv_transpose3d_bf16_kernel(
    x: S.Tensor((BATCH_SIZE, IN_CHANNELS, IN_D, IN_H, IN_W), S.bf16),
    w: S.Tensor((IN_CHANNELS, OUT_CHANNELS, K_D, K_H, K_W), S.bf16),
    b: S.Tensor((OUT_CHANNELS,), S.bf16),
    out: S.Tensor((BATCH_SIZE, OUT_CHANNELS, OUT_D, OUT_H, OUT_W), S.bf16),
):
    """ConvTranspose3d kernel: each thread computes multiple output elements.

    For output position (od, oh, ow), we find input positions that contribute:
    od = id * stride - padding + kd  =>  kd = od + padding - id * stride
    """
    tid = S.thread_id(0)
    bid = S.block_id(0)

    # Linear index to (n, oc)
    n = bid // OUT_CHANNELS
    oc = bid % OUT_CHANNELS

    # Loop over multiple spatial positions
    for t in S.range(SPATIAL_TILES):
        spatial_idx = t * THREADS + tid
        if spatial_idx < OUT_SPATIAL:
            od = spatial_idx // (OUT_H * OUT_W)
            rem = spatial_idx % (OUT_H * OUT_W)
            oh = rem // OUT_W
            ow = rem % OUT_W

            acc = S.convert(b[oc], S.f32)

            # Iterate over input channels and find contributing input positions
            for ic in S.range(IN_CHANNELS):
                # For each possible input depth, compute kernel depth position
                for id in S.range(IN_D):
                    kd = od + PADDING - id * STRIDE
                    if kd >= 0 and kd < K_D:
                        # Found valid (id, kd) pair
                        for ih in S.range(IN_H):
                            kh = oh + PADDING - ih * STRIDE
                            if kh >= 0 and kh < K_H:
                                # Found valid (ih, kh) pair
                                for iw in S.range(IN_W):
                                    kw = ow + PADDING - iw * STRIDE
                                    if kw >= 0 and kw < K_W:
                                        # Found valid (iw, kw) pair
                                        xv = S.convert(x[n, ic, id, ih, iw], S.f32)
                                        wv = S.convert(w[ic, oc, kd, kh, kw], S.f32)
                                        acc = acc + xv * wv

            out[n, oc, od, oh, ow] = S.convert(acc, S.bf16)


@substrate.jit
def logsumexp_dim1_bf16_kernel(
    x: S.Tensor((BATCH_SIZE, OUT_CHANNELS, OUT_D, OUT_H, OUT_W), S.bf16),
    out: S.Tensor((BATCH_SIZE, 1, OUT_D, OUT_H, OUT_W), S.bf16),
):
    """LogSumExp reduction over dim=1 (channel dimension)."""
    bid = S.block_id(0)

    # Each block handles one (n, od, oh, ow) position
    total_spatial = BATCH_SIZE * OUT_D * OUT_H * OUT_W
    spatial_idx = bid

    if spatial_idx < total_spatial:
        n = spatial_idx // (OUT_D * OUT_H * OUT_W)
        rem = spatial_idx % (OUT_D * OUT_H * OUT_W)
        od = rem // (OUT_H * OUT_W)
        rem2 = rem % (OUT_H * OUT_W)
        oh = rem2 // OUT_W
        ow = rem2 % OUT_W

        # Find max for numerical stability
        max_val = S.convert(x[n, 0, od, oh, ow], S.f32)
        for c in S.range(1, OUT_CHANNELS):
            v = S.convert(x[n, c, od, oh, ow], S.f32)
            if v > max_val:
                max_val = v

        # Compute sum of exp(x - max)
        log2e = S.convert(LOG2E, S.f32)
        sum_exp = S.convert(0.0, S.f32)
        for c in S.range(OUT_CHANNELS):
            v = S.convert(x[n, c, od, oh, ow], S.f32)
            diff = v - max_val
            sum_exp = sum_exp + S.exp2(diff * log2e)

        # log(sum_exp) using log2 and convert to natural log
        ln2 = S.convert(LN2, S.f32)
        result = S.log2(sum_exp) * ln2 + max_val

        out[n, 0, od, oh, ow] = S.convert(result, S.bf16)


@substrate.jit
def hardswish_sub_bias_clamp_bf16_kernel(
    x_ptr: S.Pointer(S.bf16),
    bias_ptr: S.Pointer(S.bf16),
    out_ptr: S.Pointer(S.bf16),
    n: S.u32,
):
    """
    Combined kernel: HardSwish, subtract bias, clamp to [-1, 1].
    HardSwish: x * sigmoid(x + 3) / 6
    """
    idx = S.block_id(0) * S.block_dim(0) + S.thread_id(0)

    layout = S.make_layout((n,), (1,))
    x = S.make_tensor(x_ptr, S.bf16, layout)
    bias = S.make_tensor(bias_ptr, S.bf16, layout)
    out = S.make_tensor(out_ptr, S.bf16, layout)

    if idx < n:
        xv = S.convert(x[idx], S.f32)
        bv = S.convert(bias[0], S.f32)

        # HardSwish: x * sigmoid(x + 3) / 6
        three = S.convert(3.0, S.f32)
        one = S.convert(1.0, S.f32)
        six_inv = S.convert(1.0 / 6.0, S.f32)
        log2e = S.convert(LOG2E, S.f32)
        zero_f = S.convert(0.0, S.f32)

        v_plus_3 = xv + three
        neg_v = zero_f - v_plus_3
        exp_neg = S.exp2(neg_v * log2e)
        sigmoid_val = one / (one + exp_neg)

        hardswish = xv * sigmoid_val * six_inv

        # Subtract bias
        result = hardswish - bv

        # Clamp to [-1, 1]
        min_val = S.convert(-1.0, S.f32)
        max_val = S.convert(1.0, S.f32)
        if result < min_val:
            result = min_val
        if result > max_val:
            result = max_val

        out[idx] = S.convert(result, S.bf16)


def substrate_conv_transpose3d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Launch ConvTranspose3d kernel."""
    if not x.is_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/HIP device required for Substrate kernels.")
        x = x.cuda()

    x = x.contiguous().to(torch.bfloat16)
    weight = weight.contiguous().to(torch.bfloat16)
    if bias is not None:
        bias = bias.contiguous().to(torch.bfloat16)
    else:
        bias = torch.zeros((OUT_CHANNELS,), device=x.device, dtype=torch.bfloat16)

    out = torch.empty((BATCH_SIZE, OUT_CHANNELS, OUT_D, OUT_H, OUT_W),
                      device=x.device, dtype=torch.bfloat16)

    # Launch: one block per (batch, output_channel) pair
    grid = (BATCH_SIZE * OUT_CHANNELS, 1, 1)
    block = (THREADS, 1, 1)

    conv_transpose3d_bf16_kernel[lambda: (grid, block)](x, weight, bias, out)
    return out


def substrate_logsumexp(x: torch.Tensor) -> torch.Tensor:
    """Launch LogSumExp kernel over dim=1."""
    x = x.contiguous()

    out = torch.empty((x.shape[0], 1, x.shape[2], x.shape[3], x.shape[4]),
                      device=x.device, dtype=torch.bfloat16)

    total_spatial = BATCH_SIZE * OUT_D * OUT_H * OUT_W
    grid = (total_spatial, 1, 1)
    block = (1, 1, 1)

    logsumexp_dim1_bf16_kernel[lambda: (grid, block)](x, out)
    return out


def substrate_hardswish_sub_clamp(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Launch combined HardSwish, subtract bias, clamp kernel."""
    x = x.contiguous()
    n = x.numel()

    out = torch.empty_like(x)

    grid = ((n + THREADS - 1) // THREADS, 1, 1)
    block = (THREADS, 1, 1)

    hardswish_sub_bias_clamp_bf16_kernel[lambda: (grid, block)](x, bias, out, n)
    return out


class ModelNew(nn.Module):
    """
    Optimized model with Substrate DSL kernels.
    Uses the same parameter structure as the original model.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        # Use nn.ConvTranspose3d to get proper weight initialization
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        # Final bias for subtraction
        self.bias = nn.Parameter(torch.randn(1, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Move to CUDA if needed
        if not x.is_cuda:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA/HIP device required for Substrate kernels.")
            x = x.cuda()

        # Ensure contiguous and correct dtype
        x = x.contiguous().to(torch.bfloat16)

        # ConvTranspose3d using Substrate kernel
        weight = self.conv_transpose.weight
        bias = self.conv_transpose.bias if self.conv_transpose.bias is not None else torch.zeros((OUT_CHANNELS,), device=x.device, dtype=torch.bfloat16)
        x = substrate_conv_transpose3d(x, weight, bias)

        # LogSumExp over dim=1
        x = substrate_logsumexp(x)

        # HardSwish, subtract bias, clamp
        bias_bf16 = self.bias.to(torch.bfloat16)
        x = substrate_hardswish_sub_clamp(x, bias_bf16)

        return x


batch_size = BATCH_SIZE
in_channels = IN_CHANNELS
out_channels = OUT_CHANNELS
depth, height, width = IN_D, IN_H, IN_W
kernel_size = K_D
stride = STRIDE
padding = PADDING
bias_shape = (1, 1, 1, 1)


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, bias_shape]
