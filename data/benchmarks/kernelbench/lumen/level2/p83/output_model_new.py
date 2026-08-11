import torch
import torch.nn as nn
import substrate
import substrate.language as S

# Fixed problem shape from target model/get_inputs.
BATCH_SIZE = 128
IN_CHANNELS = 3
OUT_CHANNELS = 16
IN_D, IN_H, IN_W = 16, 64, 64
KERNEL_SIZE = 3
GROUPS = 8
MIN_VALUE = 0.0
MAX_VALUE = 1.0
DROPOUT_P = 0.2
EPS = 1e-5

# Conv3d output dimensions (stride=1, padding=0)
OUT_D = IN_D - KERNEL_SIZE + 1  # 14
OUT_H = IN_H - KERNEL_SIZE + 1  # 62
OUT_W = IN_W - KERNEL_SIZE + 1  # 62

THREADS_PER_BLOCK = 256
WEIGHT_ELEMS = IN_CHANNELS * KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE  # 81
WEIGHT_LOAD_ITERS = (WEIGHT_ELEMS + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
SPATIAL_ELEMS = OUT_D * OUT_H * OUT_W  # 14 * 62 * 62 = 53816
SPATIAL_TILES = (SPATIAL_ELEMS + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

# GroupNorm constants
CHANNELS_PER_GROUP = OUT_CHANNELS // GROUPS  # 2
SPATIAL_PER_CHANNEL = OUT_D * OUT_H * OUT_W  # 53816
GROUP_SPATIAL_ELEMS = CHANNELS_PER_GROUP * SPATIAL_PER_CHANNEL  # 107632


@substrate.jit
def conv3d_3x3x3_bf16_kernel(
    x: S.Tensor((BATCH_SIZE, IN_CHANNELS, IN_D, IN_H, IN_W), S.bf16),
    w: S.Tensor((OUT_CHANNELS, IN_CHANNELS, KERNEL_SIZE, KERNEL_SIZE, KERNEL_SIZE), S.bf16),
    b: S.Tensor((OUT_CHANNELS,), S.bf16),
    out: S.Tensor((BATCH_SIZE, OUT_CHANNELS, OUT_D, OUT_H, OUT_W), S.bf16),
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
            ic = w_flat // (KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE)
            rem = w_flat % (KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE)
            kd = rem // (KERNEL_SIZE * KERNEL_SIZE)
            rem2 = rem % (KERNEL_SIZE * KERNEL_SIZE)
            kh = rem2 // KERNEL_SIZE
            kw = rem2 % KERNEL_SIZE
            s_w[w_flat] = w[oc, ic, kd, kh, kw]

    S.syncthreads()

    for t in S.range(SPATIAL_TILES):
        pos = t * THREADS_PER_BLOCK + tid
        if pos < SPATIAL_ELEMS:
            od = pos // (OUT_H * OUT_W)
            rem = pos % (OUT_H * OUT_W)
            oh = rem // OUT_W
            ow = rem % OUT_W

            acc = S.convert(b[oc], S.f32)

            for ic in S.range(IN_CHANNELS):
                for kd in S.range(KERNEL_SIZE):
                    id = od + kd
                    for kh in S.range(KERNEL_SIZE):
                        ih = oh + kh
                        for kw in S.range(KERNEL_SIZE):
                            iw = ow + kw
                            wf = ic * (KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE) + kd * (KERNEL_SIZE * KERNEL_SIZE) + kh * KERNEL_SIZE + kw
                            xv = S.convert(x[n, ic, id, ih, iw], S.f32)
                            wv = S.convert(s_w[wf], S.f32)
                            acc = acc + xv * wv

            out[n, oc, od, oh, ow] = S.convert(acc, S.bf16)


@substrate.jit
def groupnorm_bf16_kernel(
    x: S.Tensor((BATCH_SIZE, OUT_CHANNELS, OUT_D, OUT_H, OUT_W), S.bf16),
    weight: S.Tensor((OUT_CHANNELS,), S.bf16),
    bias: S.Tensor((OUT_CHANNELS,), S.bf16),
    out: S.Tensor((BATCH_SIZE, OUT_CHANNELS, OUT_D, OUT_H, OUT_W), S.bf16),
):
    # Each block handles one (batch, group) pair
    bid = S.block_id(0)

    n = bid // GROUPS
    g = bid % GROUPS

    c_start = g * CHANNELS_PER_GROUP

    # Compute mean: sum all elements in the group's channels and spatial dims
    mean_acc = S.convert(0.0, S.f32)
    for c in S.range(CHANNELS_PER_GROUP):
        ch = c_start + c
        for sd in S.range(OUT_D):
            for sh in S.range(OUT_H):
                for sw in S.range(OUT_W):
                    xv = S.convert(x[n, ch, sd, sh, sw], S.f32)
                    mean_acc = mean_acc + xv

    inv_count = S.convert(1.0 / GROUP_SPATIAL_ELEMS, S.f32)
    mean = mean_acc * inv_count

    # Compute variance
    var_acc = S.convert(0.0, S.f32)
    for c in S.range(CHANNELS_PER_GROUP):
        ch = c_start + c
        for sd in S.range(OUT_D):
            for sh in S.range(OUT_H):
                for sw in S.range(OUT_W):
                    xv = S.convert(x[n, ch, sd, sh, sw], S.f32)
                    diff = xv - mean
                    var_acc = var_acc + diff * diff

    var = var_acc * inv_count
    eps_f = S.convert(EPS, S.f32)
    one_f = S.convert(1.0, S.f32)
    inv_std = one_f / S.sqrt(var + eps_f)

    # Normalize and apply affine transform
    for c in S.range(CHANNELS_PER_GROUP):
        ch = c_start + c
        w_val = S.convert(weight[ch], S.f32)
        b_val = S.convert(bias[ch], S.f32)
        for sd in S.range(OUT_D):
            for sh in S.range(OUT_H):
                for sw in S.range(OUT_W):
                    xv = S.convert(x[n, ch, sd, sh, sw], S.f32)
                    normalized = (xv - mean) * inv_std
                    result_f = normalized * w_val + b_val
                    result_b = S.convert(result_f, S.bf16)
                    out[n, ch, sd, sh, sw] = result_b


@substrate.jit
def elementwise_min_clamp_bf16_kernel(
    x_ptr: S.Pointer(S.bf16),
    out_ptr: S.Pointer(S.bf16),
    n: S.u32,
):
    idx = S.block_id(0) * S.block_dim(0) + S.thread_id(0)
    layout = S.make_layout((n,), (1,))
    x = S.make_tensor(x_ptr, S.bf16, layout)
    out = S.make_tensor(out_ptr, S.bf16, layout)

    if idx < n:
        v = S.convert(x[idx], S.f32)
        min_val = S.convert(MIN_VALUE, S.f32)
        max_val = S.convert(MAX_VALUE, S.f32)

        # torch.min(x, min_value)
        if v > min_val:
            v = min_val

        # torch.clamp(x, min=min_value, max=max_value)
        if v < min_val:
            v = min_val
        if v > max_val:
            v = max_val

        out[idx] = S.convert(v, S.bf16)


# Dropout probability scaled as integer (p * 1000)
DROPOUT_P_SCALED = int(DROPOUT_P * 1000)  # 200 for 0.2


@substrate.jit
def dropout_bf16_kernel(
    x_ptr: S.Pointer(S.bf16),
    out_ptr: S.Pointer(S.bf16),
    n: S.u32,
    seed: S.u32,
    p_scaled: S.u32,
    training: S.i32,
):
    idx = S.block_id(0) * S.block_dim(0) + S.thread_id(0)
    layout = S.make_layout((n,), (1,))
    x = S.make_tensor(x_ptr, S.bf16, layout)
    out = S.make_tensor(out_ptr, S.bf16, layout)

    if idx < n:
        v = x[idx]
        if training == 0:
            # Eval mode: identity
            out[idx] = v
        else:
            # Training mode: simple hash-based dropout
            # Use idx and seed to generate pseudo-random number
            h = idx ^ seed
            h = (h * S.convert(2654435761, S.u32)) & S.convert(0xFFFFFFFF, S.u32)
            h = (h ^ (h >> 16)) & S.convert(0xFFFFFFFF, S.u32)
            h = (h * S.convert(2654435761, S.u32)) & S.convert(0xFFFFFFFF, S.u32)

            # Convert to integer in [0, 1000)
            rand_val = h % S.convert(1000, S.u32)

            # p_scaled = p * 1000, so keep_prob_scaled = (1000 - p_scaled)
            keep_prob_scaled = S.convert(1000, S.u32) - p_scaled
            if rand_val < keep_prob_scaled:
                # Keep element and scale by 1/(1-p)
                # scale = 1000 / keep_prob_scaled
                scale_f = S.convert(1000.0, S.f32) / S.convert(keep_prob_scaled, S.f32)
                result_f = S.convert(v, S.f32) * scale_f
                out[idx] = S.convert(result_f, S.bf16)
            else:
                # Zero out
                zero_b = S.convert(0.0, S.bf16)
                out[idx] = zero_b


def _launch_conv3d_bf16(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty((BATCH_SIZE, OUT_CHANNELS, OUT_D, OUT_H, OUT_W), device=x.device, dtype=torch.bfloat16)
    conv3d_3x3x3_bf16_kernel[
        lambda: ((BATCH_SIZE * OUT_CHANNELS, 1, 1), (THREADS_PER_BLOCK, 1, 1))
    ](x, w, b, out)
    return out


def _launch_groupnorm_bf16(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    groupnorm_bf16_kernel[
        lambda: ((BATCH_SIZE * GROUPS, 1, 1), (1, 1, 1))
    ](x, weight, bias, out)
    return out


def _launch_elementwise_min_clamp_bf16(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    grid = (n + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    elementwise_min_clamp_bf16_kernel[lambda: ((grid, 1, 1), (THREADS_PER_BLOCK, 1, 1))](
        x.view(-1), out.view(-1), n
    )
    return out


def _launch_dropout_bf16(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    grid = (n + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    seed = 42  # Fixed seed for reproducibility
    p_scaled = int(p * 1000)
    dropout_bf16_kernel[lambda: ((grid, 1, 1), (THREADS_PER_BLOCK, 1, 1))](
        x.view(-1), out.view(-1), n, seed, p_scaled, 1 if training else 0
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model using Substrate GPU kernels for:
    - Conv3d
    - GroupNorm
    - Elementwise min + clamp
    - Dropout
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, min_value, max_value, dropout_p):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.dropout_p = dropout_p
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x):
        original_device = x.device

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/HIP device is required for Substrate kernels.")

        if not x.is_cuda:
            x = x.cuda()

        # Ensure BF16
        x = x.to(torch.bfloat16).contiguous()

        # Get conv weights
        w = self.conv.weight.to(torch.bfloat16).contiguous()
        b = self.conv.bias
        if b is None:
            b = torch.zeros((OUT_CHANNELS,), device=x.device, dtype=torch.bfloat16)
        else:
            b = b.to(torch.bfloat16).contiguous()

        # Conv3d
        if x.shape != (BATCH_SIZE, IN_CHANNELS, IN_D, IN_H, IN_W):
            raise NotImplementedError(f"ModelNew currently supports input shape {(BATCH_SIZE, IN_CHANNELS, IN_D, IN_H, IN_W)}, got {tuple(x.shape)}")

        x = _launch_conv3d_bf16(x, w, b)

        # GroupNorm
        weight = self.norm.weight.to(torch.bfloat16).contiguous()
        bias = self.norm.bias.to(torch.bfloat16).contiguous()
        x = _launch_groupnorm_bf16(x, weight, bias)

        # Elementwise min + clamp
        x = _launch_elementwise_min_clamp_bf16(x)

        # Dropout
        if self.training:
            x = _launch_dropout_bf16(x, self.dropout_p, True)
        # In eval mode, dropout is identity (no need to call kernel)

        if original_device.type != "cuda":
            x = x.to(original_device)

        return x


batch_size = BATCH_SIZE
in_channels = IN_CHANNELS
out_channels = OUT_CHANNELS
depth, height, width = IN_D, IN_H, IN_W
kernel_size = KERNEL_SIZE
groups = GROUPS
min_value = MIN_VALUE
max_value = MAX_VALUE
dropout_p = DROPOUT_P


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, groups, min_value, max_value, dropout_p]
