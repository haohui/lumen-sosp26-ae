import torch
import torch.nn as nn
import torch.nn.functional as F
import substrate
import substrate.language as S

# Problem shape constants
BATCH_SIZE = 1024
IN_CHANNELS = 3
OUT_CHANNELS = 16
IN_D, IN_H, IN_W = 16, 32, 32
KERNEL_SIZE = 4

# Conv3D output shape (no padding, stride=1)
OUT_D = IN_D - KERNEL_SIZE + 1  # 13
OUT_H = IN_H - KERNEL_SIZE + 1  # 29
OUT_W = IN_W - KERNEL_SIZE + 1  # 29

# GroupNorm
NUM_GROUPS = 4
CHANNELS_PER_GROUP = OUT_CHANNELS // NUM_GROUPS  # 4

# Thread block sizes
BLOCK_SIZE = 256


# ============== Conv3D Kernel ==============
# Weight shape: (OUT_CHANNELS, IN_CHANNELS, KD, KH, KW)
WEIGHT_ELEMS = OUT_CHANNELS * IN_CHANNELS * KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE  # 16*3*4*4*4 = 3072
WEIGHT_LOAD_ITERS = (WEIGHT_ELEMS + BLOCK_SIZE - 1) // BLOCK_SIZE
SPATIAL_ELEMS = OUT_D * OUT_H * OUT_W  # 13*29*29 = 10933
SPATIAL_TILES = (SPATIAL_ELEMS + BLOCK_SIZE - 1) // BLOCK_SIZE


@substrate.jit
def conv3d_bf16_kernel(
    x_ptr: S.Pointer(S.bf16),
    w_ptr: S.Pointer(S.bf16),
    b_ptr: S.Pointer(S.bf16),
    out_ptr: S.Pointer(S.bf16),
    batch: S.u32,
    in_c: S.u32,
    out_c: S.u32,
    in_d: S.u32,
    in_h: S.u32,
    in_w: S.u32,
    out_d: S.u32,
    out_h: S.u32,
    out_w: S.u32,
    kd: S.u32,
    kh: S.u32,
    kw: S.u32,
):
    tid = S.thread_id(0)
    bid = S.block_id(0)

    n = bid // out_c
    oc = bid - n * out_c

    # Layout for input: (batch, in_c, in_d, in_h, in_w)
    x_layout = S.make_layout((batch, in_c, in_d, in_h, in_w), (in_c * in_d * in_h * in_w, in_d * in_h * in_w, in_h * in_w, in_w, 1))
    x = S.make_tensor(x_ptr, S.bf16, x_layout)

    # Layout for weights: (out_c, in_c, kd, kh, kw)
    w_layout = S.make_layout((out_c, in_c, kd, kh, kw), (in_c * kd * kh * kw, kd * kh * kw, kh * kw, kw, 1))
    w = S.make_tensor(w_ptr, S.bf16, w_layout)

    # Layout for bias: (out_c,)
    b_layout = S.make_layout((out_c,), (1,))
    b = S.make_tensor(b_ptr, S.bf16, b_layout)

    # Layout for output: (batch, out_c, out_d, out_h, out_w)
    out_layout = S.make_layout((batch, out_c, out_d, out_h, out_w), (out_c * out_d * out_h * out_w, out_d * out_h * out_w, out_h * out_w, out_w, 1))
    out = S.make_tensor(out_ptr, S.bf16, out_layout)

    # Shared memory for one output channel's weights
    s_w = S.make_shared((WEIGHT_ELEMS,), S.bf16)

    # Load weights into shared memory
    for it in S.range(WEIGHT_LOAD_ITERS):
        w_flat = it * BLOCK_SIZE + tid
        if w_flat < WEIGHT_ELEMS:
            ic = w_flat // (kd * kh * kw)
            rem = w_flat - ic * (kd * kh * kw)
            k_d = rem // (kh * kw)
            rem2 = rem - k_d * (kh * kw)
            k_h = rem2 // kw
            k_w = rem2 - k_h * kw
            s_w[w_flat] = w[oc, ic, k_d, k_h, k_w]

    S.syncthreads()

    # Compute output spatial elements
    for t in S.range(SPATIAL_TILES):
        pos = t * BLOCK_SIZE + tid
        if pos < SPATIAL_ELEMS:
            od = pos // (out_h * out_w)
            rem = pos - od * (out_h * out_w)
            oh = rem // out_w
            ow = rem - oh * out_w

            # Accumulate in f32
            acc = S.convert(b[oc], S.f32)

            # Convolution loop
            for ic in S.range(IN_CHANNELS):
                for k_d in S.range(KERNEL_SIZE):
                    id_ = od + k_d
                    for k_h in S.range(KERNEL_SIZE):
                        ih = oh + k_h
                        for k_w in S.range(KERNEL_SIZE):
                            iw = ow + k_w
                            xv = S.convert(x[n, ic, id_, ih, iw], S.f32)
                            w_flat = ic * (KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE) + k_d * (KERNEL_SIZE * KERNEL_SIZE) + k_h * KERNEL_SIZE + k_w
                            wv = S.convert(s_w[w_flat], S.f32)
                            acc = acc + xv * wv

            out[n, oc, od, oh, ow] = S.convert(acc, S.bf16)


# ============== HardSwish Kernel ==============
# HardSwish: x * hardsigmoid(x) = x * relu6(x + 3) / 6

@substrate.jit
def hardswish_bf16_kernel(
    x_ptr: S.Pointer(S.bf16),
    out_ptr: S.Pointer(S.bf16),
    n: S.u32,
):
    idx = S.block_id(0) * S.block_dim(0) + S.thread_id(0)

    layout = S.make_layout((n,), (1,))
    x = S.make_tensor(x_ptr, S.bf16, layout)
    out = S.make_tensor(out_ptr, S.bf16, layout)

    if idx < n:
        xv = S.convert(x[idx], S.f32)
        three = S.convert(3.0, S.f32)
        six = S.convert(6.0, S.f32)
        zero = S.convert(0.0, S.f32)

        # relu6(x + 3) = min(max(x + 3, 0), 6)
        shifted = xv + three
        relu6 = shifted
        if relu6 < zero:
            relu6 = zero
        if relu6 > six:
            relu6 = six

        # hardswish = x * relu6(x + 3) / 6
        result = xv * relu6 / six
        out[idx] = S.convert(result, S.bf16)


# ============== GroupNorm Kernel ==============
# Simpler approach using shared memory for mean and inv_std broadcast

@substrate.jit
def groupnorm_compute_stats_bf16_kernel(
    x_ptr: S.Pointer(S.bf16),
    mean_ptr: S.Pointer(S.f32),
    invstd_ptr: S.Pointer(S.f32),
    batch: S.u32,
    channels: S.u32,
    spatial: S.u32,
    num_groups: S.u32,
    channels_per_group: S.u32,
):
    # Each block handles one (batch, group) pair
    bid = S.block_id(0)
    n = bid // num_groups
    g = bid - n * num_groups

    # Layout: (batch, channels, spatial) - flatten spatial
    x_layout = S.make_layout((batch, channels, spatial), (channels * spatial, spatial, 1))
    x = S.make_tensor(x_ptr, S.bf16, x_layout)

    # Output layouts for stats
    mean_layout = S.make_layout((batch, num_groups), (num_groups, 1))
    invstd_layout = S.make_layout((batch, num_groups), (num_groups, 1))
    mean_out = S.make_tensor(mean_ptr, S.f32, mean_layout)
    invstd_out = S.make_tensor(invstd_ptr, S.f32, invstd_layout)

    # Shared memory for partial sums
    s_sum = S.make_shared((BLOCK_SIZE,), S.f32)
    s_sqsum = S.make_shared((BLOCK_SIZE,), S.f32)

    tid = S.thread_id(0)

    # Total elements in this group: channels_per_group * spatial
    group_elems = channels_per_group * spatial

    # Each thread computes partial sums for mean
    local_sum = S.convert(0.0, S.f32)
    iters_per_thread = (group_elems + BLOCK_SIZE - 1) // BLOCK_SIZE
    start_c = g * channels_per_group

    for it in S.range(iters_per_thread):
        elem_idx = it * BLOCK_SIZE + tid
        if elem_idx < group_elems:
            c_local = elem_idx // spatial
            s_local = elem_idx - c_local * spatial
            c_global = start_c + c_local
            v = S.convert(x[n, c_global, s_local], S.f32)
            local_sum = local_sum + v

    s_sum[tid] = local_sum
    S.syncthreads()

    # Reduce to get mean
    total_sum = S.convert(0.0, S.f32)
    if tid < BLOCK_SIZE:
        for t in S.range(BLOCK_SIZE):
            total_sum = total_sum + s_sum[t]

    if tid == 0:
        s_sum[0] = total_sum
    S.syncthreads()

    mean_val = s_sum[0] / S.convert(group_elems, S.f32)

    # Compute variance
    local_sqsum = S.convert(0.0, S.f32)
    for it in S.range(iters_per_thread):
        elem_idx = it * BLOCK_SIZE + tid
        if elem_idx < group_elems:
            c_local = elem_idx // spatial
            s_local = elem_idx - c_local * spatial
            c_global = start_c + c_local
            v = S.convert(x[n, c_global, s_local], S.f32)
            diff = v - mean_val
            local_sqsum = local_sqsum + diff * diff

    s_sqsum[tid] = local_sqsum
    S.syncthreads()

    # Reduce to get variance
    total_sqsum = S.convert(0.0, S.f32)
    if tid < BLOCK_SIZE:
        for t in S.range(BLOCK_SIZE):
            total_sqsum = total_sqsum + s_sqsum[t]

    if tid == 0:
        s_sqsum[0] = total_sqsum
    S.syncthreads()

    var_val = s_sqsum[0] / S.convert(group_elems, S.f32)

    # inv_std = 1 / sqrt(var + eps)
    eps_val = S.convert(1e-5, S.f32)
    inv_std = S.convert(1.0, S.f32) / S.sqrt(var_val + eps_val)

    # Store stats
    if tid == 0:
        mean_out[n, g] = mean_val
        invstd_out[n, g] = inv_std


@substrate.jit
def groupnorm_apply_bf16_kernel(
    x_ptr: S.Pointer(S.bf16),
    mean_ptr: S.Pointer(S.f32),
    invstd_ptr: S.Pointer(S.f32),
    gamma_ptr: S.Pointer(S.bf16),
    beta_ptr: S.Pointer(S.bf16),
    out_ptr: S.Pointer(S.bf16),
    batch: S.u32,
    channels: S.u32,
    spatial: S.u32,
    num_groups: S.u32,
    channels_per_group: S.u32,
):
    # Each thread handles one element
    idx = S.block_id(0) * S.block_dim(0) + S.thread_id(0)
    total = batch * channels * spatial

    # Layouts
    x_layout = S.make_layout((batch, channels, spatial), (channels * spatial, spatial, 1))
    x = S.make_tensor(x_ptr, S.bf16, x_layout)
    out = S.make_tensor(out_ptr, S.bf16, x_layout)

    mean_layout = S.make_layout((batch, num_groups), (num_groups, 1))
    invstd_layout = S.make_layout((batch, num_groups), (num_groups, 1))
    mean_in = S.make_tensor(mean_ptr, S.f32, mean_layout)
    invstd_in = S.make_tensor(invstd_ptr, S.f32, invstd_layout)

    gamma_layout = S.make_layout((channels,), (1,))
    beta_layout = S.make_layout((channels,), (1,))
    gamma = S.make_tensor(gamma_ptr, S.bf16, gamma_layout)
    beta = S.make_tensor(beta_ptr, S.bf16, beta_layout)

    if idx < total:
        # Compute (n, c, s) from idx
        ns = idx // spatial
        s = idx - ns * spatial
        n = ns // channels
        c = ns - n * channels
        g = c // channels_per_group

        xv = S.convert(x[n, c, s], S.f32)
        mean_val = mean_in[n, g]
        inv_std = invstd_in[n, g]

        normalized = (xv - mean_val) * inv_std
        g_val = S.convert(gamma[c], S.f32)
        b_val = S.convert(beta[c], S.f32)
        scaled = normalized * g_val + b_val
        out[n, c, s] = S.convert(scaled, S.bf16)


# ============== Mean Pooling over Spatial Dims ==============
# Input: (B, C, D, H, W) -> Output: (B, C)

@substrate.jit
def mean_spatial_bf16_kernel(
    x_ptr: S.Pointer(S.bf16),
    out_ptr: S.Pointer(S.bf16),
    batch: S.u32,
    channels: S.u32,
    spatial: S.u32,
):
    # Each block handles one (batch, channel) pair
    bid = S.block_id(0)
    n = bid // channels
    c = bid - n * channels

    # Layout: (batch, channels, spatial) - spatial is flattened D*H*W
    x_layout = S.make_layout((batch, channels, spatial), (channels * spatial, spatial, 1))
    x = S.make_tensor(x_ptr, S.bf16, x_layout)

    out_layout = S.make_layout((batch, channels), (channels, 1))
    out = S.make_tensor(out_ptr, S.bf16, out_layout)

    # Shared memory for partial sums
    s_partial = S.make_shared((BLOCK_SIZE,), S.f32)

    tid = S.thread_id(0)

    # Each thread computes partial sum
    local_sum = S.convert(0.0, S.f32)
    iters_per_thread = (spatial + BLOCK_SIZE - 1) // BLOCK_SIZE

    for it in S.range(iters_per_thread):
        s = it * BLOCK_SIZE + tid
        if s < spatial:
            local_sum = local_sum + S.convert(x[n, c, s], S.f32)

    s_partial[tid] = local_sum
    S.syncthreads()

    # Reduce
    total = S.convert(0.0, S.f32)
    if tid < BLOCK_SIZE:
        for t in S.range(BLOCK_SIZE):
            total = total + s_partial[t]

    if tid == 0:
        mean_val = total / S.convert(spatial, S.f32)
        out[n, c] = S.convert(mean_val, S.bf16)


# ============== Host Wrappers ==============

def _launch_conv3d_bf16(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    b, ic, id_, ih, iw = x.shape
    oc = weight.shape[0]
    kd, kh, kw = weight.shape[2], weight.shape[3], weight.shape[4]
    od = id_ - kd + 1
    oh = ih - kh + 1
    ow = iw - kw + 1

    out = torch.empty((b, oc, od, oh, ow), device=x.device, dtype=torch.bfloat16)

    grid = (b * oc, 1, 1)
    conv3d_bf16_kernel[lambda: (grid, (BLOCK_SIZE, 1, 1))](
        x, weight, bias, out,
        b, ic, oc, id_, ih, iw, od, oh, ow, kd, kh, kw
    )
    return out


def _launch_hardswish_bf16(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE, 1, 1)
    hardswish_bf16_kernel[lambda: (grid, (BLOCK_SIZE, 1, 1))](x, out, n)
    return out


def _launch_groupnorm_bf16(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, num_groups: int) -> torch.Tensor:
    b, c, d, h, w = x.shape
    spatial = d * h * w
    channels_per_group = c // num_groups

    # Reshape to (batch, channels, spatial)
    x_flat = x.view(b, c, spatial)
    out_flat = torch.empty_like(x_flat)

    # Allocate stats buffers
    mean_buf = torch.empty((b, num_groups), device=x.device, dtype=torch.float32)
    invstd_buf = torch.empty((b, num_groups), device=x.device, dtype=torch.float32)

    # Compute stats
    grid_stats = (b * num_groups, 1, 1)
    groupnorm_compute_stats_bf16_kernel[lambda: (grid_stats, (BLOCK_SIZE, 1, 1))](
        x_flat, mean_buf, invstd_buf,
        b, c, spatial, num_groups, channels_per_group
    )

    # Apply normalization
    total_elems = b * c * spatial
    grid_apply = ((total_elems + BLOCK_SIZE - 1) // BLOCK_SIZE, 1, 1)
    groupnorm_apply_bf16_kernel[lambda: (grid_apply, (BLOCK_SIZE, 1, 1))](
        x_flat, mean_buf, invstd_buf, gamma, beta, out_flat,
        b, c, spatial, num_groups, channels_per_group
    )

    return out_flat.view(b, c, d, h, w)


def _launch_mean_spatial_bf16(x: torch.Tensor) -> torch.Tensor:
    b, c, d, h, w = x.shape
    spatial = d * h * w

    # Reshape to (batch, channels, spatial)
    x_flat = x.view(b, c, spatial)
    out = torch.empty((b, c), device=x.device, dtype=torch.bfloat16)

    grid = (b * c, 1, 1)
    mean_spatial_bf16_kernel[lambda: (grid, (BLOCK_SIZE, 1, 1))](
        x_flat, out, b, c, spatial
    )

    return out


class ModelNew(nn.Module):
    """
    Optimized model using Substrate GPU kernels.
    Performs: Conv3D -> HardSwish -> GroupNorm -> Mean pooling
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups=4, bias=True):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.num_groups = num_groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_device = x.device
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/HIP device is required for Substrate kernels.")

        # Move to GPU if needed
        if not x.is_cuda:
            x = x.cuda()

        # Ensure contiguous and bf16
        x = x.contiguous().to(torch.bfloat16)

        # Get weights
        weight = self.conv.weight.contiguous().to(torch.bfloat16)
        bias_val = self.conv.bias
        if bias_val is None:
            bias_val = torch.zeros(self.conv.out_channels, device=x.device, dtype=torch.bfloat16)
        else:
            bias_val = bias_val.contiguous().to(torch.bfloat16)

        gamma = self.group_norm.weight.contiguous().to(torch.bfloat16)
        beta = self.group_norm.bias.contiguous().to(torch.bfloat16)

        # Conv3D
        x = _launch_conv3d_bf16(x, weight, bias_val)

        # HardSwish
        x = _launch_hardswish_bf16(x)

        # GroupNorm
        x = _launch_groupnorm_bf16(x, gamma, beta, self.num_groups)

        # Mean pooling over spatial dims
        x = _launch_mean_spatial_bf16(x)

        if orig_device.type != "cuda":
            x = x.to(orig_device)

        return x


batch_size = 1024
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 4


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
