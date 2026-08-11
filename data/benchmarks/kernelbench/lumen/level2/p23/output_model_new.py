import torch
import torch.nn as nn
import substrate
import substrate.language as S

# Fixed problem shape from target model/get_inputs.
BATCH_SIZE = 128
IN_CHANNELS = 3
OUT_CHANNELS = 24
IN_D = 24
IN_H = 32
IN_W = 32
K_D = 3
K_H = 3
K_W = 3

# Conv3d output dimensions (no padding, stride=1)
OUT_D = IN_D - K_D + 1  # 22
OUT_H = IN_H - K_H + 1  # 30
OUT_W = IN_W - K_W + 1  # 30

NUM_GROUPS = 8
CHANNELS_PER_GROUP = OUT_CHANNELS // NUM_GROUPS  # 3

THREADS_PER_BLOCK = 256

# Conv3d weight elements per output channel
WEIGHT_ELEMS = IN_CHANNELS * K_D * K_H * K_W  # 3 * 27 = 81

# Conv3d spatial elements per (batch, out_channel)
SPATIAL_ELEMS = OUT_D * OUT_H * OUT_W  # 22 * 30 * 30 = 19800

# GroupNorm spatial elements per group
GN_SPATIAL_ELEMS = CHANNELS_PER_GROUP * OUT_D * OUT_H * OUT_W  # 3 * 22 * 30 * 30 = 59400

# Mean reduction total elements per batch
MEAN_ELEMS = OUT_CHANNELS * OUT_D * OUT_H * OUT_W  # 24 * 22 * 30 * 30 = 475200

# Inverse values for mean computation
INV_GN_ELEMS = 1.0 / GN_SPATIAL_ELEMS
INV_MEAN_ELEMS = 1.0 / MEAN_ELEMS


@substrate.jit
def conv3d_bf16_kernel(
    x: S.Tensor((BATCH_SIZE, IN_CHANNELS, IN_D, IN_H, IN_W), S.bf16),
    w: S.Tensor((OUT_CHANNELS, IN_CHANNELS, K_D, K_H, K_W), S.bf16),
    b: S.Tensor((OUT_CHANNELS,), S.bf16),
    out: S.Tensor((BATCH_SIZE, OUT_CHANNELS, OUT_D, OUT_H, OUT_W), S.bf16),
):
    tid = S.thread_id(0)
    bid = S.block_id(0)

    n = bid // OUT_CHANNELS
    oc = bid % OUT_CHANNELS

    # Shared memory tile for one output channel's kernel weights.
    s_w = S.make_shared((WEIGHT_ELEMS,), S.bf16)

    # Load weights into shared memory
    WEIGHT_LOAD_ITERS = (WEIGHT_ELEMS + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    for it in S.range(WEIGHT_LOAD_ITERS):
        w_flat = it * THREADS_PER_BLOCK + tid
        if w_flat < WEIGHT_ELEMS:
            ic = w_flat // (K_D * K_H * K_W)
            rem = w_flat % (K_D * K_H * K_W)
            kd = rem // (K_H * K_W)
            rem2 = rem % (K_H * K_W)
            kh = rem2 // K_W
            kw = rem2 % K_W
            s_w[w_flat] = w[oc, ic, kd, kh, kw]

    S.syncthreads()

    # Compute output elements
    SPATIAL_TILES = (SPATIAL_ELEMS + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    for t in S.range(SPATIAL_TILES):
        pos = t * THREADS_PER_BLOCK + tid
        if pos < SPATIAL_ELEMS:
            od = pos // (OUT_H * OUT_W)
            rem = pos % (OUT_H * OUT_W)
            oh = rem // OUT_W
            ow = rem % OUT_W

            acc = S.convert(b[oc], S.f32)

            for ic in S.range(IN_CHANNELS):
                for kd in S.range(K_D):
                    id = od + kd
                    for kh in S.range(K_H):
                        ih = oh + kh
                        for kw in S.range(K_W):
                            iw = ow + kw
                            wf = ic * (K_D * K_H * K_W) + kd * (K_H * K_W) + kh * K_W + kw
                            xv = S.convert(x[n, ic, id, ih, iw], S.f32)
                            wv = S.convert(s_w[wf], S.f32)
                            acc = acc + xv * wv

            out[n, oc, od, oh, ow] = S.convert(acc, S.bf16)


@substrate.jit
def group_norm_bf16_kernel(
    x: S.Tensor((BATCH_SIZE, OUT_CHANNELS, OUT_D, OUT_H, OUT_W), S.bf16),
    gamma: S.Tensor((OUT_CHANNELS,), S.bf16),
    beta: S.Tensor((OUT_CHANNELS,), S.bf16),
    out: S.Tensor((BATCH_SIZE, OUT_CHANNELS, OUT_D, OUT_H, OUT_W), S.bf16),
):
    # Each block handles one (batch, group)
    bid = S.block_id(0)
    n = bid // NUM_GROUPS
    g = bid % NUM_GROUPS

    tid = S.thread_id(0)

    # Shared memory for block-wide reduction
    s_sum = S.make_shared((THREADS_PER_BLOCK,), S.f32)
    s_sum_sq = S.make_shared((THREADS_PER_BLOCK,), S.f32)

    # Initialize shared memory
    s_sum[tid] = S.convert(0.0, S.f32)
    s_sum_sq[tid] = S.convert(0.0, S.f32)

    # Iterate over elements assigned to this thread
    GN_ELEMS_PER_THREAD = (GN_SPATIAL_ELEMS + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    for e in S.range(GN_ELEMS_PER_THREAD):
        elem_idx = e * THREADS_PER_BLOCK + tid
        if elem_idx < GN_SPATIAL_ELEMS:
            # Decode element index to (c_in_group, od, oh, ow)
            c_in_group = elem_idx // (OUT_D * OUT_H * OUT_W)
            rem = elem_idx % (OUT_D * OUT_H * OUT_W)
            od = rem // (OUT_H * OUT_W)
            rem2 = rem % (OUT_H * OUT_W)
            oh = rem2 // OUT_W
            ow = rem2 % OUT_W

            c = g * CHANNELS_PER_GROUP + c_in_group
            val = S.convert(x[n, c, od, oh, ow], S.f32)
            s_sum[tid] = s_sum[tid] + val
            s_sum_sq[tid] = s_sum_sq[tid] + val * val

    S.syncthreads()

    # Block reduction
    for s in S.range(8):
        stride = S.convert(1 << (7 - s), S.i32)
        if tid < stride:
            s_sum[tid] = s_sum[tid] + s_sum[tid + stride]
            s_sum_sq[tid] = s_sum_sq[tid] + s_sum_sq[tid + stride]
        S.syncthreads()

    # Broadcast mean and variance from thread 0
    mean = s_sum[0]
    var = s_sum_sq[0]

    # Compute mean and variance
    inv_n = S.convert(INV_GN_ELEMS, S.f32)
    mean = mean * inv_n
    var = var * inv_n - mean * mean

    S.syncthreads()

    # Normalize and apply affine transform
    eps = S.convert(1e-5, S.f32)
    inv_std = S.amdgpu.rcp(S.sqrt(var + eps))

    for e in S.range(GN_ELEMS_PER_THREAD):
        elem_idx = e * THREADS_PER_BLOCK + tid
        if elem_idx < GN_SPATIAL_ELEMS:
            c_in_group = elem_idx // (OUT_D * OUT_H * OUT_W)
            rem = elem_idx % (OUT_D * OUT_H * OUT_W)
            od = rem // (OUT_H * OUT_W)
            rem2 = rem % (OUT_H * OUT_W)
            oh = rem2 // OUT_W
            ow = rem2 % OUT_W

            c = g * CHANNELS_PER_GROUP + c_in_group
            val = S.convert(x[n, c, od, oh, ow], S.f32)
            normalized = (val - mean) * inv_std
            g_val = S.convert(gamma[c], S.f32)
            b_val = S.convert(beta[c], S.f32)
            result = g_val * normalized + b_val
            out[n, c, od, oh, ow] = S.convert(result, S.bf16)


@substrate.jit
def mean_all_dims_bf16_kernel(
    x: S.Tensor((BATCH_SIZE, OUT_CHANNELS, OUT_D, OUT_H, OUT_W), S.bf16),
    out: S.Tensor((BATCH_SIZE,), S.bf16),
):
    # Each block handles one batch element
    n = S.block_id(0)
    tid = S.thread_id(0)

    # Shared memory for block reduction
    s_sum = S.make_shared((THREADS_PER_BLOCK,), S.f32)
    s_sum[tid] = S.convert(0.0, S.f32)

    # Iterate over elements assigned to this thread
    MEAN_ELEMS_PER_THREAD = (MEAN_ELEMS + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    for e in S.range(MEAN_ELEMS_PER_THREAD):
        elem_idx = e * THREADS_PER_BLOCK + tid
        if elem_idx < MEAN_ELEMS:
            c = elem_idx // (OUT_D * OUT_H * OUT_W)
            rem = elem_idx % (OUT_D * OUT_H * OUT_W)
            od = rem // (OUT_H * OUT_W)
            rem2 = rem % (OUT_H * OUT_W)
            oh = rem2 // OUT_W
            ow = rem2 % OUT_W

            val = S.convert(x[n, c, od, oh, ow], S.f32)
            s_sum[tid] = s_sum[tid] + val

    S.syncthreads()

    # Block reduction
    for s in S.range(8):
        stride = S.convert(1 << (7 - s), S.i32)
        if tid < stride:
            s_sum[tid] = s_sum[tid] + s_sum[tid + stride]
        S.syncthreads()

    # Thread 0 computes final mean
    if tid == 0:
        inv_n = S.convert(INV_MEAN_ELEMS, S.f32)
        result = s_sum[0] * inv_n
        out[n] = S.convert(result, S.bf16)


def _launch_conv3d_bf16(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty(
        (BATCH_SIZE, OUT_CHANNELS, OUT_D, OUT_H, OUT_W),
        device=x.device, dtype=torch.bfloat16
    )
    conv3d_bf16_kernel[
        lambda: ((BATCH_SIZE * OUT_CHANNELS, 1, 1), (THREADS_PER_BLOCK, 1, 1))
    ](x, w, b, out)
    return out


def _launch_group_norm_bf16(
    x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor
) -> torch.Tensor:
    out = torch.empty_like(x)
    group_norm_bf16_kernel[
        lambda: ((BATCH_SIZE * NUM_GROUPS, 1, 1), (THREADS_PER_BLOCK, 1, 1))
    ](x, gamma, beta, out)
    return out


def _launch_mean_all_dims(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty((BATCH_SIZE,), device=x.device, dtype=torch.bfloat16)
    mean_all_dims_bf16_kernel[
        lambda: ((BATCH_SIZE, 1, 1), (THREADS_PER_BLOCK, 1, 1))
    ](x, out)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x):
        if tuple(x.shape) != (BATCH_SIZE, IN_CHANNELS, IN_D, IN_H, IN_W):
            raise NotImplementedError(
                f"ModelNew expects input shape {(BATCH_SIZE, IN_CHANNELS, IN_D, IN_H, IN_W)}, got {tuple(x.shape)}"
            )

        orig_device = x.device
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/HIP device is required for Substrate kernels.")

        x = x.to("cuda") if not x.is_cuda else x
        x = x.to(torch.bfloat16)
        x = x.contiguous()

        # Conv3d
        w = self.conv.weight.to(device=x.device, dtype=torch.bfloat16).contiguous()
        b = self.conv.bias
        if b is None:
            b = torch.zeros((OUT_CHANNELS,), device=x.device, dtype=torch.bfloat16)
        else:
            b = b.to(device=x.device, dtype=torch.bfloat16).contiguous()

        x = _launch_conv3d_bf16(x, w, b)

        # GroupNorm
        gamma = self.group_norm.weight.to(device=x.device, dtype=torch.bfloat16).contiguous()
        beta = self.group_norm.bias.to(device=x.device, dtype=torch.bfloat16).contiguous()
        x = _launch_group_norm_bf16(x, gamma, beta)

        # Mean over all dims except batch
        out = _launch_mean_all_dims(x)

        if orig_device.type != "cuda":
            out = out.to(orig_device)
        return out


batch_size = BATCH_SIZE
in_channels = IN_CHANNELS
out_channels = OUT_CHANNELS
D, H, W = IN_D, IN_H, IN_W
kernel_size = K_D
num_groups = NUM_GROUPS


def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, num_groups]
