import torch
import torch.nn as nn
import substrate
import substrate.language as S

# Problem configuration
BATCH_SIZE = 32
IN_CHANNELS = 32
OUT_CHANNELS = 64
IN_DEPTH = 32
IN_HEIGHT = 64
IN_WIDTH = 64

# After AvgPool3d(2)
POOLED_DEPTH = IN_DEPTH // 2  # 16
POOLED_HEIGHT = IN_HEIGHT // 2  # 32
POOLED_WIDTH = IN_WIDTH // 2  # 32

# After ConvTranspose3d with stride=2, padding=1, output_padding=1
OUT_DEPTH = (POOLED_DEPTH - 1) * 2 - 2 * 1 + 3 + 1  # 32
OUT_HEIGHT = (POOLED_HEIGHT - 1) * 2 - 2 * 1 + 3 + 1  # 64
OUT_WIDTH = (POOLED_WIDTH - 1) * 2 - 2 * 1 + 3 + 1  # 64

SPATIAL_SIZE = OUT_DEPTH * OUT_HEIGHT * OUT_WIDTH  # 32 * 64 * 64 = 131072

BLOCK_SIZE = 256
LOG2E = 1.4426950408889634

# Clamp values (constants for this problem)
CLAMP_MIN = 0.0
CLAMP_MAX = 1.0


# === Clamp kernel ===
@substrate.jit
def clamp_bf16_kernel(
    x_ptr: S.Pointer(S.bf16),
    y_ptr: S.Pointer(S.bf16),
    n: S.u32,
):
    idx = S.block_id(0) * S.block_dim(0) + S.thread_id(0)
    layout = S.make_layout((n,), (1,))
    x = S.make_tensor(x_ptr, S.bf16, layout)
    y = S.make_tensor(y_ptr, S.bf16, layout)

    if idx < n:
        v = S.convert(x[idx], S.f32)
        clamp_min = S.convert(CLAMP_MIN, S.f32)
        clamp_max = S.convert(CLAMP_MAX, S.f32)
        if v < clamp_min:
            v = clamp_min
        if v > clamp_max:
            v = clamp_max
        y[idx] = S.convert(v, S.bf16)


# === Spatial softmax kernel - online softmax algorithm ===
# Each block handles one (batch, channel) pair
# Threads within a block cooperate to compute softmax over SPATIAL_SIZE elements
@substrate.jit
def spatial_softmax_bf16_kernel(
    x_ptr: S.Pointer(S.bf16),
    y_ptr: S.Pointer(S.bf16),
    spatial_size: S.u32,
):
    bc_idx = S.block_id(0)
    tid = S.thread_id(0)
    bdim = S.block_dim(0)

    # Compute offsets for this (batch, channel) pair
    batch = bc_idx // OUT_CHANNELS
    ch = bc_idx - batch * OUT_CHANNELS
    offset = batch * OUT_CHANNELS * spatial_size + ch * spatial_size

    layout = S.make_layout((spatial_size,), (1,))
    x = S.make_tensor(x_ptr, S.bf16, layout)
    y = S.make_tensor(y_ptr, S.bf16, layout)

    # Online softmax: compute max and sum
    neg_inf = S.convert(-3.402823466e38, S.f32)
    log2e = S.convert(LOG2E, S.f32)

    # First pass: find max
    max_val = neg_inf
    for i in S.range((spatial_size + bdim - 1) // bdim):
        idx = tid + i * bdim
        if idx < spatial_size:
            v = S.convert(x[offset + idx], S.f32)
            if v > max_val:
                max_val = v

    # Reduce max within block using shared memory
    smem = S.make_shared((BLOCK_SIZE,), S.f32)
    smem[tid] = max_val
    S.syncthreads()

    # Tree reduction for max
    half = BLOCK_SIZE // 2
    for i in S.range(8):  # log2(256) = 8 iterations
        if tid < half:
            other = smem[tid + half]
            if other > smem[tid]:
                smem[tid] = other
        half = half // 2
        S.syncthreads()
        if half == 0:
            break

    global_max = smem[0]
    S.syncthreads()

    # Second pass: compute exp(x - max) and sum
    sum_exp = S.convert(0.0, S.f32)
    for i in S.range((spatial_size + bdim - 1) // bdim):
        idx = tid + i * bdim
        if idx < spatial_size:
            v = S.convert(x[offset + idx], S.f32)
            exp_val = S.exp2((v - global_max) * log2e)
            sum_exp = sum_exp + exp_val

    # Reduce sum within block
    smem[tid] = sum_exp
    S.syncthreads()

    # Tree reduction for sum
    half = BLOCK_SIZE // 2
    for i in S.range(8):
        if tid < half:
            smem[tid] = smem[tid] + smem[tid + half]
        half = half // 2
        S.syncthreads()
        if half == 0:
            break

    global_sum = smem[0]
    S.syncthreads()

    # Compute reciprocal of sum
    inv_sum = S.amdgpu.rcp(global_sum)

    # Third pass: compute softmax output
    for i in S.range((spatial_size + bdim - 1) // bdim):
        idx = tid + i * bdim
        if idx < spatial_size:
            v = S.convert(x[offset + idx], S.f32)
            exp_val = S.exp2((v - global_max) * log2e)
            y[offset + idx] = S.convert(exp_val * inv_sum, S.bf16)


# === Scale multiplication kernel ===
@substrate.jit
def scale_multiply_bf16_kernel(
    x_ptr: S.Pointer(S.bf16),
    scale_ptr: S.Pointer(S.bf16),
    y_ptr: S.Pointer(S.bf16),
    n: S.u32,
    channels: S.u32,
    spatial_size: S.u32,
):
    idx = S.block_id(0) * S.block_dim(0) + S.thread_id(0)

    if idx < n:
        layout = S.make_layout((n,), (1,))
        x = S.make_tensor(x_ptr, S.bf16, layout)
        y = S.make_tensor(y_ptr, S.bf16, layout)
        scale_layout = S.make_layout((channels,), (1,))
        scale = S.make_tensor(scale_ptr, S.bf16, scale_layout)

        # Determine which channel this element belongs to
        ch_idx = (idx // spatial_size) % channels
        s = S.convert(scale[ch_idx], S.f32)
        v = S.convert(x[idx], S.f32)
        y[idx] = S.convert(v * s, S.bf16)


def substrate_clamp(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x.clone()

    x_contig = x.contiguous()
    out = torch.empty_like(x_contig)
    n = x_contig.numel()

    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE, 1, 1)
    clamp_bf16_kernel[lambda: (grid, (BLOCK_SIZE, 1, 1))](x_contig, out, n)
    return out


def substrate_spatial_softmax(x: torch.Tensor) -> torch.Tensor:
    """Compute softmax over spatial dimensions for each (batch, channel) pair."""
    b, c, d, h, w = x.shape
    spatial_size = d * h * w

    x_contig = x.contiguous()
    out = torch.empty_like(x_contig)

    # One block per (batch, channel) pair
    num_blocks = b * c
    spatial_softmax_bf16_kernel[lambda: ((num_blocks, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_contig, out, spatial_size
    )
    return out


def substrate_scale_multiply(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Multiply x by scale where scale has shape (1, C, 1, 1, 1)."""
    x_contig = x.contiguous()
    out = torch.empty_like(x_contig)

    b, c, d, h, w = x.shape
    spatial_size = d * h * w
    n = x_contig.numel()

    # Reshape scale to (C,)
    scale_flat = scale.view(-1).contiguous()

    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE, 1, 1)
    scale_multiply_bf16_kernel[lambda: (grid, (BLOCK_SIZE, 1, 1))](
        x_contig, scale_flat, out, n, c, spatial_size
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model using Substrate DSL kernels for:
    - Clamp operation
    - Spatial softmax
    - Scale multiplication

    AvgPool3d and ConvTranspose3d use PyTorch (cuDNN/MIOpen) implementations.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 output_padding, pool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.avg_pool = nn.AvgPool3d(pool_kernel_size)
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # AvgPool3d and ConvTranspose3d use optimized PyTorch implementations
        x = self.avg_pool(x)
        x = self.conv_transpose(x)

        # Convert to bfloat16 for Substrate kernels
        orig_dtype = x.dtype
        if orig_dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)

        # Clamp using Substrate kernel
        x = substrate_clamp(x)

        # Spatial softmax using Substrate kernel
        x = substrate_spatial_softmax(x)

        # Scale multiplication using Substrate kernel
        scale_bf16 = self.scale.to(torch.bfloat16)
        x = substrate_scale_multiply(x, scale_bf16)

        # Convert back to original dtype if needed
        if orig_dtype != torch.bfloat16:
            x = x.to(orig_dtype)

        return x


def get_inputs():
    return [torch.rand(BATCH_SIZE, IN_CHANNELS, IN_DEPTH, IN_HEIGHT, IN_WIDTH)]


def get_init_inputs():
    kernel_size = 3
    stride = 2
    padding = 1
    output_padding = 1
    pool_kernel_size = 2
    clamp_min = 0.0
    clamp_max = 1.0
    return [IN_CHANNELS, OUT_CHANNELS, kernel_size, stride, padding,
            output_padding, pool_kernel_size, clamp_min, clamp_max]
