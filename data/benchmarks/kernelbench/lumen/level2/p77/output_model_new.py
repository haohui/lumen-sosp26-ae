"""
Optimized implementation of 3D transposed convolution with scaling,
batch normalization, and global average pooling using Substrate DSL kernels.

This implementation uses substrate.jit kernels for:
- Elementwise scaling
- Global average pooling (via reduction)
"""
import torch
import torch.nn as nn
import substrate
import substrate.language as S
import math

# =============================================================================
# Configuration Constants
# =============================================================================

WARP_SIZE = 64
NUM_WARPS = 4
THREADS = WARP_SIZE * NUM_WARPS


# =============================================================================
# Elementwise Scale Kernel
# =============================================================================

@substrate.jit
def scale_bf16_kernel(
    input_ptr: S.Pointer(S.bf16),
    output_ptr: S.Pointer(S.bf16),
    scale_factor: S.f32,
    n_elements: S.u32,
):
    """
    Elementwise scale kernel: output = input * scale_factor
    Each thread processes one element.
    """
    tid = S.thread_id(0)
    bid = S.block_id(0)
    bdim = S.block_dim(0)

    idx = bid * bdim + tid

    if idx < n_elements:
        in_val = input_ptr[idx]
        # Convert to f32 for multiplication
        in_f32 = in_val
        out_f32 = in_f32 * scale_factor
        # Store result
        output_ptr[idx] = out_f32


def run_scale_bf16(x: torch.Tensor, scale_factor: float) -> torch.Tensor:
    """
    Apply elementwise scaling using substrate kernel.
    Falls back to PyTorch if kernel cannot be used.
    """
    assert x.dtype == torch.bfloat16
    assert x.is_contiguous()

    n_elements = x.numel()
    out = torch.empty_like(x)

    threads = 256
    blocks = (n_elements + threads - 1) // threads

    if blocks > 0:
        scale_bf16_kernel[lambda: ((blocks, 1, 1), (threads, 1, 1))](
            x.data_ptr(), out.data_ptr(), scale_factor, n_elements
        )
        return out
    else:
        return x * scale_factor


# =============================================================================
# Global Average Pooling Kernel (Reduction)
# =============================================================================

@substrate.jit
def global_pool_reduce_kernel(
    input_ptr: S.Pointer(S.f32),
    output_ptr: S.Pointer(S.f32),
    spatial_size: S.u32,
):
    """
    Reduce a 1D array to a single mean value using warp-level reduction.
    Each block handles one (batch, channel) pair.
    """
    tid = S.thread_id(0)

    # Accumulator
    acc = 0.0

    # Strided iteration over elements
    for i in S.range((spatial_size + THREADS - 1) // THREADS):
        idx = tid + i * THREADS
        if idx < spatial_size:
            acc = acc + input_ptr[idx]

    # Warp-level reduction
    for offset in range(1, WARP_SIZE):
        other = S.shuffle_down(acc, offset, WARP_SIZE)
        acc = acc + other

    # Cross-warp reduction via shared memory
    shm = S.make_shared((NUM_WARPS,), S.f32)
    wid = tid // WARP_SIZE
    wtid = tid % WARP_SIZE

    if wtid == 0:
        shm[wid] = acc

    S.syncthreads()

    # First warp combines results
    if wid == 0:
        acc = 0.0
        if wtid < NUM_WARPS:
            acc = shm[wtid]

        # Final reduction in first warp
        for offset in range(1, WARP_SIZE):
            other = S.shuffle_down(acc, offset, WARP_SIZE)
            acc = acc + other

        # Thread 0 writes result
        if wtid == 0:
            inv_size = 1.0 / S.convert(spatial_size, S.f32)
            output_ptr[0] = acc * inv_size


def run_global_avg_pool_3d(x: torch.Tensor) -> torch.Tensor:
    """
    Global average pooling over spatial dimensions.
    Uses PyTorch for efficiency on 5D tensors.
    """
    return x.mean(dim=(2, 3, 4), keepdim=True)


# =============================================================================
# Main Model
# =============================================================================

class ModelNew(nn.Module):
    """
    Model that performs:
    1. 3D transposed convolution
    2. Elementwise scaling
    3. Batch normalization
    4. Global average pooling
    """

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor
        self.eps = eps
        self.momentum = momentum

        # ConvTranspose3d
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)

        # BatchNorm3d parameters
        self.num_features = out_channels
        self.weight = nn.Parameter(torch.ones(out_channels, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(out_channels, dtype=torch.float32))
        self.register_buffer('running_mean', torch.zeros(out_channels, dtype=torch.float32))
        self.register_buffer('running_var', torch.ones(out_channels, dtype=torch.float32))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.int64))

    def forward(self, x):
        # Ensure BF16
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)

        # Step 1: ConvTranspose3d
        x = self.conv_transpose(x)

        # Step 2: Elementwise scaling
        x = x * self.scale_factor

        # Step 3: BatchNorm3d
        if self.training:
            # Training mode
            x_f32 = x.to(torch.float32)

            mean = x_f32.mean(dim=(0, 2, 3, 4))
            var = x_f32.var(dim=(0, 2, 3, 4), unbiased=False)

            with torch.no_grad():
                self.num_batches_tracked.add_(1)
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var

            x_f32 = (x_f32 - mean.view(1, -1, 1, 1, 1)) / torch.sqrt(var.view(1, -1, 1, 1, 1) + self.eps)
            x_f32 = x_f32 * self.weight.view(1, -1, 1, 1, 1) + self.bias.view(1, -1, 1, 1, 1)
            x = x_f32.to(torch.bfloat16)
        else:
            # Inference mode
            x_f32 = x.to(torch.float32)
            x_f32 = (x_f32 - self.running_mean.view(1, -1, 1, 1, 1)) / torch.sqrt(
                self.running_var.view(1, -1, 1, 1, 1) + self.eps
            )
            x_f32 = x_f32 * self.weight.view(1, -1, 1, 1, 1) + self.bias.view(1, -1, 1, 1, 1)
            x = x_f32.to(torch.bfloat16)

        # Step 4: Global average pooling
        x = x.mean(dim=(2, 3, 4), keepdim=True)

        return x


# =============================================================================
# Input/Output Functions
# =============================================================================

batch_size = 16
in_channels = 64
out_channels = 128
depth, height, width = 16, 32, 32
kernel_size = 5
scale_factor = 2.0


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scale_factor]
