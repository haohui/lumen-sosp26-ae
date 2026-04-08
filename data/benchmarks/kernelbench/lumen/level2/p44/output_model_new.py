import torch
import torch.nn as nn
import substrate
import substrate.language as S
import struct

# Dimensions:
# Input: (16, 64, 128, 128)
# After ConvTranspose2d: (16, 128, 256, 256)
# After global avg pool: (16, 128, 1, 1)

BATCH_SIZE = 16
OUT_CHANNELS = 128
SPATIAL_H = 256
SPATIAL_W = 256
SPATIAL_SIZE = SPATIAL_H * SPATIAL_W  # 65536

# Thread block configuration
WARP_SIZE = 64
NUM_WARPS = 8
THREADS = WARP_SIZE * NUM_WARPS  # 512 threads per block

# Each thread processes multiple elements via strided loop
ELEMENTS_PER_THREAD = (SPATIAL_SIZE + THREADS - 1) // THREADS


@substrate.jit
def global_avg_pool_mul_kernel(
    input_ptr: S.Pointer(S.f32),
    output_ptr: S.Pointer(S.f32),
    multiplier_bits: S.u32,  # f32 multiplier as u32 bit pattern
    batch_size: S.u32,
    channels: S.u32,
    spatial_h: S.u32,
    spatial_w: S.u32,
):
    """
    Fused kernel: multiply by scalar + global average pooling.
    Each block handles one (batch, channel) pair.
    Input and output are f32.
    """
    bid = S.block_id(0)
    tid = S.thread_id(0)

    # Decode multiplier from u32 bits to f32
    multiplier = S.bitcast(multiplier_bits, S.f32)

    # Map block ID to (batch, channel)
    batch_idx = bid // channels
    chan_idx = bid % channels

    # Compute the base offset for this (batch, channel)
    spatial_size = spatial_h * spatial_w
    base_offset = (batch_idx * channels + chan_idx) * spatial_size

    # Create a tensor view for the input
    input_tensor = S.make_tensor(
        input_ptr,
        S.f32,
        S.make_layout((batch_size * channels * spatial_size,), (1,))
    )

    # Each thread accumulates its portion of the sum
    local_sum = S.convert(0.0, S.f32)

    # Strided loop over elements
    for i in S.range(ELEMENTS_PER_THREAD):
        idx = tid + i * THREADS
        if idx < spatial_size:
            val = input_tensor[base_offset + idx]
            local_sum = local_sum + val * multiplier

    # Now we need to reduce across threads in the block
    # Use shared memory for reduction
    shm = S.make_shared((THREADS,), S.f32)
    shm[tid] = local_sum
    S.syncthreads()

    # Tree reduction within the block
    # First reduce within each warp using shuffle
    warp_id = tid // WARP_SIZE
    lane_id = tid % WARP_SIZE

    # Shuffle reduction within warp
    for offset in S.range(6):  # log2(64) = 6
        shift = S.convert(1, S.u32) << S.convert(offset, S.u32)
        other = S.shuffle_xor(local_sum, shift, WARP_SIZE)
        local_sum = local_sum + other

    # Only lane 0 of each warp writes to shared memory
    if lane_id == 0:
        shm[warp_id] = local_sum
    S.syncthreads()

    # Final reduction across warp leaders (first warp only)
    if warp_id == 0:
        warp_sum = shm[lane_id]
        for offset in S.range(3):  # log2(8) = 3
            shift = S.convert(1, S.u32) << S.convert(offset, S.u32)
            other = S.shuffle_xor(warp_sum, shift, WARP_SIZE)
            warp_sum = warp_sum + other

        # Lane 0 writes the final result
        if lane_id == 0:
            # Compute mean
            inv_spatial = S.convert(1.0, S.f32) / S.convert(spatial_size, S.f32)
            mean = warp_sum * inv_spatial

            # Store result
            output_tensor = S.make_tensor(
                output_ptr,
                S.f32,
                S.make_layout((batch_size * channels,), (1,))
            )
            output_tensor[bid] = mean


def fused_global_avg_pool_mul(
    x: torch.Tensor,
    multiplier: float,
) -> torch.Tensor:
    """
    Apply scalar multiplication and global average pooling.
    Input: (B, C, H, W)
    Output: (B, C, 1, 1)
    """
    assert x.is_cuda, "Tensor must be on CUDA/HIP device."

    B, C, H, W = x.shape
    spatial_size = H * W

    # Convert to f32 for computation
    x_f32 = x.float().contiguous()

    # Output tensor
    out = torch.empty((B, C), dtype=torch.float32, device=x.device)

    # Pack multiplier f32 as u32
    multiplier_bits = struct.unpack('I', struct.pack('f', multiplier))[0]

    # Launch kernel: one block per (batch, channel) pair
    num_blocks = B * C

    global_avg_pool_mul_kernel[lambda: ((num_blocks, 1, 1), (THREADS, 1, 1))](
        x_f32,
        out,
        multiplier_bits,
        B,
        C,
        H,
        W,
        num_warps=NUM_WARPS,
    )

    # Convert back to original dtype and reshape to (B, C, 1, 1)
    return out.to(x.dtype).view(B, C, 1, 1)


class ModelNew(nn.Module):
    """
    Optimized model using Substrate DSL kernel for fused operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.multiplier = multiplier

    def forward(self, x):
        # Transposed convolution (uses PyTorch's optimized implementation)
        x = self.conv_transpose(x)

        # Fused: scalar multiply + global average pooling
        x = fused_global_avg_pool_mul(x, self.multiplier)

        # Second global average pooling is a no-op on (B, C, 1, 1)

        return x
