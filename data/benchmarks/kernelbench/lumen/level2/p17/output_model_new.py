import torch
import torch.nn as nn
import math
import substrate
import substrate.language as S

# InstanceNorm constants
BLOCK_SIZE: S.constexpr = 256
WAVE_SIZE: S.constexpr = 64
WAVE_COUNT: S.constexpr = BLOCK_SIZE // WAVE_SIZE
EPS: S.constexpr = 1e-5


@substrate.jit
def instance_norm_stats_kernel(
    input_ptr: S.Pointer(S.bf16),
    mean_ptr: S.Pointer(S.f32),
    rstd_ptr: S.Pointer(S.f32),
    total_instances: S.i32,
    spatial_size: S.i32,
):
    """Compute mean and reciprocal std for each (batch, channel) instance."""
    tid = S.thread_id(0)
    row = S.block_id(0)
    if row >= total_instances:
        return

    lane = tid % WAVE_SIZE
    wave = tid // WAVE_SIZE
    elems_per_thread = (spatial_size + BLOCK_SIZE - 1) // BLOCK_SIZE

    input_tensor = S.make_tensor(
        input_ptr,
        S.bf16,
        S.make_layout((total_instances, spatial_size), (spatial_size, 1)),
    )
    mean_tensor = S.make_tensor(mean_ptr, S.f32, S.make_layout((total_instances,), (1,)))
    rstd_tensor = S.make_tensor(rstd_ptr, S.f32, S.make_layout((total_instances,), (1,)))
    partial_sum = S.make_shared((WAVE_COUNT,), S.f32)
    partial_sq = S.make_shared((WAVE_COUNT,), S.f32)

    local_sum = S.convert(0.0, S.f32)
    local_sq = S.convert(0.0, S.f32)

    for tile in S.range(elems_per_thread):
        elem_idx = tile * BLOCK_SIZE + tid
        if elem_idx < spatial_size:
            val = S.convert(input_tensor[row, elem_idx], S.f32)
            local_sum = local_sum + val
            local_sq = local_sq + val * val

    # Wave-level reduction for sum
    other = S.shuffle_xor(local_sum, 32, WAVE_SIZE)
    local_sum = local_sum + other
    other = S.shuffle_xor(local_sum, 16, WAVE_SIZE)
    local_sum = local_sum + other
    other = S.shuffle_xor(local_sum, 8, WAVE_SIZE)
    local_sum = local_sum + other
    other = S.shuffle_xor(local_sum, 4, WAVE_SIZE)
    local_sum = local_sum + other
    other = S.shuffle_xor(local_sum, 2, WAVE_SIZE)
    local_sum = local_sum + other
    other = S.shuffle_xor(local_sum, 1, WAVE_SIZE)
    local_sum = local_sum + other

    # Wave-level reduction for sum of squares
    other = S.shuffle_xor(local_sq, 32, WAVE_SIZE)
    local_sq = local_sq + other
    other = S.shuffle_xor(local_sq, 16, WAVE_SIZE)
    local_sq = local_sq + other
    other = S.shuffle_xor(local_sq, 8, WAVE_SIZE)
    local_sq = local_sq + other
    other = S.shuffle_xor(local_sq, 4, WAVE_SIZE)
    local_sq = local_sq + other
    other = S.shuffle_xor(local_sq, 2, WAVE_SIZE)
    local_sq = local_sq + other
    other = S.shuffle_xor(local_sq, 1, WAVE_SIZE)
    local_sq = local_sq + other

    if lane == 0:
        partial_sum[wave] = local_sum
        partial_sq[wave] = local_sq
    S.syncthreads()

    if wave == 0:
        block_sum = partial_sum[lane] if lane < WAVE_COUNT else S.convert(0.0, S.f32)
        block_sq = partial_sq[lane] if lane < WAVE_COUNT else S.convert(0.0, S.f32)

        # Block-level reduction
        other = S.shuffle_xor(block_sum, 32, WAVE_SIZE)
        block_sum = block_sum + other
        other = S.shuffle_xor(block_sum, 16, WAVE_SIZE)
        block_sum = block_sum + other
        other = S.shuffle_xor(block_sum, 8, WAVE_SIZE)
        block_sum = block_sum + other
        other = S.shuffle_xor(block_sum, 4, WAVE_SIZE)
        block_sum = block_sum + other
        other = S.shuffle_xor(block_sum, 2, WAVE_SIZE)
        block_sum = block_sum + other
        other = S.shuffle_xor(block_sum, 1, WAVE_SIZE)
        block_sum = block_sum + other

        other = S.shuffle_xor(block_sq, 32, WAVE_SIZE)
        block_sq = block_sq + other
        other = S.shuffle_xor(block_sq, 16, WAVE_SIZE)
        block_sq = block_sq + other
        other = S.shuffle_xor(block_sq, 8, WAVE_SIZE)
        block_sq = block_sq + other
        other = S.shuffle_xor(block_sq, 4, WAVE_SIZE)
        block_sq = block_sq + other
        other = S.shuffle_xor(block_sq, 2, WAVE_SIZE)
        block_sq = block_sq + other
        other = S.shuffle_xor(block_sq, 1, WAVE_SIZE)
        block_sq = block_sq + other

        if lane == 0:
            count = S.convert(spatial_size, S.f32)
            inv_count = S.amdgpu.rcp(count)
            zero = S.convert(0.0, S.f32)
            mean_val = block_sum * inv_count
            variance = block_sq * inv_count - mean_val * mean_val
            variance = variance if variance > zero else zero
            mean_tensor[row] = mean_val
            rstd_tensor[row] = S.amdgpu.rcp(S.sqrt(variance + S.convert(EPS, S.f32)))


@substrate.jit
def instance_norm_apply_kernel(
    input_ptr: S.Pointer(S.bf16),
    output_ptr: S.Pointer(S.bf16),
    scale_ptr: S.Pointer(S.f32),
    offset_ptr: S.Pointer(S.f32),
    total_instances: S.i32,
    spatial_size: S.i32,
):
    """Apply normalization: output = input * scale + offset."""
    tid = S.thread_id(0)
    row = S.block_id(0)
    if row >= total_instances:
        return

    elems_per_thread = (spatial_size + BLOCK_SIZE - 1) // BLOCK_SIZE

    input_tensor = S.make_tensor(
        input_ptr,
        S.bf16,
        S.make_layout((total_instances, spatial_size), (spatial_size, 1)),
    )
    output_tensor = S.make_tensor(
        output_ptr,
        S.bf16,
        S.make_layout((total_instances, spatial_size), (spatial_size, 1)),
    )
    scale_tensor = S.make_tensor(scale_ptr, S.f32, S.make_layout((total_instances,), (1,)))
    offset_tensor = S.make_tensor(offset_ptr, S.f32, S.make_layout((total_instances,), (1,)))

    scale = scale_tensor[row]
    offset = offset_tensor[row]

    for tile in S.range(elems_per_thread):
        elem_idx = tile * BLOCK_SIZE + tid
        if elem_idx < spatial_size:
            val = S.convert(input_tensor[row, elem_idx], S.f32)
            out_val = val * scale + offset
            output_tensor[row, elem_idx] = S.convert(out_val, S.bf16)


class ModelNew(nn.Module):
    """
    Optimized model that performs Conv2d -> InstanceNorm2d -> Divide by constant
    using Substrate DSL kernels on AMD GPU.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, divide_by: float):
        super(ModelNew, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.divide_by = divide_by

        # Conv2d weight and bias (bias=True by default in nn.Conv2d)
        self.conv_weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.conv_bias = nn.Parameter(torch.empty(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.conv_weight, a=math.sqrt(5))
        fan_in = self.conv_weight.shape[1] * self.conv_weight.shape[2] * self.conv_weight.shape[3]
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.conv_bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype

        # ============ Conv2d ============
        x_conv = torch.nn.functional.conv2d(
            x,
            self.conv_weight,
            self.conv_bias,
            stride=1,
            padding=0,
            dilation=1,
            groups=1
        )

        # ============ InstanceNorm2d + Divide using Substrate ============
        batch_size, num_features, out_h, out_w = x_conv.shape
        spatial_size = out_h * out_w

        # Convert to BF16 for kernel
        x_conv_bf16 = x_conv.to(dtype=torch.bfloat16).contiguous()

        total_instances = batch_size * num_features

        # Allocate buffers for mean and rstd
        mean_buf = torch.empty(total_instances, dtype=torch.float32, device=x.device)
        rstd_buf = torch.empty(total_instances, dtype=torch.float32, device=x.device)

        # Stats kernel
        instance_norm_stats_kernel[lambda: ((total_instances, 1, 1), (BLOCK_SIZE, 1, 1))](
            x_conv_bf16,
            mean_buf,
            rstd_buf,
            total_instances,
            spatial_size,
            num_warps=8,
        )

        # Compute scale and offset for normalization + division
        # InstanceNorm with affine=False: output = (x - mean) * rstd
        # Then divide: output = output / divide_by
        # Combined: scale = rstd / divide_by, offset = -mean * rstd / divide_by
        inv_divisor = 1.0 / self.divide_by
        scale_buf = (rstd_buf * inv_divisor).contiguous()
        offset_buf = (-mean_buf * rstd_buf * inv_divisor).contiguous()

        # Output tensor
        final_output = torch.empty_like(x_conv_bf16)

        # Apply kernel
        instance_norm_apply_kernel[lambda: ((total_instances, 1, 1), (BLOCK_SIZE, 1, 1))](
            x_conv_bf16,
            final_output,
            scale_buf,
            offset_buf,
            total_instances,
            spatial_size,
            num_warps=8,
        )

        # Convert back to original dtype if needed
        if original_dtype != torch.bfloat16:
            final_output = final_output.to(original_dtype)

        return final_output
