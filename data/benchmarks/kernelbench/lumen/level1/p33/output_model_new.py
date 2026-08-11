import torch
import torch.nn as nn
import substrate
import substrate.language as S

BLOCK_SIZE: S.constexpr = 256
VEC_SIZE: S.constexpr = 8
U32_PER_VEC: S.constexpr = 4
EPS: S.constexpr = 1e-5


@substrate.jit
def batchnorm_apply_kernel(
    x_ptr: S.Pointer(S.bf16),
    scale_ptr: S.Pointer(S.f32),
    offset_ptr: S.Pointer(S.f32),
    out_ptr: S.Pointer(S.bf16),
    features: S.i32,
    blocks_per_row: S.i32,
    total_u32: S.i32,
    range_bytes: S.u32,
):
    tid = S.thread_id(0)
    bid = S.block_id(0)
    row = bid // blocks_per_row
    block_in_row = bid - row * blocks_per_row
    vec_idx = block_in_row * BLOCK_SIZE + tid
    c = row % features

    flat_input = S.make_tensor(x_ptr, S.bf16, S.make_layout((total_u32 * 2,), (1,)))
    flat_output = S.make_tensor(out_ptr, S.bf16, S.make_layout((total_u32 * 2,), (1,)))
    input_u32 = S.view(flat_input, S.u32, S.make_layout((total_u32,), (1,)))
    output_u32 = S.view(flat_output, S.u32, S.make_layout((total_u32,), (1,)))
    input_rsrc = S.amdgpu.make_rsrc(input_u32, range_bytes)
    output_rsrc = S.amdgpu.make_rsrc(output_u32, range_bytes)

    param_layout = S.make_layout((features,), (1,))
    scale_tensor = S.make_tensor(scale_ptr, S.f32, param_layout)
    offset_tensor = S.make_tensor(offset_ptr, S.f32, param_layout)

    scale = scale_tensor[c]
    offset = offset_tensor[c]
    out_vec = S.make_local((VEC_SIZE,), S.bf16)
    zero_u32 = S.convert(0, S.u32)
    byte_offset = S.convert((row * blocks_per_row * BLOCK_SIZE + vec_idx) * VEC_SIZE * 2, S.u32)
    packed = S.amdgpu.raw_buffer_load_x4(input_rsrc, byte_offset, zero_u32, 0)
    vals = S.view(packed, S.Tensor((VEC_SIZE,), S.bf16))
    for j in S.range(VEC_SIZE):
        x_val = S.convert(vals[j], S.f32)
        out_vec[j] = S.convert(x_val * scale + offset, S.bf16)
    packed_out = S.view(out_vec, S.Tensor((U32_PER_VEC,), S.u32))
    S.amdgpu.raw_buffer_store_x4(packed_out, output_rsrc, byte_offset, zero_u32, 0)


def substrate_batchnorm2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    training: bool,
) -> torch.Tensor:
    """Apply BatchNorm2d using Substrate kernels."""
    assert x.is_cuda, "Input must be on CUDA/HIP device"

    batch_size, features, dim1, dim2 = x.shape
    spatial_size = dim1 * dim2
    x_contiguous = x.contiguous()

    # Allocate output
    out = torch.empty_like(x_contiguous)

    if training:
        var_bf16, mean_bf16 = torch.var_mean(x_contiguous, dim=(0, 2, 3), correction=0)
        mean = mean_bf16.float().contiguous()
        var = var_bf16.float().contiguous()
    else:
        mean = running_mean.float().contiguous()
        var = running_var.float().contiguous()

    scale = weight.float() / torch.sqrt(var + EPS)
    offset = bias.float() - mean * scale

    total_instances = batch_size * features
    blocks_per_row = spatial_size // (BLOCK_SIZE * VEC_SIZE)
    total_u32 = x_contiguous.numel() // 2

    if spatial_size % (BLOCK_SIZE * VEC_SIZE) != 0:
        out = (
            x_contiguous.float() * scale.view(1, features, 1, 1)
            + offset.view(1, features, 1, 1)
        ).to(torch.bfloat16)
        return out, mean, var

    batchnorm_apply_kernel[lambda: ((total_instances * blocks_per_row, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_contiguous,
        scale.contiguous(),
        offset.contiguous(),
        out,
        features,
        blocks_per_row,
        total_u32,
        x_contiguous.numel() * 2,
        num_warps=4,
    )

    return out, mean, var


class ModelNew(nn.Module):
    """
    Optimized model that performs Batch Normalization using Substrate DSL.
    """
    def __init__(self, num_features: int):
        """
        Initializes the BatchNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
        """
        super(ModelNew, self).__init__()
        self.num_features = num_features
        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        # Running statistics (for inference)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Batch Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Batch Normalization applied, same shape as input.
        """
        # Convert to BF16 if needed
        input_dtype = x.dtype
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)

        out, batch_mean, batch_var = substrate_batchnorm2d(
            x,
            self.weight,
            self.bias,
            self.running_mean,
            self.running_var,
            self.training,
        )

        # Convert back to original dtype if needed
        if input_dtype != torch.bfloat16:
            out = out.to(input_dtype)

        # Update running statistics in training mode
        if self.training:
            with torch.no_grad():
                momentum = 0.1  # Default PyTorch momentum
                sample_count = x.shape[0] * x.shape[2] * x.shape[3]
                correction = (
                    float(sample_count) / float(sample_count - 1)
                    if sample_count > 1
                    else 1.0
                )
                unbiased_var = batch_var * correction
                self.running_mean.mul_(1 - momentum).add_(batch_mean.to(self.running_mean.dtype), alpha=momentum)
                self.running_var.mul_(1 - momentum).add_(unbiased_var.to(self.running_var.dtype), alpha=momentum)
                self.num_batches_tracked.add_(1)

        return out
