import torch
import torch.nn as nn
import substrate
import substrate.language as S

BLOCK_SIZE: S.constexpr = 256
VEC_SIZE: S.constexpr = 8
U32_PER_VEC: S.constexpr = 4
WAVE_SIZE: S.constexpr = 64
WAVE_COUNT: S.constexpr = BLOCK_SIZE // WAVE_SIZE
EPS: S.constexpr = 1e-5


@substrate.jit
def groupnorm_apply_kernel(
    x_ptr: S.Pointer(S.bf16),
    scale_ptr: S.Pointer(S.f32),
    offset_ptr: S.Pointer(S.f32),
    out_ptr: S.Pointer(S.bf16),
    total_instances: S.i32,
    blocks_per_row: S.i32,
    total_u32: S.i32,
    range_bytes: S.u32,
):
    tid = S.thread_id(0)
    bid = S.block_id(0)
    row = bid // blocks_per_row
    block_in_row = bid - row * blocks_per_row
    vec_idx = block_in_row * BLOCK_SIZE + tid

    flat_input = S.make_tensor(x_ptr, S.bf16, S.make_layout((total_u32 * 2,), (1,)))
    flat_output = S.make_tensor(out_ptr, S.bf16, S.make_layout((total_u32 * 2,), (1,)))
    input_u32 = S.view(flat_input, S.u32, S.make_layout((total_u32,), (1,)))
    output_u32 = S.view(flat_output, S.u32, S.make_layout((total_u32,), (1,)))
    input_rsrc = S.amdgpu.make_rsrc(input_u32, range_bytes)
    output_rsrc = S.amdgpu.make_rsrc(output_u32, range_bytes)
    scale_tensor = S.make_tensor(scale_ptr, S.f32, S.make_layout((total_instances,), (1,)))
    offset_tensor = S.make_tensor(offset_ptr, S.f32, S.make_layout((total_instances,), (1,)))

    scale = scale_tensor[row]
    offset = offset_tensor[row]
    out_vec = S.make_local((VEC_SIZE,), S.bf16)
    zero_u32 = S.convert(0, S.u32)
    byte_offset = S.convert((row * blocks_per_row * BLOCK_SIZE + vec_idx) * VEC_SIZE * 2, S.u32)
    packed = S.amdgpu.raw_buffer_load_x4(input_rsrc, byte_offset, zero_u32, 0)
    vals = S.view(packed, S.Tensor((VEC_SIZE,), S.bf16))
    for j in S.range(VEC_SIZE):
        val = S.convert(vals[j], S.f32)
        out_vec[j] = S.convert(val * scale + offset, S.bf16)
    packed_out = S.view(out_vec, S.Tensor((U32_PER_VEC,), S.u32))
    S.amdgpu.raw_buffer_store_x4(packed_out, output_rsrc, byte_offset, zero_u32, 0)


def substrate_groupnorm(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    num_groups: int,
    eps: float = 1e-5,
) -> torch.Tensor:
    assert x.is_cuda, "Tensors must be on CUDA/HIP device."
    assert x.dtype == torch.bfloat16, "Input tensor must be bfloat16"

    batch_size, features, dim1, dim2 = x.shape
    spatial_size = dim1 * dim2
    assert spatial_size % VEC_SIZE == 0, "Spatial size must be divisible by 8."
    assert spatial_size % (BLOCK_SIZE * VEC_SIZE) == 0, "Spatial size must align with block tiles."
    assert features % num_groups == 0, "features must be divisible by num_groups"

    x_contiguous = x.contiguous()
    total_groups = batch_size * num_groups
    total_instances = batch_size * features
    blocks_per_row = spatial_size // (BLOCK_SIZE * VEC_SIZE)
    channels_per_group = features // num_groups
    out = torch.empty_like(x_contiguous)
    channels_per_group = features // num_groups
    x_group = x_contiguous.view(batch_size, num_groups, channels_per_group, dim1, dim2)
    var_bf16, mean_bf16 = torch.var_mean(x_group, dim=(2, 3, 4), correction=0)
    mean = mean_bf16.float().contiguous().view(-1)
    rstd = torch.rsqrt(var_bf16.float().contiguous().view(-1) + eps)

    mean_2d = mean.view(batch_size, num_groups)
    rstd_2d = rstd.view(batch_size, num_groups)
    group_index = torch.arange(features, device=x.device) // channels_per_group
    gamma_f32 = gamma.float().view(1, features)
    beta_f32 = beta.float().view(1, features)
    scale = (rstd_2d[:, group_index] * gamma_f32).contiguous().view(-1)
    offset = (beta_f32 - mean_2d[:, group_index] * scale.view(batch_size, features)).contiguous().view(-1)

    groupnorm_apply_kernel[lambda: ((total_instances * blocks_per_row, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_contiguous,
        scale,
        offset,
        out,
        total_instances,
        blocks_per_row,
        x_contiguous.numel() // 2,
        x_contiguous.numel() * 2,
        num_warps=4,
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, num_features: int, num_groups: int):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.num_groups = num_groups
        self.weight = nn.Parameter(torch.ones(num_features, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(num_features, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)
        output = substrate_groupnorm(x, self.weight, self.bias, self.num_groups)
        return output if input_dtype == torch.bfloat16 else output.to(input_dtype)
