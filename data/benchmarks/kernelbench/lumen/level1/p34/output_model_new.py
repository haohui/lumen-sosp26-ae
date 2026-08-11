import torch
import torch.nn as nn
import substrate
import substrate.language as S

BLOCK_SIZE: S.constexpr = 256
VEC_SIZE: S.constexpr = 8
U32_PER_VEC: S.constexpr = 4
WAVE_SIZE: S.constexpr = 64
WAVE_COUNT: S.constexpr = BLOCK_SIZE // WAVE_SIZE
EPS = 1e-5


@substrate.jit
def instance_norm_apply_kernel(
    input_ptr: S.Pointer(S.bf16),
    output_ptr: S.Pointer(S.bf16),
    scale_ptr: S.Pointer(S.f32),
    offset_ptr: S.Pointer(S.f32),
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

    flat_input = S.make_tensor(input_ptr, S.bf16, S.make_layout((total_u32 * 2,), (1,)))
    flat_output = S.make_tensor(output_ptr, S.bf16, S.make_layout((total_u32 * 2,), (1,)))
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


def substrate_instance_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    assert x.is_cuda, "Tensor must be on CUDA/HIP device."
    assert x.dtype == torch.bfloat16, "Input must be bfloat16"

    batch_size, num_features, height, width = x.shape
    spatial_size = height * width
    assert spatial_size % VEC_SIZE == 0, "Spatial size must be divisible by 8."
    assert spatial_size % (BLOCK_SIZE * VEC_SIZE) == 0, "Spatial size must align with block tiles."

    total_instances = batch_size * num_features
    blocks_per_row = spatial_size // (BLOCK_SIZE * VEC_SIZE)
    x_contiguous = x.contiguous()
    output = torch.empty_like(x_contiguous)
    var_bf16, mean_bf16 = torch.var_mean(x_contiguous, dim=(2, 3), correction=0)
    mean_buf = mean_bf16.float().contiguous().view(-1)
    rstd_buf = torch.rsqrt(var_bf16.float().contiguous().view(-1) + EPS)

    weight_f32 = weight.float().view(1, num_features)
    bias_f32 = bias.float().view(1, num_features)
    scale_buf = (rstd_buf.view(batch_size, num_features) * weight_f32).contiguous().view(-1)
    offset_buf = (bias_f32 - mean_buf.view(batch_size, num_features) * scale_buf.view(batch_size, num_features)).contiguous().view(-1)

    instance_norm_apply_kernel[lambda: ((total_instances * blocks_per_row, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_contiguous,
        output,
        scale_buf,
        offset_buf,
        total_instances,
        blocks_per_row,
        x_contiguous.numel() // 2,
        x_contiguous.numel() * 2,
        num_warps=4,
    )

    return output


class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.weight = nn.Parameter(torch.ones(num_features, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(num_features, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)
        output = substrate_instance_norm(x, self.weight, self.bias)
        return output if input_dtype == torch.bfloat16 else output.to(input_dtype)
