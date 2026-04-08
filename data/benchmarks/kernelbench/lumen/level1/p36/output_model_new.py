import torch
import substrate
import substrate.language as S

BLOCK_SIZE: S.constexpr = 256
VEC_SIZE: S.constexpr = 8
U32_PER_VEC: S.constexpr = 4


@substrate.jit
def rms_norm_apply_kernel(
    input_ptr: S.Pointer(S.bf16),
    scale_ptr: S.Pointer(S.bf16),
    output_ptr: S.Pointer(S.bf16),
    blocks_per_batch: S.i32,
    scale_vec_dim: S.i32,
    input_range_bytes: S.u32,
    scale_range_bytes: S.u32,
    output_range_bytes: S.u32,
):
    tid = S.thread_id(0)
    block = S.block_id(0)
    batch_idx = block // blocks_per_batch
    block_in_batch = block - batch_idx * blocks_per_batch
    scale_blocks_per_feature = scale_vec_dim // BLOCK_SIZE
    scale_block = block_in_batch - (block_in_batch // scale_blocks_per_feature) * scale_blocks_per_feature

    flat_input = S.make_tensor(input_ptr, S.bf16, S.make_layout((input_range_bytes // 2,), (1,)))
    flat_output = S.make_tensor(output_ptr, S.bf16, S.make_layout((output_range_bytes // 2,), (1,)))
    flat_scale = S.make_tensor(scale_ptr, S.bf16, S.make_layout((scale_range_bytes // 2,), (1,)))
    input_u32 = S.view(flat_input, S.u32, S.make_layout((input_range_bytes // 4,), (1,)))
    output_u32 = S.view(flat_output, S.u32, S.make_layout((output_range_bytes // 4,), (1,)))
    scale_u32 = S.view(flat_scale, S.u32, S.make_layout((scale_range_bytes // 4,), (1,)))
    input_rsrc = S.amdgpu.make_rsrc(input_u32, input_range_bytes)
    output_rsrc = S.amdgpu.make_rsrc(output_u32, output_range_bytes)
    scale_rsrc = S.amdgpu.make_rsrc(scale_u32, scale_range_bytes)

    zero_u32 = S.convert(0, S.u32)
    input_byte_offset = S.convert((block * BLOCK_SIZE + tid) * VEC_SIZE * 2, S.u32)
    scale_byte_offset = S.convert((batch_idx * scale_vec_dim + scale_block * BLOCK_SIZE + tid) * VEC_SIZE * 2, S.u32)
    packed = S.amdgpu.raw_buffer_load_x4(input_rsrc, input_byte_offset, zero_u32, 0)
    scale_packed = S.amdgpu.raw_buffer_load_x4(scale_rsrc, scale_byte_offset, zero_u32, 0)
    vals = S.view(packed, S.Tensor((VEC_SIZE,), S.bf16))
    scale_vals = S.view(scale_packed, S.Tensor((VEC_SIZE,), S.bf16))
    out_vec = S.make_local((VEC_SIZE,), S.bf16)

    for i in S.range(VEC_SIZE):
        out_vec[i] = S.convert(S.convert(vals[i], S.f32) * S.convert(scale_vals[i], S.f32), S.bf16)

    packed_out = S.view(out_vec, S.Tensor((U32_PER_VEC,), S.u32))
    S.amdgpu.raw_buffer_store_x4(packed_out, output_rsrc, input_byte_offset, zero_u32, 0)


def substrate_rms_norm(x: torch.Tensor, num_features: int, eps: float) -> torch.Tensor:
    assert x.is_cuda, "Tensors must be on CUDA/HIP device."

    original_dtype = x.dtype
    x_bf16 = x.contiguous().to(torch.bfloat16)
    batch_size, features, dim1, dim2 = x_bf16.shape
    assert features == num_features, "Feature count mismatch"

    spatial = dim1 * dim2
    elements_per_batch = features * spatial
    total_elements = x_bf16.numel()
    output = torch.empty_like(x_bf16)

    norm = torch.norm(x_bf16, p=2, dim=1).float()
    scale = (
        torch.sqrt(torch.tensor(float(num_features), device=x.device, dtype=torch.float32))
        / torch.clamp_min(norm, eps)
    ).to(torch.bfloat16).contiguous()
    total_vectors = total_elements // VEC_SIZE
    blocks_per_batch = (elements_per_batch // VEC_SIZE) // BLOCK_SIZE
    scale_vec_dim = spatial // VEC_SIZE
    rms_norm_apply_kernel[lambda: ((total_vectors // BLOCK_SIZE, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_bf16,
        scale.view(-1),
        output,
        blocks_per_batch,
        scale_vec_dim,
        x_bf16.numel() * 2,
        scale.numel() * 2,
        x_bf16.numel() * 2,
    )

    return output.to(dtype=original_dtype)


class ModelNew(torch.nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return substrate_rms_norm(x, self.num_features, self.eps)
