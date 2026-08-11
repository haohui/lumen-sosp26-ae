import torch
import substrate
import substrate.language as S

BLOCK_SIZE: S.constexpr = 256
VEC_SIZE: S.constexpr = 8
U32_PER_VEC: S.constexpr = 4


@substrate.jit
def l1_norm_vector_kernel(
    input_ptr: S.Pointer(S.bf16),
    scale_ptr: S.Pointer(S.f32),
    output_ptr: S.Pointer(S.bf16),
    batch_size: S.i32,
    dim: S.i32,
    vec_dim: S.i32,
    total_u32: S.i32,
    range_bytes: S.u32,
):
    idx = S.block_id(0) * BLOCK_SIZE + S.thread_id(0)
    flat_bf16 = S.make_tensor(input_ptr, S.bf16, S.make_layout((total_u32 * 2,), (1,)))
    flat_out = S.make_tensor(output_ptr, S.bf16, S.make_layout((total_u32 * 2,), (1,)))
    flat_u32 = S.view(flat_bf16, S.u32, S.make_layout((total_u32,), (1,)))
    out_u32 = S.view(flat_out, S.u32, S.make_layout((total_u32,), (1,)))
    input_rsrc = S.amdgpu.make_rsrc(flat_u32, range_bytes)
    output_rsrc = S.amdgpu.make_rsrc(out_u32, range_bytes)
    scale_tensor = S.make_tensor(scale_ptr, S.f32, S.make_layout((batch_size,), (1,)))

    row = idx // vec_dim
    vec_in_row = idx - row * vec_dim
    scale = scale_tensor[row]
    zero_u32 = S.convert(0, S.u32)
    byte_offset = S.convert((row * dim + vec_in_row * VEC_SIZE) * 2, S.u32)
    packed = S.amdgpu.raw_buffer_load_x4(input_rsrc, byte_offset, zero_u32, 0)
    vals = S.view(packed, S.Tensor((VEC_SIZE,), S.bf16))
    out_vec = S.make_local((VEC_SIZE,), S.bf16)

    for i in S.range(VEC_SIZE):
        out_vec[i] = S.convert(S.convert(vals[i], S.f32) * scale, S.bf16)

    packed_out = S.view(out_vec, S.Tensor((U32_PER_VEC,), S.u32))
    S.amdgpu.raw_buffer_store_x4(packed_out, output_rsrc, byte_offset, zero_u32, 0)


@substrate.jit
def l1_norm_tail_kernel(
    input_ptr: S.Pointer(S.bf16),
    scale_ptr: S.Pointer(S.f32),
    output_ptr: S.Pointer(S.bf16),
    batch_size: S.i32,
    dim: S.i32,
    vec_dim: S.i32,
    tail: S.i32,
):
    idx = S.block_id(0) * BLOCK_SIZE + S.thread_id(0)
    total_tail = batch_size * tail
    if idx < total_tail:
        row = idx // tail
        tail_idx = idx - row * tail
        element_idx = row * dim + vec_dim * VEC_SIZE + tail_idx
        input_tensor = S.make_tensor(input_ptr, S.bf16, S.make_layout((batch_size * dim,), (1,)))
        output_tensor = S.make_tensor(output_ptr, S.bf16, S.make_layout((batch_size * dim,), (1,)))
        scale_tensor = S.make_tensor(scale_ptr, S.f32, S.make_layout((batch_size,), (1,)))
        output_tensor[element_idx] = S.convert(S.convert(input_tensor[element_idx], S.f32) * scale_tensor[row], S.bf16)


def substrate_l1_norm(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Tensors must be on CUDA/HIP device."

    original_dtype = x.dtype
    x_bf16 = x.to(dtype=torch.bfloat16, device=x.device).contiguous()
    batch_size, dim = x_bf16.shape
    vec_dim = dim // VEC_SIZE
    tail = dim - vec_dim * VEC_SIZE
    output = torch.empty_like(x_bf16)

    scale = torch.reciprocal(torch.mean(torch.abs(x_bf16).float(), dim=1)).contiguous()
    total_vectors = batch_size * vec_dim
    if total_vectors:
        total_u32 = x_bf16.numel() // 2
        l1_norm_vector_kernel[lambda: ((total_vectors // BLOCK_SIZE, 1, 1), (BLOCK_SIZE, 1, 1))](
            x_bf16,
            scale,
            output,
            batch_size,
            dim,
            vec_dim,
            total_u32,
            x_bf16.numel() * 2,
        )

    if tail:
        total_tail = batch_size * tail
        tail_blocks = (total_tail + BLOCK_SIZE - 1) // BLOCK_SIZE
        l1_norm_tail_kernel[lambda: ((tail_blocks, 1, 1), (BLOCK_SIZE, 1, 1))](
            x_bf16,
            scale,
            output,
            batch_size,
            dim,
            vec_dim,
            tail,
        )

    return output.to(dtype=original_dtype)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return substrate_l1_norm(x)
