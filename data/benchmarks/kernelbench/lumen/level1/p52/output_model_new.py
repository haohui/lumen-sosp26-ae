import torch
import substrate
import substrate.language as S

BLOCK_SIZE: S.constexpr = 256
VEC_SIZE: S.constexpr = 8
U32_PER_VEC: S.constexpr = 4


@substrate.jit
def argmin_reduction_vec_kernel(
    input_ptr: S.Pointer(S.bf16),
    output_ptr: S.Pointer(S.i32),
    batch_size: S.i32,
    dim1: S.i32,
    dim2: S.i32,
    dim2_vectors: S.i32,
    input_range_bytes: S.u32,
    output_range_bytes: S.u32,
):
    slot = S.block_id(0) * BLOCK_SIZE + S.thread_id(0)
    batch_idx = slot // dim2_vectors
    vec_idx = slot - batch_idx * dim2_vectors

    input_flat = S.make_tensor(input_ptr, S.bf16, S.make_layout((batch_size * dim1 * dim2,), (1,)))
    output_flat = S.make_tensor(output_ptr, S.i32, S.make_layout((batch_size * dim2,), (1,)))
    input_rsrc = S.amdgpu.make_rsrc(input_flat, input_range_bytes)
    output_rsrc = S.amdgpu.make_rsrc(output_flat, output_range_bytes)

    zero_u32 = S.convert(0, S.u32)
    first_offset = S.convert(((batch_idx * dim1) * dim2 + vec_idx * VEC_SIZE) * 2, S.u32)
    current_min = S.make_local((VEC_SIZE,), S.bf16)
    current_argmin = S.make_local((VEC_SIZE,), S.i32)
    first_vals = S.view(
        S.amdgpu.raw_buffer_load_x4(input_rsrc, first_offset, zero_u32, 0),
        S.Tensor((VEC_SIZE,), S.bf16),
    )

    for j in S.range(VEC_SIZE):
        current_min[j] = first_vals[j]
        current_argmin[j] = S.convert(0, S.i32)

    for i in S.range(1, dim1):
        vals = S.view(
            S.amdgpu.raw_buffer_load_x4(
                input_rsrc,
                S.convert(((batch_idx * dim1 + i) * dim2 + vec_idx * VEC_SIZE) * 2, S.u32),
                zero_u32,
                0,
            ),
            S.Tensor((VEC_SIZE,), S.bf16),
        )
        i_i32 = S.convert(i, S.i32)
        for j in S.range(VEC_SIZE):
            is_less = vals[j] < current_min[j]
            current_min[j] = vals[j] if is_less else current_min[j]
            current_argmin[j] = i_i32 if is_less else current_argmin[j]

    out_byte_offset = S.convert((batch_idx * dim2 + vec_idx * VEC_SIZE) * 4, S.u32)
    chunk0 = S.make_local((U32_PER_VEC,), S.u32)
    chunk1 = S.make_local((U32_PER_VEC,), S.u32)
    for j in S.range(U32_PER_VEC):
        chunk0[j] = S.convert(current_argmin[j], S.u32)
        chunk1[j] = S.convert(current_argmin[j + U32_PER_VEC], S.u32)
    S.amdgpu.raw_buffer_store_x4(chunk0, output_rsrc, out_byte_offset, zero_u32, 0)
    S.amdgpu.raw_buffer_store_x4(
        chunk1,
        output_rsrc,
        out_byte_offset + S.convert(U32_PER_VEC * 4, S.u32),
        zero_u32,
        0,
    )


@substrate.jit
def argmin_reduction_scalar_tail_kernel(
    input_ptr: S.Pointer(S.bf16),
    output_ptr: S.Pointer(S.i32),
    batch_size: S.i32,
    dim1: S.i32,
    dim2: S.i32,
    dim2_vectors: S.i32,
    scalar_tail: S.i32,
):
    gid = S.block_id(0) * BLOCK_SIZE + S.thread_id(0)
    total_tail = batch_size * scalar_tail

    if gid < total_tail:
        batch_idx = gid // scalar_tail
        tail_idx = gid - batch_idx * scalar_tail
        dim2_idx = dim2_vectors * VEC_SIZE + tail_idx

        input_flat = S.make_tensor(input_ptr, S.bf16, S.make_layout((batch_size * dim1 * dim2,), (1,)))
        output_flat = S.make_tensor(output_ptr, S.i32, S.make_layout((batch_size * dim2,), (1,)))

        current_min = input_flat[batch_idx * dim1 * dim2 + dim2_idx]
        current_argmin = S.convert(0, S.i32)
        for i in S.range(1, dim1):
            val = input_flat[(batch_idx * dim1 + i) * dim2 + dim2_idx]
            is_less = val < current_min
            current_min = val if is_less else current_min
            current_argmin = S.convert(i, S.i32) if is_less else current_argmin

        output_flat[batch_idx * dim2 + dim2_idx] = current_argmin


def substrate_argmin(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda, "Tensors must be on CUDA/HIP device."
    assert x.dtype == torch.bfloat16, "Input tensor must be bfloat16"

    batch_size = x.shape[0]
    dim1 = x.shape[1]
    dim2 = x.shape[2]

    x_contiguous = x.contiguous()
    output_i32 = torch.empty((batch_size, dim2), dtype=torch.int32, device=x.device)
    x_flat = x_contiguous.view(-1)
    output_flat = output_i32.view(-1)

    dim2_vectors = dim2 // VEC_SIZE
    scalar_tail = dim2 - dim2_vectors * VEC_SIZE

    if dim2_vectors:
        total_slots = batch_size * dim2_vectors
        num_blocks = (total_slots + BLOCK_SIZE - 1) // BLOCK_SIZE
        argmin_reduction_vec_kernel[lambda: ((num_blocks, 1, 1), (BLOCK_SIZE, 1, 1))](
            x_flat,
            output_flat,
            batch_size,
            dim1,
            dim2,
            dim2_vectors,
            x_flat.numel() * x_flat.element_size(),
            output_flat.numel() * output_flat.element_size(),
        )

    if scalar_tail:
        total_tail = batch_size * scalar_tail
        num_blocks = (total_tail + BLOCK_SIZE - 1) // BLOCK_SIZE
        argmin_reduction_scalar_tail_kernel[lambda: ((num_blocks, 1, 1), (BLOCK_SIZE, 1, 1))](
            x_flat,
            output_flat,
            batch_size,
            dim1,
            dim2,
            dim2_vectors,
            scalar_tail,
        )

    return output_i32.to(torch.long)


class ModelNew(torch.nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_bf16 = x.to(torch.bfloat16)
        return substrate_argmin(x_bf16, self.dim)
