import torch
import torch.nn as nn
import substrate
import substrate.language as S

BLOCK_SIZE: S.constexpr = 256
VEC_SIZE: S.constexpr = 8
U32_PER_VEC: S.constexpr = 4


@substrate.jit
def sum_reduction_vec_kernel(
    input_ptr: S.Pointer(S.bf16),
    output_ptr: S.Pointer(S.bf16),
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
    output_flat = S.make_tensor(output_ptr, S.bf16, S.make_layout((batch_size * dim2,), (1,)))
    input_rsrc = S.amdgpu.make_rsrc(input_flat, input_range_bytes)
    output_rsrc = S.amdgpu.make_rsrc(output_flat, output_range_bytes)

    zero_u32 = S.convert(0, S.u32)
    acc = S.make_local((VEC_SIZE,), S.f32)
    result = S.make_local((VEC_SIZE,), S.bf16)

    for j in S.range(VEC_SIZE):
        acc[j] = S.convert(0.0, S.f32)

    for i in S.range(dim1):
        elem_offset = (batch_idx * dim1 + i) * dim2 + vec_idx * VEC_SIZE
        byte_offset = S.convert(elem_offset * 2, S.u32)
        packed = S.amdgpu.raw_buffer_load_x4(input_rsrc, byte_offset, zero_u32, 0)
        vals = S.view(packed, S.Tensor((VEC_SIZE,), S.bf16))
        for j in S.range(VEC_SIZE):
            acc[j] = acc[j] + S.convert(vals[j], S.f32)

    for j in S.range(VEC_SIZE):
        result[j] = S.convert(acc[j], S.bf16)

    out_byte_offset = S.convert((batch_idx * dim2 + vec_idx * VEC_SIZE) * 2, S.u32)
    packed_out = S.view(result, S.Tensor((U32_PER_VEC,), S.u32))
    S.amdgpu.raw_buffer_store_x4(packed_out, output_rsrc, out_byte_offset, zero_u32, 0)


@substrate.jit
def sum_reduction_scalar_tail_kernel(
    input_ptr: S.Pointer(S.bf16),
    output_ptr: S.Pointer(S.bf16),
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
        output_flat = S.make_tensor(output_ptr, S.bf16, S.make_layout((batch_size * dim2,), (1,)))

        current_sum = S.convert(0.0, S.f32)
        for i in S.range(dim1):
            elem_idx = (batch_idx * dim1 + i) * dim2 + dim2_idx
            current_sum = current_sum + S.convert(input_flat[elem_idx], S.f32)

        output_flat[batch_idx * dim2 + dim2_idx] = S.convert(current_sum, S.bf16)


def substrate_sum(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda, "Tensors must be on CUDA/HIP device."
    assert x.dtype == torch.bfloat16, "Input tensor must be bfloat16"

    batch_size = x.shape[0]
    dim1 = x.shape[1]
    dim2 = x.shape[2]

    x_contiguous = x.contiguous()
    output = torch.empty((batch_size, 1, dim2), dtype=torch.bfloat16, device=x.device)
    x_flat = x_contiguous.view(-1)
    output_flat = output.view(-1)

    dim2_vectors = dim2 // VEC_SIZE
    scalar_tail = dim2 - dim2_vectors * VEC_SIZE

    if dim2_vectors:
        total_slots = batch_size * dim2_vectors
        num_blocks = (total_slots + BLOCK_SIZE - 1) // BLOCK_SIZE
        sum_reduction_vec_kernel[lambda: ((num_blocks, 1, 1), (BLOCK_SIZE, 1, 1))](
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
        sum_reduction_scalar_tail_kernel[lambda: ((num_blocks, 1, 1), (BLOCK_SIZE, 1, 1))](
            x_flat,
            output_flat,
            batch_size,
            dim1,
            dim2,
            dim2_vectors,
            scalar_tail,
        )

    return output


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return substrate_sum(x, self.dim)
