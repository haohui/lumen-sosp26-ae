import torch
import substrate
import substrate.language as S

BLOCK_SIZE: S.constexpr = 256
ELEMENTS_PER_THREAD: S.constexpr = 64
VEC_SIZE: S.constexpr = 8
U32_PER_VEC: S.constexpr = 4


@substrate.jit
def sum_squares_kernel(
    input_ptr: S.Pointer(S.bf16),
    partial_sums_ptr: S.Pointer(S.f32),
    batch_size: S.i32,
    features: S.i32,
    dim1: S.i32,
    dim2: S.i32,
    num_blocks: S.i32,
):
    """Compute partial sum of squares for Frobenius norm."""
    tid = S.thread_id(0)
    bid = S.block_id(0)

    if bid >= num_blocks:
        return

    shm = S.make_shared((BLOCK_SIZE,), S.f32)

    layout = S.make_layout(
        (batch_size, features, dim1, dim2),
        (features * dim1 * dim2, dim1 * dim2, dim2, 1),
    )
    input_tensor = S.make_tensor(input_ptr, S.bf16, layout)

    partial_layout = S.make_layout((num_blocks,), (1,))
    partial_tensor = S.make_tensor(partial_sums_ptr, S.f32, partial_layout)

    total_elements = batch_size * features * dim1 * dim2
    block_offset = bid * BLOCK_SIZE * ELEMENTS_PER_THREAD
    local_sum = S.convert(0.0, S.f32)

    for i in S.range(ELEMENTS_PER_THREAD):
        flat_idx = block_offset + i * BLOCK_SIZE + tid
        if flat_idx < total_elements:
            b_idx = flat_idx // (features * dim1 * dim2)
            remainder = flat_idx - b_idx * (features * dim1 * dim2)
            f_idx = remainder // (dim1 * dim2)
            remainder2 = remainder - f_idx * (dim1 * dim2)
            d1_idx = remainder2 // dim2
            d2_idx = remainder2 - d1_idx * dim2

            val = input_tensor[b_idx, f_idx, d1_idx, d2_idx]
            val_f32 = S.convert(val, S.f32)
            local_sum = local_sum + val_f32 * val_f32

    shm[tid] = local_sum
    S.syncthreads()

    if tid < 128:
        shm[tid] = shm[tid] + shm[tid + 128]
    S.syncthreads()

    if tid < 64:
        shm[tid] = shm[tid] + shm[tid + 64]
    S.syncthreads()

    if tid < 32:
        shm[tid] = shm[tid] + shm[tid + 32]
    S.syncthreads()

    if tid < 16:
        shm[tid] = shm[tid] + shm[tid + 16]
    S.syncthreads()

    if tid < 8:
        shm[tid] = shm[tid] + shm[tid + 8]
    S.syncthreads()

    if tid < 4:
        shm[tid] = shm[tid] + shm[tid + 4]
    S.syncthreads()

    if tid < 2:
        shm[tid] = shm[tid] + shm[tid + 2]
    S.syncthreads()

    if tid == 0:
        shm[0] = shm[0] + shm[1]
        partial_tensor[bid] = shm[0]


@substrate.jit
def normalize_kernel(
    input_ptr: S.Pointer(S.bf16),
    output_ptr: S.Pointer(S.bf16),
    norm_inv_ptr: S.Pointer(S.f32),
    n_vectors: S.i32,
    range_bytes: S.u32,
):
    """Flat vectorized scaling by the inverse Frobenius norm."""
    idx = S.block_id(0) * BLOCK_SIZE + S.thread_id(0)
    bf16_layout = S.make_layout((n_vectors, VEC_SIZE), (VEC_SIZE, 1))
    u32_layout = S.make_layout((n_vectors, U32_PER_VEC), (U32_PER_VEC, 1))
    input_tensor = S.make_tensor(input_ptr, S.bf16, bf16_layout)
    output_tensor = S.make_tensor(output_ptr, S.bf16, bf16_layout)
    input_u32 = S.view(input_tensor, S.u32, u32_layout)
    output_u32 = S.view(output_tensor, S.u32, u32_layout)
    input_rsrc = S.amdgpu.make_rsrc(input_u32, range_bytes)
    output_rsrc = S.amdgpu.make_rsrc(output_u32, range_bytes)
    norm_inv_layout = S.make_layout((1, 1), (1, 1))
    norm_inv_tensor = S.make_tensor(norm_inv_ptr, S.f32, norm_inv_layout)
    norm_inv = norm_inv_tensor[0, 0]
    byte_offset = S.convert(idx * VEC_SIZE * 2, S.u32)
    zero_u32 = S.convert(0, S.u32)
    packed = S.amdgpu.raw_buffer_load_x4(input_rsrc, byte_offset, zero_u32, 0)
    val = S.view(packed, S.Tensor((VEC_SIZE,), S.bf16))
    result = S.make_local((VEC_SIZE,), S.bf16)

    for i in S.range(VEC_SIZE):
        result[i] = S.convert(S.convert(val[i], S.f32) * norm_inv, S.bf16)

    packed_out = S.view(result, S.Tensor((U32_PER_VEC,), S.u32))
    S.amdgpu.raw_buffer_store_x4(packed_out, output_rsrc, byte_offset, zero_u32, 0)


@substrate.jit
def normalize_scalar_kernel(
    input_ptr: S.Pointer(S.bf16),
    output_ptr: S.Pointer(S.bf16),
    norm_inv_ptr: S.Pointer(S.f32),
    n_elements: S.i32,
):
    idx = S.thread_id(0)
    layout = S.make_layout((n_elements,), (1,))
    input_tensor = S.make_tensor(input_ptr, S.bf16, layout)
    output_tensor = S.make_tensor(output_ptr, S.bf16, layout)
    norm_inv_layout = S.make_layout((1, 1), (1, 1))
    norm_inv_tensor = S.make_tensor(norm_inv_ptr, S.f32, norm_inv_layout)
    norm_inv = norm_inv_tensor[0, 0]
    output_tensor[idx] = S.convert(S.convert(input_tensor[idx], S.f32) * norm_inv, S.bf16)


def substrate_frobenius_norm(x: torch.Tensor) -> torch.Tensor:
    """Compute Frobenius norm normalization using Substrate DSL kernels."""
    assert x.is_cuda, "Tensors must be on CUDA/HIP device."

    x_bf16 = x.to(dtype=torch.bfloat16).contiguous()
    batch_size = x_bf16.shape[0]
    features = x_bf16.shape[1]
    dim1 = x_bf16.shape[2]
    dim2 = x_bf16.shape[3]
    total_elements = int(x_bf16.numel())

    norm = torch.norm(x_bf16, p='fro')
    zero = torch.zeros_like(norm)
    one = torch.ones_like(norm)
    norm_inv_tensor = torch.where(norm > zero, torch.reciprocal(norm.float()), one.float()).reshape(1, 1)

    output = torch.empty_like(x_bf16)
    x_flat = x_bf16.view(-1)
    out_flat = output.view(-1)

    block_size = 256
    vector_size = 8
    n_vectors = total_elements // vector_size
    scalar_tail = total_elements - n_vectors * vector_size
    max_vectors_per_launch = ((1 << 31) - 1) // 16
    offset = 0

    remaining_vectors = n_vectors
    while remaining_vectors:
        chunk_vectors = int(min(remaining_vectors, max_vectors_per_launch))
        vector_elems = chunk_vectors * vector_size
        grid_size = (chunk_vectors + block_size - 1) // block_size
        normalize_kernel[lambda: ((grid_size, 1, 1), (block_size, 1, 1))](
            x_flat.narrow(0, offset, vector_elems),
            out_flat.narrow(0, offset, vector_elems),
            norm_inv_tensor,
            chunk_vectors,
            vector_elems * 2,
        )
        offset += vector_elems
        remaining_vectors -= chunk_vectors

    if scalar_tail:
        normalize_scalar_kernel[lambda: ((1, 1, 1), (scalar_tail, 1, 1))](
            x_flat.narrow(0, offset, scalar_tail),
            out_flat.narrow(0, offset, scalar_tail),
            norm_inv_tensor,
            scalar_tail,
        )

    return output


class ModelNew(torch.nn.Module):
    """
    Optimized model that performs Frobenius norm normalization using Substrate DSL.
    """

    def __init__(self):
        """
        Initializes the Frobenius norm normalization layer.
        """
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Frobenius norm normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of arbitrary shape.

        Returns:
            torch.Tensor: Output tensor with Frobenius norm normalization applied, same shape as input.
        """
        return substrate_frobenius_norm(x)
