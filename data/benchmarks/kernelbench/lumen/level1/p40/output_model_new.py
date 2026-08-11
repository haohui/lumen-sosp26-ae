import torch
import substrate
import substrate.language as S

BLOCK_SIZE: S.constexpr = 256
ELEMENTS_PER_THREAD: S.constexpr = 64
VEC_SIZE: S.constexpr = 8
U32_PER_VEC: S.constexpr = 4
EPS: S.constexpr = 1e-5


@substrate.jit
def layer_norm_stats_kernel(
    input_ptr: S.Pointer(S.bf16),
    partial_sum_ptr: S.Pointer(S.f32),
    partial_sum_sq_ptr: S.Pointer(S.f32),
    batch_size: S.i32,
    features: S.i32,
    dim1: S.i32,
    dim2: S.i32,
    blocks_per_batch: S.i32,
):
    tid = S.thread_id(0)
    bid = S.block_id(0)

    total_blocks = batch_size * blocks_per_batch
    if bid >= total_blocks:
        return

    batch_idx = bid // blocks_per_batch
    block_in_batch = bid - batch_idx * blocks_per_batch

    spatial = dim1 * dim2
    elements_per_batch = features * spatial
    block_chunk = BLOCK_SIZE * ELEMENTS_PER_THREAD
    chunk_base = block_in_batch * block_chunk

    shm_sum = S.make_shared((BLOCK_SIZE,), S.f32)
    shm_sum_sq = S.make_shared((BLOCK_SIZE,), S.f32)

    layout = S.make_layout(
        (batch_size, features, dim1, dim2),
        (features * dim1 * dim2, dim1 * dim2, dim2, 1),
    )
    input_tensor = S.make_tensor(input_ptr, S.bf16, layout)

    local_sum = S.convert(0.0, S.f32)
    local_sum_sq = S.convert(0.0, S.f32)

    for i in S.range(ELEMENTS_PER_THREAD):
        linear_idx = chunk_base + i * BLOCK_SIZE + tid
        if linear_idx < elements_per_batch:
            f_idx = linear_idx // spatial
            rem = linear_idx - f_idx * spatial
            d1_idx = rem // dim2
            d2_idx = rem - d1_idx * dim2

            val = S.convert(input_tensor[batch_idx, f_idx, d1_idx, d2_idx], S.f32)
            local_sum = local_sum + val
            local_sum_sq = local_sum_sq + val * val

    shm_sum[tid] = local_sum
    shm_sum_sq[tid] = local_sum_sq
    S.syncthreads()

    if tid < 128:
        shm_sum[tid] = shm_sum[tid] + shm_sum[tid + 128]
        shm_sum_sq[tid] = shm_sum_sq[tid] + shm_sum_sq[tid + 128]
    S.syncthreads()

    if tid < 64:
        shm_sum[tid] = shm_sum[tid] + shm_sum[tid + 64]
        shm_sum_sq[tid] = shm_sum_sq[tid] + shm_sum_sq[tid + 64]
    S.syncthreads()

    if tid < 32:
        shm_sum[tid] = shm_sum[tid] + shm_sum[tid + 32]
        shm_sum_sq[tid] = shm_sum_sq[tid] + shm_sum_sq[tid + 32]
    S.syncthreads()

    if tid < 16:
        shm_sum[tid] = shm_sum[tid] + shm_sum[tid + 16]
        shm_sum_sq[tid] = shm_sum_sq[tid] + shm_sum_sq[tid + 16]
    S.syncthreads()

    if tid < 8:
        shm_sum[tid] = shm_sum[tid] + shm_sum[tid + 8]
        shm_sum_sq[tid] = shm_sum_sq[tid] + shm_sum_sq[tid + 8]
    S.syncthreads()

    if tid < 4:
        shm_sum[tid] = shm_sum[tid] + shm_sum[tid + 4]
        shm_sum_sq[tid] = shm_sum_sq[tid] + shm_sum_sq[tid + 4]
    S.syncthreads()

    if tid < 2:
        shm_sum[tid] = shm_sum[tid] + shm_sum[tid + 2]
        shm_sum_sq[tid] = shm_sum_sq[tid] + shm_sum_sq[tid + 2]
    S.syncthreads()

    if tid == 0:
        partial_layout = S.make_layout((total_blocks,), (1,))
        partial_sum = S.make_tensor(partial_sum_ptr, S.f32, partial_layout)
        partial_sum_sq = S.make_tensor(partial_sum_sq_ptr, S.f32, partial_layout)

        partial_sum[bid] = shm_sum[0] + shm_sum[1]
        partial_sum_sq[bid] = shm_sum_sq[0] + shm_sum_sq[1]


@substrate.jit
def layer_norm_apply_kernel(
    input_ptr: S.Pointer(S.bf16),
    output_ptr: S.Pointer(S.bf16),
    weight_ptr: S.Pointer(S.bf16),
    bias_ptr: S.Pointer(S.bf16),
    mean_ptr: S.Pointer(S.f32),
    inv_std_ptr: S.Pointer(S.f32),
    batch_size: S.i32,
    features: S.i32,
    dim1: S.i32,
    dim2: S.i32,
    range_bytes_x: S.u32,
    range_bytes_wb: S.u32,
):
    tid = S.thread_id(0)
    spatial = dim1 * dim2
    elements_per_batch = features * spatial
    vec_dim = elements_per_batch // VEC_SIZE
    idx = S.block_id(0) * BLOCK_SIZE + tid
    batch_idx = idx // vec_dim
    vec_idx = idx - batch_idx * vec_dim

    input_bf16 = S.make_tensor(
        input_ptr,
        S.bf16,
        S.make_layout((batch_size, vec_dim, VEC_SIZE), (elements_per_batch, VEC_SIZE, 1)),
    )
    output_bf16 = S.make_tensor(
        output_ptr,
        S.bf16,
        S.make_layout((batch_size, vec_dim, VEC_SIZE), (elements_per_batch, VEC_SIZE, 1)),
    )
    wb_bf16 = S.make_tensor(
        weight_ptr,
        S.bf16,
        S.make_layout((vec_dim, VEC_SIZE), (VEC_SIZE, 1)),
    )
    bias_bf16 = S.make_tensor(
        bias_ptr,
        S.bf16,
        S.make_layout((vec_dim, VEC_SIZE), (VEC_SIZE, 1)),
    )
    input_u32 = S.view(
        input_bf16,
        S.u32,
        S.make_layout((batch_size, vec_dim, U32_PER_VEC), (vec_dim * U32_PER_VEC, U32_PER_VEC, 1)),
    )
    output_u32 = S.view(
        output_bf16,
        S.u32,
        S.make_layout((batch_size, vec_dim, U32_PER_VEC), (vec_dim * U32_PER_VEC, U32_PER_VEC, 1)),
    )
    wb_u32 = S.view(
        wb_bf16,
        S.u32,
        S.make_layout((vec_dim, U32_PER_VEC), (U32_PER_VEC, 1)),
    )
    bias_u32 = S.view(
        bias_bf16,
        S.u32,
        S.make_layout((vec_dim, U32_PER_VEC), (U32_PER_VEC, 1)),
    )
    input_rsrc = S.amdgpu.make_rsrc(input_u32, range_bytes_x)
    output_rsrc = S.amdgpu.make_rsrc(output_u32, range_bytes_x)
    weight_rsrc = S.amdgpu.make_rsrc(wb_u32, range_bytes_wb)
    bias_rsrc = S.amdgpu.make_rsrc(bias_u32, range_bytes_wb)
    stats_layout = S.make_layout((batch_size,), (1,))
    mean_tensor = S.make_tensor(mean_ptr, S.f32, stats_layout)
    inv_std_tensor = S.make_tensor(inv_std_ptr, S.f32, stats_layout)
    mean_val = mean_tensor[batch_idx]
    inv_std_val = inv_std_tensor[batch_idx]
    zero_u32 = S.convert(0, S.u32)
    data_offset = S.convert((batch_idx * vec_dim + vec_idx) * VEC_SIZE * 2, S.u32)
    wb_offset = S.convert(vec_idx * VEC_SIZE * 2, S.u32)
    x_packed = S.amdgpu.raw_buffer_load_x4(input_rsrc, data_offset, zero_u32, 0)
    w_packed = S.amdgpu.raw_buffer_load_x4(weight_rsrc, wb_offset, zero_u32, 0)
    b_packed = S.amdgpu.raw_buffer_load_x4(bias_rsrc, wb_offset, zero_u32, 0)
    x_vals = S.view(x_packed, S.Tensor((VEC_SIZE,), S.bf16))
    w_vals = S.view(w_packed, S.Tensor((VEC_SIZE,), S.bf16))
    b_vals = S.view(b_packed, S.Tensor((VEC_SIZE,), S.bf16))
    out_vec = S.make_local((VEC_SIZE,), S.bf16)

    for j in S.range(VEC_SIZE):
        x_val = S.convert(x_vals[j], S.f32)
        w_val = S.convert(w_vals[j], S.f32)
        b_val = S.convert(b_vals[j], S.f32)
        out_val = (x_val - mean_val) * inv_std_val
        out_val = out_val * w_val + b_val
        out_vec[j] = S.convert(out_val, S.bf16)

    packed_out = S.view(out_vec, S.Tensor((U32_PER_VEC,), S.u32))
    S.amdgpu.raw_buffer_store_x4(packed_out, output_rsrc, data_offset, zero_u32, 0)


def substrate_layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Tensors must be on CUDA/HIP device."

    x_contiguous = x.contiguous().to(torch.bfloat16)
    weight_contiguous = weight.contiguous().to(torch.bfloat16)
    bias_contiguous = bias.contiguous().to(torch.bfloat16)

    batch_size = x_contiguous.shape[0]
    features = x_contiguous.shape[1]
    dim1 = x_contiguous.shape[2]
    dim2 = x_contiguous.shape[3]

    spatial = dim1 * dim2
    elements_per_batch = features * spatial
    block_chunk = BLOCK_SIZE * ELEMENTS_PER_THREAD
    blocks_per_batch = (elements_per_batch + block_chunk - 1) // block_chunk
    total_stats_blocks = batch_size * blocks_per_batch

    partial_sum = torch.empty(total_stats_blocks, dtype=torch.float32, device=x.device)
    partial_sum_sq = torch.empty(total_stats_blocks, dtype=torch.float32, device=x.device)

    layer_norm_stats_kernel[lambda: ((total_stats_blocks, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_contiguous,
        partial_sum,
        partial_sum_sq,
        batch_size,
        features,
        dim1,
        dim2,
        blocks_per_batch,
    )

    partial_sum_2d = partial_sum.view(batch_size, blocks_per_batch)
    partial_sum_sq_2d = partial_sum_sq.view(batch_size, blocks_per_batch)
    sum_per_batch = partial_sum_2d.sum(dim=1)
    sum_sq_per_batch = partial_sum_sq_2d.sum(dim=1)

    count = float(elements_per_batch)
    mean = sum_per_batch / count
    var = sum_sq_per_batch / count - mean * mean
    inv_std = torch.rsqrt(torch.clamp_min(var, 0.0) + EPS)

    output = torch.empty_like(x_contiguous)
    total_vectors = batch_size * (elements_per_batch // VEC_SIZE)
    num_blocks = total_vectors // BLOCK_SIZE

    layer_norm_apply_kernel[lambda: ((num_blocks, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_contiguous,
        output,
        weight_contiguous,
        bias_contiguous,
        mean.contiguous(),
        inv_std.contiguous(),
        batch_size,
        features,
        dim1,
        dim2,
        x_contiguous.numel() * 2,
        weight_contiguous.numel() * 2,
    )

    return output


class ModelNew(torch.nn.Module):
    """
    Optimized model that performs Layer Normalization using Substrate DSL.
    """

    def __init__(self, normalized_shape: tuple):
        super(ModelNew, self).__init__()
        self.normalized_shape = normalized_shape
        self.weight = torch.nn.Parameter(torch.ones(normalized_shape, dtype=torch.bfloat16))
        self.bias = torch.nn.Parameter(torch.zeros(normalized_shape, dtype=torch.bfloat16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_bf16 = x.to(dtype=torch.bfloat16)
        weight_bf16 = self.weight.to(dtype=torch.bfloat16)
        bias_bf16 = self.bias.to(dtype=torch.bfloat16)
        return substrate_layer_norm(x_bf16, weight_bf16, bias_bf16)
