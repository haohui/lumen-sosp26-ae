import math

import torch
import substrate
import substrate.language as S

BLOCK_SIZE: S.constexpr = 256
VEC_SIZE: S.constexpr = 8
U32_PER_VEC: S.constexpr = 4
WAVE_SIZE: S.constexpr = 64
WAVE_COUNT: S.constexpr = BLOCK_SIZE // WAVE_SIZE
EXP2_SCALE = math.log2(math.e)


@substrate.jit
def softmax_kernel(
    input_ptr: S.Pointer(S.bf16),
    output_ptr: S.Pointer(S.bf16),
    batch_size: S.i32,
    dim: S.i32,
    range_bytes: S.u32,
):
    tid = S.thread_id(0)
    row_idx = S.block_id(0)

    if row_idx >= batch_size:
        return

    lane = tid % WAVE_SIZE
    wave = tid // WAVE_SIZE
    vec_dim = dim // VEC_SIZE
    vecs_per_thread = vec_dim // BLOCK_SIZE
    bf16_layout = S.make_layout((batch_size, vec_dim, VEC_SIZE), (dim, VEC_SIZE, 1))
    u32_layout = S.make_layout((batch_size, vec_dim, U32_PER_VEC), (vec_dim * U32_PER_VEC, U32_PER_VEC, 1))
    input_tensor = S.make_tensor(input_ptr, S.bf16, bf16_layout)
    output_tensor = S.make_tensor(output_ptr, S.bf16, bf16_layout)
    input_u32 = S.view(input_tensor, S.u32, u32_layout)
    output_u32 = S.view(output_tensor, S.u32, u32_layout)
    input_rsrc = S.amdgpu.make_rsrc(input_u32, range_bytes)
    output_rsrc = S.amdgpu.make_rsrc(output_u32, range_bytes)
    partial_max = S.make_shared((WAVE_COUNT,), S.f32)
    partial_sum = S.make_shared((WAVE_COUNT,), S.f32)
    exp2_scale = S.convert(EXP2_SCALE, S.f32)
    neg_inf = S.convert(-1.0e30, S.f32)
    zero_u32 = S.convert(0, S.u32)

    local_max = neg_inf
    local_sum = S.convert(0.0, S.f32)
    for i in S.range(vecs_per_thread):
        vec_idx = i * BLOCK_SIZE + tid
        byte_offset = S.convert((row_idx * vec_dim + vec_idx) * VEC_SIZE * 2, S.u32)
        packed = S.amdgpu.raw_buffer_load_x4(input_rsrc, byte_offset, zero_u32, 0)
        vals = S.view(packed, S.Tensor((VEC_SIZE,), S.bf16))
        for j in S.range(VEC_SIZE):
            val_f32 = S.convert(vals[j], S.f32)
            next_max = val_f32 if val_f32 > local_max else local_max
            local_sum = (
                local_sum * S.exp2((local_max - next_max) * exp2_scale)
                + S.exp2((val_f32 - next_max) * exp2_scale)
            )
            local_max = next_max

    other_max = S.shuffle_xor(local_max, 32, WAVE_SIZE)
    other_sum = S.shuffle_xor(local_sum, 32, WAVE_SIZE)
    next_max = other_max if other_max > local_max else local_max
    local_sum = local_sum * S.exp2((local_max - next_max) * exp2_scale) + other_sum * S.exp2((other_max - next_max) * exp2_scale)
    local_max = next_max
    other_max = S.shuffle_xor(local_max, 16, WAVE_SIZE)
    other_sum = S.shuffle_xor(local_sum, 16, WAVE_SIZE)
    next_max = other_max if other_max > local_max else local_max
    local_sum = local_sum * S.exp2((local_max - next_max) * exp2_scale) + other_sum * S.exp2((other_max - next_max) * exp2_scale)
    local_max = next_max
    other_max = S.shuffle_xor(local_max, 8, WAVE_SIZE)
    other_sum = S.shuffle_xor(local_sum, 8, WAVE_SIZE)
    next_max = other_max if other_max > local_max else local_max
    local_sum = local_sum * S.exp2((local_max - next_max) * exp2_scale) + other_sum * S.exp2((other_max - next_max) * exp2_scale)
    local_max = next_max
    other_max = S.shuffle_xor(local_max, 4, WAVE_SIZE)
    other_sum = S.shuffle_xor(local_sum, 4, WAVE_SIZE)
    next_max = other_max if other_max > local_max else local_max
    local_sum = local_sum * S.exp2((local_max - next_max) * exp2_scale) + other_sum * S.exp2((other_max - next_max) * exp2_scale)
    local_max = next_max
    other_max = S.shuffle_xor(local_max, 2, WAVE_SIZE)
    other_sum = S.shuffle_xor(local_sum, 2, WAVE_SIZE)
    next_max = other_max if other_max > local_max else local_max
    local_sum = local_sum * S.exp2((local_max - next_max) * exp2_scale) + other_sum * S.exp2((other_max - next_max) * exp2_scale)
    local_max = next_max
    other_max = S.shuffle_xor(local_max, 1, WAVE_SIZE)
    other_sum = S.shuffle_xor(local_sum, 1, WAVE_SIZE)
    next_max = other_max if other_max > local_max else local_max
    local_sum = local_sum * S.exp2((local_max - next_max) * exp2_scale) + other_sum * S.exp2((other_max - next_max) * exp2_scale)
    local_max = next_max

    if lane == 0:
        partial_max[wave] = local_max
        partial_sum[wave] = local_sum
    S.syncthreads()

    if wave == 0:
        block_max = partial_max[lane] if lane < WAVE_COUNT else neg_inf
        block_sum = partial_sum[lane] if lane < WAVE_COUNT else S.convert(0.0, S.f32)
        other_max = S.shuffle_xor(block_max, 32, WAVE_SIZE)
        other_sum = S.shuffle_xor(block_sum, 32, WAVE_SIZE)
        next_max = other_max if other_max > block_max else block_max
        block_sum = block_sum * S.exp2((block_max - next_max) * exp2_scale) + other_sum * S.exp2((other_max - next_max) * exp2_scale)
        block_max = next_max
        other_max = S.shuffle_xor(block_max, 16, WAVE_SIZE)
        other_sum = S.shuffle_xor(block_sum, 16, WAVE_SIZE)
        next_max = other_max if other_max > block_max else block_max
        block_sum = block_sum * S.exp2((block_max - next_max) * exp2_scale) + other_sum * S.exp2((other_max - next_max) * exp2_scale)
        block_max = next_max
        other_max = S.shuffle_xor(block_max, 8, WAVE_SIZE)
        other_sum = S.shuffle_xor(block_sum, 8, WAVE_SIZE)
        next_max = other_max if other_max > block_max else block_max
        block_sum = block_sum * S.exp2((block_max - next_max) * exp2_scale) + other_sum * S.exp2((other_max - next_max) * exp2_scale)
        block_max = next_max
        other_max = S.shuffle_xor(block_max, 4, WAVE_SIZE)
        other_sum = S.shuffle_xor(block_sum, 4, WAVE_SIZE)
        next_max = other_max if other_max > block_max else block_max
        block_sum = block_sum * S.exp2((block_max - next_max) * exp2_scale) + other_sum * S.exp2((other_max - next_max) * exp2_scale)
        block_max = next_max
        other_max = S.shuffle_xor(block_max, 2, WAVE_SIZE)
        other_sum = S.shuffle_xor(block_sum, 2, WAVE_SIZE)
        next_max = other_max if other_max > block_max else block_max
        block_sum = block_sum * S.exp2((block_max - next_max) * exp2_scale) + other_sum * S.exp2((other_max - next_max) * exp2_scale)
        block_max = next_max
        other_max = S.shuffle_xor(block_max, 1, WAVE_SIZE)
        other_sum = S.shuffle_xor(block_sum, 1, WAVE_SIZE)
        next_max = other_max if other_max > block_max else block_max
        block_sum = block_sum * S.exp2((block_max - next_max) * exp2_scale) + other_sum * S.exp2((other_max - next_max) * exp2_scale)
        block_max = next_max
        if lane == 0:
            partial_max[0] = block_max
            partial_sum[0] = block_sum
    S.syncthreads()
    row_max = partial_max[0]
    inv_row_sum = S.amdgpu.rcp(partial_sum[0])
    out_vec = S.make_local((VEC_SIZE,), S.bf16)
    for i in S.range(vecs_per_thread):
        vec_idx = i * BLOCK_SIZE + tid
        byte_offset = S.convert((row_idx * vec_dim + vec_idx) * VEC_SIZE * 2, S.u32)
        packed = S.amdgpu.raw_buffer_load_x4(input_rsrc, byte_offset, zero_u32, 0)
        vals = S.view(packed, S.Tensor((VEC_SIZE,), S.bf16))
        for j in S.range(VEC_SIZE):
            exp_val = S.exp2((S.convert(vals[j], S.f32) - row_max) * exp2_scale)
            out_vec[j] = S.convert(exp_val * inv_row_sum, S.bf16)
        packed_out = S.view(out_vec, S.Tensor((U32_PER_VEC,), S.u32))
        S.amdgpu.raw_buffer_store_x4(packed_out, output_rsrc, byte_offset, zero_u32, 0)


def substrate_softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Tensors must be on CUDA/HIP device."
    assert x.dim() == 2, "Input must be 2D tensor."

    original_dtype = x.dtype
    x_contiguous = x.contiguous().to(torch.bfloat16)
    batch_size = x_contiguous.shape[0]
    dim = x_contiguous.shape[1]
    assert dim % VEC_SIZE == 0, "This kernel expects dim to be divisible by 8."
    assert (dim // VEC_SIZE) % BLOCK_SIZE == 0, "This kernel expects vec_dim to be divisible by block size."

    output = torch.empty_like(x_contiguous)
    softmax_kernel[lambda: ((batch_size, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_contiguous, output, batch_size, dim, x_contiguous.numel() * 2, num_warps=4
    )
    return output.to(dtype=original_dtype)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return substrate_softmax(x)
