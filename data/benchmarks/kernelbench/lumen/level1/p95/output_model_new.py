import math

import torch
import substrate
import substrate.language as S

BLOCK_SIZE: S.constexpr = 256
VEC_SIZE: S.constexpr = 8
VECTORS_PER_THREAD: S.constexpr = 2
NUM_CLASSES: S.constexpr = 4096
VECTORS_PER_ROW: S.constexpr = NUM_CLASSES // VEC_SIZE
WAVE_SIZE: S.constexpr = 64
WAVE_COUNT: S.constexpr = BLOCK_SIZE // WAVE_SIZE
REDUCE_BLOCK: S.constexpr = 256
REDUCE_ELEMS_PER_THREAD: S.constexpr = 128
EXP2_SCALE = math.log2(math.e)


@substrate.jit
def cross_entropy_row_kernel(
    predictions_ptr: S.Pointer(S.bf16),
    targets_ptr: S.Pointer(S.i64),
    row_losses_ptr: S.Pointer(S.f32),
    batch_size: S.i32,
    input_range_bytes: S.u32,
):
    tid = S.thread_id(0)
    row_idx = S.block_id(0)
    lane = tid % WAVE_SIZE
    wave = tid // WAVE_SIZE
    exp2_scale = S.convert(EXP2_SCALE, S.f32)

    predictions = S.make_tensor(
        predictions_ptr,
        S.bf16,
        S.make_layout((batch_size * NUM_CLASSES,), (1,)),
    )
    targets = S.make_tensor(targets_ptr, S.i64, S.make_layout((batch_size,), (1,)))
    row_losses = S.make_tensor(row_losses_ptr, S.f32, S.make_layout((batch_size,), (1,)))
    pred_rsrc = S.amdgpu.make_rsrc(predictions, input_range_bytes)

    partial_max = S.make_shared((WAVE_COUNT,), S.f32)
    partial_sum = S.make_shared((WAVE_COUNT,), S.f32)
    partial_target = S.make_shared((WAVE_COUNT,), S.f32)

    zero_u32 = S.convert(0, S.u32)
    neg_inf = S.convert(-1.0e30, S.f32)
    zero = S.convert(0.0, S.f32)

    target_i32 = S.convert(targets[row_idx], S.i32)
    target_vec = target_i32 // VEC_SIZE
    target_lane = target_i32 - target_vec * VEC_SIZE

    local_max = neg_inf
    local_sum = zero
    local_target = zero

    for i in S.range(VECTORS_PER_THREAD):
        vec_idx = tid + i * BLOCK_SIZE
        vals = S.view(
            S.amdgpu.raw_buffer_load_x4(
                pred_rsrc,
                S.convert((row_idx * NUM_CLASSES + vec_idx * VEC_SIZE) * 2, S.u32),
                zero_u32,
                0,
            ),
            S.Tensor((VEC_SIZE,), S.bf16),
        )

        if target_vec == vec_idx:
            local_target = S.convert(vals[target_lane], S.f32)

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
    local_target = local_target + S.shuffle_xor(local_target, 32, WAVE_SIZE)
    other_max = S.shuffle_xor(local_max, 16, WAVE_SIZE)
    other_sum = S.shuffle_xor(local_sum, 16, WAVE_SIZE)
    next_max = other_max if other_max > local_max else local_max
    local_sum = local_sum * S.exp2((local_max - next_max) * exp2_scale) + other_sum * S.exp2((other_max - next_max) * exp2_scale)
    local_max = next_max
    local_target = local_target + S.shuffle_xor(local_target, 16, WAVE_SIZE)
    other_max = S.shuffle_xor(local_max, 8, WAVE_SIZE)
    other_sum = S.shuffle_xor(local_sum, 8, WAVE_SIZE)
    next_max = other_max if other_max > local_max else local_max
    local_sum = local_sum * S.exp2((local_max - next_max) * exp2_scale) + other_sum * S.exp2((other_max - next_max) * exp2_scale)
    local_max = next_max
    local_target = local_target + S.shuffle_xor(local_target, 8, WAVE_SIZE)
    other_max = S.shuffle_xor(local_max, 4, WAVE_SIZE)
    other_sum = S.shuffle_xor(local_sum, 4, WAVE_SIZE)
    next_max = other_max if other_max > local_max else local_max
    local_sum = local_sum * S.exp2((local_max - next_max) * exp2_scale) + other_sum * S.exp2((other_max - next_max) * exp2_scale)
    local_max = next_max
    local_target = local_target + S.shuffle_xor(local_target, 4, WAVE_SIZE)
    other_max = S.shuffle_xor(local_max, 2, WAVE_SIZE)
    other_sum = S.shuffle_xor(local_sum, 2, WAVE_SIZE)
    next_max = other_max if other_max > local_max else local_max
    local_sum = local_sum * S.exp2((local_max - next_max) * exp2_scale) + other_sum * S.exp2((other_max - next_max) * exp2_scale)
    local_max = next_max
    local_target = local_target + S.shuffle_xor(local_target, 2, WAVE_SIZE)
    other_max = S.shuffle_xor(local_max, 1, WAVE_SIZE)
    other_sum = S.shuffle_xor(local_sum, 1, WAVE_SIZE)
    next_max = other_max if other_max > local_max else local_max
    local_sum = local_sum * S.exp2((local_max - next_max) * exp2_scale) + other_sum * S.exp2((other_max - next_max) * exp2_scale)
    local_max = next_max
    local_target = local_target + S.shuffle_xor(local_target, 1, WAVE_SIZE)

    if lane == 0:
        partial_max[wave] = local_max
        partial_sum[wave] = local_sum
        partial_target[wave] = local_target
    S.syncthreads()

    if wave == 0:
        block_max = partial_max[lane] if lane < WAVE_COUNT else neg_inf
        block_sum = partial_sum[lane] if lane < WAVE_COUNT else zero
        block_target = partial_target[lane] if lane < WAVE_COUNT else zero

        other_max = S.shuffle_xor(block_max, 32, WAVE_SIZE)
        other_sum = S.shuffle_xor(block_sum, 32, WAVE_SIZE)
        next_max = other_max if other_max > block_max else block_max
        block_sum = block_sum * S.exp2((block_max - next_max) * exp2_scale) + other_sum * S.exp2((other_max - next_max) * exp2_scale)
        block_max = next_max
        block_target = block_target + S.shuffle_xor(block_target, 32, WAVE_SIZE)
        other_max = S.shuffle_xor(block_max, 16, WAVE_SIZE)
        other_sum = S.shuffle_xor(block_sum, 16, WAVE_SIZE)
        next_max = other_max if other_max > block_max else block_max
        block_sum = block_sum * S.exp2((block_max - next_max) * exp2_scale) + other_sum * S.exp2((other_max - next_max) * exp2_scale)
        block_max = next_max
        block_target = block_target + S.shuffle_xor(block_target, 16, WAVE_SIZE)
        other_max = S.shuffle_xor(block_max, 8, WAVE_SIZE)
        other_sum = S.shuffle_xor(block_sum, 8, WAVE_SIZE)
        next_max = other_max if other_max > block_max else block_max
        block_sum = block_sum * S.exp2((block_max - next_max) * exp2_scale) + other_sum * S.exp2((other_max - next_max) * exp2_scale)
        block_max = next_max
        block_target = block_target + S.shuffle_xor(block_target, 8, WAVE_SIZE)
        other_max = S.shuffle_xor(block_max, 4, WAVE_SIZE)
        other_sum = S.shuffle_xor(block_sum, 4, WAVE_SIZE)
        next_max = other_max if other_max > block_max else block_max
        block_sum = block_sum * S.exp2((block_max - next_max) * exp2_scale) + other_sum * S.exp2((other_max - next_max) * exp2_scale)
        block_max = next_max
        block_target = block_target + S.shuffle_xor(block_target, 4, WAVE_SIZE)
        other_max = S.shuffle_xor(block_max, 2, WAVE_SIZE)
        other_sum = S.shuffle_xor(block_sum, 2, WAVE_SIZE)
        next_max = other_max if other_max > block_max else block_max
        block_sum = block_sum * S.exp2((block_max - next_max) * exp2_scale) + other_sum * S.exp2((other_max - next_max) * exp2_scale)
        block_max = next_max
        block_target = block_target + S.shuffle_xor(block_target, 2, WAVE_SIZE)
        other_max = S.shuffle_xor(block_max, 1, WAVE_SIZE)
        other_sum = S.shuffle_xor(block_sum, 1, WAVE_SIZE)
        next_max = other_max if other_max > block_max else block_max
        block_sum = block_sum * S.exp2((block_max - next_max) * exp2_scale) + other_sum * S.exp2((other_max - next_max) * exp2_scale)
        block_max = next_max
        block_target = block_target + S.shuffle_xor(block_target, 1, WAVE_SIZE)

        if lane == 0:
            row_losses[row_idx] = block_max + S.log(block_sum) - block_target


@substrate.jit
def sum_reduce_kernel(
    input_ptr: S.Pointer(S.f32),
    output_ptr: S.Pointer(S.f32),
    size: S.i32,
):
    tid = S.thread_id(0)

    input_tensor = S.make_tensor(input_ptr, S.f32, S.make_layout((size,), (1,)))
    output_tensor = S.make_tensor(output_ptr, S.f32, S.make_layout((1,), (1,)))
    shm = S.make_shared((REDUCE_BLOCK,), S.f32)

    local_sum = S.convert(0.0, S.f32)
    for i in S.range(REDUCE_ELEMS_PER_THREAD):
        idx = tid * REDUCE_ELEMS_PER_THREAD + i
        if idx < size:
            local_sum = local_sum + input_tensor[idx]

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
    if tid < 1:
        shm[tid] = shm[tid] + shm[tid + 1]

    if tid == 0:
        output_tensor[0] = shm[0]


def substrate_cross_entropy(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA/HIP device."

    orig_dtype = predictions.dtype
    pred_bf16 = predictions.to(dtype=torch.bfloat16, device=predictions.device).contiguous()
    targets_cont = targets.to(dtype=torch.long, device=targets.device).contiguous()
    batch_size, num_classes = pred_bf16.shape

    if num_classes != NUM_CLASSES:
        return torch.nn.functional.cross_entropy(predictions, targets).to(dtype=orig_dtype)

    row_losses = torch.empty((batch_size,), dtype=torch.float32, device=predictions.device)
    cross_entropy_row_kernel[lambda: ((batch_size, 1, 1), (BLOCK_SIZE, 1, 1))](
        pred_bf16.view(-1),
        targets_cont,
        row_losses,
        batch_size,
        pred_bf16.numel() * pred_bf16.element_size(),
        num_warps=4,
    )

    total_loss = torch.empty((1,), dtype=torch.float32, device=predictions.device)
    sum_reduce_kernel[lambda: ((1, 1, 1), (REDUCE_BLOCK, 1, 1))](
        row_losses,
        total_loss,
        batch_size,
        num_warps=4,
    )

    mean_loss = total_loss[0] / batch_size
    return mean_loss.to(dtype=orig_dtype)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return substrate_cross_entropy(predictions, targets)
