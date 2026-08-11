import torch
import substrate
import substrate.language as S

BLOCK_SIZE: S.constexpr = 256
VEC_SIZE: S.constexpr = 8
ROW_SIZE: S.constexpr = 16384
VECS_PER_THREAD: S.constexpr = 8


@substrate.jit
def kl_div_row_sum_kernel(
    log_predictions_ptr: S.Pointer(S.bf16),
    targets_ptr: S.Pointer(S.bf16),
    target_log_targets_ptr: S.Pointer(S.bf16),
    row_sums_ptr: S.Pointer(S.f32),
    batch_size: S.i32,
    input_range_bytes: S.u32,
):
    tid = S.thread_id(0)
    row = S.block_id(0)

    log_predictions = S.make_tensor(
        log_predictions_ptr,
        S.bf16,
        S.make_layout((batch_size * ROW_SIZE,), (1,)),
    )
    targets = S.make_tensor(
        targets_ptr,
        S.bf16,
        S.make_layout((batch_size * ROW_SIZE,), (1,)),
    )
    target_log_targets = S.make_tensor(
        target_log_targets_ptr,
        S.bf16,
        S.make_layout((batch_size * ROW_SIZE,), (1,)),
    )
    row_sums = S.make_tensor(row_sums_ptr, S.f32, S.make_layout((batch_size,), (1,)))
    pred_rsrc = S.amdgpu.make_rsrc(log_predictions, input_range_bytes)
    targ_rsrc = S.amdgpu.make_rsrc(targets, input_range_bytes)
    tlt_rsrc = S.amdgpu.make_rsrc(target_log_targets, input_range_bytes)
    shm = S.make_shared((BLOCK_SIZE,), S.f32)

    zero_u32 = S.convert(0, S.u32)
    local_sum = S.convert(0.0, S.f32)
    base_elem = row * ROW_SIZE + tid * VEC_SIZE

    for i in S.range(VECS_PER_THREAD):
        elem_offset = base_elem + i * BLOCK_SIZE * VEC_SIZE
        byte_offset = S.convert(elem_offset * 2, S.u32)
        log_pred_vals = S.view(
            S.amdgpu.raw_buffer_load_x4(pred_rsrc, byte_offset, zero_u32, 0),
            S.Tensor((VEC_SIZE,), S.bf16),
        )
        target_vals = S.view(
            S.amdgpu.raw_buffer_load_x4(targ_rsrc, byte_offset, zero_u32, 0),
            S.Tensor((VEC_SIZE,), S.bf16),
        )
        target_log_target_vals = S.view(
            S.amdgpu.raw_buffer_load_x4(tlt_rsrc, byte_offset, zero_u32, 0),
            S.Tensor((VEC_SIZE,), S.bf16),
        )
        for j in S.range(VEC_SIZE):
            target = S.convert(target_vals[j], S.f32)
            log_pred = S.convert(log_pred_vals[j], S.f32)
            target_log_target = S.convert(target_log_target_vals[j], S.f32)
            local_sum = local_sum + target_log_target - target * log_pred

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
        row_sums[row] = shm[0]


@substrate.jit
def reduce_row_sums_kernel(
    row_sums_ptr: S.Pointer(S.f32),
    output_ptr: S.Pointer(S.f32),
    size: S.i32,
    num_iterations: S.i32,
):
    tid = S.thread_id(0)
    row_sums = S.make_tensor(row_sums_ptr, S.f32, S.make_layout((size,), (1,)))
    output = S.make_tensor(output_ptr, S.f32, S.make_layout((1,), (1,)))
    shm = S.make_shared((BLOCK_SIZE,), S.f32)
    local_sum = S.convert(0.0, S.f32)
    idx = tid

    for _ in S.range(num_iterations):
        if idx < size:
            local_sum = local_sum + row_sums[idx]
        idx = idx + BLOCK_SIZE

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
        output[0] = shm[0]


def substrate_kl_div(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA/HIP device."
    assert predictions.dtype == torch.bfloat16, "Input must be bfloat16"

    batch_size, dim = predictions.shape
    if dim != ROW_SIZE:
        return torch.nn.functional.kl_div(torch.log(predictions), targets, reduction="batchmean").to(torch.bfloat16)

    log_predictions = torch.log(predictions)
    log_preds_cont = log_predictions.contiguous()
    targets_cont = targets.contiguous()
    target_log_targets = (targets_cont * torch.log(targets_cont)).contiguous()
    row_sums = torch.empty(batch_size, dtype=torch.float32, device=predictions.device)
    total_sum = torch.empty(1, dtype=torch.float32, device=predictions.device)
    num_iterations = (batch_size + BLOCK_SIZE - 1) // BLOCK_SIZE

    kl_div_row_sum_kernel[lambda: ((batch_size, 1, 1), (BLOCK_SIZE, 1, 1))](
        log_preds_cont.view(-1),
        targets_cont.view(-1),
        target_log_targets.view(-1),
        row_sums,
        batch_size,
        log_preds_cont.numel() * log_preds_cont.element_size(),
    )
    reduce_row_sums_kernel[lambda: ((1, 1, 1), (BLOCK_SIZE, 1, 1))](
        row_sums,
        total_sum,
        batch_size,
        num_iterations,
    )

    return (total_sum[0] / batch_size).to(torch.bfloat16)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return substrate_kl_div(predictions, targets)
