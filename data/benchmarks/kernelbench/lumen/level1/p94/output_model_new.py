import torch
import substrate
import substrate.language as S

BLOCK_SIZE: S.constexpr = 256
VEC_SIZE: S.constexpr = 8
U32_PER_VEC: S.constexpr = 4
ROW_SIZE: S.constexpr = 32768
VECS_PER_THREAD: S.constexpr = 16
ROWS_PER_CHUNK = 16384


@substrate.jit
def mse_row_sum_kernel(
    predictions_ptr: S.Pointer(S.bf16),
    targets_ptr: S.Pointer(S.bf16),
    row_sums_ptr: S.Pointer(S.f32),
    rows_in_chunk: S.i32,
    input_range_bytes: S.u32,
):
    tid = S.thread_id(0)
    row = S.block_id(0)

    predictions = S.make_tensor(
        predictions_ptr,
        S.bf16,
        S.make_layout((rows_in_chunk * ROW_SIZE,), (1,)),
    )
    targets = S.make_tensor(
        targets_ptr,
        S.bf16,
        S.make_layout((rows_in_chunk * ROW_SIZE,), (1,)),
    )
    row_sums = S.make_tensor(row_sums_ptr, S.f32, S.make_layout((rows_in_chunk,), (1,)))
    pred_rsrc = S.amdgpu.make_rsrc(predictions, input_range_bytes)
    targ_rsrc = S.amdgpu.make_rsrc(targets, input_range_bytes)
    shm = S.make_shared((BLOCK_SIZE,), S.f32)

    zero_u32 = S.convert(0, S.u32)
    local_sum = S.convert(0.0, S.f32)
    base_elem = row * ROW_SIZE + tid * VEC_SIZE

    for i in S.range(VECS_PER_THREAD):
        elem_offset = base_elem + i * BLOCK_SIZE * VEC_SIZE
        byte_offset = S.convert(elem_offset * 2, S.u32)
        pred_vals = S.view(
            S.amdgpu.raw_buffer_load_x4(pred_rsrc, byte_offset, zero_u32, 0),
            S.Tensor((VEC_SIZE,), S.bf16),
        )
        targ_vals = S.view(
            S.amdgpu.raw_buffer_load_x4(targ_rsrc, byte_offset, zero_u32, 0),
            S.Tensor((VEC_SIZE,), S.bf16),
        )
        for j in S.range(VEC_SIZE):
            diff = S.convert(pred_vals[j], S.f32) - S.convert(targ_vals[j], S.f32)
            local_sum = local_sum + diff * diff

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


def substrate_mse_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA/HIP device."
    assert predictions.shape == targets.shape, "Input shapes must match."

    orig_dtype = predictions.dtype
    pred_bf16 = predictions.to(dtype=torch.bfloat16, device=predictions.device).contiguous()
    targ_bf16 = targets.to(dtype=torch.bfloat16, device=targets.device).contiguous()
    batch_size, dim = pred_bf16.shape

    if dim != ROW_SIZE:
        return torch.mean((predictions - targets) ** 2)

    total_sum = torch.zeros(1, dtype=torch.float32, device=predictions.device)
    total_elements = pred_bf16.numel()

    for row_start in range(0, batch_size, ROWS_PER_CHUNK):
        rows_in_chunk = min(ROWS_PER_CHUNK, batch_size - row_start)
        pred_chunk = pred_bf16.narrow(0, row_start, rows_in_chunk)
        targ_chunk = targ_bf16.narrow(0, row_start, rows_in_chunk)
        row_sums = torch.empty(rows_in_chunk, dtype=torch.float32, device=predictions.device)
        chunk_sum = torch.empty(1, dtype=torch.float32, device=predictions.device)
        num_iterations = (rows_in_chunk + BLOCK_SIZE - 1) // BLOCK_SIZE

        mse_row_sum_kernel[lambda: ((rows_in_chunk, 1, 1), (BLOCK_SIZE, 1, 1))](
            pred_chunk.view(-1),
            targ_chunk.view(-1),
            row_sums,
            rows_in_chunk,
            pred_chunk.numel() * pred_chunk.element_size(),
        )
        reduce_row_sums_kernel[lambda: ((1, 1, 1), (BLOCK_SIZE, 1, 1))](
            row_sums,
            chunk_sum,
            rows_in_chunk,
            num_iterations,
        )
        total_sum += chunk_sum

    return (total_sum[0] / total_elements).to(orig_dtype)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return substrate_mse_loss(predictions, targets)
