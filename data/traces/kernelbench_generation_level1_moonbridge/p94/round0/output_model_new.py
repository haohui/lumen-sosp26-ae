import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE: al.constexpr = 256
TILE_SIZE: al.constexpr = 131072


@avelang.jit
def mse_reduce_kernel(
    pred_ptr: al.Pointer(al.bf16),
    target_ptr: al.Pointer(al.bf16),
    partial_out_ptr: al.Pointer(al.f32),
    N: al.i32,
    num_tiles: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    if bid < num_tiles:
        smem = al.make_shared((BLOCK_SIZE,), al.f32)

        layout = al.make_layout((N,), (1,))
        pred = al.make_tensor(pred_ptr, al.bf16, layout)
        target = al.make_tensor(target_ptr, al.bf16, layout)

        tile_start = bid * TILE_SIZE
        tile_end = tile_start + TILE_SIZE
        if tile_end > N:
            tile_end = N

        local_sum = al.convert(0.0, al.f32)
        for i in al.range(tile_start + tid, tile_end, BLOCK_SIZE):
            diff = al.convert(pred[i], al.f32) - al.convert(target[i], al.f32)
            local_sum = local_sum + diff * diff

        smem[tid] = local_sum
        al.syncthreads()

        if tid < 128:
            smem[tid] = smem[tid] + smem[tid + 128]
        al.syncthreads()
        if tid < 64:
            smem[tid] = smem[tid] + smem[tid + 64]
        al.syncthreads()
        if tid < 32:
            smem[tid] = smem[tid] + smem[tid + 32]
        al.syncthreads()
        if tid < 16:
            smem[tid] = smem[tid] + smem[tid + 16]
        al.syncthreads()
        if tid < 8:
            smem[tid] = smem[tid] + smem[tid + 8]
        al.syncthreads()
        if tid < 4:
            smem[tid] = smem[tid] + smem[tid + 4]
        al.syncthreads()
        if tid < 2:
            smem[tid] = smem[tid] + smem[tid + 2]
        al.syncthreads()
        if tid < 1:
            smem[tid] = smem[tid] + smem[tid + 1]

        if tid == 0:
            partial_layout = al.make_layout((num_tiles,), (1,))
            partials = al.make_tensor(partial_out_ptr, al.f32, partial_layout)
            partials[bid] = smem[0]


@avelang.jit
def mse_aggregate_kernel(
    partial_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.bf16),
    num_tiles: al.i32,
    total_N: al.i32,
):
    tid = al.thread_id(0)
    smem = al.make_shared((BLOCK_SIZE,), al.f32)

    layout = al.make_layout((num_tiles,), (1,))
    partials = al.make_tensor(partial_ptr, al.f32, layout)

    local_sum = al.convert(0.0, al.f32)
    for i in al.range(tid, num_tiles, BLOCK_SIZE):
        local_sum = local_sum + partials[i]

    smem[tid] = local_sum
    al.syncthreads()

    if tid < 128:
        smem[tid] = smem[tid] + smem[tid + 128]
    al.syncthreads()
    if tid < 64:
        smem[tid] = smem[tid] + smem[tid + 64]
    al.syncthreads()
    if tid < 32:
        smem[tid] = smem[tid] + smem[tid + 32]
    al.syncthreads()
    if tid < 16:
        smem[tid] = smem[tid] + smem[tid + 16]
    al.syncthreads()
    if tid < 8:
        smem[tid] = smem[tid] + smem[tid + 8]
    al.syncthreads()
    if tid < 4:
        smem[tid] = smem[tid] + smem[tid + 4]
    al.syncthreads()
    if tid < 2:
        smem[tid] = smem[tid] + smem[tid + 2]
    al.syncthreads()
    if tid < 1:
        smem[tid] = smem[tid] + smem[tid + 1]

    if tid == 0:
        N_f32 = al.convert(total_N, al.f32)
        mean = smem[0] / N_f32
        out_layout = al.make_layout((1,), (1,))
        out = al.make_tensor(out_ptr, al.bf16, out_layout)
        out[0] = al.convert(mean, al.bf16)


def _to_bf16_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(torch.bfloat16)
    return t.contiguous().cuda().to(torch.bfloat16)


def avelang_mse(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")
    assert predictions.shape == targets.shape, "Predictions and targets must have the same shape."

    pred_bf16 = _to_bf16_contiguous(predictions)
    target_bf16 = _to_bf16_contiguous(targets)

    N = pred_bf16.numel()
    num_tiles = (N + TILE_SIZE - 1) // TILE_SIZE

    partial_sums = torch.empty((num_tiles,), dtype=torch.float32, device=pred_bf16.device)

    mse_reduce_kernel[lambda: ((num_tiles, 1, 1), (BLOCK_SIZE, 1, 1))](
        pred_bf16, target_bf16, partial_sums, N, num_tiles
    )

    out = torch.empty((1,), dtype=torch.bfloat16, device=pred_bf16.device)
    mse_aggregate_kernel[lambda: ((1, 1, 1), (BLOCK_SIZE, 1, 1))](
        partial_sums, out, num_tiles, N
    )

    return out.reshape(())


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return avelang_mse(predictions, targets)
