import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE = 256
NUM_BLOCKS = 8192


@avelang.jit
def hinge_loss_partial_kernel(
    pred_ptr: al.Pointer(al.bf16),
    target_ptr: al.Pointer(al.bf16),
    partial_sum_ptr: al.Pointer(al.f32),
    M: al.i32,
    N: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    pred_layout = al.make_layout((M, N), (N, 1))
    pred = al.make_tensor(pred_ptr, al.bf16, pred_layout)
    target_layout = al.make_layout((M,), (1,))
    target = al.make_tensor(target_ptr, al.bf16, target_layout)

    thread_sum = al.convert(0, al.f32)
    total_elems = M * N
    start_idx = bid * 256 + tid
    stride = 256 * 8192

    for idx in al.range(start_idx, total_elems, stride):
        row = idx // N
        col = idx - row * N
        p = al.convert(pred[row, col], al.f32)
        t = al.convert(target[row], al.f32)
        tmp = al.convert(1, al.f32) - p * t
        if tmp > al.convert(0, al.f32):
            thread_sum = thread_sum + tmp

    shared = al.make_shared((256,), al.f32)
    shared[tid] = thread_sum
    al.syncthreads()

    if tid < 128:
        shared[tid] = shared[tid] + shared[tid + 128]
    al.syncthreads()
    if tid < 64:
        shared[tid] = shared[tid] + shared[tid + 64]
    al.syncthreads()
    if tid < 32:
        shared[tid] = shared[tid] + shared[tid + 32]
    al.syncthreads()
    if tid < 16:
        shared[tid] = shared[tid] + shared[tid + 16]
    al.syncthreads()
    if tid < 8:
        shared[tid] = shared[tid] + shared[tid + 8]
    al.syncthreads()
    if tid < 4:
        shared[tid] = shared[tid] + shared[tid + 4]
    al.syncthreads()
    if tid < 2:
        shared[tid] = shared[tid] + shared[tid + 2]
    al.syncthreads()
    if tid == 0:
        shared[0] = shared[0] + shared[1]

    if tid == 0:
        ps_layout = al.make_layout((8192,), (1,))
        ps = al.make_tensor(partial_sum_ptr, al.f32, ps_layout)
        ps[bid] = shared[0]


def avelang_hinge_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    assert predictions.is_cuda and targets.is_cuda
    M, N_val = predictions.shape

    pred_bf16 = predictions.to(torch.bfloat16).contiguous()
    target_bf16 = targets.to(torch.bfloat16).contiguous()

    partial_sums = torch.zeros(NUM_BLOCKS, dtype=torch.float32, device=predictions.device)

    hinge_loss_partial_kernel[lambda: ((NUM_BLOCKS, 1, 1), (BLOCK_SIZE, 1, 1))](
        pred_bf16, target_bf16, partial_sums, M, N_val
    )

    total = partial_sums.sum()
    total_elements = M * N_val
    mean_val = total / total_elements
    return mean_val.to(torch.bfloat16).reshape(())


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return avelang_hinge_loss(predictions, targets)
