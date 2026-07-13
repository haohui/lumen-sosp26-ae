import torch
import torch.nn as nn
import avelang
import avelang.language as al

batch_size = 8192 * 2
input_shape = (8192 * 2,)
dim = 1


def get_inputs():
    scale = torch.rand(())
    return [(torch.rand(batch_size, *input_shape)*scale).softmax(dim=-1), torch.rand(batch_size, *input_shape).softmax(dim=-1)]

def get_init_inputs():
    return []


@avelang.jit
def kl_div_row_kernel(
    pred_ptr: al.Pointer(al.bf16),
    tgt_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.f32),
    num_cols: al.i32,
    batch_size: al.i32,
    BLOCK_SIZE: al.constexpr,
):
    row_id = al.block_id(0)
    tid = al.thread_id(0)

    if row_id < batch_size:
        layout_2d = al.make_layout((batch_size, num_cols), (num_cols, 1))
        pred = al.make_tensor(pred_ptr, al.bf16, layout_2d)
        tgt = al.make_tensor(tgt_ptr, al.bf16, layout_2d)

        shared = al.make_shared((BLOCK_SIZE,), al.f32)

        acc = al.convert(0.0, al.f32)
        for col in al.range(tid, num_cols, BLOCK_SIZE):
            tgt_bf16 = tgt[row_id, col]
            tgt_val = al.convert(tgt_bf16, al.f32)
            if tgt_val > al.convert(0.0, al.f32):
                pred_bf16 = pred[row_id, col]
                pred_val = al.convert(pred_bf16, al.f32)
                log_tgt = al.log(tgt_val)
                log_pred = al.log(pred_val)
                term = tgt_val * (log_tgt - log_pred)
                acc = acc + term

        shared[tid] = acc
        al.syncthreads()

        if tid < 64:
            partial = shared[tid]
            for i in al.range(1, 4):
                partial = partial + shared[tid + i * 64]
            shared[tid] = partial
        al.syncthreads()

        if tid == 0:
            final_sum = shared[0]
            for i in al.range(1, 64):
                final_sum = final_sum + shared[i]

            out_layout = al.make_layout((batch_size,), (1,))
            out = al.make_tensor(out_ptr, al.f32, out_layout)
            out[row_id] = final_sum


@avelang.jit
def final_reduce_kernel(
    in_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.f32),
    batch_size: al.i32,
    BLOCK_SIZE: al.constexpr,
):
    tid = al.thread_id(0)

    in_layout = al.make_layout((batch_size,), (1,))
    in_tensor = al.make_tensor(in_ptr, al.f32, in_layout)

    shared = al.make_shared((BLOCK_SIZE,), al.f32)

    acc = al.convert(0.0, al.f32)
    for i in al.range(tid, batch_size, BLOCK_SIZE):
        acc = acc + in_tensor[i]

    shared[tid] = acc
    al.syncthreads()

    if tid < 64:
        partial = shared[tid]
        for i in al.range(1, 4):
            partial = partial + shared[tid + i * 64]
        shared[tid] = partial
    al.syncthreads()

    if tid == 0:
        final_sum = shared[0]
        for i in al.range(1, 64):
            final_sum = final_sum + shared[i]
        result = final_sum / al.convert(batch_size, al.f32)

        out_layout = al.make_layout((1,), (1,))
        out = al.make_tensor(out_ptr, al.f32, out_layout)
        out[0] = result


def avelang_kl_div(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA/HIP device."
    assert predictions.shape == targets.shape, "Shapes must match."

    batch, num_cols = predictions.shape

    predictions_bf16 = predictions.to(torch.bfloat16).contiguous()
    targets_bf16 = targets.to(torch.bfloat16).contiguous()

    BLOCK_SIZE = 256

    row_sums = torch.empty(batch, dtype=torch.float32, device=predictions.device)

    kl_div_row_kernel[lambda: ((batch, 1, 1), (BLOCK_SIZE, 1, 1))](
        predictions_bf16, targets_bf16, row_sums, num_cols, batch, BLOCK_SIZE
    )

    result = torch.empty(1, dtype=torch.float32, device=predictions.device)
    final_reduce_kernel[lambda: ((1, 1, 1), (BLOCK_SIZE, 1, 1))](
        row_sums, result, batch, BLOCK_SIZE
    )

    return result.reshape(()).to(predictions.dtype)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, predictions, targets):
        return avelang_kl_div(predictions, targets)
