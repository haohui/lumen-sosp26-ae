import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE: al.constexpr = 256


@avelang.jit
def exclusive_cumsum_kernel(
    input_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    n_rows: al.i32,
    n_cols: al.i32,
):
    row = al.block_id(0)
    tid = al.thread_id(0)

    if row >= n_rows - 1:
        return

    layout_in = al.make_layout((n_rows, n_cols), (n_cols, 1))
    input_tensor = al.make_tensor(input_ptr, al.bf16, layout_in)

    layout_out = al.make_layout((n_rows - 1, n_cols + 1), (n_cols + 1, 1))
    output_tensor = al.make_tensor(output_ptr, al.bf16, layout_out)

    per_thread = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
    start_col = tid * per_thread

    # Phase 1: local inclusive scan in FP32 with BF16 output
    local_sum = al.convert(0.0, al.f32)
    for j in al.range(per_thread):
        col = start_col + j
        if col < n_cols:
            val = al.convert(input_tensor[row, col], al.f32)
            local_sum = local_sum + val
            output_tensor[row, col + 1] = al.convert(local_sum, al.bf16)

    if tid == 0:
        output_tensor[row, 0] = al.convert(0.0, al.bf16)

    # Phase 2: Blelloch exclusive scan of thread partial sums in FP32
    thread_sums = al.make_shared((BLOCK_SIZE,), al.f32)
    thread_sums[tid] = local_sum
    al.syncthreads()

    # Up-sweep
    offset = 1
    for _ in al.range(16):
        if offset < BLOCK_SIZE:
            idx = (tid + 1) * offset * 2 - 1
            if idx < BLOCK_SIZE:
                thread_sums[idx] = thread_sums[idx] + thread_sums[idx - offset]
        offset = offset * 2
        al.syncthreads()

    # Clear last element
    if tid == BLOCK_SIZE - 1:
        thread_sums[BLOCK_SIZE - 1] = al.convert(0.0, al.f32)
    al.syncthreads()

    # Down-sweep
    offset = BLOCK_SIZE // 2
    for _ in al.range(16):
        if offset > 0:
            idx = (tid + 1) * offset * 2 - 1
            if idx < BLOCK_SIZE:
                t = thread_sums[idx - offset]
                thread_sums[idx - offset] = thread_sums[idx]
                thread_sums[idx] = thread_sums[idx] + t
        offset = offset // 2
        al.syncthreads()

    # Phase 3: add block-level prefix
    block_prefix = thread_sums[tid]
    for j in al.range(per_thread):
        col = start_col + j
        if col < n_cols:
            cur = al.convert(output_tensor[row, col + 1], al.f32)
            output_tensor[row, col + 1] = al.convert(cur + block_prefix, al.bf16)


def avelang_exclusive_cumsum(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda, "Tensors must be on CUDA/HIP device."
    assert x.dtype == torch.bfloat16, "Input tensor must be bfloat16"

    x_contiguous = x.contiguous()
    n_rows = x.shape[0]
    n_cols = x.shape[1]

    output_rows = n_rows - 1
    output_cols = n_cols + 1
    output = torch.empty((output_rows, output_cols), dtype=torch.bfloat16, device=x.device)

    num_blocks = output_rows
    exclusive_cumsum_kernel[lambda: ((num_blocks, 1, 1), (256, 1, 1))](
        x_contiguous, output, n_rows, n_cols
    )

    return output


class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        return avelang_exclusive_cumsum(x, self.dim)
