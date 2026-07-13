import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE: al.constexpr = 256


@avelang.jit
def logsoftmax_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    batch_size: al.i32,
    dim_size: al.i32,
):
    """
    Single-kernel LogSoftmax with fused max+sumexp reduction.
    Each thread computes (local_max, local_sumexp) online in one pass,
    then tree-reduce to get global (max, sumexp).
    Final pass applies: output = x - max - log(sumexp).
    """
    tid = al.thread_id(0)
    bid = al.block_id(0)

    if bid < batch_size:
        base = bid * dim_size

        layout_in = al.make_layout((batch_size * dim_size,), (1,))
        x = al.make_tensor(x_ptr, al.bf16, layout_in)

        layout_out = al.make_layout((batch_size * dim_size,), (1,))
        out = al.make_tensor(out_ptr, al.bf16, layout_out)

        smem_max = al.make_shared((BLOCK_SIZE,), al.f32)
        smem_sum = al.make_shared((BLOCK_SIZE,), al.f32)

        # === Phase 1: per-thread online (max, sumexp) in a single pass ===
        local_max = al.convert(-1.0e10, al.f32)
        local_sum = al.convert(0.0, al.f32)

        for i in al.range(tid, dim_size, BLOCK_SIZE):
            idx = base + i
            val = al.convert(x[idx], al.f32)
            if val > local_max:
                local_sum = local_sum * al.exp(local_max - val) + al.convert(1.0, al.f32)
                local_max = val
            else:
                local_sum = local_sum + al.exp(val - local_max)

        smem_max[tid] = local_max
        smem_sum[tid] = local_sum
        al.syncthreads()

        # === Phase 2: tree-reduce (max, sumexp) pairs ===
        offset = 128
        for _ in al.range(0, 8):
            if tid < offset:
                max_a = smem_max[tid]
                max_b = smem_max[tid + offset]
                sum_a = smem_sum[tid]
                sum_b = smem_sum[tid + offset]
                if max_a > max_b:
                    smem_max[tid] = max_a
                    smem_sum[tid] = sum_a + sum_b * al.exp(max_b - max_a)
                else:
                    smem_max[tid] = max_b
                    smem_sum[tid] = sum_b + sum_a * al.exp(max_a - max_b)
            offset = offset // 2
            al.syncthreads()

        global_max = smem_max[0]
        global_sum = smem_sum[0]
        log_sum = al.log(global_sum)

        # === Phase 3: apply LogSoftmax ===
        for i in al.range(tid, dim_size, BLOCK_SIZE):
            idx = base + i
            val = al.convert(x[idx], al.f32)
            result = val - global_max - log_sum
            out[idx] = al.convert(result, al.bf16)


def avelang_logsoftmax_impl(x_bf16: torch.Tensor) -> torch.Tensor:
    """Apply LogSoftmax along dim=1 using AveLang kernel. Input must be BF16 and contiguous."""
    batch_size_val = x_bf16.shape[0]
    dim_size_val = x_bf16.shape[1]

    out = torch.empty_like(x_bf16)

    logsoftmax_kernel[lambda: ((batch_size_val, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_bf16, out, batch_size_val, dim_size_val
    )

    return out


def avelang_logsoftmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Apply LogSoftmax along the given dimension using AveLang DSL."""
    assert x.ndim == 2, "Only 2D tensors are supported"
    assert x.is_cuda, "Input tensor must be on CUDA/HIP device"

    x_bf16 = x.to(dtype=torch.bfloat16).contiguous()

    if dim == 0:
        x_t = x_bf16.T.contiguous()
        result_t = avelang_logsoftmax_impl(x_t)
        result = result_t.T.contiguous()
    else:
        result = avelang_logsoftmax_impl(x_bf16)

    return result.to(dtype=x.dtype)


class ModelNew(nn.Module):
    def __init__(self, dim: int = 1):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_logsoftmax(x, self.dim)
