import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE: al.constexpr = 256


@avelang.jit
def softmax_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    batch_size: al.i32,
    dim: al.i32,
):
    """
    Compute row-wise softmax in a single kernel with online max+sum reduction.
    Each block handles one row:
      Pass 1: online max (bf16 comparison) + sum (f32 accumulation) with strided iteration.
      Pass 2: normalize and write output.
    """
    tid = al.thread_id(0)
    bid = al.block_id(0)

    if bid < batch_size:
        layout_in = al.make_layout((batch_size, dim), (dim, 1))
        x = al.make_tensor(x_ptr, al.bf16, layout_in)

        layout_out = al.make_layout((batch_size, dim), (dim, 1))
        out = al.make_tensor(out_ptr, al.bf16, layout_out)

        smem_max = al.make_shared((BLOCK_SIZE,), al.bf16)
        smem_sum = al.make_shared((BLOCK_SIZE,), al.f32)

        # --- Pass 1: online max+sum ---
        first_val = x[bid, tid]
        local_max = first_val
        local_sum = al.convert(1.0, al.f32)

        for i in al.range(tid + BLOCK_SIZE, dim, BLOCK_SIZE):
            val = x[bid, i]
            new_max = val if val > local_max else local_max

            old_max_f32 = al.convert(local_max, al.f32)
            new_max_f32 = al.convert(new_max, al.f32)
            val_f32 = al.convert(val, al.f32)

            rescale = al.exp(old_max_f32 - new_max_f32)
            local_sum = local_sum * rescale + al.exp(val_f32 - new_max_f32)
            local_max = new_max

        smem_max[tid] = local_max
        smem_sum[tid] = local_sum
        al.syncthreads()

        # Tree reduction: merge (max, sum) pairs
        stride = 128
        for _ in al.range(0, 8):
            if tid < stride:
                other_max = smem_max[tid + stride]
                my_max = smem_max[tid]
                my_sum = smem_sum[tid]
                other_sum = smem_sum[tid + stride]

                winner_max = other_max if other_max > my_max else my_max
                loser_max = my_max if other_max > my_max else other_max
                winner_sum = other_sum if other_max > my_max else my_sum
                loser_sum = my_sum if other_max > my_max else other_sum

                loser_f32 = al.convert(loser_max, al.f32)
                winner_f32 = al.convert(winner_max, al.f32)
                rescale = al.exp(loser_f32 - winner_f32)
                merged_sum = winner_sum + loser_sum * rescale

                smem_max[tid] = winner_max
                smem_sum[tid] = merged_sum
            al.syncthreads()
            stride = stride // 2

        row_max = al.convert(smem_max[0], al.f32)
        row_sum = smem_sum[0]

        # --- Pass 2: normalize and write output ---
        for i in al.range(tid, dim, BLOCK_SIZE):
            val = al.convert(x[bid, i], al.f32)
            exp_val = al.exp(val - row_max)
            out[bid, i] = al.convert(exp_val / row_sum, al.bf16)


def avelang_softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA/HIP device."
    assert x.dtype == torch.bfloat16, "Input tensor must be bfloat16."
    assert x.ndim == 2, "Input tensor must be 2D (batch_size, dim)."

    batch_size = x.shape[0]
    dim_val = x.shape[1]

    x_contig = x.contiguous()
    out = torch.empty_like(x_contig)

    softmax_kernel[lambda: ((batch_size, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_contig, out, batch_size, dim_val
    )

    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs a Softmax activation using AveLang DSL.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_softmax(x)
