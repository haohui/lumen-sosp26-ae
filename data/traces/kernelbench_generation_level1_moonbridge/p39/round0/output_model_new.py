import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE: al.constexpr = 256


@avelang.jit
def l2norm_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    batch_size: al.i32,
    N: al.i32,
):
    """
    Single-kernel L2 normalization: reduce sum_sq, compute norm, apply.
    Launch: grid = (batch_size, 1, 1), block = (BLOCK_SIZE, 1, 1)
    """
    tid = al.thread_id(0)
    bid = al.block_id(0)

    if bid < batch_size:
        smem_sq = al.make_shared((BLOCK_SIZE,), al.f32)

        layout_x = al.make_layout((batch_size * N,), (1,))
        x = al.make_tensor(x_ptr, al.bf16, layout_x)

        base = bid * N

        # Phase 1: strided reduction — sum of squares
        local_sq = al.convert(0.0, al.f32)
        for i in al.range(tid, N, BLOCK_SIZE):
            idx = base + i
            val = al.convert(x[idx], al.f32)
            local_sq = local_sq + val * val

        smem_sq[tid] = local_sq
        al.syncthreads()

        # Tree reduction
        if tid < 128:
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 128]
        al.syncthreads()
        if tid < 64:
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 64]
        al.syncthreads()
        if tid < 32:
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 32]
        al.syncthreads()
        if tid < 16:
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 16]
        al.syncthreads()
        if tid < 8:
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 8]
        al.syncthreads()
        if tid < 4:
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 4]
        al.syncthreads()
        if tid < 2:
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 2]
        al.syncthreads()
        if tid < 1:
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 1]

        # Thread 0 computes norm, broadcast via smem_sq[0]
        if tid == 0:
            smem_sq[0] = al.sqrt(smem_sq[0])
        al.syncthreads()

        norm_val = smem_sq[0]

        # Phase 2: apply normalization
        layout_out = al.make_layout((batch_size * N,), (1,))
        ot = al.make_tensor(out_ptr, al.bf16, layout_out)

        for i in al.range(tid, N, BLOCK_SIZE):
            idx = base + i
            x_val = al.convert(x[idx], al.f32)
            result = x_val / norm_val
            ot[idx] = al.convert(result, al.bf16)


def avelang_l2norm(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Tensors must be on CUDA/HIP device."
    assert x.dtype == torch.bfloat16, "Input tensor must be bfloat16"

    batch_size = x.shape[0]
    N = x.shape[1]

    x_contig = x.contiguous()
    out = torch.empty_like(x_contig)

    l2norm_kernel[lambda: ((batch_size, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_contig, out, batch_size, N
    )

    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs L2 normalization using AveLang DSL.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_bf16 = x.to(dtype=torch.bfloat16, device=x.device).contiguous()
        result = avelang_l2norm(x_bf16)
        return result.to(x.dtype)
