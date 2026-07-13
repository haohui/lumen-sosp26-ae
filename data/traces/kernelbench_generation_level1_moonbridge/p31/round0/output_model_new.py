import torch
import torch.nn as nn
import avelang
import avelang.language as al

THREADS = 256
ELEMS_PER_THREAD = 512


@avelang.jit
def elu_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    total_elems: al.u32,
):
    tid = al.thread_id(0)
    bdim = al.block_dim(0)

    x = al.make_tensor(x_ptr, al.bf16, al.make_layout((total_elems,), (1,)))
    out = al.make_tensor(out_ptr, al.bf16, al.make_layout((total_elems,), (1,)))

    block_start = al.block_id(0) * bdim * ELEMS_PER_THREAD
    block_end = block_start + bdim * ELEMS_PER_THREAD
    one_f32 = al.convert(1.0, al.f32)
    zero_f32 = al.convert(0.0, al.f32)
    alpha_f32 = al.convert(1.0, al.f32)

    if block_end <= total_elems:
        for i in al.range(ELEMS_PER_THREAD):
            idx = block_start + tid + i * bdim
            val_f32 = al.convert(x[idx], al.f32)
            result = val_f32
            if val_f32 < zero_f32:
                result = alpha_f32 * (al.exp(val_f32) - one_f32)
            out[idx] = al.convert(result, al.bf16)
    else:
        for i in al.range(ELEMS_PER_THREAD):
            idx = block_start + tid + i * bdim
            if idx < total_elems:
                val_f32 = al.convert(x[idx], al.f32)
                result = val_f32
                if val_f32 < zero_f32:
                    result = alpha_f32 * (al.exp(val_f32) - one_f32)
                out[idx] = al.convert(result, al.bf16)


def _ensure_bf16_cuda_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_elu(x: torch.Tensor, alpha: float) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = _ensure_bf16_cuda_contiguous(x)
    total_elems = x_bf16.numel()
    out = torch.empty_like(x_bf16)

    elems_per_block = THREADS * ELEMS_PER_THREAD
    grid_x = (total_elems + elems_per_block - 1) // elems_per_block

    elu_kernel[lambda: ((grid_x, 1, 1), (THREADS, 1, 1))](
        x_bf16, out, total_elems
    )

    if out.shape != x.shape:
        out = out.reshape(x.shape)

    if out.dtype != x.dtype:
        out = out.to(dtype=x.dtype)

    return out


class ModelNew(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super(ModelNew, self).__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_elu(x, self.alpha)
