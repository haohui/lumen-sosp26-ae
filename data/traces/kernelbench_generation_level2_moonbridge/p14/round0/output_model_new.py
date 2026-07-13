import torch
import torch.nn as nn
import avelang
import avelang.language as al
from avelang_kernels.amdgpu_gemm import gemm_pipeline_transposed_b

# ── Problem dimensions ──────────────────────────────────────────────
batch_size = 1024
input_size = 8192
hidden_size = 8192
scaling_factor = 1.5

# ── Reduction tiling ────────────────────────────────────────────────
REDUCE_BLOCK_SIZE: al.constexpr = 256


# ══════════════════════════════════════════════════════════════════════
# Kernel: Row-wise sum reduction + scaling
# ══════════════════════════════════════════════════════════════════════


@avelang.jit
def sum_reduce_scale_kernel(
    in_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    m: al.u32,
    n: al.u32,
    scale: al.f32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    if bid < m:
        smem = al.make_shared((REDUCE_BLOCK_SIZE,), al.f32)

        layout_in = al.make_layout((m, n), (n, 1))
        in_tensor = al.make_tensor(in_ptr, al.bf16, layout_in)

        local_sum = al.convert(0.0, al.f32)
        for i in al.range(tid, n, REDUCE_BLOCK_SIZE):
            val = al.convert(in_tensor[bid, i], al.f32)
            local_sum = local_sum + val

        smem[tid] = local_sum
        al.syncthreads()

        # Tree reduction in shared memory
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
            result = smem[0] * scale
            layout_out = al.make_layout((m, 1), (1, 1))
            out_tensor = al.make_tensor(out_ptr, al.bf16, layout_out)
            out_tensor[bid, 0] = al.convert(result, al.bf16)


# ══════════════════════════════════════════════════════════════════════
# Host wrapper
# ══════════════════════════════════════════════════════════════════════


def _prepare_bf16_cuda_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_gemm_sum_scale(
    x: torch.Tensor,
    weight: torch.Tensor,
    scaling_factor: float,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    orig_dtype = x.dtype

    x_bf16 = _prepare_bf16_cuda_contiguous(x)
    w_bf16 = _prepare_bf16_cuda_contiguous(weight)

    m, k = x_bf16.shape
    n, wk = w_bf16.shape
    if wk != k:
        raise ValueError(f"Weight/input K mismatch: x has K={k}, weight has K={wk}")

    # Phase 1: GEMM using verified amdgpu_gemm kernel → (m, n) bf16
    gemm_out = gemm_pipeline_transposed_b(x_bf16, w_bf16)

    # Phase 2: row-sum reduction + scaling → (m, 1) bf16
    scale = scaling_factor / 2.0
    final_out = torch.empty((m, 1), device=x_bf16.device, dtype=torch.bfloat16)
    grid_reduce = (m, 1, 1)
    sum_reduce_scale_kernel[lambda: (grid_reduce, (REDUCE_BLOCK_SIZE, 1, 1))](
        gemm_out, final_out, m, n, scale
    )

    # Convert back to original dtype
    return final_out.to(dtype=orig_dtype)


# ══════════════════════════════════════════════════════════════════════
# ModelNew entrypoint
# ══════════════════════════════════════════════════════════════════════


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        return avelang_gemm_sum_scale(x, self.weight, self.scaling_factor)


def get_inputs():
    return [torch.rand(batch_size, input_size)]


def get_init_inputs():
    return [input_size, hidden_size, scaling_factor]
