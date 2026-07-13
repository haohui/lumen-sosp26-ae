import torch
import torch.nn as nn
import avelang
import avelang.language as al

from avelang_kernels.amdgpu_gemm import gemm_pipeline_transposed_b

REDUCTION_BLOCK = 256


@avelang.jit
def logsumexp_reduce_epilogue_kernel(
    in_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    m: al.u32,
    n: al.u32,
):
    tid = al.thread_id(0)
    row_idx = al.block_id(0)

    if row_idx < m:
        layout_in = al.make_layout((m, n), (n, 1))
        x = al.make_tensor(in_ptr, al.bf16, layout_in)
        bias_layout = al.make_layout((n,), (1,))
        b = al.make_tensor(bias_ptr, al.bf16, bias_layout)

        smem = al.make_shared((REDUCTION_BLOCK,), al.f32)

        # Phase 1: find row max (with bias)
        neg_inf = al.convert(-1.0e30, al.f32)
        local_max = neg_inf
        for i in al.range(tid, n, REDUCTION_BLOCK):
            val = al.convert(x[row_idx, i], al.f32) + al.convert(b[i], al.f32)
            if val > local_max:
                local_max = val

        smem[tid] = local_max
        al.syncthreads()

        if tid < 128:
            other = smem[tid + 128]
            if other > smem[tid]:
                smem[tid] = other
        al.syncthreads()
        if tid < 64:
            other = smem[tid + 64]
            if other > smem[tid]:
                smem[tid] = other
        al.syncthreads()
        if tid < 32:
            other = smem[tid + 32]
            if other > smem[tid]:
                smem[tid] = other
        al.syncthreads()
        if tid < 16:
            other = smem[tid + 16]
            if other > smem[tid]:
                smem[tid] = other
        al.syncthreads()
        if tid < 8:
            other = smem[tid + 8]
            if other > smem[tid]:
                smem[tid] = other
        al.syncthreads()
        if tid < 4:
            other = smem[tid + 4]
            if other > smem[tid]:
                smem[tid] = other
        al.syncthreads()
        if tid < 2:
            other = smem[tid + 2]
            if other > smem[tid]:
                smem[tid] = other
        al.syncthreads()
        if tid < 1:
            other = smem[tid + 1]
            if other > smem[tid]:
                smem[tid] = other
        al.syncthreads()

        row_max = smem[0]

        # Phase 2: compute sum(exp(x + bias - max))
        zero = al.convert(0.0, al.f32)
        local_sum = zero
        for i in al.range(tid, n, REDUCTION_BLOCK):
            val = al.convert(x[row_idx, i], al.f32) + al.convert(b[i], al.f32)
            local_sum = local_sum + al.exp(val - row_max)

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
        al.syncthreads()

        row_sum = smem[0]

        if tid == 0:
            # LogSumExp
            result = al.log(row_sum) + row_max

            # LeakyReLU x2 (negative_slope = 0.01)
            neg_slope = al.convert(0.01, al.f32)
            if result > zero:
                result = result
            else:
                result = neg_slope * result

            if result > zero:
                result = result
            else:
                result = neg_slope * result

            # GELU x2 using tanh approximation
            sqrt_2_pi = al.convert(0.7978845608028654, al.f32)
            coeff = al.convert(0.044715, al.f32)
            half = al.convert(0.5, al.f32)
            one = al.convert(1.0, al.f32)

            x2 = result * result
            x3 = x2 * result
            inner = sqrt_2_pi * (result + coeff * x3)
            result = half * result * (one + al.tanh(inner))

            x2 = result * result
            x3 = x2 * result
            inner = sqrt_2_pi * (result + coeff * x3)
            result = half * result * (one + al.tanh(inner))

            layout_out = al.make_layout((m, 1), (1, 1))
            out = al.make_tensor(out_ptr, al.bf16, layout_out)
            out[row_idx, 0] = al.convert(result, al.bf16)


def _prepare_bf16_cuda_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_gemm_logsumexp(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    m, k_in = x.shape
    n, k_w = weight.shape
    if k_in != k_w:
        raise ValueError(
            f"Weight/input K mismatch: x has K={k_in}, weight has K={k_w}"
        )

    x_bf16 = _prepare_bf16_cuda_contiguous(x)
    w_bf16 = _prepare_bf16_cuda_contiguous(weight)
    b_bf16 = _prepare_bf16_cuda_contiguous(bias)

    # Stage 1: GEMM using proven AveLang pipeline kernel
    gemm_out = gemm_pipeline_transposed_b(x_bf16, w_bf16)

    # Stage 2: Bias-add + LogSumExp reduction + LeakyReLU + GELU
    out = torch.empty((m, 1), device=x_bf16.device, dtype=torch.bfloat16)
    logsumexp_reduce_epilogue_kernel[lambda: ((m, 1, 1), (REDUCTION_BLOCK, 1, 1))](
        gemm_out, b_bf16, out, m, n
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        if self.linear.bias is not None:
            bias = self.linear.bias
        else:
            bias = torch.zeros(self.linear.out_features, device=x.device, dtype=torch.bfloat16)
        return avelang_gemm_logsumexp(x, self.linear.weight, bias)


batch_size = 1024
in_features = 8192
out_features = 8192


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features]
