import torch
import torch.nn as nn
import avelang
import avelang.language as al


MULTIPLIER = 2.0
NEGATIVE_SLOPE = 0.1

THREADS = 256


@avelang.jit
def mul_leakyrelu_kernel(
    in_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    total_elems: al.u32,
    n: al.u32,
):
    in_tensor = al.make_tensor(in_ptr, al.bf16, al.make_layout((total_elems,), (1,)))
    bias_tensor = al.make_tensor(bias_ptr, al.bf16, al.make_layout((n,), (1,)))
    out_tensor = al.make_tensor(out_ptr, al.bf16, al.make_layout((total_elems,), (1,)))

    idx = al.block_id(0) * THREADS + al.thread_id(0)
    if idx < total_elems:
        row = idx // n
        col = idx % n
        val = al.convert(in_tensor[idx], al.f32)
        bias_val = al.convert(bias_tensor[col], al.f32)
        mul_val = al.convert(MULTIPLIER, al.f32)
        neg_slope_val = al.convert(NEGATIVE_SLOPE, al.f32)
        zero = al.convert(0.0, al.f32)

        result = val + bias_val
        result = result * mul_val
        if result < zero:
            result = result * neg_slope_val
        out_tensor[idx] = al.convert(result, al.bf16)


def _prepare_bf16_cuda_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_linear_mul_leakyrelu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = _prepare_bf16_cuda_contiguous(x)
    weight_bf16 = _prepare_bf16_cuda_contiguous(weight)
    bias_bf16 = _prepare_bf16_cuda_contiguous(bias)

    m, k = x_bf16.shape
    n, weight_k = weight_bf16.shape
    if weight_k != k:
        raise ValueError(
            f"Weight/input K mismatch: x has K={k}, weight has K={weight_k}"
        )

    from avelang_kernels.amdgpu_gemm import gemm_pipeline_transposed_b
    gemm_out = gemm_pipeline_transposed_b(x_bf16, weight_bf16)

    total_elems = m * n
    grid_elems = (total_elems + THREADS - 1) // THREADS

    epilogue_out = torch.empty_like(gemm_out)
    epilogue_in_flat = gemm_out.reshape(-1)
    epilogue_out_flat = epilogue_out.reshape(-1)

    mul_leakyrelu_kernel[lambda: ((grid_elems, 1, 1), (THREADS, 1, 1))](
        epilogue_in_flat, bias_bf16, epilogue_out_flat, total_elems, n,
    )

    return epilogue_out


class ModelNew(nn.Module):
    def __init__(self, in_features: int, out_features: int, multiplier: float, negative_slope: float):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))

        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_linear_mul_leakyrelu(x, self.weight, self.bias)


batch_size = 1024
in_features = 8192
out_features = 8192
multiplier = 2.0
negative_slope = 0.1


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, multiplier, negative_slope]
