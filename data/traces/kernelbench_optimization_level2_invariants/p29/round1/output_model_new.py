import importlib.util
import torch
import torch.nn as nn
import avelang
import avelang.language as al

spec = importlib.util.spec_from_file_location(
    "amdgpu_gemm", "/avelang/python/avelang_kernels/amdgpu_gemm.py"
)
amdgpu_gemm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(amdgpu_gemm)
gemm_pipeline_transposed_b = amdgpu_gemm.gemm_pipeline_transposed_b

_NUM_THREADS = 256


@avelang.jit
def fused_bias_double_mish_kernel(
    in_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    total_elems: al.i32,
    N: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)
    block_dim = al.block_dim(0)
    idx = bid * block_dim + tid

    layout_in = al.make_layout((total_elems,), (1,))
    in_t = al.make_tensor(in_ptr, al.bf16, layout_in)
    layout_bias = al.make_layout((N,), (1,))
    bias_t = al.make_tensor(bias_ptr, al.bf16, layout_bias)
    layout_out = al.make_layout((total_elems,), (1,))
    out_t = al.make_tensor(out_ptr, al.bf16, layout_out)

    if idx < total_elems:
        col = idx % N
        val = al.convert(in_t[idx], al.f32)
        bias_val = al.convert(bias_t[col], al.f32)
        val = val + bias_val
        one = al.convert(1.0, al.f32)
        sp = al.log(one + al.exp(val))
        val = val * al.tanh(sp)
        sp2 = al.log(one + al.exp(val))
        val = val * al.tanh(sp2)
        out_t[idx] = al.convert(val, al.bf16)


def _launch_fused_bias_mish(
    x: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
) -> None:
    total_elems = x.numel()
    N = x.shape[1]
    grid = (total_elems + _NUM_THREADS - 1) // _NUM_THREADS

    fused_bias_double_mish_kernel[
        lambda: ((grid, 1, 1), (_NUM_THREADS, 1, 1))
    ](
        x.data_ptr(),
        bias.data_ptr(),
        out.data_ptr(),
        total_elems,
        N,
    )


class ModelNew(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_bf16 = x.to(dtype=torch.bfloat16).contiguous()
        w_nk = self.linear.weight.to(
            device=x.device, dtype=torch.bfloat16
        ).contiguous()

        m = x_bf16.shape[0]
        n = w_nk.shape[0]

        gemm_out = gemm_pipeline_transposed_b(x_bf16, w_nk)

        bias = self.linear.bias.to(
            device=x.device, dtype=torch.bfloat16
        ).contiguous()
        out = torch.empty(
            (m, n), dtype=torch.bfloat16, device=x.device
        )
        _launch_fused_bias_mish(gemm_out, bias, out)
        return out
