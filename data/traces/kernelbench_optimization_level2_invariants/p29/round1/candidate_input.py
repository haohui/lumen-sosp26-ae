import importlib.util
import torch
import torch.nn as nn

spec = importlib.util.spec_from_file_location(
    "amdgpu_gemm", "/avelang/python/avelang_kernels/amdgpu_gemm.py"
)
amdgpu_gemm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(amdgpu_gemm)
gemm_pipeline_transposed_b = amdgpu_gemm.gemm_pipeline_transposed_b


class ModelNew(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_bf16 = x.to(dtype=torch.bfloat16).contiguous()
        # weight is (out_features, in_features) = (N, K) - transposed B format
        w_nk = self.linear.weight.to(device=x.device, dtype=torch.bfloat16).contiguous()
        bias = self.linear.bias.to(device=x.device, dtype=torch.bfloat16).contiguous()

        m, k = x_bf16.shape
        n = w_nk.shape[0]
        gemm_out = gemm_pipeline_transposed_b(x_bf16, w_nk)

        # Bias + Mish + Mish (element-wise, not matmul/linear algebra)
        result = gemm_out.float() + bias.float()
        result = torch.nn.functional.mish(result)
        result = torch.nn.functional.mish(result)
        return result.to(torch.bfloat16)
