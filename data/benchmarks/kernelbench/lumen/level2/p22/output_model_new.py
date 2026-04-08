import torch
import torch.nn as nn


@torch.jit.script
def fused_post_gemm(x: torch.Tensor, scale: float, clamp_min: float, clamp_max: float) -> torch.Tensor:
    """Fused post-GEMM operations: scale*2, clamp, logsumexp, mish."""
    # Fused scale: x * 2 * scale (combines x * scale and x + x)
    x = x * scale
    x = torch.clamp(x, clamp_min, clamp_max)
    x = torch.logsumexp(x, dim=1, keepdim=True)
    x = x * torch.nn.functional.mish(x)
    return x


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super().__init__()
        self.matmul = nn.Linear(input_size, hidden_size)
        self.scale_factor = scale_factor
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

        # Fused scale: x * scale_factor followed by x + x = x * (2 * scale_factor)
        self.fused_scale = 2.0 * scale_factor  # 4.0

    def forward(self, x):
        # Use PyTorch's native linear layer for correct precision
        x = self.matmul(x)

        # Use JIT-compiled fused post-GEMM operations
        x = fused_post_gemm(x, self.fused_scale, self.clamp_min, self.clamp_max)

        return x


batch_size = 1024
input_size = 8192
hidden_size = 8192
scale_factor = 2.0
clamp_min = -10.0
clamp_max = 10.0


def get_inputs():
    return [torch.rand(batch_size, input_size)]


def get_init_inputs():
    return [input_size, hidden_size, scale_factor, clamp_min, clamp_max]
