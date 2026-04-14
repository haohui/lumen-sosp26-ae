import torch
import torch.nn as nn

try:
    # Current aiter layout.
    from aiter.ops.triton.mha import flash_attn_func
except ModuleNotFoundError:
    # Older aiter layout.
    from aiter.ops.triton.attention.mha import flash_attn_func


class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, q_bshd: torch.Tensor, k_bshd: torch.Tensor, v_bshd: torch.Tensor) -> torch.Tensor:
        return flash_attn_func(
            q_bshd,
            k_bshd,
            v_bshd,
            dropout_p=0.0,
            causal=True,
        )
