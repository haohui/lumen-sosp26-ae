import torch
import torch.nn as nn

_FLASH_ATTN_FUNC = None


def _flash_attn_func():
    global _FLASH_ATTN_FUNC
    if _FLASH_ATTN_FUNC is not None:
        return _FLASH_ATTN_FUNC
    try:
        # Current aiter layout.
        from aiter.ops.triton.mha import flash_attn_func
    except ModuleNotFoundError:
        # Older aiter layout.
        from aiter.ops.triton.attention.mha import flash_attn_func
    _FLASH_ATTN_FUNC = flash_attn_func
    return flash_attn_func


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        q_bshd: torch.Tensor,
        k_bshd: torch.Tensor,
        v_bshd: torch.Tensor,
    ) -> torch.Tensor:
        return _flash_attn_func()(
            q_bshd,
            k_bshd,
            v_bshd,
            dropout_p=0.0,
            causal=True,
        )
