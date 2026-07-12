import torch
import torch.nn as nn

from .attn_06_inst_scheduling import flash_attn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        q_bshd: torch.Tensor,
        k_bshd: torch.Tensor,
        v_bshd: torch.Tensor,
    ) -> torch.Tensor:
        if q_bshd.ndim != 4 or k_bshd.ndim != 4 or v_bshd.ndim != 4:
            raise ValueError("q, k, and v must have shape [batch, sequence, heads, dim]")

        batch_size, seq_len = q_bshd.shape[:2]
        if k_bshd.shape[:2] != (batch_size, seq_len) or v_bshd.shape[:2] != (
            batch_size,
            seq_len,
        ):
            raise ValueError("q, k, and v must have matching batch and sequence dimensions")

        q_packed = q_bshd.contiguous().view(-1, *q_bshd.shape[2:])
        k_packed = k_bshd.contiguous().view(-1, *k_bshd.shape[2:])
        v_packed = v_bshd.contiguous().view(-1, *v_bshd.shape[2:])
        seq_ptr = torch.arange(
            batch_size + 1,
            device=q_bshd.device,
            dtype=torch.int32,
        ).mul_(seq_len)

        out_packed = flash_attn(q_packed, k_packed, v_packed, seq_ptr)
        return out_packed.view_as(q_bshd)
