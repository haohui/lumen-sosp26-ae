#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

try:
    import torch
except Exception:
    torch = None


@dataclass
class SharedInputs:
    q_bshd: "torch.Tensor"
    k_bshd: "torch.Tensor"
    v_bshd: "torch.Tensor"


def attention_tflops(*, batch_size: int, seq_len: int, num_q_heads: int, head_dim: int, causal: bool, ms: float) -> float:
    if ms <= 0.0:
        return float("nan")
    flops = 4.0 * batch_size * num_q_heads * seq_len * seq_len * head_dim
    if causal:
        flops /= 2.0
    return flops / (ms * 1.0e-3) / 1.0e12


def build_shared_inputs(
    *,
    seq_lens: List[int],
    batch_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    device: "torch.device",
    dtype: "torch.dtype",
    seed: int,
) -> Dict[int, SharedInputs]:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    out: Dict[int, SharedInputs] = {}
    for s in seq_lens:
        out[s] = SharedInputs(
            q_bshd=torch.randn((batch_size, s, num_q_heads, head_dim), device=device, dtype=dtype, generator=g),
            k_bshd=torch.randn((batch_size, s, num_kv_heads, head_dim), device=device, dtype=dtype, generator=g),
            v_bshd=torch.randn((batch_size, s, num_kv_heads, head_dim), device=device, dtype=dtype, generator=g),
        )
    return out
