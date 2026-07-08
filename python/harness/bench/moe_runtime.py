#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

try:
    import torch
except Exception:
    torch = None


BLOCK_N = 128
BLOCK_K = 128


@dataclass
class SharedInputs:
    input_q: "torch.Tensor"
    topk_weights: "torch.Tensor"
    topk_ids: "torch.Tensor"
    input_scale: "torch.Tensor"


def build_shared_inputs(
    *,
    seq_lens: List[int],
    dim: int,
    experts: int,
    topk: int,
    input_dtype: "torch.dtype",
    device: "torch.device",
    seed: int,
) -> Dict[int, SharedInputs]:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    out: Dict[int, SharedInputs] = {}
    hidden_blocks = dim // BLOCK_K

    for s in seq_lens:
        input_q = torch.randn((s, dim), dtype=torch.float32, device=device, generator=g).mul_(1.0).add_(0.1)
        input_q = input_q.to(input_dtype).contiguous()

        input_scale = (
            torch.randn((s, hidden_blocks), dtype=torch.float32, device=device, generator=g).mul_(2e-2).add_(1e-1)
        ).clamp_min_(1e-8).contiguous()

        scores = torch.randn((s, experts), dtype=torch.float32, device=device, generator=g)
        topk_val, topk_idx = torch.topk(scores, k=topk, dim=-1, largest=True, sorted=True)
        topk_ids = topk_idx.to(torch.int32).contiguous()
        topk_weights = torch.softmax(topk_val, dim=-1).to(torch.float32).contiguous()

        out[s] = SharedInputs(
            input_q=input_q,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            input_scale=input_scale,
        )
    return out


def build_shared_weights(
    *,
    dim: int,
    inter_dim: int,
    experts: int,
    input_dtype: "torch.dtype",
    device: "torch.device",
    seed: int,
) -> Dict[str, "torch.Tensor"]:
    g = torch.Generator(device=device)
    g.manual_seed(seed + 17)

    w1_q = torch.randn((experts, inter_dim * 2, dim), dtype=torch.float32, device=device, generator=g).mul_(8.0).to(input_dtype)
    w2_q = torch.randn((experts, dim, inter_dim), dtype=torch.float32, device=device, generator=g).mul_(8.0).to(input_dtype)

    fc1_scale = (
        torch.randn(
            (experts, ((inter_dim * 2) // BLOCK_N) * (dim // BLOCK_K)),
            dtype=torch.float32,
            device=device,
            generator=g,
        ).mul_(2e-3).add_(1e-2)
    ).clamp_min_(1e-8)

    fc2_scale = (
        torch.randn(
            (experts, (dim // BLOCK_N) * (inter_dim // BLOCK_K)),
            dtype=torch.float32,
            device=device,
            generator=g,
        ).mul_(2e-3).add_(1e-2)
    ).clamp_min_(1e-8)

    return {
        "w1_q": w1_q.contiguous(),
        "w2_q": w2_q.contiguous(),
        "fc1_scale": fc1_scale.contiguous(),
        "fc2_scale": fc2_scale.contiguous(),
    }
