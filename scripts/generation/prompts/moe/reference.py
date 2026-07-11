#!/usr/bin/env python3
"""Pure PyTorch reference MoE implementation (blockscale FP8 g1u1, SiLU)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

BLOCK_N = 128
BLOCK_K = 128


@dataclass
class MoeConfig:
    dim: int = 7168
    inter_dim: int = 2048
    experts: int = 32
    topk: int = 4


class FusedMoESiluTorchOps:
    """Reference implementation equivalent to test-side PyTorch ref."""

    def __init__(self, out_dtype: torch.dtype = torch.bfloat16):
        self.block_n = BLOCK_N
        self.block_k = BLOCK_K
        self.out_dtype = out_dtype

    def _dequantize_input(self, input_q: torch.Tensor, input_scale: torch.Tensor) -> torch.Tensor:
        tokens, model_dim = input_q.shape
        blocks = input_q.to(torch.float32).view(tokens, model_dim // self.block_k, self.block_k)
        return (blocks * input_scale.to(torch.float32).unsqueeze(-1)).reshape(tokens, model_dim)

    def _dequantize_weight(self, weight_q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        experts, rows, cols = weight_q.shape
        row_blocks = rows // self.block_n
        col_blocks = cols // self.block_k

        dense_q = weight_q.to(torch.float32)
        blocks = (
            dense_q.view(experts, row_blocks, self.block_n, col_blocks, self.block_k)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )
        scaled_blocks = blocks * scale.to(torch.float32).view(experts, row_blocks, col_blocks, 1, 1)
        return (
            scaled_blocks.permute(0, 1, 3, 2, 4)
            .reshape(experts, rows, cols)
            .contiguous()
        )

    def forward(
        self,
        input_q: torch.Tensor,
        w1_q: torch.Tensor,
        w2_q: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        input_scale: torch.Tensor,
        fc1_scale: torch.Tensor,
        fc2_scale: torch.Tensor,
    ) -> torch.Tensor:
        tokens, model_dim = input_q.shape
        _, _, inter_dim = w2_q.shape

        input_f = self._dequantize_input(input_q, input_scale)
        topk_weights_f = topk_weights.to(torch.float32)

        # Capture-friendly reference: avoid per-expert dynamic torch.where branches.
        w1_f = self._dequantize_weight(w1_q, fc1_scale)  # [E, 2I, D]
        w2_f = self._dequantize_weight(w2_q, fc2_scale)  # [E, D, I]

        stage1_all = torch.einsum("td,eid->tei", input_f, w1_f)  # [T, E, 2I]
        gate, up = stage1_all.split([inter_dim, inter_dim], dim=-1)
        activated = F.silu(gate) * up  # [T, E, I]
        route_all = torch.einsum("tei,edi->ted", activated, w2_f)  # [T, E, D]

        gather_index = topk_ids.to(torch.int64).unsqueeze(-1).expand(tokens, topk_ids.shape[1], model_dim)
        selected = torch.gather(route_all, dim=1, index=gather_index)  # [T, K, D]
        out = (selected * topk_weights_f.unsqueeze(-1)).sum(dim=1)  # [T, D]
        return out.to(self.out_dtype)


def make_inputs(
    *,
    tokens: int,
    cfg: MoeConfig,
    device: torch.device,
    seed: int,
) -> dict[str, torch.Tensor]:
    if cfg.dim % BLOCK_K != 0 or cfg.dim % BLOCK_N != 0:
        raise ValueError(f"dim must be divisible by {BLOCK_N}/{BLOCK_K}, got {cfg.dim}")
    if cfg.inter_dim % BLOCK_K != 0 or (cfg.inter_dim * 2) % BLOCK_N != 0:
        raise ValueError(f"inter_dim must match blockshape, got {cfg.inter_dim}")
    if cfg.topk > cfg.experts:
        raise ValueError(f"topk ({cfg.topk}) must be <= experts ({cfg.experts})")

    g = torch.Generator(device=str(device))
    g.manual_seed(seed + tokens)

    input_q = (
        torch.randn((tokens, cfg.dim), dtype=torch.float32, device=device, generator=g) * 1.0 + 0.1
    ).to(torch.float8_e4m3fnuz)
    w1_q = torch.randn(
        (cfg.experts, cfg.inter_dim * 2, cfg.dim),
        dtype=torch.float32,
        device=device,
        generator=g,
    ).mul_(8.0).to(torch.float8_e4m3fnuz)
    w2_q = torch.randn(
        (cfg.experts, cfg.dim, cfg.inter_dim),
        dtype=torch.float32,
        device=device,
        generator=g,
    ).mul_(8.0).to(torch.float8_e4m3fnuz)

    def _rand_pos(shape: tuple[int, ...], mean: float, std: float) -> torch.Tensor:
        x = torch.randn(shape, dtype=torch.float32, device=device, generator=g) * std + mean
        return x.clamp_min(1e-8)

    input_scale = _rand_pos((tokens, cfg.dim // BLOCK_K), mean=1e-1, std=2e-2)
    fc1_scale = _rand_pos(
        (cfg.experts, ((cfg.inter_dim * 2) // BLOCK_N) * (cfg.dim // BLOCK_K)),
        mean=1e-2,
        std=2e-3,
    )
    fc2_scale = _rand_pos(
        (cfg.experts, (cfg.dim // BLOCK_N) * (cfg.inter_dim // BLOCK_K)),
        mean=1e-2,
        std=2e-3,
    )

    # Match MoE routing semantics: one token selects top-k unique experts.
    scores = torch.randn((tokens, cfg.experts), dtype=torch.float32, device=device, generator=g)
    topk_val, topk_idx = torch.topk(scores, k=cfg.topk, dim=-1, largest=True, sorted=True)
    topk_ids = topk_idx.to(torch.int32)
    topk_weights = torch.softmax(topk_val, dim=-1).to(torch.float32)

    return {
        "input_q": input_q,
        "w1_q": w1_q,
        "w2_q": w2_q,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "input_scale": input_scale,
        "fc1_scale": fc1_scale,
        "fc2_scale": fc2_scale,
    }


def _eval_cfg() -> dict[str, Any]:
    cfg = globals().get("EVAL_CONFIG", {})
    if not isinstance(cfg, dict):
        return {}
    return cfg


def _cfg_int(cfg: dict[str, Any], key: str, default: int, minimum: int = 1) -> int:
    raw = cfg.get(key, default)
    try:
        val = int(raw)
    except Exception:
        val = int(default)
    return max(minimum, val)


def _cfg_device(cfg: dict[str, Any]) -> torch.device:
    d = str(cfg.get("device", "cuda:0")).strip()
    if not d:
        d = "cuda:0"
    if d.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"EVAL_CONFIG.device={d!r} requires CUDA/HIP runtime")
    return torch.device(d)


class Model(nn.Module):
    """KernelBench/CudaForge/KernelFalcon-compatible reference wrapper."""

    def __init__(self):
        super().__init__()
        cfg = _eval_cfg()
        self.cfg = MoeConfig(
            dim=_cfg_int(cfg, "dim", 7168, minimum=128),
            inter_dim=_cfg_int(cfg, "inter_dim", 2048, minimum=128),
            experts=_cfg_int(cfg, "experts", 32, minimum=1),
            topk=_cfg_int(cfg, "topk", 4, minimum=1),
        )
        if self.cfg.topk > self.cfg.experts:
            self.cfg.topk = self.cfg.experts
        self.ref_impl = FusedMoESiluTorchOps(out_dtype=torch.bfloat16)

    def forward(
        self,
        input_q: torch.Tensor,
        w1_q: torch.Tensor,
        w2_q: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        input_scale: torch.Tensor,
        fc1_scale: torch.Tensor,
        fc2_scale: torch.Tensor,
    ) -> torch.Tensor:
        return self.ref_impl.forward(
            input_q=input_q,
            w1_q=w1_q,
            w2_q=w2_q,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            input_scale=input_scale,
            fc1_scale=fc1_scale,
            fc2_scale=fc2_scale,
        )


def get_init_inputs():
    return []


def get_inputs():
    cfg = _eval_cfg()
    task_cfg = MoeConfig(
        dim=_cfg_int(cfg, "dim", 7168, minimum=128),
        inter_dim=_cfg_int(cfg, "inter_dim", 2048, minimum=128),
        experts=_cfg_int(cfg, "experts", 32, minimum=1),
        topk=_cfg_int(cfg, "topk", 4, minimum=1),
    )
    if task_cfg.topk > task_cfg.experts:
        task_cfg.topk = task_cfg.experts

    tokens = _cfg_int(cfg, "tokens", 256, minimum=1)
    seed = _cfg_int(cfg, "seed", 20260317, minimum=0)
    device = _cfg_device(cfg)

    data = make_inputs(tokens=tokens, cfg=task_cfg, device=device, seed=seed)
    return [
        data["input_q"],
        data["w1_q"],
        data["w2_q"],
        data["topk_weights"],
        data["topk_ids"],
        data["input_scale"],
        data["fc1_scale"],
        data["fc2_scale"],
    ]
