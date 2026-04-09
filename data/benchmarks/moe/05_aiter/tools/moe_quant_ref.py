#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

BLOCK_N = 128
BLOCK_K = 128
ROUTE_GROUP_SIZE = 32
FP8_E4M3_MAX = 240.0


@dataclass
class MoeConfig:
    dim: int = 7168
    inter_dim: int = 2048
    experts: int = 32
    topk: int = 4


class MoeSingleOpTorchRef:
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
        return scaled_blocks.permute(0, 1, 3, 2, 4).reshape(experts, rows, cols).contiguous()

    def _dequantize_weight_expert(
        self,
        weight_q_expert: torch.Tensor,
        scale_expert: torch.Tensor,
    ) -> torch.Tensor:
        rows, cols = weight_q_expert.shape
        row_blocks = rows // self.block_n
        col_blocks = cols // self.block_k
        blocks = (
            weight_q_expert.to(torch.float32)
            .view(row_blocks, self.block_n, col_blocks, self.block_k)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        scaled_blocks = blocks * scale_expert.to(torch.float32).view(row_blocks, col_blocks, 1, 1)
        return scaled_blocks.permute(0, 2, 1, 3).reshape(rows, cols).contiguous()

    def _quantize_and_shuffle(self, stage1_act: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rows, cols = stage1_act.shape
        if cols % self.block_k == 0:
            x = stage1_act.to(torch.float32).view(rows, cols // self.block_k, self.block_k)
        else:
            x = stage1_act.to(torch.float32).view(rows, 1, cols)
        amax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
        quant_scale = FP8_E4M3_MAX / amax
        dequant_scale = 1.0 / quant_scale

        x_scaled = x * quant_scale
        if hasattr(torch, "float8_e4m3fnuz"):
            quantized = x_scaled.to(torch.float8_e4m3fnuz)
        else:
            quantized = torch.clamp(torch.round(x_scaled), -FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float32)
        return quantized, dequant_scale

    def _dequantize_stage_bridge(self, quantized_act: torch.Tensor, dequant_scale: torch.Tensor) -> torch.Tensor:
        return (quantized_act.to(torch.float32) * dequant_scale).reshape(quantized_act.shape[0], -1).contiguous()

    def _build_sorted_routes(
        self,
        topk_ids_i64: torch.Tensor,
        topk_weights_f: torch.Tensor,
        num_experts: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens, topk = topk_ids_i64.shape
        max_num_tokens_padded = tokens * topk + num_experts * ROUTE_GROUP_SIZE - topk
        max_num_m_blocks = (max_num_tokens_padded + ROUTE_GROUP_SIZE - 1) // ROUTE_GROUP_SIZE

        init_val = (topk << 24) | tokens
        sorted_token_ids = torch.full(
            (max_num_tokens_padded,),
            init_val,
            dtype=torch.int64,
            device=topk_ids_i64.device,
        )
        sorted_weights = torch.zeros((max_num_tokens_padded,), dtype=torch.float32, device=topk_weights_f.device)
        sorted_expert_ids = torch.full(
            (max_num_m_blocks,),
            -1,
            dtype=torch.int64,
            device=topk_ids_i64.device,
        )

        sorted_ids_begin = 0
        sorted_expert_ids_begin = 0
        for expert in range(num_experts):
            mask = topk_ids_i64.eq(expert)
            if not bool(mask.any()):
                continue

            token_ids, slot_ids = torch.nonzero(mask, as_tuple=True)
            tokens_num = int(token_ids.numel())

            route_ids = (slot_ids.to(torch.int64) << 24) | token_ids.to(torch.int64)
            sorted_token_ids[sorted_ids_begin : sorted_ids_begin + tokens_num] = route_ids
            sorted_weights[sorted_ids_begin : sorted_ids_begin + tokens_num] = topk_weights_f[token_ids, slot_ids]

            sorted_expert_ids_num = (tokens_num + ROUTE_GROUP_SIZE - 1) // ROUTE_GROUP_SIZE
            tokens_num_pad = sorted_expert_ids_num * ROUTE_GROUP_SIZE
            sorted_expert_ids[sorted_expert_ids_begin : sorted_expert_ids_begin + sorted_expert_ids_num] = expert

            sorted_ids_begin += tokens_num_pad
            sorted_expert_ids_begin += sorted_expert_ids_num

        num_valid_ids = torch.empty((2,), dtype=torch.int64, device=topk_ids_i64.device)
        num_valid_ids[0] = sorted_ids_begin
        num_valid_ids[1] = tokens
        return sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids

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
        experts, _, inter_dim = w2_q.shape

        input_f = self._dequantize_input(input_q, input_scale)

        topk_ids_i64 = topk_ids.to(torch.int64).contiguous()
        topk_weights_f = topk_weights.to(torch.float32).contiguous()

        out = torch.zeros((tokens, model_dim), dtype=torch.float32, device=input_q.device)
        sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids = self._build_sorted_routes(
            topk_ids_i64=topk_ids_i64,
            topk_weights_f=topk_weights_f,
            num_experts=experts,
        )
        num_valid_routes = int(num_valid_ids[0].item())
        if num_valid_routes == 0:
            return out.to(self.out_dtype)

        valid_route_idx = torch.arange(num_valid_routes, dtype=torch.int64, device=input_q.device)
        route_group_idx = torch.div(valid_route_idx, ROUTE_GROUP_SIZE, rounding_mode="floor")
        route_expert_ids = sorted_expert_ids.index_select(0, route_group_idx)
        route_token_ids = sorted_token_ids[:num_valid_routes] & 0x00FFFFFF
        route_weights = sorted_weights[:num_valid_routes]
        route_token_valid = route_token_ids.lt(tokens)

        for expert in range(experts):
            route_mask = route_expert_ids.eq(expert) & route_token_valid
            if not bool(route_mask.any()):
                continue

            token_ids = route_token_ids[route_mask]
            weights_e = route_weights[route_mask].unsqueeze(-1)
            x_e = input_f.index_select(0, token_ids)

            w1_e = self._dequantize_weight_expert(w1_q[expert], fc1_scale[expert])
            stage1 = x_e @ w1_e.transpose(0, 1)
            gate, up = stage1.split(inter_dim, dim=-1)
            activated = F.silu(gate) * up
            bridge_q, bridge_dq = self._quantize_and_shuffle(activated)
            activated = self._dequantize_stage_bridge(bridge_q, bridge_dq)

            w2_e = self._dequantize_weight_expert(w2_q[expert], fc2_scale[expert])
            stage2 = activated @ w2_e.transpose(0, 1)
            out.index_add_(0, token_ids, stage2 * weights_e)

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

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        raw_cfg = globals().get("EVAL_CONFIG", {})
        cfg = raw_cfg if isinstance(raw_cfg, dict) else {}

        def as_int(key: str, default: int, minimum: int = 1) -> int:
            raw = cfg.get(key, default)
            try:
                val = int(raw)
            except Exception:
                val = int(default)
            return max(minimum, val)

        self.cfg = MoeConfig(
            dim=as_int("dim", 7168, minimum=128),
            inter_dim=as_int("inter_dim", 2048, minimum=128),
            experts=as_int("experts", 32, minimum=1),
            topk=as_int("topk", 4, minimum=1),
        )
        if self.cfg.topk > self.cfg.experts:
            self.cfg.topk = self.cfg.experts
        self.ref_impl = MoeSingleOpTorchRef(out_dtype=torch.bfloat16)

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
    raw_cfg = globals().get("EVAL_CONFIG", {})
    cfg = raw_cfg if isinstance(raw_cfg, dict) else {}

    def as_int(key: str, default: int, minimum: int = 1) -> int:
        raw = cfg.get(key, default)
        try:
            val = int(raw)
        except Exception:
            val = int(default)
        return max(minimum, val)

    task_cfg = MoeConfig(
        dim=as_int("dim", 7168, minimum=128),
        inter_dim=as_int("inter_dim", 2048, minimum=128),
        experts=as_int("experts", 32, minimum=1),
        topk=as_int("topk", 4, minimum=1),
    )
    if task_cfg.topk > task_cfg.experts:
        task_cfg.topk = task_cfg.experts

    tokens = as_int("tokens", 1024, minimum=1)
    seed = as_int("seed", 20260317, minimum=0)
    d = str(cfg.get("device", "cuda:0")).strip()
    if not d:
        d = "cuda:0"
    if d.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"EVAL_CONFIG.device={d!r} requires CUDA/HIP runtime")
    device = torch.device(d)

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


def run(
    input_q,
    w1_q,
    w2_q,
    topk_weights,
    topk_ids,
    input_scale,
    fc1_scale,
    fc2_scale,
):
    model = Model()
    with torch.inference_mode():
        return model(
            input_q,
            w1_q,
            w2_q,
            topk_weights,
            topk_ids,
            input_scale,
            fc1_scale,
            fc2_scale,
        )
