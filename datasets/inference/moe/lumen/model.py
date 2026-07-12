#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable, List

import torch


DIRECT_EXIT_AFTER_SUCCESS = True
ROUTE_GROUP_SIZE = 32
_THIS_DIR = Path(__file__).resolve().parent
_MOE_MODULE = "fused_moe.py"


def _load_kernel_module():
    path = _THIS_DIR / _MOE_MODULE
    module_name = f"lumen_moe_{path.stem}_{abs(hash(str(path.resolve()))):x}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import LUMEN MoE module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _build_sorted_routes(
    *,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens, topk = topk_ids.shape
    max_num_tokens_padded = tokens * topk + experts * ROUTE_GROUP_SIZE - topk
    max_num_m_blocks = (
        max_num_tokens_padded + ROUTE_GROUP_SIZE - 1
    ) // ROUTE_GROUP_SIZE

    init_val = (topk << 24) | tokens
    sorted_token_ids = torch.full(
        (max_num_tokens_padded,),
        init_val,
        dtype=torch.int32,
        device=topk_ids.device,
    )
    sorted_weights = torch.zeros(
        (max_num_tokens_padded,),
        dtype=torch.float32,
        device=topk_weights.device,
    )
    sorted_expert_ids = torch.full(
        (max_num_m_blocks,),
        -1,
        dtype=torch.int32,
        device=topk_ids.device,
    )

    sorted_ids_begin = 0
    sorted_expert_ids_begin = 0
    for expert in range(experts):
        mask = topk_ids.eq(expert)
        if not bool(mask.any()):
            continue
        token_ids, slot_ids = torch.nonzero(mask, as_tuple=True)
        tokens_num = int(token_ids.numel())

        route_ids = (slot_ids.to(torch.int32) << 24) | token_ids.to(torch.int32)
        sorted_token_ids[sorted_ids_begin : sorted_ids_begin + tokens_num] = route_ids
        sorted_weights[sorted_ids_begin : sorted_ids_begin + tokens_num] = topk_weights[
            token_ids,
            slot_ids,
        ]

        sorted_expert_ids_num = (tokens_num + ROUTE_GROUP_SIZE - 1) // ROUTE_GROUP_SIZE
        tokens_num_pad = sorted_expert_ids_num * ROUTE_GROUP_SIZE
        sorted_expert_ids[
            sorted_expert_ids_begin : sorted_expert_ids_begin + sorted_expert_ids_num
        ] = expert

        sorted_ids_begin += tokens_num_pad
        sorted_expert_ids_begin += sorted_expert_ids_num

    num_valid_ids = torch.empty((2,), dtype=torch.int32, device=topk_ids.device)
    num_valid_ids[0] = sorted_ids_begin
    num_valid_ids[1] = tokens
    return (
        sorted_token_ids.contiguous(),
        sorted_weights.contiguous(),
        sorted_expert_ids.contiguous(),
        num_valid_ids.contiguous(),
    )


def _validate_shared_input(shared_input: Any, *, seq_len: int, dim: int, topk: int):
    for name in ("input_q", "topk_weights", "topk_ids", "input_scale"):
        if not hasattr(shared_input, name):
            raise ValueError(f"shared_input missing field: {name}")
    x = shared_input
    if tuple(x.input_q.shape) != (seq_len, dim):
        raise ValueError(
            f"input_q shape mismatch: expected {(seq_len, dim)}, "
            f"got {tuple(x.input_q.shape)}"
        )
    if tuple(x.topk_weights.shape) != (seq_len, topk):
        raise ValueError(
            f"topk_weights shape mismatch: expected {(seq_len, topk)}, "
            f"got {tuple(x.topk_weights.shape)}"
        )
    if tuple(x.topk_ids.shape) != (seq_len, topk):
        raise ValueError(
            f"topk_ids shape mismatch: expected {(seq_len, topk)}, "
            f"got {tuple(x.topk_ids.shape)}"
        )
    return x


def _validate_shared_weights(
    shared_weights: Any,
    *,
    experts: int,
    dim: int,
    inter_dim: int,
):
    if not isinstance(shared_weights, dict):
        raise ValueError(
            f"shared_weights must be dict, got {type(shared_weights).__name__}"
        )
    for name in ("w1_q", "w2_q", "fc1_scale", "fc2_scale"):
        if name not in shared_weights:
            raise ValueError(f"shared_weights missing key: {name}")
    if tuple(shared_weights["w1_q"].shape) != (experts, inter_dim * 2, dim):
        raise ValueError(
            f"w1_q shape mismatch: expected {(experts, inter_dim * 2, dim)}, "
            f"got {tuple(shared_weights['w1_q'].shape)}"
        )
    if tuple(shared_weights["w2_q"].shape) != (experts, dim, inter_dim):
        raise ValueError(
            f"w2_q shape mismatch: expected {(experts, dim, inter_dim)}, "
            f"got {tuple(shared_weights['w2_q'].shape)}"
        )
    return shared_weights


class Model:
    def __init__(self, *, variant: str):
        if variant != "lumen":
            raise ValueError(f"unsupported LUMEN variant: {variant}")
        self.variant = variant
        self._mod = None
        self._out_cache = {}

    def _kernel_module(self):
        if self._mod is None:
            self._mod = _load_kernel_module()
        return self._mod

    def build_cases(
        self,
        *,
        seq_len: int,
        shared_input: Any,
        shared_weights: dict[str, torch.Tensor],
        dim: int,
        inter_dim: int,
        experts: int,
        topk: int,
        input_dtype: str,
    ) -> List[Callable[[], None]]:
        if input_dtype != "fp8":
            raise ValueError("LUMEN MoE benchmark currently supports only fp8 input")

        x = _validate_shared_input(
            shared_input,
            seq_len=seq_len,
            dim=dim,
            topk=topk,
        )
        w = _validate_shared_weights(
            shared_weights,
            experts=experts,
            dim=dim,
            inter_dim=inter_dim,
        )
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids = (
            _build_sorted_routes(
                topk_ids=x.topk_ids,
                topk_weights=x.topk_weights,
                experts=experts,
            )
        )
        key = (seq_len, dim, x.input_q.device)
        out = self._out_cache.get(key)
        if out is None:
            out = torch.empty((seq_len, dim), dtype=torch.bfloat16, device=x.input_q.device)
            self._out_cache[key] = out

        fn = getattr(self._kernel_module(), "fused_moe_fp8_blockscale_g1u1")

        def run_lumen() -> None:
            fn(
                x.input_q,
                w["w1_q"],
                w["w2_q"],
                sorted_ids,
                sorted_weights,
                sorted_expert_ids,
                num_valid_ids,
                topk,
                x.input_scale,
                w["fc1_scale"],
                w["fc2_scale"],
                out=out,
            )

        return [run_lumen]
