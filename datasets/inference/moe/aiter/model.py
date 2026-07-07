#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from types import SimpleNamespace
from typing import Any, Callable, List

try:
    import torch
except Exception:
    torch = None


ROUTE_GROUP_SIZE = 32
_AITER_RUNTIME_CACHE = None


def resolve_backends(args: Any) -> List[str]:
    if bool(getattr(args, "run_aiter", False)):
        return ["asm"]
    return []


def _load_aiter_runtime():
    global _AITER_RUNTIME_CACHE
    if _AITER_RUNTIME_CACHE is not None:
        return _AITER_RUNTIME_CACHE

    import aiter
    from aiter.ops.shuffle import shuffle_weight

    _AITER_RUNTIME_CACHE = {
        "aiter": aiter,
        "shuffle_weight": shuffle_weight,
    }
    return _AITER_RUNTIME_CACHE


def _validate_shared_input(shared_input: Any, *, seq_len: int, dim: int, topk: int):
    # Contract: all MoE baselines consume one unified input pack built in
    # python/harness/bench/moe_runtime.py. No random input construction here.
    for name in ("input_q", "topk_weights", "topk_ids", "input_scale"):
        if not hasattr(shared_input, name):
            raise ValueError(f"shared_input missing field: {name}")

    x = shared_input
    if x.input_q.shape != (seq_len, dim):
        raise ValueError(f"input_q shape mismatch: expected {(seq_len, dim)}, got {tuple(x.input_q.shape)}")
    if x.topk_weights.shape != (seq_len, topk):
        raise ValueError(
            f"topk_weights shape mismatch: expected {(seq_len, topk)}, got {tuple(x.topk_weights.shape)}"
        )
    if x.topk_ids.shape != (seq_len, topk):
        raise ValueError(f"topk_ids shape mismatch: expected {(seq_len, topk)}, got {tuple(x.topk_ids.shape)}")
    if x.input_scale.shape[0] != seq_len:
        raise ValueError(f"input_scale shape mismatch on dim0: expected {seq_len}, got {x.input_scale.shape[0]}")
    return x


def _validate_shared_weights(shared_weights: Any, *, experts: int, dim: int, inter_dim: int):
    # Same rule as shared_input: weights come from unified outer construction only.
    if not isinstance(shared_weights, dict):
        raise ValueError(f"shared_weights must be dict, got {type(shared_weights).__name__}")
    for name in ("w1_q", "w2_q", "fc1_scale", "fc2_scale"):
        if name not in shared_weights:
            raise ValueError(f"shared_weights missing key: {name}")

    w = shared_weights
    if tuple(w["w1_q"].shape) != (experts, inter_dim * 2, dim):
        raise ValueError(
            f"w1_q shape mismatch: expected {(experts, inter_dim * 2, dim)}, got {tuple(w['w1_q'].shape)}"
        )
    if tuple(w["w2_q"].shape) != (experts, dim, inter_dim):
        raise ValueError(
            f"w2_q shape mismatch: expected {(experts, dim, inter_dim)}, got {tuple(w['w2_q'].shape)}"
        )
    return w


def _build_asm_sorted_routes(
    *,
    topk_ids: "torch.Tensor",
    topk_weights: "torch.Tensor",
    experts: int,
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    tokens, topk = topk_ids.shape
    max_num_tokens_padded = tokens * topk + experts * ROUTE_GROUP_SIZE - topk
    max_num_m_blocks = (max_num_tokens_padded + ROUTE_GROUP_SIZE - 1) // ROUTE_GROUP_SIZE

    init_val = (topk << 24) | tokens
    sorted_token_ids = torch.full(
        (max_num_tokens_padded,),
        init_val,
        dtype=torch.int64,
        device=topk_ids.device,
    )
    sorted_weights = torch.zeros((max_num_tokens_padded,), dtype=torch.float32, device=topk_weights.device)
    sorted_expert_ids = torch.full(
        (max_num_m_blocks,),
        -1,
        dtype=torch.int64,
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

        route_ids = (slot_ids.to(torch.int64) << 24) | token_ids.to(torch.int64)
        sorted_token_ids[sorted_ids_begin : sorted_ids_begin + tokens_num] = route_ids
        sorted_weights[sorted_ids_begin : sorted_ids_begin + tokens_num] = topk_weights[token_ids, slot_ids]

        sorted_expert_ids_num = (tokens_num + ROUTE_GROUP_SIZE - 1) // ROUTE_GROUP_SIZE
        tokens_num_pad = sorted_expert_ids_num * ROUTE_GROUP_SIZE
        sorted_expert_ids[sorted_expert_ids_begin : sorted_expert_ids_begin + sorted_expert_ids_num] = expert

        sorted_ids_begin += tokens_num_pad
        sorted_expert_ids_begin += sorted_expert_ids_num

    num_valid_ids = torch.empty((2,), dtype=torch.int64, device=topk_ids.device)
    num_valid_ids[0] = sorted_ids_begin
    num_valid_ids[1] = tokens
    return sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids


def build_cases_for_seq(
    *,
    args: Any,
    seq_len: int,
    shared_input,
    shared_weights,
    backends: List[str] | None = None,
) -> List[Callable[[], None]]:
    if torch is None:
        raise RuntimeError("torch is required")

    resolved = list(backends) if backends is not None else resolve_backends(args)
    if not resolved:
        return []
    valid = {"asm"}
    unknown = [b for b in resolved if b not in valid]
    if unknown:
        raise ValueError(f"unsupported AITER backends: {unknown}")

    rt = _load_aiter_runtime()
    aiter = rt["aiter"]
    shuffle_weight = rt["shuffle_weight"]

    experts = int(args.experts)
    topk = int(args.topk)
    dim = int(args.dim)
    inter_dim = int(args.inter_dim)
    x = _validate_shared_input(shared_input, seq_len=seq_len, dim=dim, topk=topk)
    w = _validate_shared_weights(shared_weights, experts=experts, dim=dim, inter_dim=inter_dim)

    w1_shuf = shuffle_weight(w["w1_q"], (16, 16))
    w2_shuf = shuffle_weight(w["w2_q"], (16, 16))
    input_scale_t = x.input_scale.t().contiguous()

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids = _build_asm_sorted_routes(
        topk_ids=x.topk_ids.to(torch.int64).contiguous(),
        topk_weights=x.topk_weights.to(torch.float32).contiguous(),
        experts=experts,
    )
    sorted_ids_i32 = sorted_ids.to(torch.int32).contiguous()
    sorted_weights_f32 = sorted_weights.to(torch.float32).contiguous()
    sorted_expert_ids_i32 = sorted_expert_ids.to(torch.int32).contiguous()
    num_valid_ids_i32 = num_valid_ids.to(torch.int32).contiguous()

    out_asm = torch.zeros((seq_len, dim), dtype=torch.bfloat16, device=x.input_q.device)

    def run_asm() -> None:
        out_asm.zero_()
        aiter.fmoe_fp8_blockscale_g1u1(
            out_asm,
            x.input_q,
            w1_shuf,
            w2_shuf,
            sorted_ids_i32,
            sorted_weights_f32,
            sorted_expert_ids_i32,
            num_valid_ids_i32,
            topk,
            input_scale_t,
            w["fc1_scale"],
            w["fc2_scale"],
            "",
            128,
            128,
            None,
        )

    out: List[Callable[[], None]] = []
    if "asm" in resolved:
        out.append(run_asm)
    return out


class Model:
    def __init__(self, *, variant: str):
        if variant != "aiter":
            raise ValueError(f"unsupported AITER variant: {variant}")
        self.variant = variant

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
        args = SimpleNamespace(
            run_aiter=True,
            dim=dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            input_dtype=input_dtype,
        )
        return build_cases_for_seq(
            args=args,
            seq_len=seq_len,
            shared_input=shared_input,
            shared_weights=shared_weights,
        )


def _parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MoE AITER entry (called by benchmark_moe_unified_graph.py)")
    p.add_argument("--run-aiter", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse(argv)
    print(",".join(resolve_backends(args)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
