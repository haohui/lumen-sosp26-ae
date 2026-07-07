#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from backend_moe import (
    BACKENDS,
    build_shared_inputs,
    build_shared_weights,
    parse_dtype,
    validate_config,
)
from cli_utils import select_backend
from config import MOE_DEFAULTS, MOE_WORKLOADS, TIMER_DEFAULTS, benchmark_root

try:
    import torch
except Exception:
    torch = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MoE benchmark with CUDA Graph timing")
    p.add_argument("--backend", choices=sorted(BACKENDS), default="aiter")
    p.add_argument("--seed", type=int, default=20260319)
    p.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=list(MOE_WORKLOADS),
    )
    p.add_argument("--dim", type=int, default=MOE_DEFAULTS["dim"])
    p.add_argument("--inter-dim", type=int, default=MOE_DEFAULTS["inter_dim"])
    p.add_argument("--experts", type=int, default=MOE_DEFAULTS["experts"])
    p.add_argument("--topk", type=int, default=MOE_DEFAULTS["topk"])
    p.add_argument(
        "--input-dtype",
        type=str,
        default=MOE_DEFAULTS["input_dtype"],
        choices=["fp8", "bf16"],
    )
    p.add_argument("--warmup", type=int, default=TIMER_DEFAULTS["warmup"])
    p.add_argument("--repeat", type=int, default=TIMER_DEFAULTS["repeat"])
    p.add_argument("--graph-iters", type=int, default=TIMER_DEFAULTS["graph_iters"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    token_counts = args.tokens
    validate_config(
        dim=args.dim,
        inter_dim=args.inter_dim,
        experts=args.experts,
        topk=args.topk,
    )

    if torch is None:
        raise RuntimeError("torch is required")

    repo_root = Path(__file__).resolve().parents[2]
    moe_root = benchmark_root(repo_root) / "moe"

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    input_dtype_name = args.input_dtype.strip().lower()
    input_dtype = parse_dtype(input_dtype_name)
    shared_inputs = build_shared_inputs(
        seq_lens=token_counts,
        dim=args.dim,
        experts=args.experts,
        topk=args.topk,
        input_dtype=input_dtype,
        device=device,
        seed=args.seed,
    )
    shared_weights = build_shared_weights(
        dim=args.dim,
        inter_dim=args.inter_dim,
        experts=args.experts,
        input_dtype=input_dtype,
        device=device,
        seed=args.seed,
    )

    selected = select_backend(BACKENDS, args.backend)
    selected(
        moe_root=moe_root,
        token_counts=token_counts,
        shared_inputs=shared_inputs,
        shared_weights=shared_weights,
        dim=args.dim,
        inter_dim=args.inter_dim,
        experts=args.experts,
        topk=args.topk,
        input_dtype_name=input_dtype_name,
        warmup=args.warmup,
        repeat=args.repeat,
        graph_iters=args.graph_iters,
    )


if __name__ == "__main__":
    main()
