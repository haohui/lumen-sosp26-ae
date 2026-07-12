#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

from backend_moe import (
    BACKENDS,
    build_shared_inputs,
    build_shared_weights,
    parse_dtype,
    run_backend,
    validate_config,
)
from cli_utils import add_timer_args, cuda_runtime
from config import MOE_DEFAULTS, MOE_WORKLOADS, TIMER_DEFAULTS, benchmark_root
from paths import resolve_repo_root


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
    add_timer_args(p, TIMER_DEFAULTS)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    validate_config(
        dim=args.dim,
        inter_dim=args.inter_dim,
        experts=args.experts,
        topk=args.topk,
    )

    device, input_dtype_name, input_dtype = cuda_runtime(
        seed=args.seed, dtype_name=args.input_dtype, parse_dtype=parse_dtype
    )
    shared_inputs = build_shared_inputs(
        seq_lens=args.tokens,
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

    run_backend(
        backend=args.backend,
        moe_root=benchmark_root(resolve_repo_root()) / "moe",
        token_counts=args.tokens,
        shared_inputs=shared_inputs,
        shared_weights=shared_weights,
        device=device,
        input_dtype=input_dtype,
        dim=args.dim,
        inter_dim=args.inter_dim,
        experts=args.experts,
        topk=args.topk,
        input_dtype_name=input_dtype_name,
        warmup=args.warmup,
        repeat=args.repeat,
        graph_iters=args.graph_iters,
    )
    if args.backend == "lumen":
        # Avelang can abort during Python/C-extension teardown after a successful
        # run; exit directly once JSONL output has been flushed.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
