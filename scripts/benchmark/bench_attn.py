#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

from backend_attn import BACKENDS, build_shared_inputs, parse_dtype, run_backend
from cli_utils import add_timer_args, cuda_runtime
from config import (
    ATTENTION_DEFAULTS,
    ATTENTION_WORKLOADS,
    TIMER_DEFAULTS,
    benchmark_root,
)
from paths import resolve_repo_root


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Attention benchmark with CUDA Graph timing"
    )
    p.add_argument("--backend", choices=sorted(BACKENDS), default="triton")
    p.add_argument("--seed", type=int, default=20260314)
    p.add_argument(
        "--dtype",
        type=str,
        default=ATTENTION_DEFAULTS["dtype"],
        choices=["bf16", "fp16"],
    )
    p.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=list(ATTENTION_WORKLOADS),
    )
    p.add_argument("--batch-size", type=int, default=ATTENTION_DEFAULTS["batch_size"])
    p.add_argument("--num-q-heads", type=int, default=ATTENTION_DEFAULTS["num_q_heads"])
    p.add_argument(
        "--num-kv-heads",
        type=int,
        default=ATTENTION_DEFAULTS["num_kv_heads"],
    )
    p.add_argument("--head-dim", type=int, default=ATTENTION_DEFAULTS["head_dim"])
    add_timer_args(p, TIMER_DEFAULTS)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device, dtype_name, dtype = cuda_runtime(
        seed=args.seed, dtype_name=args.dtype, parse_dtype=parse_dtype
    )
    shared = build_shared_inputs(
        seq_lens=args.seq_lens,
        batch_size=args.batch_size,
        num_q_heads=args.num_q_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        device=device,
        dtype=dtype,
        seed=args.seed,
    )
    run_backend(
        backend=args.backend,
        attn_root=benchmark_root(resolve_repo_root()) / "attention",
        seq_lens=args.seq_lens,
        shared=shared,
        device=device,
        dtype=dtype,
        dtype_name=dtype_name,
        batch_size=args.batch_size,
        num_q_heads=args.num_q_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        causal=bool(ATTENTION_DEFAULTS["causal"]),
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
