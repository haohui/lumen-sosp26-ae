#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from backend_attn import BACKENDS, build_shared_inputs, parse_dtype
from cli_utils import select_backend
from config import (
    ATTENTION_DEFAULTS,
    ATTENTION_WORKLOADS,
    TIMER_DEFAULTS,
    benchmark_root,
)

try:
    import torch
except Exception:
    torch = None


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
    p.add_argument("--warmup", type=int, default=TIMER_DEFAULTS["warmup"])
    p.add_argument("--repeat", type=int, default=TIMER_DEFAULTS["repeat"])
    p.add_argument("--graph-iters", type=int, default=TIMER_DEFAULTS["graph_iters"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seq_lens = args.seq_lens
    repo_root = Path(__file__).resolve().parents[2]
    attn_root = benchmark_root(repo_root) / "attn"

    if torch is None:
        raise RuntimeError("torch is required")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype_name = args.dtype.strip().lower()
    dtype = parse_dtype(dtype_name)
    causal = bool(ATTENTION_DEFAULTS["causal"])
    shared = build_shared_inputs(
        seq_lens=seq_lens,
        batch_size=args.batch_size,
        num_q_heads=args.num_q_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        device=device,
        dtype=dtype,
        seed=args.seed,
    )
    selected = select_backend(BACKENDS, args.backend)
    selected(
        attn_root=attn_root,
        seq_lens=seq_lens,
        shared=shared,
        device=device,
        dtype=dtype,
        dtype_name=dtype_name,
        batch_size=args.batch_size,
        num_q_heads=args.num_q_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        causal=causal,
        warmup=args.warmup,
        repeat=args.repeat,
        graph_iters=args.graph_iters,
    )


if __name__ == "__main__":
    main()
