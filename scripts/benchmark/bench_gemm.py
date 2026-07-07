#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from backend_gemm import BACKENDS, build_shared_inputs, parse_dtype
from cli_utils import select_backend
from config import GEMM_DEFAULTS, GEMM_WORKLOADS, TIMER_DEFAULTS, benchmark_root

try:
    import torch
except Exception:
    torch = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GEMM benchmark with CUDA Graph timing")
    p.add_argument("--backend", choices=sorted(BACKENDS), default="triton")
    p.add_argument("--seed", type=int, default=20260312)
    p.add_argument(
        "--dtype",
        type=str,
        default=GEMM_DEFAULTS["dtype"],
        choices=["bf16", "fp16"],
    )
    p.add_argument(
        "--matrix-sizes",
        type=int,
        nargs="+",
        default=list(GEMM_WORKLOADS),
    )
    p.add_argument("--warmup", type=int, default=TIMER_DEFAULTS["warmup"])
    p.add_argument("--repeat", type=int, default=TIMER_DEFAULTS["repeat"])
    p.add_argument("--graph-iters", type=int, default=TIMER_DEFAULTS["graph_iters"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    matrix_sizes = args.matrix_sizes
    repo_root = Path(__file__).resolve().parents[2]
    gemm_root = benchmark_root(repo_root) / "gemm"

    if torch is None:
        raise RuntimeError("torch is required")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype_name = args.dtype.strip().lower()
    dtype = parse_dtype(dtype_name)
    shared = build_shared_inputs(
        sizes=matrix_sizes,
        device=device,
        dtype=dtype,
        seed=args.seed,
    )
    selected = select_backend(BACKENDS, args.backend)
    selected(
        gemm_root=gemm_root,
        matrix_sizes=matrix_sizes,
        shared=shared,
        device=device,
        dtype=dtype,
        dtype_name=dtype_name,
        warmup=args.warmup,
        repeat=args.repeat,
        graph_iters=args.graph_iters,
    )


if __name__ == "__main__":
    main()
