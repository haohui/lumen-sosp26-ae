#!/usr/bin/env python3
from __future__ import annotations

import argparse

from backend_gemm import BACKENDS, build_shared_inputs, parse_dtype, run_backend
from cli_utils import add_timer_args, cuda_runtime
from config import GEMM_DEFAULTS, GEMM_WORKLOADS, TIMER_DEFAULTS, benchmark_root
from paths import resolve_repo_root


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
    p.add_argument(
        "--check-correctness",
        action="store_true",
        help="Compare each backend result with torch.matmul before benchmarking.",
    )
    add_timer_args(p, TIMER_DEFAULTS)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device, dtype_name, dtype = cuda_runtime(
        seed=args.seed, dtype_name=args.dtype, parse_dtype=parse_dtype
    )
    shared = build_shared_inputs(
        sizes=args.matrix_sizes,
        device=device,
        dtype=dtype,
        seed=args.seed,
    )
    run_backend(
        backend=args.backend,
        gemm_root=benchmark_root(resolve_repo_root()) / "gemm",
        matrix_sizes=args.matrix_sizes,
        shared=shared,
        device=device,
        dtype=dtype,
        dtype_name=dtype_name,
        warmup=args.warmup,
        repeat=args.repeat,
        graph_iters=args.graph_iters,
        check_correctness=args.check_correctness,
    )


if __name__ == "__main__":
    main()
