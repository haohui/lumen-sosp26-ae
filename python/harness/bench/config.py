#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

OUTPUT_ROOT = Path(__file__).resolve().parent / "output"


def benchmark_root(repo_root: Path) -> Path:
    return Path(os.environ.get("AE_BENCHMARK_ROOT", repo_root / "data" / "benchmarks"))

WORKLOADS = [1024, 2048, 4096, 8192, 16384]
GEMM_WORKLOADS = list(WORKLOADS)
ATTENTION_WORKLOADS = list(WORKLOADS)
MOE_WORKLOADS = list(WORKLOADS)

GEMM_DEFAULTS = {
    "dtype": "bf16",
}

ATTENTION_DEFAULTS = {
    "dtype": "bf16",
    "batch_size": 16,
    "num_q_heads": 8,
    "num_kv_heads": 1,
    "head_dim": 128,
    "causal": True,
}

MOE_DEFAULTS = {
    "dim": 7168,
    "inter_dim": 2048,
    "experts": 32,
    "topk": 4,
    "input_dtype": "fp8",
}

TIMER_DEFAULTS = {
    "warmup": 10,
    "warmup_ms": 1000.0,
    "repeat_ms": 5000.0,
    "min_graph_ms": 300.0,
    "graph_iters": 1,
    "timer_trials": 9,
    "min_replays": 1,
    "max_replays": 10,
    "max_graph_iters": 100,
    "pre_capture_iters": 3,
}

BASELINE_COLUMNS = [
    "Lumen",
    "HipBlasLt",
    "HipKittens",
    "AITER",
    "Triton",
]
