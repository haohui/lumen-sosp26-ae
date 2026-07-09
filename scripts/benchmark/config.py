#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


def benchmark_root(repo_root: Path) -> Path:
    return repo_root / "datasets" / "inference"


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
    "repeat": 100,
    "graph_iters": 1,
}
