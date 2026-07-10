#!/usr/bin/env python3
from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any


def select_backend(
    backends: Mapping[str, Callable[..., Any]],
    name: str,
) -> Callable[..., Any]:
    try:
        return backends[name]
    except KeyError as e:
        expected = ", ".join(sorted(backends))
        raise RuntimeError(
            f"unknown backend {name!r}; expected one of: {expected}"
        ) from e


def emit_jsonl(record: Mapping[str, Any]) -> None:
    print(json.dumps(record, separators=(",", ":")), flush=True)


def add_timer_args(parser, defaults: Mapping[str, int]) -> None:
    parser.add_argument("--warmup", type=int, default=defaults["warmup"])
    parser.add_argument("--repeat", type=int, default=defaults["repeat"])
    parser.add_argument("--graph-iters", type=int, default=defaults["graph_iters"])


def cuda_runtime(*, seed: int, dtype_name: str, parse_dtype):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required") from exc

    torch.manual_seed(seed)
    normalized_dtype_name = dtype_name.strip().lower()
    dtype = parse_dtype(normalized_dtype_name)
    return torch.device("cuda"), normalized_dtype_name, dtype
