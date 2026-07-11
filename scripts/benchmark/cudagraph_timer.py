#!/usr/bin/env python3
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class CUDAGraphTimingResult:
    mean_ms: float


def benchmark_with_cudagraph(
    fn: Callable[[], Any],
    *,
    warmup: int,
    repeat: int,
    graph_iters: int,
) -> CUDAGraphTimingResult:
    warmup = max(0, int(warmup))
    repeat = max(1, int(repeat))
    graph_iters = max(1, int(graph_iters))

    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(graph_iters):
                fn()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeat):
            graph.replay()
        end.record()
        torch.cuda.synchronize()

    elapsed_ms = float(start.elapsed_time(end))
    return CUDAGraphTimingResult(mean_ms=elapsed_ms / float(repeat * graph_iters))
