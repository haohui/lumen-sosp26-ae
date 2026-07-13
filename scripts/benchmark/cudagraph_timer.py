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
    replay_warmup: int | None = None,
    measure_trials: int = 2,
) -> CUDAGraphTimingResult:
    warmup = max(0, int(warmup))
    repeat = max(1, int(repeat))
    graph_iters = max(1, int(graph_iters))
    if replay_warmup is None:
        replay_warmup = max(1, min(5, warmup))
    else:
        replay_warmup = max(0, int(replay_warmup))
    measure_trials = max(1, int(measure_trials))

    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        captured_outputs = []
        with torch.cuda.graph(graph):
            for _ in range(graph_iters):
                captured_outputs.append(fn())
        torch.cuda.synchronize()

        for _ in range(replay_warmup):
            graph.replay()
        torch.cuda.synchronize()

        elapsed_ms = []
        for _ in range(measure_trials):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repeat):
                graph.replay()
            end.record()
            torch.cuda.synchronize()
            elapsed_ms.append(float(start.elapsed_time(end)))

    return CUDAGraphTimingResult(
        mean_ms=min(elapsed_ms) / float(repeat * graph_iters)
    )
