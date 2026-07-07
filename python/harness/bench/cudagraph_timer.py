#!/usr/bin/env python3
"""Publication-grade CUDA Graph timing utility."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import torch


@dataclass
class CUDAGraphTimingResult:
    samples_ms: list[float]
    mean_ms: float
    median_ms: float
    stdev_ms: float
    min_ms: float
    max_ms: float
    p10_ms: float
    p90_ms: float
    cv: float
    warmup_calls: int
    total_calls_per_sample: int
    graph_iters: int
    num_replays: int
    eager_probe_ms: float | None
    suspicious: bool
    suspicious_reason: str | None


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    if q <= 0.0:
        return float(min(values))
    if q >= 1.0:
        return float(max(values))
    sv = sorted(values)
    pos = (len(sv) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sv[lo])
    w = pos - lo
    return float((1.0 - w) * sv[lo] + w * sv[hi])


def _event_probe_ms(
    fn: Callable[[], None],
    *,
    device: "torch.device",
    probe_calls: int,
) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(device=device)
    start.record()
    for _ in range(max(1, probe_calls)):
        fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / float(max(1, probe_calls))


def benchmark_with_cudagraph(
    fn: Callable[..., Any],
    *,
    device: "torch.device",
    fn_args: Sequence[Any] = (),
    fn_kwargs: Mapping[str, Any] | None = None,
    warmup: int = 10,
    warmup_ms: float | None = None,
    graph_iters: int = 10,
    pre_capture_iters: int = 3,
    trial_count: int = 9,
    min_measure_ms: float = 200.0,
    min_graph_ms: float = 300.0,
    min_replays: int = 5,
    max_replays: int = 200000,
    max_graph_iters: int = 16384,
    fixed_repeat_calls: int | None = None,
    use_default_stream: bool = True,
    setup_fn: Callable[[], None] | None = None,
    probe_calls: int = 20,
) -> CUDAGraphTimingResult:
    """Time fn with CUDA Graph replay and robust statistics."""
    call_kwargs = dict(fn_kwargs or {})

    def call_once() -> None:
        fn(*fn_args, **call_kwargs)

    if setup_fn is not None:
        setup_fn()

    # Warmup eager path first, then synchronize to isolate capture/measure.
    warmup_calls = max(1, int(warmup))
    for _ in range(warmup_calls):
        call_once()
    # Extend warmup to satisfy target warmup duration if requested.
    if warmup_ms is not None and warmup_ms > 0.0:
        est_ms = _event_probe_ms(call_once, device=device, probe_calls=5)
        if est_ms > 0.0:
            target_calls = int(math.ceil(float(warmup_ms) / est_ms))
            target_calls = max(target_calls, 1)
            extra_calls = max(0, target_calls - warmup_calls)
            for _ in range(extra_calls):
                call_once()
            warmup_calls += extra_calls
    torch.cuda.synchronize(device=device)

    min_replays = max(1, int(min_replays))
    max_replays = max(1, int(max_replays))
    if max_replays < min_replays:
        max_replays = min_replays

    # Long-graph policy:
    # 1) Use an eager probe to estimate per-call latency.
    # 2) Inflate graph_iters so one capture contains enough calls (min_graph_ms target).
    # 3) Bound total replays by [min_replays, max_replays].
    effective_graph_iters = max(1, int(graph_iters))
    est_call_ms = _event_probe_ms(call_once, device=device, probe_calls=20)
    if est_call_ms <= 0.0:
        est_call_ms = float("inf")

    if fixed_repeat_calls is not None:
        desired_total_calls = max(1, int(fixed_repeat_calls))
    else:
        # Keep total graph executions long enough to cover min_measure_ms.
        desired_total_calls = int(math.ceil(float(max(1.0, min_measure_ms)) / est_call_ms))
        desired_total_calls = max(1, desired_total_calls)

    # Enforce a minimum graph size target (by call count) so timing is less noisy.
    if min_graph_ms > 0.0 and est_call_ms < float("inf"):
        min_graph_calls = max(1, int(math.ceil(float(min_graph_ms) / est_call_ms)))
    else:
        min_graph_calls = 1

    if fixed_repeat_calls is None:
        # Keep per-sample replay count controlled by min_measure_ms and avoid too many replays.
        min_graph_calls = max(min_graph_calls, int(math.ceil(float(desired_total_calls) / float(max_replays))))
    effective_graph_iters = max(effective_graph_iters, min_graph_calls)
    if max_graph_iters > 0:
        effective_graph_iters = min(effective_graph_iters, int(max_graph_iters))

    if fixed_repeat_calls is None:
        num_replays = int(math.ceil(float(desired_total_calls) / float(effective_graph_iters)))
        if num_replays < min_replays:
            num_replays = min_replays
        num_replays = min(num_replays, max_replays)
    else:
        num_replays = max(1, int(math.ceil(float(desired_total_calls) / float(effective_graph_iters))))
        num_replays = max(min_replays, min(num_replays, max_replays))

    graph = torch.cuda.CUDAGraph()
    graph_stream = (
        torch.cuda.default_stream(device=device)
        if use_default_stream
        else torch.cuda.Stream(device=device)
    )
    caller_stream = torch.cuda.current_stream(device=device)
    total_calls = max(1, num_replays * effective_graph_iters)
    samples: list[float] = []

    try:
        graph_stream.wait_stream(caller_stream)
        with torch.cuda.stream(graph_stream):
            for _ in range(max(0, int(pre_capture_iters))):
                call_once()
            with torch.cuda.graph(graph):
                for _ in range(effective_graph_iters):
                    call_once()
        caller_stream.wait_stream(graph_stream)
        torch.cuda.synchronize(device=device)

        # Trial loop.
        for _ in range(max(1, int(trial_count))):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(graph_stream):
                start.record()
                for _ in range(num_replays):
                    graph.replay()
                end.record()
            caller_stream.wait_stream(graph_stream)
            torch.cuda.synchronize(device=device)
            total_ms = float(start.elapsed_time(end))
            samples.append(total_ms / float(total_calls))
    except RuntimeError as e:
        raise RuntimeError(f"CUDA graph capture/replay failed: {type(e).__name__}: {e}") from e

    eager_probe_ms = _event_probe_ms(call_once, device=device, probe_calls=probe_calls)
    median_ms = float(statistics.median(samples))
    if median_ms <= 0.0:
        raise RuntimeError("empty CUDA graph detected: replay latency is zero")

    mean_ms = float(statistics.fmean(samples))
    stdev_ms = 0.0 if len(samples) <= 1 else float(statistics.stdev(samples))
    min_ms = float(min(samples))
    max_ms = float(max(samples))
    p10_ms = _quantile(samples, 0.10)
    p90_ms = _quantile(samples, 0.90)
    cv = 0.0 if mean_ms == 0.0 else float(stdev_ms / mean_ms)

    return CUDAGraphTimingResult(
        samples_ms=samples,
        mean_ms=mean_ms,
        median_ms=median_ms,
        stdev_ms=stdev_ms,
        min_ms=min_ms,
        max_ms=max_ms,
        p10_ms=p10_ms,
        p90_ms=p90_ms,
        cv=cv,
        warmup_calls=warmup_calls,
        total_calls_per_sample=total_calls,
        graph_iters=effective_graph_iters,
        num_replays=num_replays,
        eager_probe_ms=eager_probe_ms,
        suspicious=False,
        suspicious_reason=None,
    )
