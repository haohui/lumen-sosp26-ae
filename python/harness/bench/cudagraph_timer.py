#!/usr/bin/env python3
"""Publication-grade CUDA Graph timing utility."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import torch


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


@dataclass
class CUDAGraphTimingResult:
    samples_ms: list[float]
    warmup_calls: int
    total_calls_per_sample: int
    graph_iters: int
    num_replays: int
    eager_probe_ms: float | None
    suspicious: bool
    suspicious_reason: str | None

    @property
    def mean_ms(self) -> float:
        return float(statistics.fmean(self.samples_ms))

    @property
    def median_ms(self) -> float:
        return float(statistics.median(self.samples_ms))

    @property
    def stdev_ms(self) -> float:
        if len(self.samples_ms) <= 1:
            return 0.0
        return float(statistics.stdev(self.samples_ms))

    @property
    def min_ms(self) -> float:
        return float(min(self.samples_ms))

    @property
    def max_ms(self) -> float:
        return float(max(self.samples_ms))

    @property
    def p10_ms(self) -> float:
        return _quantile(self.samples_ms, 0.10)

    @property
    def p90_ms(self) -> float:
        return _quantile(self.samples_ms, 0.90)

    @property
    def cv(self) -> float:
        mean = self.mean_ms
        if mean == 0.0:
            return 0.0
        return float(self.stdev_ms / mean)


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
    min_replays: int = 5,
    max_replays: int = 200000,
    fixed_repeat_calls: int | None = None,
    use_default_stream: bool = True,
    setup_fn: Callable[[], None] | None = None,
    probe_calls: int = 20,
    suspicious_ratio_threshold: float = 0.25,
    allow_suspicious: bool = True,
) -> CUDAGraphTimingResult:
    """Time fn with CUDA Graph replay and robust statistics.

    The function captures one graph and measures replay latency across multiple
    independent trials. Each trial records one event range around many replays
    to reduce event quantization noise.
    """
    call_kwargs = dict(fn_kwargs or {})

    def _call_once() -> None:
        fn(*fn_args, **call_kwargs)

    if setup_fn is not None:
        setup_fn()

    # Warmup eager path first, then synchronize to isolate capture/measure.
    warmup_calls = max(1, warmup)
    for _ in range(warmup_calls):
        _call_once()
    # Extend warmup to satisfy target warmup duration if requested.
    if warmup_ms is not None and warmup_ms > 0.0:
        est_ms = _event_probe_ms(_call_once, device=device, probe_calls=5)
        if est_ms > 0.0:
            target_calls = int(math.ceil(float(warmup_ms) / est_ms))
            target_calls = max(target_calls, 1)
            extra_calls = max(0, target_calls - warmup_calls)
            for _ in range(extra_calls):
                _call_once()
            warmup_calls += extra_calls
    torch.cuda.synchronize(device=device)

    min_replays = max(1, int(min_replays))
    max_replays = max(1, int(max_replays))
    if max_replays < min_replays:
        max_replays = min_replays

    # "Long graph + bounded replay" policy:
    # - Repeat budget is still controlled by min_measure_ms (repeat_ms).
    # - We first estimate total calls needed to reach the budget.
    # - Then we enlarge graph_iters so each replay contains a long graph,
    #   while keeping replay count >= min_replays (commonly 5).
    # This avoids tiny-graph/high-replay timing bias for very fast kernels.
    effective_graph_iters = max(1, int(graph_iters))
    if fixed_repeat_calls is not None:
        desired_total_calls = max(1, int(fixed_repeat_calls))
    else:
        est_call_ms = _event_probe_ms(_call_once, device=device, probe_calls=5)
        if est_call_ms > 0.0:
            desired_total_calls = int(math.ceil(float(max(1.0, min_measure_ms)) / est_call_ms))
            desired_total_calls = max(desired_total_calls, 1)
        else:
            desired_total_calls = max(1, effective_graph_iters * min_replays)

    if fixed_repeat_calls is None:
        # Make graph as large as needed so replay count tends toward min_replays.
        # graph_iters acts as a lower bound; auto policy can increase it.
        effective_graph_iters = max(
            effective_graph_iters,
            int(math.ceil(float(desired_total_calls) / float(min_replays))),
        )

    num_replays = int(math.ceil(float(desired_total_calls) / float(effective_graph_iters)))
    num_replays = max(min_replays, min(num_replays, max_replays))

    graph = torch.cuda.CUDAGraph()
    graph_stream = (
        torch.cuda.default_stream(device=device)
        if use_default_stream
        else torch.cuda.Stream(device=device)
    )
    caller_stream = torch.cuda.current_stream(device=device)
    graph_stream.wait_stream(caller_stream)
    with torch.cuda.stream(graph_stream):
        for _ in range(max(0, pre_capture_iters)):
            _call_once()
        with torch.cuda.graph(graph):
            for _ in range(effective_graph_iters):
                _call_once()
    caller_stream.wait_stream(graph_stream)
    torch.cuda.synchronize(device=device)

    total_calls = max(1, num_replays * effective_graph_iters)

    # Trial loop.
    samples: list[float] = []
    for _ in range(max(1, trial_count)):
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

    # Detect suspiciously small graph numbers (often empty graph / wrong stream).
    eager_probe_ms = _event_probe_ms(_call_once, device=device, probe_calls=probe_calls)
    median_ms = float(statistics.median(samples))
    suspicious = False
    suspicious_reason: str | None = None
    if eager_probe_ms > 0.0 and median_ms < suspicious_ratio_threshold * eager_probe_ms:
        suspicious = True
        suspicious_reason = (
            f"graph median {median_ms:.6f} ms is < "
            f"{suspicious_ratio_threshold:.2f}x eager probe {eager_probe_ms:.6f} ms"
        )
        if not allow_suspicious:
            raise RuntimeError(f"suspicious CUDA graph timing: {suspicious_reason}")

    return CUDAGraphTimingResult(
        samples_ms=samples,
        warmup_calls=warmup_calls,
        total_calls_per_sample=total_calls,
        graph_iters=effective_graph_iters,
        num_replays=num_replays,
        eager_probe_ms=eager_probe_ms,
        suspicious=suspicious,
        suspicious_reason=suspicious_reason,
    )


def time_with_cudagraph(
    fn: Callable[..., Any],
    *,
    device: "torch.device",
    fn_args: Sequence[Any] = (),
    fn_kwargs: Mapping[str, Any] | None = None,
    warmup: int = 10,
    warmup_ms: float | None = None,
    repeat: int = 50,
    graph_iters: int = 10,
    pre_capture_iters: int = 3,
    use_default_stream: bool = True,
    setup_fn: Callable[[], None] | None = None,
) -> float:
    """Backward-compatible wrapper returning one scalar ms/call."""
    result = benchmark_with_cudagraph(
        fn=fn,
        device=device,
        fn_args=fn_args,
        fn_kwargs=fn_kwargs,
        warmup=warmup,
        warmup_ms=warmup_ms,
        graph_iters=graph_iters,
        pre_capture_iters=pre_capture_iters,
        trial_count=1,
        fixed_repeat_calls=repeat,
        use_default_stream=use_default_stream,
        setup_fn=setup_fn,
        allow_suspicious=True,
    )
    return result.median_ms
