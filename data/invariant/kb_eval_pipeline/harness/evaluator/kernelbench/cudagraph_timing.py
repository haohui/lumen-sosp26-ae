from typing import Any

import torch

def benchmark_with_cudagraph(
    kernel_fn: callable,
    args: list[Any],
    num_warmup: int = 3,
    num_trials: int = 10,
    verbose: bool = True,
    device: torch.device | None = None,
    graph_iters: int = 32,
    pre_capture_iters: int = 3,
) -> dict:
    """
    Benchmark a kernel by capturing a fixed-size CUDA Graph and replaying it.

    The returned samples are normalized to milliseconds per logical kernel call.
    The graph captures a constant number of kernel calls (`graph_iters`), then
    replays enough times so the total number of logical kernel calls is at least
    `num_trials`.
    """
    if device is None:
        if verbose:
            print(f"Using current device: {torch.cuda.current_device()}")
        device = torch.cuda.current_device()

    graph_iters = max(1, graph_iters)
    requested_num_trials = max(1, num_trials)
    num_warmup = max(1, num_warmup)

    with torch.cuda.device(device):
        for _ in range(num_warmup):
            kernel_fn(*args)
        torch.cuda.synchronize(device=device)

        if verbose:
            print(
                f"[Profiling] Using device: {device} {torch.cuda.get_device_name(device)}, "
                f"warm up {num_warmup}, requested trials {requested_num_trials}, graph iters {graph_iters}"
            )

        graph = torch.cuda.CUDAGraph()
        graph_stream = torch.cuda.Stream(device=device)
        caller_stream = torch.cuda.current_stream(device=device)
        graph_stream.wait_stream(caller_stream)

        with torch.cuda.stream(graph_stream):
            for _ in range(max(0, pre_capture_iters)):
                kernel_fn(*args)
            with torch.cuda.graph(graph):
                for _ in range(graph_iters):
                    kernel_fn(*args)
        caller_stream.wait_stream(graph_stream)
        torch.cuda.synchronize(device=device)

        num_replays = max(1, (requested_num_trials + graph_iters - 1) // graph_iters)
        actual_num_trials = num_replays * graph_iters
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(graph_stream):
            start_event.record()
            for _ in range(num_replays):
                graph.replay()
            end_event.record()
        caller_stream.wait_stream(graph_stream)
        torch.cuda.synchronize(device=device)

        elapsed_time_ms = float(start_event.elapsed_time(end_event)) / float(actual_num_trials)
        if verbose:
            print(f"Graph batch: {elapsed_time_ms:.3g} ms over {actual_num_trials} calls")

    return {
        "samples_ms": [elapsed_time_ms],
        "timing_mode": "cudagraph",
        "requested_num_trials": requested_num_trials,
        "num_trials": actual_num_trials,
    }


def time_execution_with_cudagraph(
    kernel_fn: callable,
    args: list[Any],
    num_warmup: int = 3,
    num_trials: int = 10,
    discard_first: int = 1,  # ignored; replay trials are already stable after capture
    verbose: bool = True,
    device: torch.device | None = None,
) -> list[float]:
    result = benchmark_with_cudagraph(
        kernel_fn,
        args,
        num_warmup=num_warmup,
        num_trials=num_trials,
        verbose=verbose,
        device=device,
    )
    return result
