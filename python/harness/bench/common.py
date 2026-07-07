#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import csv
import importlib.util
import os
import time
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, Iterable, List

from cudagraph_timer import CUDAGraphTimingResult, benchmark_with_cudagraph

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def parse_int_csv(spec: str, *, name: str) -> List[int]:
    vals: List[int] = []
    seen: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            v = int(token)
        except ValueError as e:
            raise ValueError(f"invalid {name} token: {token}") from e
        if v <= 0:
            raise ValueError(f"{name} must be positive: {v}")
        if v in seen:
            continue
        seen.add(v)
        vals.append(v)
    return vals


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def parse_cpu_cores(spec: str) -> List[int]:
    cores: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            lo_s, hi_s = token.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if hi < lo:
                raise ValueError(f"invalid CPU core range: {token}")
            cores.update(range(lo, hi + 1))
        else:
            cores.add(int(token))
    if not cores:
        raise ValueError("no CPU cores parsed from --cpu-cores")
    return sorted(cores)


def apply_cpu_affinity(spec: str) -> List[int]:
    if not spec:
        return []
    if not hasattr(os, "sched_setaffinity"):
        raise RuntimeError("CPU affinity is not supported on this platform")
    allowed = sorted(os.sched_getaffinity(0))
    cores = parse_cpu_cores(spec)
    invalid = [c for c in cores if c not in allowed]
    if invalid:
        raise ValueError(f"requested CPU cores not allowed in this runtime: {invalid}")
    os.sched_setaffinity(0, set(cores))
    return sorted(os.sched_getaffinity(0))


def apply_visible_devices(hip_visible_devices: str) -> str:
    if not hip_visible_devices:
        return ""
    normalized = ",".join(x.strip() for x in hip_visible_devices.split(",") if x.strip())
    if not normalized:
        raise ValueError("invalid --hip-visible-devices value")
    os.environ["HIP_VISIBLE_DEVICES"] = normalized
    os.environ.pop("ROCR_VISIBLE_DEVICES", None)
    return normalized


def validate_device_local_index(device: str, visible_devices: str) -> None:
    if not visible_devices or not device.startswith("cuda:"):
        return
    visible_count = len([x for x in visible_devices.split(",") if x])
    local_idx = int(device.split(":", 1)[1])
    if local_idx < 0 or local_idx >= visible_count:
        raise ValueError(
            f"--device {device} is out of range for visible devices [{visible_devices}] "
            f"(use local index cuda:0..cuda:{visible_count - 1})"
        )


def configure_sync_wait_mode(*, device: "torch.device", mode: str) -> int:
    if torch is None:
        raise RuntimeError("torch is required to configure sync wait mode")
    mode_norm = str(mode).strip().lower()
    if mode_norm == "block":
        mode_norm = "blocking"
    flags = {
        "auto": 0,
        "spin": 1,
        "yield": 2,
        "blocking": 4,
    }
    if mode_norm not in flags:
        raise ValueError(f"invalid --sync-wait-mode: {mode}")
    if not str(device).startswith("cuda"):
        return -1

    lib = None
    runtime_name = ""
    set_fn_name = ""
    get_fn_name = ""
    load_errors: List[str] = []
    for candidate_runtime, candidate_lib, candidate_set, candidate_get in (
        ("HIP", "libamdhip64.so", "hipSetDeviceFlags", "hipGetDeviceFlags"),
        ("CUDA", "libcudart.so", "cudaSetDeviceFlags", "cudaGetDeviceFlags"),
    ):
        try:
            lib = ctypes.CDLL(candidate_lib)
            runtime_name = candidate_runtime
            set_fn_name = candidate_set
            get_fn_name = candidate_get
            break
        except OSError as e:
            load_errors.append(f"{candidate_lib}: {e}")
    if lib is None:
        warnings.warn(
            f"could not load HIP or CUDA runtime library; sync wait mode {mode_norm!r} was not applied "
            f"({'; '.join(load_errors)})",
            RuntimeWarning,
        )
        return -1

    try:
        set_device_flags = getattr(lib, set_fn_name)
        get_device_flags = getattr(lib, get_fn_name)
    except AttributeError as e:
        warnings.warn(
            f"{runtime_name} runtime does not expose device flag APIs; sync wait mode {mode_norm!r} was not applied: {e}",
            RuntimeWarning,
        )
        return -1
    set_device_flags.argtypes = [ctypes.c_uint]
    set_device_flags.restype = ctypes.c_int
    get_device_flags.argtypes = [ctypes.POINTER(ctypes.c_uint)]
    get_device_flags.restype = ctypes.c_int

    torch.cuda.set_device(device)
    rc = int(set_device_flags(flags[mode_norm]))
    if rc != 0:
        warnings.warn(
            f"{set_fn_name}({flags[mode_norm]}) failed with error code {rc}; sync wait mode {mode_norm!r} was not applied",
            RuntimeWarning,
        )
        return -1

    got = ctypes.c_uint(0)
    rc_get = int(get_device_flags(ctypes.byref(got)))
    if rc_get != 0:
        warnings.warn(f"{get_fn_name} failed with error code {rc_get}", RuntimeWarning)
        return -1
    return int(got.value)


def load_module(path: Path) -> ModuleType:
    module_name = f"bench_dyn_{path.stem}_{abs(hash(str(path.resolve()))):x}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses may inspect sys.modules during class decoration (Python 3.12)
    import sys

    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def build_model_fn(module: ModuleType, *, device: "torch.device", dtype: "torch.dtype | None" = None):
    if hasattr(module, "ModelNew"):
        model = module.ModelNew()
    elif hasattr(module, "Model"):
        model = module.Model()
    elif hasattr(module, "kernel_function"):
        return lambda *xs: module.kernel_function(*xs)
    elif hasattr(module, "run"):
        return lambda *xs: module.run(*xs)
    else:
        raise RuntimeError("expected one of: ModelNew, Model, kernel_function, run")

    if hasattr(model, "to"):
        if dtype is None:
            model = model.to(device=device)
        else:
            model = model.to(device=device, dtype=dtype)
    return lambda *xs: model(*xs)


def add_common_runtime_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--hip-visible-devices", type=str, default="")
    p.add_argument("--cpu-cores", type=str, default="")
    p.add_argument(
        "--sync-wait-mode",
        type=str,
        default="blocking",
        choices=["auto", "spin", "yield", "blocking", "block"],
    )
    p.add_argument("--seed", type=int, default=20260312)


def add_timer_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--warmup-ms", type=float, default=1000.0)
    p.add_argument("--graph-iters", type=int, default=1)
    p.add_argument("--repeat", type=int, default=0)
    p.add_argument("--timer-trials", type=int, default=9)
    p.add_argument("--repeat-ms", type=float, default=5000.0)
    p.add_argument("--measure-ms", type=float, default=None)
    p.add_argument("--min-replays", type=int, default=1)
    p.add_argument("--max-replays", type=int, default=10)
    p.add_argument("--pre-capture-iters", type=int, default=3)


def enable_default_flags(args: argparse.Namespace, all_flags: Iterable[str], default_true_flags: Iterable[str] | None = None) -> None:
    flags = list(all_flags)
    if any(bool(getattr(args, f)) for f in flags):
        return
    defaults = list(default_true_flags) if default_true_flags is not None else flags
    for f in defaults:
        setattr(args, f, True)


def effective_repeat_ms(args: Any) -> float:
    return float(args.repeat_ms if getattr(args, "measure_ms", None) is None else args.measure_ms)


def build_csv_row(
    *,
    domain: str,
    baseline: str,
    workload: int,
    mean_ms: float,
    tflops: float,
    status: str,
    kernel_entry: str,
    timestamp_utc: str,
    run_id: str,
) -> Dict[str, Any]:
    return {
        "domain": domain,
        "baseline": baseline,
        "workload": int(workload),
        "mean_ms": float(mean_ms),
        "tflops": float(tflops),
        "status": status,
        "kernel_entry": kernel_entry,
        "timestamp_utc": timestamp_utc,
        "run_id": run_id,
    }


def maybe_write_csv(*, csv_out: Path | None, rows: List[Dict[str, Any]]) -> None:
    if csv_out is None:
        return
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        csv_out.write_text("", encoding="utf-8")
        return
    fieldnames = [
        "domain",
        "baseline",
        "workload",
        "mean_ms",
        "tflops",
        "status",
        "kernel_entry",
        "timestamp_utc",
        "run_id",
    ]
    with csv_out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def timing_fields(timing: CUDAGraphTimingResult, *, tflops_median: float) -> Dict[str, Any]:
    return {
        "status": "ok",
        "median_ms": timing.median_ms,
        "mean_ms": timing.mean_ms,
        "stdev_ms": timing.stdev_ms,
        "min_ms": timing.min_ms,
        "max_ms": timing.max_ms,
        "p10_ms": timing.p10_ms,
        "p90_ms": timing.p90_ms,
        "cv": timing.cv,
        "tflops_median": tflops_median,
        "eager_probe_ms": timing.eager_probe_ms,
        "suspicious": timing.suspicious,
        "suspicious_reason": timing.suspicious_reason,
        "num_replays": timing.num_replays,
        "graph_iters": timing.graph_iters,
        "warmup_calls": timing.warmup_calls,
        "total_calls_per_sample": timing.total_calls_per_sample,
        "trial_count": len(timing.samples_ms),
        "samples_ms": timing.samples_ms,
    }


def timer_kwargs(args: Any) -> Dict[str, Any]:
    # Hard caps requested by experiment policy:
    # - at most 100 kernel calls per captured graph
    # - at most 10 graph replays per timing sample
    replay_cap = 10

    fixed_repeat_calls: int | None = None
    if int(getattr(args, "repeat", 0)) > 0:
        fixed_repeat_calls = int(args.repeat)
    measure_ms = effective_repeat_ms(args)
    graph_iters_req = max(1, int(args.graph_iters))
    # Default cap is 100, but allow explicit larger user request (e.g. 1000).
    graph_iters_cap = max(100, graph_iters_req)
    graph_iters = min(graph_iters_req, graph_iters_cap)
    min_replays = min(int(args.min_replays), replay_cap)
    max_replays = min(int(args.max_replays), replay_cap)
    if max_replays < 1:
        max_replays = 1
    if min_replays < 1:
        min_replays = 1
    if max_replays < min_replays:
        min_replays = max_replays
    return {
        "warmup": int(args.warmup),
        "warmup_ms": float(args.warmup_ms),
        "graph_iters": graph_iters,
        "min_graph_ms": 300.0,
        "pre_capture_iters": int(args.pre_capture_iters),
        "trial_count": int(args.timer_trials),
        "min_measure_ms": measure_ms,
        "min_replays": min_replays,
        "max_replays": max_replays,
        "max_graph_iters": graph_iters_cap,
        "fixed_repeat_calls": fixed_repeat_calls,
    }


def time_call(call: Callable[[], None], *, device: "torch.device", args: Any) -> CUDAGraphTimingResult:
    if torch is not None:
        def wrapped() -> None:
            with torch.inference_mode():
                call()

        return benchmark_with_cudagraph(fn=wrapped, device=device, **timer_kwargs(args))
    return benchmark_with_cudagraph(fn=call, device=device, **timer_kwargs(args))


def parse_dtype(name: str) -> "torch.dtype":
    n = name.strip().lower()
    if n == "bf16":
        return torch.bfloat16
    if n == "fp16":
        return torch.float16
    if n == "fp8":
        dt = getattr(torch, "float8_e4m3fnuz", None) or getattr(torch, "float8_e4m3fn", None)
        if dt is None:
            raise RuntimeError("torch float8 dtype is unavailable")
        return dt
    raise ValueError(f"unsupported dtype: {name}")
