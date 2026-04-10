#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import socket
import time
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


def collect_env_metadata(device: "torch.device") -> Dict[str, Any]:
    props = torch.cuda.get_device_properties(device)
    return {
        "hostname": socket.gethostname(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_hip_version": getattr(torch.version, "hip", None),
        "device_index": int(device.index if device.index is not None else 0),
        "device_name": props.name,
        "total_memory_gb": float(props.total_memory) / (1024.0**3),
        "multi_processor_count": int(getattr(props, "multi_processor_count", 0)),
    }


def add_common_runtime_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--hip-visible-devices", type=str, default="")
    p.add_argument("--cpu-cores", type=str, default="")
    p.add_argument("--seed", type=int, default=20260312)


def add_timer_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--warmup-ms", type=float, default=1000.0)
    p.add_argument("--graph-iters", type=int, default=10)
    p.add_argument("--repeat", type=int, default=0)
    p.add_argument("--timer-trials", type=int, default=9)
    p.add_argument("--repeat-ms", type=float, default=5000.0)
    p.add_argument("--measure-ms", type=float, default=None)
    p.add_argument("--min-replays", type=int, default=5)
    p.add_argument("--max-replays", type=int, default=200000)
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


def maybe_write_json(*, json_out: Path | None, device: "torch.device", config: Dict[str, Any], rows: List[Dict[str, Any]]) -> None:
    if json_out is None:
        return
    payload = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": config,
        "environment": collect_env_metadata(device),
        "rows": rows,
    }
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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
    fixed_repeat_calls: int | None = None
    if int(getattr(args, "repeat", 0)) > 0:
        fixed_repeat_calls = int(args.repeat)
    measure_ms = effective_repeat_ms(args)
    return {
        "warmup": int(args.warmup),
        "warmup_ms": float(args.warmup_ms),
        "graph_iters": int(args.graph_iters),
        "pre_capture_iters": int(args.pre_capture_iters),
        "trial_count": int(args.timer_trials),
        "min_measure_ms": measure_ms,
        "min_replays": int(args.min_replays),
        "max_replays": int(args.max_replays),
        "fixed_repeat_calls": fixed_repeat_calls,
    }


def time_call(call: Callable[[], None], *, device: "torch.device", args: Any) -> CUDAGraphTimingResult:
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


def csv_from_ints(vals: Iterable[int]) -> str:
    return ",".join(str(x) for x in vals)
