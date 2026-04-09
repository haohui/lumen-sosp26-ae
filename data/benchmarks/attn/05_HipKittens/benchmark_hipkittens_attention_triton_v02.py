#!/usr/bin/env python3
"""Benchmark HipKittens attention path via triton_baseline_v02 forward kernel."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
CUDA_GRAPH_TIMER_DIR = REPO_ROOT / "script"
TRITON_BASELINE_V02 = (
    REPO_ROOT
    / "third_party"
    / "HipKittens"
    / "analysis"
    / "baselines"
    / "attn"
    / "triton_baseline_v02.py"
)

if str(CUDA_GRAPH_TIMER_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(CUDA_GRAPH_TIMER_DIR))
from cudagraph_timer import CUDAGraphTimingResult, benchmark_with_cudagraph


@dataclass
class SharedInputs:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    o: torch.Tensor
    metadata: Any


def _parse_cpu_cores(spec: str) -> List[int]:
    cores: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            lo_s, hi_s = token.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if hi < lo:
                raise ValueError(f"invalid cpu core range token: {token}")
            cores.update(range(lo, hi + 1))
        else:
            cores.add(int(token))
    if not cores:
        raise ValueError("no CPU cores parsed from --cpu-cores")
    return sorted(cores)


def _apply_cpu_affinity(spec: str) -> List[int]:
    if not spec:
        return []
    if not hasattr(os, "sched_setaffinity"):
        raise RuntimeError("CPU affinity is not supported on this platform")
    allowed = sorted(os.sched_getaffinity(0))
    cores = _parse_cpu_cores(spec)
    invalid = [c for c in cores if c not in allowed]
    if invalid:
        raise ValueError(f"requested CPU cores are not allowed: {invalid}")
    os.sched_setaffinity(0, set(cores))
    return sorted(os.sched_getaffinity(0))


def _apply_visible_devices(hip_visible_devices: str) -> str:
    if not hip_visible_devices:
        return ""
    normalized = ",".join(x.strip() for x in hip_visible_devices.split(",") if x.strip())
    if not normalized:
        raise ValueError("invalid --hip-visible-devices value")
    os.environ["HIP_VISIBLE_DEVICES"] = normalized
    if "ROCR_VISIBLE_DEVICES" in os.environ:
        os.environ.pop("ROCR_VISIBLE_DEVICES")
    return normalized


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _attention_tflops(
    *,
    batch_size: int,
    seq_len: int,
    num_q_heads: int,
    head_dim: int,
    causal: bool,
    ms: float,
) -> float:
    if ms <= 0.0:
        return float("nan")
    if causal:
        flops = 2.0 * batch_size * num_q_heads * seq_len * seq_len * head_dim
    else:
        flops = 4.0 * batch_size * num_q_heads * seq_len * seq_len * head_dim
    return flops / (ms * 1.0e-3) / 1.0e12


def _timer_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    fixed_repeat_calls: int | None = None
    if args.repeat > 0:
        fixed_repeat_calls = args.repeat
    return {
        "warmup": args.warmup,
        "warmup_ms": args.warmup_ms,
        "graph_iters": args.graph_iters,
        "pre_capture_iters": args.pre_capture_iters,
        "trial_count": args.timer_trials,
        "min_measure_ms": args.repeat_ms if args.measure_ms is None else args.measure_ms,
        "min_replays": args.min_replays,
        "max_replays": args.max_replays,
        "fixed_repeat_calls": fixed_repeat_calls,
        "suspicious_ratio_threshold": args.suspicious_ratio,
        "allow_suspicious": args.allow_suspicious_graph,
    }


def _time_one(call: Callable[[], None], *, device: torch.device, args: argparse.Namespace) -> CUDAGraphTimingResult:
    return benchmark_with_cudagraph(
        fn=call,
        device=device,
        **_timer_kwargs(args),
    )


def _make_row(
    *,
    seq_len: int,
    args: argparse.Namespace,
    timing: CUDAGraphTimingResult | None,
    error: str | None = None,
) -> Dict[str, Any]:
    if timing is None:
        return {
            "baseline": "hipkittens::triton_baseline_v02::attention_fwd",
            "kernel_path": str(TRITON_BASELINE_V02),
            "seq_len": seq_len,
            "status": "error",
            "error": error,
        }

    return {
        "baseline": "hipkittens::triton_baseline_v02::attention_fwd",
        "kernel_path": str(TRITON_BASELINE_V02),
        "seq_len": seq_len,
        "batch_size": args.batch_size,
        "num_q_heads": args.num_q_heads,
        "num_kv_heads": args.num_kv_heads,
        "head_dim": args.head_dim,
        "causal": args.causal,
        "status": "ok",
        "median_ms": timing.median_ms,
        "mean_ms": timing.mean_ms,
        "stdev_ms": timing.stdev_ms,
        "min_ms": timing.min_ms,
        "max_ms": timing.max_ms,
        "p10_ms": timing.p10_ms,
        "p90_ms": timing.p90_ms,
        "cv": timing.cv,
        "tflops_median": _attention_tflops(
            batch_size=args.batch_size,
            seq_len=seq_len,
            num_q_heads=args.num_q_heads,
            head_dim=args.head_dim,
            causal=args.causal,
            ms=timing.median_ms,
        ),
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HipKittens attention benchmark (triton_baseline_v02 forward path).")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--hip-visible-devices", type=str, default="")
    p.add_argument("--cpu-cores", type=str, default="")
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--seq-lens", type=str, default="1024,2048,4096,8192,16384")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-q-heads", type=int, default=8)
    p.add_argument("--num-kv-heads", type=int, default=1)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--layout", type=str, default="bhsd", choices=["bhsd", "bshd"])
    p.add_argument("--causal", dest="causal", action="store_true", default=True)
    p.add_argument("--non-causal", dest="causal", action="store_false")
    p.add_argument("--seed", type=int, default=20260324)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--warmup-ms", type=float, default=200.0)
    p.add_argument("--graph-iters", type=int, default=10)
    p.add_argument("--repeat", type=int, default=0)
    p.add_argument("--timer-trials", type=int, default=9)
    p.add_argument("--repeat-ms", type=float, default=1000.0)
    p.add_argument("--measure-ms", type=float, default=None)
    p.add_argument("--min-replays", type=int, default=5)
    p.add_argument("--max-replays", type=int, default=200000)
    p.add_argument("--pre-capture-iters", type=int, default=3)
    p.add_argument("--suspicious-ratio", type=float, default=0.25)
    p.add_argument("--allow-suspicious-graph", action="store_true")
    p.add_argument("--json-out", type=Path, default=SCRIPT_DIR / "hipkittens_attention_results.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    visible_devices = _apply_visible_devices(args.hip_visible_devices)
    cpu_affinity = _apply_cpu_affinity(args.cpu_cores)
    seq_lens = [int(x.strip()) for x in args.seq_lens.split(",") if x.strip()]

    if visible_devices and args.device.startswith("cuda:"):
        visible_count = len([x for x in visible_devices.split(",") if x])
        local_idx = int(args.device.split(":", 1)[1])
        if local_idx < 0 or local_idx >= visible_count:
            raise ValueError(
                f"--device {args.device} is out of range for visible devices [{visible_devices}] "
                f"(use local index cuda:0..cuda:{visible_count - 1})"
            )

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    hk_module = _load_module(TRITON_BASELINE_V02)
    shared: Dict[int, SharedInputs] = {}
    for s in seq_lens:
        q, k, v, metadata = hk_module.input_helper(
            args.batch_size,
            args.num_q_heads,
            args.num_kv_heads,
            s,
            s,
            args.head_dim,
            dtype,
            args.layout,
            requires_grad=False,
        )
        if args.causal:
            metadata.need_causal()
        o = torch.empty_like(q)
        shared[s] = SharedInputs(q=q, k=k, v=v, o=o, metadata=metadata)

    rows: List[Dict[str, Any]] = []
    for s in seq_lens:
        x = shared[s]

        def call() -> None:
            hk_module.attention(x.q, x.k, x.v, x.o, x.metadata)

        try:
            timing = _time_one(call, device=device, args=args)
            row = _make_row(seq_len=s, args=args, timing=timing)
        except Exception as e:
            row = _make_row(seq_len=s, args=args, timing=None, error=f"{type(e).__name__}: {e}")
        rows.append(row)
        if row["status"] == "ok":
            print(
                f"S={s:>6} | mean={row['mean_ms']:.4f} ms | med={row['median_ms']:.4f} ms | "
                f"tflops={row['tflops_median']:.2f} | suspicious={row['suspicious']}"
            )
        else:
            print(f"S={s:>6} | error | {row['error']}")

    props = torch.cuda.get_device_properties(device)
    report = {
        "created_at_utc": __import__("datetime").datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": (
            "HipKittens repo attention path benchmarked via analysis/baselines/attn/triton_baseline_v02.py "
            "(forward, causal)."
        ),
        "config": {
            "device": args.device,
            "dtype": args.dtype,
            "hip_visible_devices": visible_devices,
            "cpu_cores": args.cpu_cores,
            "cpu_affinity_applied": cpu_affinity,
            "seq_lens": seq_lens,
            "batch_size": args.batch_size,
            "num_q_heads": args.num_q_heads,
            "num_kv_heads": args.num_kv_heads,
            "head_dim": args.head_dim,
            "layout": args.layout,
            "causal": args.causal,
            "seed": args.seed,
            "warmup": args.warmup,
            "warmup_ms": args.warmup_ms,
            "graph_iters": args.graph_iters,
            "repeat": args.repeat,
            "timer_trials": args.timer_trials,
            "repeat_ms": args.repeat_ms,
            "measure_ms": args.measure_ms,
            "min_replays": args.min_replays,
            "max_replays": args.max_replays,
            "pre_capture_iters": args.pre_capture_iters,
            "suspicious_ratio": args.suspicious_ratio,
            "allow_suspicious_graph": args.allow_suspicious_graph,
            "triton_baseline_path": str(TRITON_BASELINE_V02),
        },
        "environment": {
            "hostname": socket.gethostname(),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "torch_hip_version": getattr(torch.version, "hip", None),
            "device_index": int(device.index if device.index is not None else 0),
            "device_name": props.name,
            "total_memory_gb": float(props.total_memory) / (1024.0**3),
            "multi_processor_count": int(getattr(props, "multi_processor_count", 0)),
        },
        "rows": rows,
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[json] wrote {args.json_out}")


if __name__ == "__main__":
    main()
