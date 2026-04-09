#!/usr/bin/env python3
"""Unified MoE benchmark with publication-grade CUDA Graph timing."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, List

from cudagraph_timer import CUDAGraphTimingResult, benchmark_with_cudagraph

try:
    import torch
except Exception:
    torch = None


BLOCK_N = 128
BLOCK_K = 128


@dataclass
class SharedInputs:
    input_q: "torch.Tensor"
    topk_weights: "torch.Tensor"
    topk_ids: "torch.Tensor"
    input_scale: "torch.Tensor"


def _moe_tflops(*, tokens: int, dim: int, inter_dim: int, topk: int, ms: float) -> float:
    if ms <= 0.0:
        return float("nan")
    # Dense-style estimate used by existing MoE benchmark scripts.
    flops = 6.0 * float(tokens) * float(topk) * float(dim) * float(inter_dim)
    return flops / (ms * 1.0e-3) / 1.0e12


def _load_module(path: Path) -> ModuleType:
    module_name = f"kb_dyn_{path.stem}_{abs(hash(str(path.resolve()))):x}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    # Python 3.12 dataclasses may inspect sys.modules during class decoration.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _build_model_fn(module: ModuleType, device: "torch.device"):
    if hasattr(module, "ModelNew"):
        model = module.ModelNew()
        if hasattr(model, "to"):
            model = model.to(device=device)
        return lambda *xs: model(*xs)
    if hasattr(module, "Model"):
        model = module.Model()
        if hasattr(model, "to"):
            model = model.to(device=device)
        return lambda *xs: model(*xs)
    if hasattr(module, "kernel_function"):
        return lambda *xs: module.kernel_function(*xs)
    if hasattr(module, "run"):
        return lambda *xs: module.run(*xs)
    raise RuntimeError("expected one of: ModelNew, Model, kernel_function, run")


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
                raise ValueError(f"Invalid cpu core range token: {token}")
            cores.update(range(lo, hi + 1))
        else:
            cores.add(int(token))
    if not cores:
        raise ValueError("No CPU cores parsed from --cpu-cores")
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
        raise ValueError(f"Requested CPU cores not allowed in this runtime: {invalid}")
    os.sched_setaffinity(0, set(cores))
    return sorted(os.sched_getaffinity(0))


def _apply_visible_devices(hip_visible_devices: str) -> str:
    if not hip_visible_devices:
        return ""
    normalized = ",".join(x.strip() for x in hip_visible_devices.split(",") if x.strip())
    if not normalized:
        raise ValueError("Invalid --hip-visible-devices value")
    os.environ["HIP_VISIBLE_DEVICES"] = normalized
    if "ROCR_VISIBLE_DEVICES" in os.environ:
        os.environ.pop("ROCR_VISIBLE_DEVICES")
    return normalized


def _resolve_input_dtype(name: str) -> "torch.dtype":
    n = name.strip().lower()
    if n == "bf16":
        return torch.bfloat16
    if n == "fp8":
        dt = getattr(torch, "float8_e4m3fnuz", None)
        if dt is None:
            dt = getattr(torch, "float8_e4m3fn", None)
        if dt is None:
            raise RuntimeError("Requested --input-dtype=fp8 but torch float8 dtype is unavailable")
        return dt
    raise ValueError(f"Unsupported input dtype: {name}")


def _build_shared_inputs(
    *,
    seq_lens: List[int],
    dim: int,
    experts: int,
    topk: int,
    input_dtype: "torch.dtype",
    device: "torch.device",
    seed: int,
) -> Dict[int, SharedInputs]:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    out: Dict[int, SharedInputs] = {}
    hidden_blocks = dim // BLOCK_K

    for s in seq_lens:
        input_q = torch.randn((s, dim), dtype=torch.float32, device=device, generator=g).mul_(1.0).add_(0.1)
        input_q = input_q.to(input_dtype).contiguous()

        input_scale = (
            torch.randn((s, hidden_blocks), dtype=torch.float32, device=device, generator=g).mul_(2e-2).add_(1e-1)
        ).clamp_min_(1e-8).contiguous()

        scores = torch.randn((s, experts), dtype=torch.float32, device=device, generator=g)
        topk_val, topk_idx = torch.topk(scores, k=topk, dim=-1, largest=True, sorted=True)
        topk_ids = topk_idx.to(torch.int32).contiguous()
        topk_weights = torch.softmax(topk_val, dim=-1).to(torch.float32).contiguous()

        out[s] = SharedInputs(
            input_q=input_q,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            input_scale=input_scale,
        )
    return out


def _build_shared_weights(
    *,
    dim: int,
    inter_dim: int,
    experts: int,
    input_dtype: "torch.dtype",
    device: "torch.device",
    seed: int,
) -> Dict[str, "torch.Tensor"]:
    g = torch.Generator(device=device)
    g.manual_seed(seed + 17)

    w1_q = torch.randn(
        (experts, inter_dim * 2, dim),
        dtype=torch.float32,
        device=device,
        generator=g,
    ).mul_(8.0).to(input_dtype).contiguous()

    w2_q = torch.randn(
        (experts, dim, inter_dim),
        dtype=torch.float32,
        device=device,
        generator=g,
    ).mul_(8.0).to(input_dtype).contiguous()

    fc1_scale = (
        torch.randn(
            (experts, ((inter_dim * 2) // BLOCK_N) * (dim // BLOCK_K)),
            dtype=torch.float32,
            device=device,
            generator=g,
        ).mul_(2e-3).add_(1e-2)
    ).clamp_min_(1e-8).contiguous()

    fc2_scale = (
        torch.randn(
            (experts, (dim // BLOCK_N) * (inter_dim // BLOCK_K)),
            dtype=torch.float32,
            device=device,
            generator=g,
        ).mul_(2e-3).add_(1e-2)
    ).clamp_min_(1e-8).contiguous()

    return {
        "w1_q": w1_q,
        "w2_q": w2_q,
        "fc1_scale": fc1_scale,
        "fc2_scale": fc2_scale,
    }


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


def _time_one(
    call: Callable[[], None],
    *,
    device: "torch.device",
    args: argparse.Namespace,
) -> CUDAGraphTimingResult:
    if args.disable_cudagraph:
        return _time_one_eager(call, device=device, args=args)
    try:
        graph_timing = benchmark_with_cudagraph(
            fn=call,
            device=device,
            **_timer_kwargs(args),
        )
        if graph_timing.suspicious and args.fallback_eager_on_suspicious_graph:
            reason = graph_timing.suspicious_reason or "unknown"
            print(f"[warn] cudagraph timing marked suspicious ({reason}); fallback to eager timer")
            eager_timing = _time_one_eager(call, device=device, args=args)
            eager_timing.suspicious = False
            eager_timing.suspicious_reason = f"eager_fallback_after_suspicious_graph: {reason}"
            return eager_timing
        return graph_timing
    except Exception as e:
        msg = str(e).lower()
        if "capturing" not in msg and "streamcapture" not in msg:
            raise
        if args.fallback_eager_on_capture_fail:
            print(f"[warn] cudagraph capture failed ({type(e).__name__}); fallback to eager timer")
            return _time_one_eager(call, device=device, args=args)
        raise RuntimeError(f"cudagraph capture failed and eager fallback disabled: {type(e).__name__}: {e}")


@dataclass
class _EagerTiming:
    median_ms: float
    mean_ms: float
    stdev_ms: float
    min_ms: float
    max_ms: float
    p10_ms: float
    p90_ms: float
    cv: float
    eager_probe_ms: float | None
    suspicious: bool
    suspicious_reason: str
    num_replays: int
    graph_iters: int
    warmup_calls: int
    total_calls_per_sample: int
    samples_ms: List[float]


def _time_one_eager(
    call: Callable[[], None],
    *,
    device: "torch.device",
    args: argparse.Namespace,
) -> _EagerTiming:
    warmup_calls = max(1, int(args.warmup))
    for _ in range(warmup_calls):
        call()
    torch.cuda.synchronize(device=device)

    target_warmup_ms = max(0.0, float(args.warmup_ms))
    if target_warmup_ms > 0.0:
        elapsed_ms = 0.0
        ev_warmup_start = torch.cuda.Event(enable_timing=True)
        ev_warmup_end = torch.cuda.Event(enable_timing=True)
        while elapsed_ms < target_warmup_ms:
            ev_warmup_start.record()
            call()
            ev_warmup_end.record()
            torch.cuda.synchronize(device=device)
            elapsed_ms += float(ev_warmup_start.elapsed_time(ev_warmup_end))
            warmup_calls += 1

    ev_start = torch.cuda.Event(enable_timing=True)
    ev_end = torch.cuda.Event(enable_timing=True)
    ev_start.record()
    call()
    ev_end.record()
    torch.cuda.synchronize(device=device)
    probe_ms = float(ev_start.elapsed_time(ev_end))

    if int(args.repeat) > 0:
        replays = int(args.repeat)
    else:
        target_ms = float(args.repeat_ms if args.measure_ms is None else args.measure_ms)
        base = max(1e-3, probe_ms)
        replays = int(round(target_ms / base))
        replays = max(int(args.min_replays), replays)
        replays = min(int(args.max_replays), replays)
        replays = max(1, replays)

    samples: List[float] = []
    trials = max(1, int(args.timer_trials))
    for _ in range(trials):
        ev_start.record()
        for _ in range(replays):
            call()
        ev_end.record()
        torch.cuda.synchronize(device=device)
        total_ms = float(ev_start.elapsed_time(ev_end))
        samples.append(total_ms / float(replays))

    samples_sorted = sorted(samples)
    n = len(samples_sorted)
    mean_ms = float(statistics.fmean(samples_sorted))
    stdev_ms = float(statistics.pstdev(samples_sorted)) if n > 1 else 0.0
    median_ms = float(statistics.median(samples_sorted))
    p10_ms = samples_sorted[max(0, int(0.1 * (n - 1)))]
    p90_ms = samples_sorted[min(n - 1, int(0.9 * (n - 1)))]
    cv = float(stdev_ms / mean_ms) if mean_ms > 0 else 0.0
    return _EagerTiming(
        median_ms=median_ms,
        mean_ms=mean_ms,
        stdev_ms=stdev_ms,
        min_ms=float(samples_sorted[0]),
        max_ms=float(samples_sorted[-1]),
        p10_ms=float(p10_ms),
        p90_ms=float(p90_ms),
        cv=cv,
        eager_probe_ms=probe_ms,
        suspicious=False,
        suspicious_reason="eager_fallback",
        num_replays=replays,
        graph_iters=0,
        warmup_calls=warmup_calls,
        total_calls_per_sample=replays,
        samples_ms=[float(x) for x in samples],
    )


def _run_model_once(
    fn_model: Callable[..., Any],
    *,
    x: SharedInputs,
    shared_weights: Dict[str, "torch.Tensor"],
) -> "torch.Tensor":
    with torch.inference_mode():
        y = fn_model(
            x.input_q,
            shared_weights["w1_q"],
            shared_weights["w2_q"],
            x.topk_weights,
            x.topk_ids,
            x.input_scale,
            shared_weights["fc1_scale"],
            shared_weights["fc2_scale"],
        )
    if not isinstance(y, torch.Tensor):
        raise RuntimeError(f"kernel output must be torch.Tensor, got {type(y).__name__}")
    return y


def _max_rel_err(actual: "torch.Tensor", expect: "torch.Tensor") -> float:
    diff = (actual - expect).abs()
    denom = torch.maximum(expect.abs(), torch.tensor(1e-6, device=expect.device))
    return float((diff / denom).max().item())


def _make_row(
    *,
    baseline: str,
    kernel_path: str,
    seq_len: int,
    dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    timing: CUDAGraphTimingResult | None,
    error: str | None = None,
) -> Dict[str, Any]:
    if timing is None:
        return {
            "baseline": baseline,
            "kernel_path": kernel_path,
            "seq_len": seq_len,
            "status": "error",
            "error": error,
        }
    timing_mode = "eager" if isinstance(timing, _EagerTiming) else "cudagraph"
    median_ms = timing.median_ms
    return {
        "baseline": baseline,
        "kernel_path": kernel_path,
        "seq_len": seq_len,
        "dim": dim,
        "inter_dim": inter_dim,
        "experts": experts,
        "topk": topk,
        "status": "ok",
        "timing_mode": timing_mode,
        "median_ms": median_ms,
        "mean_ms": timing.mean_ms,
        "stdev_ms": timing.stdev_ms,
        "min_ms": timing.min_ms,
        "max_ms": timing.max_ms,
        "p10_ms": timing.p10_ms,
        "p90_ms": timing.p90_ms,
        "cv": timing.cv,
        "tflops_median": _moe_tflops(tokens=seq_len, dim=dim, inter_dim=inter_dim, topk=topk, ms=median_ms),
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


def _print_row(row: Dict[str, Any]) -> None:
    s = int(row["seq_len"])
    if row.get("status") != "ok":
        print(f"  T={s:>6} | error | {row.get('error', 'unknown')}")
        return
    flag = " [SUSPICIOUS]" if row.get("suspicious") else ""
    line = (
        f"  T={s:>6} | med={row['median_ms']:>8.4f} ms | "
        f"p10/p90={row['p10_ms']:.4f}/{row['p90_ms']:.4f} | "
        f"cv={100.0 * row['cv']:.2f}% | {row['tflops_median']:>8.2f} TFLOPS | "
        f"trials={row['trial_count']} replays={row['num_replays']}{flag}"
    )
    corr = row.get("correctness")
    if isinstance(corr, dict):
        if corr.get("status") == "ok":
            line += (
                f" | chk_abs={corr['max_abs_err']:.3e}"
                f" rel={corr['max_rel_err']:.3e}"
                f" pass={bool(corr['allclose'])}"
            )
        else:
            line += f" | chk_error={corr.get('error', 'unknown')}"
    print(line)


def _build_reference_outputs(
    reference_kernel: Path,
    *,
    seq_lens: List[int],
    shared_inputs: Dict[int, SharedInputs],
    shared_weights: Dict[str, "torch.Tensor"],
    device: "torch.device",
) -> Dict[int, "torch.Tensor"]:
    module = _load_module(reference_kernel)
    fn_model = _build_model_fn(module, device=device)
    out: Dict[int, "torch.Tensor"] = {}
    print(f"\n[reference] {reference_kernel}")
    for s in seq_lens:
        y = _run_model_once(fn_model, x=shared_inputs[s], shared_weights=shared_weights)
        out[s] = y.detach()
        print(f"  T={s:>6} | ready")
    return out


def run_python_kernel_baseline(
    name: str,
    kernel_path: Path,
    *,
    seq_lens: List[int],
    shared_inputs: Dict[int, SharedInputs],
    shared_weights: Dict[str, "torch.Tensor"],
    device: "torch.device",
    args: argparse.Namespace,
    reference_outputs: Dict[int, "torch.Tensor"] | None = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    module = _load_module(kernel_path)
    fn_model = _build_model_fn(module, device=device)
    print(f"\n[{name}] {kernel_path}")
    for s in seq_lens:
        x = shared_inputs[s]

        def call() -> None:
            _run_model_once(fn_model, x=x, shared_weights=shared_weights)

        try:
            timing = _time_one(call, device=device, args=args)
            row = _make_row(
                baseline=name,
                kernel_path=str(kernel_path),
                seq_len=s,
                dim=args.dim,
                inter_dim=args.inter_dim,
                experts=args.experts,
                topk=args.topk,
                timing=timing,
            )
        except Exception as e:
            row = _make_row(
                baseline=name,
                kernel_path=str(kernel_path),
                seq_len=s,
                dim=args.dim,
                inter_dim=args.inter_dim,
                experts=args.experts,
                topk=args.topk,
                timing=None,
                error=f"{type(e).__name__}: {e}",
            )

        if row.get("status") == "ok" and reference_outputs is not None:
            try:
                actual = _run_model_once(fn_model, x=x, shared_weights=shared_weights).to(torch.float32)
                expect = reference_outputs[s].to(torch.float32)
                abs_err = float((actual - expect).abs().max().item())
                rel_err = _max_rel_err(actual, expect)
                ok = bool(
                    torch.allclose(
                        actual,
                        expect,
                        atol=float(args.correctness_atol),
                        rtol=float(args.correctness_rtol),
                    )
                )
                row["correctness"] = {
                    "status": "ok",
                    "reference_kernel": str(args.reference_kernel) if args.reference_kernel else "",
                    "max_abs_err": abs_err,
                    "max_rel_err": rel_err,
                    "allclose": ok,
                    "atol": float(args.correctness_atol),
                    "rtol": float(args.correctness_rtol),
                }
            except Exception as e:
                row["correctness"] = {
                    "status": "error",
                    "reference_kernel": str(args.reference_kernel) if args.reference_kernel else "",
                    "error": f"{type(e).__name__}: {e}",
                }
        _print_row(row)
        rows.append(row)
    return rows


def _collect_env_metadata(device: "torch.device") -> Dict[str, Any]:
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


def _run_aiter_backend_helper(
    *,
    args: argparse.Namespace,
    seq_lens: List[int],
) -> List[Dict[str, Any]]:
    if args.skip_aiter_backends:
        return []

    helper = Path(args.aiter_helper_script)
    if not helper.exists():
        print(f"[warn] skip aiter backends: helper script missing: {helper}")
        return []

    if args.json_out is not None:
        out_dir = args.json_out.parent
    else:
        out_dir = Path(tempfile.mkdtemp(prefix="moe_aiter_helper_"))
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"moe_aiter_backends_{int(time.time())}"
    cmd = [
        sys.executable,
        str(helper),
        "--device",
        args.device,
        "--dim",
        str(args.dim),
        "--inter-dim",
        str(args.inter_dim),
        "--experts",
        str(args.experts),
        "--topk",
        str(args.topk),
        "--ms-list",
        ",".join(str(x) for x in seq_lens),
        "--backends",
        "asm,triton",
        "--warmup",
        str(args.warmup),
        "--warmup-ms",
        str(args.warmup_ms),
        "--repeat-ms",
        str(args.repeat_ms if args.measure_ms is None else args.measure_ms),
        "--trials",
        str(args.timer_trials),
        "--graph-iters",
        str(args.graph_iters),
        "--pre-capture-iters",
        str(args.pre_capture_iters),
        "--min-replays",
        str(args.min_replays),
        "--max-replays",
        str(args.max_replays),
        "--out-dir",
        str(out_dir),
        "--out-prefix",
        prefix,
    ]
    if args.disable_cudagraph:
        cmd.append("--disable-cudagraph")
    print(f"\n[aiter_backends] helper={helper}")
    # Ensure helper can import the shared timer module when invoked from nested paths.
    env = os.environ.copy()
    bench_dir = str(Path(__file__).resolve().parent)
    py_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{bench_dir}:{py_path}" if py_path else bench_dir
    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as first_err:
        # Known failure mode in source-mode aiter: stale enum/module ABI mismatch:
        # "could not convert default argument 'activation: ActivationType' ..."
        print(
            "[warn] aiter helper failed on first attempt; trying auto-repair "
            "(rebuild module_aiter_enum/module_moe_asm/module_moe_sorting)",
            flush=True,
        )
        repair_py = (
            "import pathlib\n"
            "import aiter\n"
            "jit = pathlib.Path(aiter.__file__).resolve().parent / 'jit'\n"
            "for name in ('module_aiter_enum.so','module_moe_asm.so','module_moe_sorting.so'):\n"
            "    p = jit / name\n"
            "    if p.exists():\n"
            "        p.unlink()\n"
            "        print(f'[aiter_repair] removed {p}')\n"
        )
        subprocess.run([sys.executable, "-c", repair_py], check=True, env=env)
        retry_env = env.copy()
        retry_env["AITER_REBUILD"] = "2"
        subprocess.run(cmd, check=True, env=retry_env)

    candidates = sorted(out_dir.glob(f"{prefix}_*.json"))
    if not candidates:
        print("[warn] aiter helper produced no json result")
        return []

    helper_payload = json.loads(candidates[-1].read_text(encoding="utf-8"))
    helper_rows = helper_payload.get("rows", [])
    out_rows: List[Dict[str, Any]] = []
    for r in helper_rows:
        backend = str(r.get("backend", "")).strip().lower()
        if backend == "asm":
            baseline = "AITER (asm)"
            kernel_path = "data/benchmarks/moe/05_aiter/ASM/src/moe_op.py"
        elif backend == "triton":
            baseline = "Triton (aiter backend)"
            kernel_path = "data/benchmarks/moe/05_aiter/Triton/src/moe_op.py"
        else:
            continue
        try:
            seq_len = int(r["tokens"])
            mean_ms = float(r["mean_ms"])
            median_ms = float(r["median_ms"])
            stdev_ms = float(r["stdev_ms"])
            p10_ms = float(r["p10_ms"])
            p90_ms = float(r["p90_ms"])
            cv = float(r["cv"])
            num_replays = int(r["num_replays"])
            graph_iters = int(r["graph_iters"])
            total_calls = int(r["total_calls_per_sample"])
        except Exception as e:
            print(f"[warn] malformed aiter helper row skipped: {type(e).__name__}: {e}")
            continue
        row = {
            "baseline": baseline,
            "kernel_path": kernel_path,
            "seq_len": seq_len,
            "dim": args.dim,
            "inter_dim": args.inter_dim,
            "experts": args.experts,
            "topk": args.topk,
            "status": "ok",
            "timing_mode": "cudagraph" if graph_iters > 0 else "eager",
            "median_ms": median_ms,
            "mean_ms": mean_ms,
            "stdev_ms": stdev_ms,
            "min_ms": float(r.get("min_ms", median_ms)),
            "max_ms": float(r.get("max_ms", median_ms)),
            "p10_ms": p10_ms,
            "p90_ms": p90_ms,
            "cv": cv,
            "tflops_median": _moe_tflops(
                tokens=seq_len,
                dim=args.dim,
                inter_dim=args.inter_dim,
                topk=args.topk,
                ms=median_ms,
            ),
            "eager_probe_ms": None,
            "suspicious": str(r.get("suspicious", "False")).lower() == "true",
            "suspicious_reason": str(r.get("suspicious_reason", "")),
            "num_replays": num_replays,
            "graph_iters": graph_iters,
            "warmup_calls": int(args.warmup),
            "total_calls_per_sample": total_calls,
            "trial_count": int(args.timer_trials),
            "samples_ms": [],
        }
        print(f"  T={seq_len:>6} | {baseline} | mean={mean_ms:.6f} ms")
        out_rows.append(row)
    return out_rows


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    moe_root = repo_root / "data" / "benchmarks" / "moe"
    p = argparse.ArgumentParser(
        description="Unified MoE timer with publication-grade CUDA Graph statistics."
    )
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument(
        "--hip-visible-devices",
        type=str,
        default="",
        help="Set HIP_VISIBLE_DEVICES for this process (e.g. '1' or '1,3').",
    )
    p.add_argument(
        "--cpu-cores",
        type=str,
        default="",
        help="Pin benchmark process to CPU cores, e.g. '0-15' or '0-7,16-23'.",
    )
    p.add_argument(
        "--seq-lens",
        type=str,
        default="1024,2048,4096,8192,16384",
        help="Comma-separated token lengths.",
    )
    p.add_argument("--dim", type=int, default=7168)
    p.add_argument("--inter-dim", type=int, default=2048)
    p.add_argument("--experts", type=int, default=32)
    p.add_argument("--topk", type=int, default=4)
    p.add_argument(
        "--input-dtype",
        type=str,
        default="fp8",
        choices=["fp8", "bf16"],
        help="Input/weight quantized tensor dtype used for benchmark inputs.",
    )
    p.add_argument("--seed", type=int, default=20260319)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--warmup-ms", type=float, default=200.0)
    p.add_argument("--graph-iters", type=int, default=10)
    p.add_argument(
        "--repeat",
        type=int,
        default=0,
        help="Fixed total calls per sample (legacy mode). If 0, use --measure-ms auto sizing.",
    )
    p.add_argument("--timer-trials", type=int, default=9)
    p.add_argument("--repeat-ms", type=float, default=1000.0)
    p.add_argument("--measure-ms", type=float, default=None)
    p.add_argument("--min-replays", type=int, default=5)
    p.add_argument("--max-replays", type=int, default=200000)
    p.add_argument("--pre-capture-iters", type=int, default=3)
    p.add_argument("--disable-cudagraph", action="store_true")
    p.add_argument(
        "--fallback-eager-on-capture-fail",
        action="store_true",
        help="If cudagraph capture fails, fallback to eager event timing instead of erroring out.",
    )
    p.add_argument(
        "--fallback-eager-on-suspicious-graph",
        action="store_true",
        help="If cudagraph timing is marked suspicious, fallback to eager event timing.",
    )
    p.add_argument("--suspicious-ratio", type=float, default=0.25)
    p.add_argument("--allow-suspicious-graph", action="store_true")
    default_baselines = [
        moe_root / "01_kernelbench" / "best_kernel.py",
        moe_root / "02_cudaforge" / "best_kernel.py",
        moe_root / "03_kernelfalcon" / "best_kernel.py",
        moe_root / "04_ksearch" / "best_kernel.py",
    ]
    p.add_argument(
        "--baseline-kernel",
        action="append",
        type=Path,
        default=None,
        help="Path to baseline MoE kernel python file; can be passed multiple times.",
    )
    p.add_argument(
        "--check-correctness",
        action="store_true",
        help="Compare each baseline output against --reference-kernel on shared inputs.",
    )
    p.add_argument(
        "--reference-kernel",
        type=Path,
        default=moe_root / "01_kernelbench" / "best_kernel.py",
        help="Reference kernel used by correctness checks.",
    )
    p.add_argument("--correctness-atol", type=float, default=0.5)
    p.add_argument("--correctness-rtol", type=float, default=0.05)
    p.add_argument(
        "--skip-aiter-backends",
        action="store_true",
        help="Skip AITER asm/triton backend rows in this MoE run.",
    )
    p.add_argument(
        "--aiter-helper-script",
        type=Path,
        default=moe_root / "05_aiter" / "tools" / "bench_moe_aiter_backends_cudagraph.py",
        help="Helper script path for AITER asm/triton MoE rows.",
    )
    p.add_argument("--json-out", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true")

    args = p.parse_args()
    if args.baseline_kernel is None:
        args.baseline_kernel = default_baselines
    return args


def main() -> None:
    args = parse_args()
    visible_devices = _apply_visible_devices(args.hip_visible_devices)
    cpu_affinity = _apply_cpu_affinity(args.cpu_cores)
    seq_lens = [int(x.strip()) for x in args.seq_lens.split(",") if x.strip()]
    baseline_paths = [Path(p) for p in args.baseline_kernel]

    if args.topk > args.experts:
        raise ValueError(f"topk ({args.topk}) must be <= experts ({args.experts})")
    if args.dim % BLOCK_K != 0 or args.dim % BLOCK_N != 0:
        raise ValueError(f"dim must be divisible by {BLOCK_N}/{BLOCK_K}, got {args.dim}")
    if args.inter_dim % BLOCK_K != 0:
        raise ValueError(f"inter_dim must be divisible by {BLOCK_K}, got {args.inter_dim}")

    if visible_devices:
        visible_count = len([x for x in visible_devices.split(",") if x])
        if args.device.startswith("cuda:"):
            local_idx = int(args.device.split(":", 1)[1])
            if local_idx < 0 or local_idx >= visible_count:
                raise ValueError(
                    f"--device {args.device} is out of range for visible devices [{visible_devices}] "
                    f"(use local index cuda:0..cuda:{visible_count - 1})"
                )

    effective_repeat_ms = args.repeat_ms if args.measure_ms is None else args.measure_ms
    mode = f"fixed_calls={args.repeat}" if args.repeat > 0 else f"auto_repeat_ms={effective_repeat_ms}"
    print(
        f"[config] device={args.device} seq_lens={seq_lens} "
        f"dim={args.dim} inter_dim={args.inter_dim} experts={args.experts} topk={args.topk} "
        f"input_dtype={args.input_dtype} seed={args.seed} warmup={args.warmup} "
        f"graph_iters={args.graph_iters} timer_trials={args.timer_trials} mode={mode}"
    )
    print(
        f"[config] suspicious_ratio={args.suspicious_ratio} "
        f"allow_suspicious_graph={args.allow_suspicious_graph}"
    )
    print(
        f"[config] skip_aiter_backends={bool(args.skip_aiter_backends)} "
        f"aiter_helper_script={args.aiter_helper_script}"
    )
    if visible_devices:
        print(f"[config] HIP_VISIBLE_DEVICES={visible_devices}")
    if cpu_affinity:
        affinity_label = ",".join(str(c) for c in cpu_affinity)
        print(f"[config] CPU_AFFINITY={affinity_label}")
    print(f"[discover] baseline_kernels={len(baseline_paths)}")

    if args.dry_run:
        for p in baseline_paths:
            print(f"[baseline] {p}")
        return

    if torch is None:
        raise RuntimeError("torch is required. Activate ROCm/PyTorch environment first.")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    input_dtype = _resolve_input_dtype(args.input_dtype)
    shared_inputs = _build_shared_inputs(
        seq_lens=seq_lens,
        dim=args.dim,
        experts=args.experts,
        topk=args.topk,
        input_dtype=input_dtype,
        device=device,
        seed=args.seed,
    )
    shared_weights = _build_shared_weights(
        dim=args.dim,
        inter_dim=args.inter_dim,
        experts=args.experts,
        input_dtype=input_dtype,
        device=device,
        seed=args.seed,
    )

    rows: List[Dict[str, Any]] = []
    reference_outputs: Dict[int, "torch.Tensor"] | None = None
    if args.check_correctness:
        if not Path(args.reference_kernel).exists():
            raise FileNotFoundError(f"reference kernel not found: {args.reference_kernel}")
        reference_outputs = _build_reference_outputs(
            Path(args.reference_kernel),
            seq_lens=seq_lens,
            shared_inputs=shared_inputs,
            shared_weights=shared_weights,
            device=device,
        )

    for i, p in enumerate(baseline_paths, start=1):
        name = f"baseline_{i}"
        if not p.exists():
            print(f"\n[{name}] missing: {p}")
            for s in seq_lens:
                row = _make_row(
                    baseline=name,
                    kernel_path=str(p),
                    seq_len=s,
                    dim=args.dim,
                    inter_dim=args.inter_dim,
                    experts=args.experts,
                    topk=args.topk,
                    timing=None,
                    error="missing kernel file",
                )
                _print_row(row)
                rows.append(row)
            continue
        rows.extend(
            run_python_kernel_baseline(
                name=name,
                kernel_path=p,
                seq_lens=seq_lens,
                shared_inputs=shared_inputs,
                shared_weights=shared_weights,
                device=device,
                args=args,
                reference_outputs=reference_outputs,
            )
        )

    try:
        rows.extend(
            _run_aiter_backend_helper(
                args=args,
                seq_lens=seq_lens,
            )
        )
    except Exception as e:
        print(f"[warn] aiter backend helper failed: {type(e).__name__}: {e}")

    if args.json_out:
        payload = {
            "config": {
                "device": args.device,
                "hip_visible_devices": visible_devices,
                "cpu_affinity": cpu_affinity,
                "seq_lens": seq_lens,
                "dim": args.dim,
                "inter_dim": args.inter_dim,
                "experts": args.experts,
                "topk": args.topk,
                "input_dtype": args.input_dtype,
                "seed": args.seed,
                "warmup": args.warmup,
                "warmup_ms": args.warmup_ms,
                "graph_iters": args.graph_iters,
                "repeat": args.repeat,
                "repeat_ms": effective_repeat_ms,
                "timer_trials": args.timer_trials,
                "min_replays": args.min_replays,
                "max_replays": args.max_replays,
                "pre_capture_iters": args.pre_capture_iters,
                "disable_cudagraph": bool(args.disable_cudagraph),
                "suspicious_ratio": args.suspicious_ratio,
                "allow_suspicious_graph": args.allow_suspicious_graph,
                "baseline_kernels": [str(p) for p in baseline_paths],
                "check_correctness": bool(args.check_correctness),
                "reference_kernel": str(args.reference_kernel) if args.reference_kernel else None,
                "correctness_atol": float(args.correctness_atol),
                "correctness_rtol": float(args.correctness_rtol),
                "skip_aiter_backends": bool(args.skip_aiter_backends),
                "aiter_helper_script": str(args.aiter_helper_script),
            },
            "environment": _collect_env_metadata(device),
            "rows": rows,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2))
        print(f"[saved] {args.json_out}")


if __name__ == "__main__":
    main()
