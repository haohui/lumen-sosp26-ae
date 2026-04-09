#!/usr/bin/env python3
"""Unified GEMM benchmark with publication-grade CUDA Graph timing."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, List, Tuple

from cudagraph_timer import CUDAGraphTimingResult, benchmark_with_cudagraph

try:
    import torch
except Exception:
    torch = None


@dataclass
class SharedInputs:
    a_mk: "torch.Tensor"
    b_kn: "torch.Tensor"


def _tflops(m: int, n: int, k: int, ms: float) -> float:
    return (2.0 * m * n * k) / (ms * 1.0e-3) / 1.0e12


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_model_fn(module: ModuleType, device: "torch.device", dtype: "torch.dtype"):
    if hasattr(module, "ModelNew"):
        model = module.ModelNew().to(device=device, dtype=dtype)
        return lambda a, b: model(a, b)
    if hasattr(module, "Model"):
        model = module.Model().to(device=device, dtype=dtype)
        return lambda a, b: model(a, b)
    if hasattr(module, "kernel_function"):
        return lambda a, b: module.kernel_function(a, b)
    raise RuntimeError("expected one of: ModelNew, Model, kernel_function")


def _discover_python_files(root: Path) -> List[Path]:
    out: List[Path] = []
    for p in sorted(root.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        if p.name.startswith("benchmark_"):
            continue
        out.append(p)
    return out


def _discover_cu_files(root: Path) -> List[Path]:
    return sorted(root.rglob("*.cu"))


def _parse_cpu_cores(spec: str) -> List[int]:
    cores: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            parts = token.split("-", 1)
            if len(parts) != 2 or not parts[0] or not parts[1]:
                raise ValueError(f"Invalid cpu core range token: {token}")
            lo, hi = int(parts[0]), int(parts[1])
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
    # Keep one source of truth for PyTorch HIP parser.
    if "ROCR_VISIBLE_DEVICES" in os.environ:
        os.environ.pop("ROCR_VISIBLE_DEVICES")
    return normalized


def _build_shared_inputs(
    *,
    sizes: List[int],
    device: "torch.device",
    dtype: "torch.dtype",
    seed: int,
) -> Dict[int, SharedInputs]:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    shared: Dict[int, SharedInputs] = {}
    for s in sizes:
        a_mk = torch.randn((s, s), device=device, dtype=dtype, generator=g)
        b_kn = torch.randn((s, s), device=device, dtype=dtype, generator=g)
        shared[s] = SharedInputs(a_mk=a_mk, b_kn=b_kn)
    return shared


def _load_hipblaslt_internal_module(gemm_root: Path):
    from torch.utils.cpp_extension import load_inline

    baseline_src = gemm_root / "06_hipblaslt" / "src" / "hipblaslt_internal_ext.cpp"
    src = baseline_src
    if not src.exists():
        raise FileNotFoundError(f"missing source: {src}")

    vendor_lib_dir = gemm_root / "06_hipblaslt" / "runtime_libs"
    system_rocm_lib = Path("/opt/rocm/lib")
    system_rocm_711_lib = Path("/opt/rocm-7.1.1/lib")
    rocm_lib_dirs = [
        system_rocm_lib,
        system_rocm_711_lib,
        Path("/opt/rocm-6.4.3/lib"),
        vendor_lib_dir,
    ]
    lib_dirs: List[Path] = []
    for p in rocm_lib_dirs:
        if p.exists() and p.is_dir() and p not in lib_dirs:
            lib_dirs.append(p)
    if not lib_dirs:
        raise FileNotFoundError(
            f"missing ROCm libs: tried {[str(p) for p in rocm_lib_dirs]}"
        )

    ldflags: List[str] = []
    for d in lib_dirs:
        ldflags.extend([f"-L{d}", f"-Wl,-rpath,{d}"])
    ldflags.extend(["-lhipblaslt", "-lhipblas", "-lrocblas", "-lamdhip64"])

    # Pin hipBLASLt to the runtime library directory that contains Tensile data.
    tensile_candidates = [
        system_rocm_lib / "hipblaslt" / "library",
        system_rocm_711_lib / "hipblaslt" / "library",
        vendor_lib_dir / "hipblaslt" / "library",
    ]
    for d in tensile_candidates:
        if (d / "TensileLibrary_lazy_gfx942.dat").exists():
            os.environ["HIPBLASLT_TENSILE_LIBPATH"] = str(d)
            break

    os.environ.setdefault("CXX", "hipcc")
    os.environ.setdefault("MAX_JOBS", "4")
    cpp_src = src.read_text(encoding="utf-8")
    return load_inline(
        name="kb_hipblaslt_internal_ext_mini",
        cpp_sources=cpp_src,
        functions=["hipblaslt_bf16_mm_out"],
        extra_cflags=["-O3"],
        extra_ldflags=ldflags,
        with_cuda=False,
        verbose=False,
    )


def _resolve_hipkittens_kernel(case_n: int, kernels_dir: Path) -> Tuple[str, Path]:
    """Resolve a prebuilt HipKittens extension module for a given matrix size."""
    module_name = f"tk_kernel_{case_n}_mini"
    matches = sorted(kernels_dir.glob(f"{module_name}*.so"))
    if not matches:
        raise FileNotFoundError(
            f"missing prebuilt HipKittens module for N={case_n}: "
            f"expected {module_name}*.so under {kernels_dir}"
        )
    so_path = max(matches, key=lambda p: p.stat().st_mtime)
    return module_name, so_path


def _load_so_module(name: str, so_path: Path):
    if name in sys.modules:
        del sys.modules[name]
    importlib.invalidate_caches()
    spec = importlib.util.spec_from_file_location(name, str(so_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import extension: {so_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[name] = mod
    return mod


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
    return benchmark_with_cudagraph(
        fn=call,
        device=device,
        **_timer_kwargs(args),
    )


def _make_row(
    *,
    baseline: str,
    kernel_path: str,
    size: int,
    timing: CUDAGraphTimingResult | None,
    error: str | None = None,
) -> Dict[str, Any]:
    if timing is None:
        return {
            "baseline": baseline,
            "kernel_path": kernel_path,
            "m": size,
            "n": size,
            "k": size,
            "status": "error",
            "error": error,
        }
    median_ms = timing.median_ms
    return {
        "baseline": baseline,
        "kernel_path": kernel_path,
        "m": size,
        "n": size,
        "k": size,
        "status": "ok",
        "median_ms": median_ms,
        "mean_ms": timing.mean_ms,
        "stdev_ms": timing.stdev_ms,
        "min_ms": timing.min_ms,
        "max_ms": timing.max_ms,
        "p10_ms": timing.p10_ms,
        "p90_ms": timing.p90_ms,
        "cv": timing.cv,
        "tflops_median": _tflops(size, size, size, median_ms),
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
    s = int(row["m"])
    if row.get("status") != "ok":
        print(f"  M=N=K={s:>5} | error | {row.get('error', 'unknown')}")
        return
    flag = " [SUSPICIOUS]" if row.get("suspicious") else ""
    print(
        f"  M=N=K={s:>5} | med={row['median_ms']:>8.4f} ms | "
        f"p10/p90={row['p10_ms']:.4f}/{row['p90_ms']:.4f} | "
        f"cv={100.0 * row['cv']:.2f}% | {row['tflops_median']:>8.2f} TFLOPS | "
        f"trials={row['trial_count']} replays={row['num_replays']} graph_iters={row['graph_iters']}{flag}"
    )


def run_python_kernel_baseline(
    name: str,
    kernel_path: Path,
    *,
    sizes: List[int],
    shared_inputs: Dict[int, SharedInputs],
    device: "torch.device",
    dtype: "torch.dtype",
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    module = _load_module(kernel_path)
    fn_model = _build_model_fn(module, device=device, dtype=dtype)
    print(f"\n[{name}] {kernel_path}")
    for s in sizes:
        x = shared_inputs[s]

        def call() -> None:
            fn_model(x.a_mk, x.b_kn)

        try:
            timing = _time_one(call, device=device, args=args)
            row = _make_row(baseline=name, kernel_path=str(kernel_path), size=s, timing=timing)
        except Exception as e:
            row = _make_row(
                baseline=name,
                kernel_path=str(kernel_path),
                size=s,
                timing=None,
                error=f"{type(e).__name__}: {e}",
            )
        _print_row(row)
        rows.append(row)
    return rows


def run_aiter_baseline(
    *,
    sizes: List[int],
    shared_inputs: Dict[int, SharedInputs],
    device: "torch.device",
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    import aiter

    rows: List[Dict[str, Any]] = []
    name = "aiter"
    print(f"\n[{name}]")
    for s in sizes:
        x = shared_inputs[s]
        a = x.a_mk
        b_t = x.b_kn.t().contiguous()  # aiter expects transposed B layout
        y = torch.empty((s, s), device=device, dtype=torch.float32)

        def call() -> None:
            aiter.gemm_a16w16_asm(a, b_t, y)

        try:
            timing = _time_one(call, device=device, args=args)
            row = _make_row(baseline=name, kernel_path="aiter.gemm_a16w16_asm", size=s, timing=timing)
        except Exception as e:
            row = _make_row(
                baseline=name,
                kernel_path="aiter.gemm_a16w16_asm",
                size=s,
                timing=None,
                error=f"{type(e).__name__}: {e}",
            )
        _print_row(row)
        rows.append(row)
    return rows


def run_hipblaslt_baseline(
    *,
    gemm_root: Path,
    sizes: List[int],
    shared_inputs: Dict[int, SharedInputs],
    device: "torch.device",
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    name = "hipblaslt-internal"
    print(f"\n[{name}]")
    mod = _load_hipblaslt_internal_module(gemm_root)
    for s in sizes:
        x = shared_inputs[s]
        d = torch.empty((s, s), dtype=torch.bfloat16, device=device)

        def call() -> None:
            mod.hipblaslt_bf16_mm_out(x.a_mk, x.b_kn, d)

        try:
            timing = _time_one(call, device=device, args=args)
            row = _make_row(
                baseline=name,
                kernel_path=(
                    "data/benchmarks/gemm/06_hipblaslt/src/hipblaslt_internal_ext.cpp"
                    "::hipblaslt_bf16_mm_out"
                ),
                size=s,
                timing=timing,
            )
        except Exception as e:
            row = _make_row(
                baseline=name,
                kernel_path=(
                    "data/benchmarks/gemm/06_hipblaslt/src/hipblaslt_internal_ext.cpp"
                    "::hipblaslt_bf16_mm_out"
                ),
                size=s,
                timing=None,
                error=f"{type(e).__name__}: {e}",
            )
        _print_row(row)
        rows.append(row)
    return rows


def run_hipkittens_baseline(
    *,
    sizes: List[int],
    shared_inputs: Dict[int, SharedInputs],
    device: "torch.device",
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    name = "hipkittens"
    print(f"\n[{name}]")
    kernels_dir = Path(args.hipkittens_kernels_dir).resolve()
    for s in sizes:
        try:
            module_name, so_path = _resolve_hipkittens_kernel(s, kernels_dir)
            mod = _load_so_module(module_name, so_path)
        except Exception as e:
            row = _make_row(
                baseline=name,
                kernel_path=str(kernels_dir),
                size=s,
                timing=None,
                error=f"{type(e).__name__}: {e}",
            )
            _print_row(row)
            rows.append(row)
            continue

        x = shared_inputs[s]
        b_t = x.b_kn.t().contiguous()
        c = torch.empty((s, s), device=device, dtype=x.a_mk.dtype)

        def call() -> None:
            stream_ptr = int(torch.cuda.current_stream(device=device).cuda_stream)
            mod.dispatch_micro(x.a_mk, b_t, c, stream_ptr)

        try:
            timing = _time_one(call, device=device, args=args)
            row = _make_row(baseline=name, kernel_path=str(so_path), size=s, timing=timing)
        except Exception as e:
            row = _make_row(
                baseline=name,
                kernel_path=str(so_path),
                size=s,
                timing=None,
                error=f"{type(e).__name__}: {e}",
            )
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


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    gemm_root = repo_root / "data" / "benchmarks" / "gemm"
    p = argparse.ArgumentParser(
        description="Unified GEMM timer with publication-grade CUDA Graph statistics."
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
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--sizes", type=str, default="1024,2048,4096,8192,9216,14592,16384")
    p.add_argument("--seed", type=int, default=20260312)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--warmup-ms", type=float, default=200.0, help="Target eager warmup duration in ms.")
    p.add_argument("--graph-iters", type=int, default=10)
    p.add_argument(
        "--repeat",
        type=int,
        default=0,
        help="Fixed total calls per sample (legacy mode). If 0, use --measure-ms auto sizing.",
    )
    p.add_argument("--timer-trials", type=int, default=9)
    p.add_argument("--repeat-ms", type=float, default=1000.0, help="Target timed replay duration per trial in ms.")
    p.add_argument(
        "--measure-ms",
        type=float,
        default=None,
        help="Deprecated alias for --repeat-ms; if set, overrides --repeat-ms.",
    )
    p.add_argument("--min-replays", type=int, default=5)
    p.add_argument("--max-replays", type=int, default=200000)
    p.add_argument("--pre-capture-iters", type=int, default=3)
    p.add_argument("--suspicious-ratio", type=float, default=0.25)
    p.add_argument(
        "--allow-suspicious-graph",
        action="store_true",
        help="Keep suspicious graph results instead of treating them as errors.",
    )
    p.add_argument("--run-aiter", action="store_true")
    p.add_argument("--run-hipblaslt", action="store_true")
    p.add_argument("--run-hipkittens", action="store_true")
    p.add_argument("--gemm-kernel-root", type=Path, default=gemm_root)
    p.add_argument(
        "--hipkittens-kernels-dir",
        type=Path,
        default=gemm_root / "08_hipketten" / "build_hipkittens_mini",
    )
    p.add_argument("--json-out", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    visible_devices = _apply_visible_devices(args.hip_visible_devices)
    cpu_affinity = _apply_cpu_affinity(args.cpu_cores)
    sizes = [int(x.strip()) for x in args.sizes.split(",") if x.strip()]
    repo_root = Path(__file__).resolve().parents[3]
    gemm_root = repo_root / "data" / "benchmarks" / "gemm"

    if visible_devices:
        visible_count = len([x for x in visible_devices.split(",") if x])
        if args.device.startswith("cuda:"):
            local_idx = int(args.device.split(":", 1)[1])
            if local_idx < 0 or local_idx >= visible_count:
                raise ValueError(
                    f"--device {args.device} is out of range for visible devices [{visible_devices}] "
                    f"(use local index cuda:0..cuda:{visible_count - 1})"
                )

    py_kernels = _discover_python_files(args.gemm_kernel_root)
    cu_kernels = _discover_cu_files(args.gemm_kernel_root)

    effective_repeat_ms = args.repeat_ms if args.measure_ms is None else args.measure_ms
    mode = f"fixed_calls={args.repeat}" if args.repeat > 0 else f"auto_repeat_ms={effective_repeat_ms}"
    print(
        f"[config] device={args.device} dtype={args.dtype} sizes={sizes} seed={args.seed} "
        f"warmup={args.warmup} graph_iters={args.graph_iters} timer_trials={args.timer_trials} "
        f"mode={mode} warmup_ms={args.warmup_ms} pre_capture_iters={args.pre_capture_iters}"
    )
    print(
        f"[config] suspicious_ratio={args.suspicious_ratio} "
        f"allow_suspicious_graph={args.allow_suspicious_graph}"
    )
    if visible_devices:
        print(f"[config] HIP_VISIBLE_DEVICES={visible_devices}")
    if cpu_affinity:
        affinity_label = ",".join(str(c) for c in cpu_affinity)
        print(f"[config] CPU_AFFINITY={affinity_label}")
    print(f"[discover] python_kernels={len(py_kernels)} cu_kernels={len(cu_kernels)}")

    if args.dry_run:
        for p in py_kernels:
            print(f"[py] {p}")
        for p in cu_kernels:
            print(f"[cu-skip] {p} (raw .cu needs a dedicated wrapper/harness)")
        if args.run_aiter:
            print("[dry-run] aiter enabled")
        if args.run_hipblaslt:
            print("[dry-run] hipblaslt enabled")
        if args.run_hipkittens:
            print("[dry-run] hipkittens enabled")
        return

    if torch is None:
        raise RuntimeError("torch is required. Activate ROCm/PyTorch environment first.")

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    shared_inputs = _build_shared_inputs(sizes=sizes, device=device, dtype=dtype, seed=args.seed)

    rows: List[Dict[str, Any]] = []
    if args.run_aiter:
        try:
            rows.extend(
                run_aiter_baseline(
                    sizes=sizes,
                    shared_inputs=shared_inputs,
                    device=device,
                    args=args,
                )
            )
        except Exception as e:
            print(f"[skip] aiter failed: {type(e).__name__}: {e}")

    if args.run_hipblaslt:
        try:
            rows.extend(
                run_hipblaslt_baseline(
                    gemm_root=gemm_root,
                    sizes=sizes,
                    shared_inputs=shared_inputs,
                    device=device,
                    args=args,
                )
            )
        except Exception as e:
            print(f"[skip] hipblaslt failed: {type(e).__name__}: {e}")

    if args.run_hipkittens:
        try:
            rows.extend(
                run_hipkittens_baseline(
                    sizes=sizes,
                    shared_inputs=shared_inputs,
                    device=device,
                    args=args,
                )
            )
        except Exception as e:
            print(f"[skip] hipkittens failed: {type(e).__name__}: {e}")

    for p in py_kernels:
        name = f"baseline::{p.relative_to(args.gemm_kernel_root)}"
        try:
            rows.extend(
                run_python_kernel_baseline(
                    name=name,
                    kernel_path=p,
                    sizes=sizes,
                    shared_inputs=shared_inputs,
                    device=device,
                    dtype=dtype,
                    args=args,
                )
            )
        except Exception as e:
            print(f"\n[{name}] skip | {type(e).__name__}: {e}")

    for p in cu_kernels:
        print(
            f"\n[baseline::{p.relative_to(args.gemm_kernel_root)}] "
            "skip | raw .cu needs dedicated wrapper/harness"
        )

    if args.json_out is not None:
        payload = {
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "config": {
                "device": args.device,
                "dtype": args.dtype,
                "hip_visible_devices": visible_devices,
                "cpu_cores": args.cpu_cores,
                "cpu_affinity_applied": cpu_affinity,
                "sizes": sizes,
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
                "run_aiter": args.run_aiter,
                "run_hipblaslt": args.run_hipblaslt,
                "run_hipkittens": args.run_hipkittens,
                "hipkittens_kernels_dir": str(args.hipkittens_kernels_dir),
                "gemm_kernel_root": str(args.gemm_kernel_root),
            },
            "environment": _collect_env_metadata(device),
            "rows": rows,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\n[json] wrote {args.json_out}")


if __name__ == "__main__":
    main()
