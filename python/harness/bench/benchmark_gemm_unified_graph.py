#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from common import (
    add_common_runtime_args,
    add_timer_args,
    apply_cpu_affinity,
    apply_visible_devices,
    build_model_fn,
    effective_repeat_ms,
    enable_default_flags,
    load_module,
    maybe_write_json,
    parse_dtype,
    parse_int_csv,
    time_call,
    timing_fields,
    validate_device_local_index,
)
from gemm_runtime import (
    load_extension_module,
    load_hipblaslt_internal_module,
    resolve_hipkittens_kernel,
)

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


def _build_shared_inputs(*, sizes: List[int], device: "torch.device", dtype: "torch.dtype", seed: int) -> Dict[int, SharedInputs]:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    out: Dict[int, SharedInputs] = {}
    for s in sizes:
        out[s] = SharedInputs(
            a_mk=torch.randn((s, s), device=device, dtype=dtype, generator=g),
            b_kn=torch.randn((s, s), device=device, dtype=dtype, generator=g),
        )
    return out


def _row(*, baseline: str, kernel_path: str, size: int, timing) -> Dict[str, Any]:
    return {
        "baseline": baseline,
        "kernel_path": kernel_path,
        "m": size,
        "n": size,
        "k": size,
        **timing_fields(timing, tflops_median=_tflops(size, size, size, timing.median_ms)),
    }


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    gemm_root = repo_root / "data" / "benchmarks" / "gemm"

    p = argparse.ArgumentParser(description="GEMM benchmark with CUDA Graph timing")
    add_common_runtime_args(p)
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--sizes", type=str, default="1024,2048,4096,8192,16384")
    add_timer_args(p)

    p.add_argument("--run-aiter", action="store_true")
    p.add_argument("--run-hipblaslt", action="store_true")
    p.add_argument("--run-hipkittens", action="store_true")
    p.add_argument("--run-kernelbench", action="store_true")
    p.add_argument("--run-cudaforge", action="store_true")
    p.add_argument("--run-kernelfalcon", action="store_true")
    p.add_argument("--run-ksearch", action="store_true")
    p.add_argument("--run-triton", action="store_true")

    p.add_argument("--aiter-kernel", type=Path, default=gemm_root / "05_aiter" / "best_kernel.py")
    p.add_argument(
        "--hipkittens-kernels-dir",
        type=Path,
        default=gemm_root / "08_hipketten" / "build_hipkittens_mini",
    )
    p.add_argument("--json-out", type=Path, default=None)

    args = p.parse_args()
    enable_default_flags(
        args,
        [
            "run_aiter",
            "run_hipblaslt",
            "run_hipkittens",
            "run_kernelbench",
            "run_cudaforge",
            "run_kernelfalcon",
            "run_ksearch",
            "run_triton",
        ],
    )
    return args


def main() -> None:
    args = parse_args()
    visible = apply_visible_devices(args.hip_visible_devices)
    cpu_aff = apply_cpu_affinity(args.cpu_cores)
    validate_device_local_index(args.device, visible)

    sizes = parse_int_csv(args.sizes, name="sizes")
    repo_root = Path(__file__).resolve().parents[3]
    gemm_root = repo_root / "data" / "benchmarks" / "gemm"

    py_baselines: List[tuple[str, Path, bool]] = [
        ("kernelbench", gemm_root / "01_kernelbench" / "best_kernel.py", args.run_kernelbench),
        ("cudaforge", gemm_root / "02_cudaforge" / "best_kernel.py", args.run_cudaforge),
        ("kernelfalcon", gemm_root / "03_kernelfalcon" / "best_kernel.py", args.run_kernelfalcon),
        ("ksearch", gemm_root / "04_ksearch" / "best_kernel.py", args.run_ksearch),
        ("triton", gemm_root / "07_triton" / "best_kernel.py", args.run_triton),
    ]

    if torch is None:
        raise RuntimeError("torch is required")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = parse_dtype(args.dtype)
    if args.run_hipblaslt and dtype is not torch.bfloat16:
        raise ValueError(
            "--run-hipblaslt requires --dtype bf16; "
            "hipblaslt_bf16_mm_out only supports bf16 inputs"
        )
    shared = _build_shared_inputs(sizes=sizes, device=device, dtype=dtype, seed=args.seed)

    rows: List[Dict[str, Any]] = []

    if args.run_aiter:
        mod = load_module(args.aiter_kernel)
        fn = build_model_fn(mod, device=device, dtype=dtype)
        for s in sizes:
            x = shared[s]
            timing = time_call(lambda: fn(x.a_mk, x.b_kn), device=device, args=args)
            row = _row(baseline="aiter", kernel_path=str(args.aiter_kernel), size=s, timing=timing)
            rows.append(row)

    if args.run_hipblaslt:
        hipblaslt_mod = load_hipblaslt_internal_module(gemm_root)
        for s in sizes:
            x = shared[s]
            out = torch.empty((s, s), dtype=torch.bfloat16, device=device)
            timing = time_call(lambda: hipblaslt_mod.hipblaslt_bf16_mm_out(x.a_mk, x.b_kn, out), device=device, args=args)
            row = _row(
                baseline="hipblaslt",
                kernel_path="data/benchmarks/gemm/06_hipblaslt/src/hipblaslt_internal_ext.cpp::hipblaslt_bf16_mm_out",
                size=s,
                timing=timing,
            )
            rows.append(row)

    if args.run_hipkittens:
        kernels_dir = args.hipkittens_kernels_dir.resolve()
        for s in sizes:
            x = shared[s]
            module_name, so_path = resolve_hipkittens_kernel(s, kernels_dir)
            mod = load_extension_module(module_name, so_path)
            b_t = x.b_kn.t().contiguous()
            out = torch.empty((s, s), device=device, dtype=x.a_mk.dtype)
            dispatch = mod.dispatch_micro
            dispatch_doc = str(getattr(dispatch, "__doc__", "") or "")
            takes_stream_ptr = "arg3" in dispatch_doc
            if takes_stream_ptr:
                call = lambda: dispatch(
                    x.a_mk,
                    b_t,
                    out,
                    int(torch.cuda.current_stream(device=device).cuda_stream),
                )
            else:
                call = lambda: dispatch(x.a_mk, b_t, out)
            timing = time_call(
                call,
                device=device,
                args=args,
            )
            row = _row(baseline="hipkittens", kernel_path=str(so_path), size=s, timing=timing)
            rows.append(row)

    for name, path, on in py_baselines:
        if not on:
            continue
        mod = load_module(path)
        fn = build_model_fn(mod, device=device, dtype=dtype)
        for s in sizes:
            x = shared[s]
            timing = time_call(lambda: fn(x.a_mk, x.b_kn), device=device, args=args)
            row = _row(baseline=name, kernel_path=str(path), size=s, timing=timing)
            rows.append(row)

    maybe_write_json(
        json_out=args.json_out,
        device=device,
        config={
            "device": args.device,
            "dtype": args.dtype,
            "hip_visible_devices": visible,
            "cpu_affinity": cpu_aff,
            "sizes": sizes,
            "seed": args.seed,
            "warmup": args.warmup,
            "warmup_ms": args.warmup_ms,
            "graph_iters": args.graph_iters,
            "repeat": args.repeat,
            "repeat_ms": effective_repeat_ms(args),
            "timer_trials": args.timer_trials,
            "min_replays": args.min_replays,
            "max_replays": args.max_replays,
            "pre_capture_iters": args.pre_capture_iters,
            "run_aiter": args.run_aiter,
            "run_hipblaslt": args.run_hipblaslt,
            "run_hipkittens": args.run_hipkittens,
            "run_kernelbench": args.run_kernelbench,
            "run_cudaforge": args.run_cudaforge,
            "run_kernelfalcon": args.run_kernelfalcon,
            "run_ksearch": args.run_ksearch,
            "run_triton": args.run_triton,
            "aiter_kernel": str(args.aiter_kernel),
            "hipkittens_kernels_dir": str(args.hipkittens_kernels_dir),
        },
        rows=rows,
    )


if __name__ == "__main__":
    main()
