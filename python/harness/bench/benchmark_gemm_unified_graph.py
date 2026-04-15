#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from common import (
    add_common_runtime_args,
    add_timer_args,
    apply_cpu_affinity,
    apply_visible_devices,
    build_csv_row,
    build_model_fn,
    enable_default_flags,
    load_module,
    maybe_write_csv,
    now_utc,
    parse_dtype,
    parse_int_csv,
    time_call,
    timing_fields,
    validate_device_local_index,
)
from gemm_runtime import (
    SharedInputs,
    build_shared_inputs,
    gemm_tflops,
    load_hipblaslt_internal_module,
)

try:
    import torch
except Exception:
    torch = None


def _row(*, baseline: str, kernel_path: str, size: int, timing) -> Dict[str, Any]:
    return {
        "baseline": baseline,
        "kernel_path": kernel_path,
        "m": size,
        "n": size,
        "k": size,
        **timing_fields(timing, tflops_median=gemm_tflops(size, size, size, timing.median_ms)),
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
    p.add_argument("--csv-out", type=Path, default=None)
    p.add_argument("--run-id", type=str, default="")

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
    apply_cpu_affinity(args.cpu_cores)
    validate_device_local_index(args.device, visible)

    sizes = parse_int_csv(args.sizes, name="sizes")
    repo_root = Path(__file__).resolve().parents[3]
    gemm_root = repo_root / "data" / "benchmarks" / "gemm"

    py_baselines: List[tuple[str, Path, bool]] = [
        ("aiter", args.aiter_kernel, args.run_aiter),
        ("hipkittens", gemm_root / "08_hipketten" / "best_kernel.py", args.run_hipkittens),
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
        args.run_hipblaslt = False
    shared = build_shared_inputs(sizes=sizes, device=device, dtype=dtype, seed=args.seed)

    timestamp = now_utc()
    rows: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []

    if args.run_hipblaslt:
        hipblaslt_mod = load_hipblaslt_internal_module(gemm_root)
        for s in sizes:
            x = shared[s]
            out = torch.empty((s, s), dtype=torch.bfloat16, device=device)
            # hipblaslt_bf16_mm_out directly consumes B:[N,K] and computes A @ B^T.
            timing = time_call(lambda: hipblaslt_mod.hipblaslt_bf16_mm_out(x.a_mk, x.b_nk, out), device=device, args=args)
            row = _row(
                baseline="hipblaslt",
                kernel_path="data/benchmarks/gemm/06_hipblaslt/hipblaslt_internal_ext.cpp::hipblaslt_bf16_mm_out",
                size=s,
                timing=timing,
            )
            rows.append(row)
            csv_rows.append(
                build_csv_row(
                    domain="gemm",
                    baseline="hipblaslt",
                    workload=s,
                    mean_ms=timing.mean_ms,
                    tflops=gemm_tflops(s, s, s, timing.mean_ms),
                    status=row["status"],
                    kernel_entry=row["kernel_path"],
                    timestamp_utc=timestamp,
                    run_id=args.run_id,
                )
            )

    for name, path, on in py_baselines:
        if not on:
            continue
        mod = load_module(path)
        fn = build_model_fn(mod, device=device, dtype=dtype)
        for s in sizes:
            x = shared[s]
            # Unified contract: all Python baselines consume B:[N,K] and compute A @ B^T directly.
            timing = time_call(lambda: fn(x.a_mk, x.b_nk), device=device, args=args)
            row = _row(baseline=name, kernel_path=str(path), size=s, timing=timing)
            rows.append(row)
            csv_rows.append(
                build_csv_row(
                    domain="gemm",
                    baseline=name,
                    workload=s,
                    mean_ms=timing.mean_ms,
                    tflops=gemm_tflops(s, s, s, timing.mean_ms),
                    status=row["status"],
                    kernel_entry=str(path),
                    timestamp_utc=timestamp,
                    run_id=args.run_id,
                )
            )

    maybe_write_csv(csv_out=args.csv_out, rows=csv_rows)


if __name__ == "__main__":
    main()
