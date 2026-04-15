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
from moe_runtime import (
    BLOCK_K,
    BLOCK_N,
    build_shared_inputs,
    build_shared_weights,
    run_model_once,
)

try:
    import torch
except Exception:
    torch = None


def _moe_tflops(*, tokens: int, dim: int, inter_dim: int, topk: int, ms: float) -> float:
    if ms <= 0.0:
        return float("nan")
    flops = 6.0 * float(tokens) * float(topk) * float(dim) * float(inter_dim)
    return flops / (ms * 1.0e-3) / 1.0e12


def _row(*, baseline: str, kernel_path: str, seq_len: int, args: argparse.Namespace, timing) -> Dict[str, Any]:
    return {
        "baseline": baseline,
        "kernel_path": kernel_path,
        "seq_len": seq_len,
        "dim": args.dim,
        "inter_dim": args.inter_dim,
        "experts": args.experts,
        "topk": args.topk,
        "timing_mode": "cudagraph",
        **timing_fields(
            timing,
            tflops_median=_moe_tflops(
                tokens=seq_len,
                dim=args.dim,
                inter_dim=args.inter_dim,
                topk=args.topk,
                ms=timing.median_ms,
            ),
        ),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MoE benchmark with CUDA Graph timing")
    add_common_runtime_args(p)
    p.set_defaults(seed=20260319)
    p.add_argument("--seq-lens", type=str, default="1024,2048,4096,8192,16384")
    p.add_argument("--dim", type=int, default=7168)
    p.add_argument("--inter-dim", type=int, default=2048)
    p.add_argument("--experts", type=int, default=32)
    p.add_argument("--topk", type=int, default=4)
    p.add_argument("--input-dtype", type=str, default="fp8", choices=["fp8", "bf16"])
    add_timer_args(p)

    p.add_argument("--run-kernelbench", action="store_true")
    p.add_argument("--run-cudaforge", action="store_true")
    p.add_argument("--run-kernelfalcon", action="store_true")
    p.add_argument("--run-ksearch", action="store_true")
    p.add_argument("--run-aiter", action="store_true")
    p.add_argument("--run-aiter-asm", action="store_true")
    p.add_argument("--run-aiter-triton", action="store_true")
    p.add_argument("--csv-out", type=Path, default=None)
    p.add_argument("--run-id", type=str, default="")

    args = p.parse_args()
    enable_default_flags(
        args,
        [
            "run_kernelbench",
            "run_cudaforge",
            "run_kernelfalcon",
            "run_ksearch",
            "run_aiter",
            "run_aiter_asm",
            "run_aiter_triton",
        ],
        default_true_flags=[
            "run_kernelbench",
            "run_cudaforge",
            "run_kernelfalcon",
            "run_ksearch",
            "run_aiter",
        ],
    )
    return args


def main() -> None:
    args = parse_args()
    visible = apply_visible_devices(args.hip_visible_devices)
    apply_cpu_affinity(args.cpu_cores)
    validate_device_local_index(args.device, visible)

    seq_lens = parse_int_csv(args.seq_lens, name="seq-lens")
    if args.topk > args.experts:
        raise ValueError(f"topk ({args.topk}) must be <= experts ({args.experts})")
    if args.dim % BLOCK_K != 0 or args.dim % BLOCK_N != 0:
        raise ValueError(f"dim must be divisible by {BLOCK_N}/{BLOCK_K}, got {args.dim}")
    if args.inter_dim % BLOCK_K != 0:
        raise ValueError(f"inter_dim must be divisible by {BLOCK_K}, got {args.inter_dim}")

    repo_root = Path(__file__).resolve().parents[3]
    moe_root = repo_root / "data" / "benchmarks" / "moe"
    aiter_entry = moe_root / "05_aiter" / "run_aiter.py"
    aiter_mod = load_module(aiter_entry)
    build_aiter_cases = getattr(aiter_mod, "build_cases_for_seq", None)
    resolve_aiter_backends = getattr(aiter_mod, "resolve_backends", None)
    if not callable(build_aiter_cases):
        raise RuntimeError(f"missing build_cases_for_seq() in {aiter_entry}")
    if not callable(resolve_aiter_backends):
        raise RuntimeError(f"missing resolve_backends() in {aiter_entry}")

    aiter_backends = resolve_aiter_backends(args)
    if not isinstance(aiter_backends, list):
        raise RuntimeError(f"resolve_backends() must return list, got {type(aiter_backends).__name__}")
    aiter_backends = [str(x) for x in aiter_backends]
    baseline_defs: List[tuple[str, Path, bool]] = [
        ("kernelbench", moe_root / "01_kernelbench" / "best_kernel.py", args.run_kernelbench),
        ("cudaforge", moe_root / "02_cudaforge" / "best_kernel.py", args.run_cudaforge),
        ("kernelfalcon", moe_root / "03_kernelfalcon" / "best_kernel.py", args.run_kernelfalcon),
        ("ksearch", moe_root / "04_ksearch" / "best_kernel.py", args.run_ksearch),
    ]

    if torch is None:
        raise RuntimeError("torch is required")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    input_dtype = parse_dtype(args.input_dtype)
    shared_inputs = build_shared_inputs(
        seq_lens=seq_lens,
        dim=args.dim,
        experts=args.experts,
        topk=args.topk,
        input_dtype=input_dtype,
        device=device,
        seed=args.seed,
    )
    shared_weights = build_shared_weights(
        dim=args.dim,
        inter_dim=args.inter_dim,
        experts=args.experts,
        input_dtype=input_dtype,
        device=device,
        seed=args.seed,
    )

    timestamp = now_utc()
    rows: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []
    for name, path, on in baseline_defs:
        if not on:
            continue
        mod = load_module(path)
        fn = build_model_fn(mod, device=device)
        for s in seq_lens:
            x = shared_inputs[s]
            timing = time_call(lambda: run_model_once(fn, x, shared_weights), device=device, args=args)
            row = _row(baseline=name, kernel_path=str(path), seq_len=s, args=args, timing=timing)
            rows.append(row)
            csv_rows.append(
                build_csv_row(
                    domain="moe",
                    baseline=name,
                    workload=s,
                    mean_ms=timing.mean_ms,
                    tflops=_moe_tflops(tokens=s, dim=args.dim, inter_dim=args.inter_dim, topk=args.topk, ms=timing.mean_ms),
                    status=row["status"],
                    kernel_entry=str(path),
                    timestamp_utc=timestamp,
                    run_id=args.run_id,
                )
            )
    if aiter_backends:
        for s in seq_lens:
            cases = build_aiter_cases(
                args=args,
                seq_len=s,
                shared_input=shared_inputs[s],
                shared_weights=shared_weights,
                backends=aiter_backends,
            )
            for case in cases:
                timing = time_call(case.fn, device=device, args=args)
                row = _row(
                    baseline=case.baseline,
                    kernel_path=case.kernel_path,
                    seq_len=s,
                    args=args,
                    timing=timing,
                )
                rows.append(row)
                csv_rows.append(
                    build_csv_row(
                        domain="moe",
                        baseline=case.baseline,
                        workload=s,
                        mean_ms=timing.mean_ms,
                        tflops=_moe_tflops(tokens=s, dim=args.dim, inter_dim=args.inter_dim, topk=args.topk, ms=timing.mean_ms),
                        status=row["status"],
                        kernel_entry=case.kernel_path,
                        timestamp_utc=timestamp,
                        run_id=args.run_id,
                    )
                )

    maybe_write_csv(csv_out=args.csv_out, rows=csv_rows)


if __name__ == "__main__":
    main()
