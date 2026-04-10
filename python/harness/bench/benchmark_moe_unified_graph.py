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
from moe_runtime import (
    BLOCK_K,
    BLOCK_N,
    build_shared_inputs,
    build_shared_weights,
    run_aiter_backends,
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


def _resolve_aiter_backends(args: argparse.Namespace) -> List[str]:
    if args.run_aiter:
        return ["asm", "triton"]
    out: List[str] = []
    if args.run_aiter_asm:
        out.append("asm")
    if args.run_aiter_triton:
        out.append("triton")
    return out


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    moe_root = repo_root / "data" / "benchmarks" / "moe"

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

    p.add_argument(
        "--aiter-helper-script",
        type=Path,
        default=moe_root / "05_aiter" / "tools" / "bench_moe_aiter_backends_cudagraph.py",
    )
    p.add_argument("--json-out", type=Path, default=None)

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
    cpu_aff = apply_cpu_affinity(args.cpu_cores)
    validate_device_local_index(args.device, visible)

    seq_lens = parse_int_csv(args.seq_lens, name="seq-lens")
    aiter_backends = _resolve_aiter_backends(args)

    if args.topk > args.experts:
        raise ValueError(f"topk ({args.topk}) must be <= experts ({args.experts})")
    if args.dim % BLOCK_K != 0 or args.dim % BLOCK_N != 0:
        raise ValueError(f"dim must be divisible by {BLOCK_N}/{BLOCK_K}, got {args.dim}")
    if args.inter_dim % BLOCK_K != 0:
        raise ValueError(f"inter_dim must be divisible by {BLOCK_K}, got {args.inter_dim}")

    repo_root = Path(__file__).resolve().parents[3]
    moe_root = repo_root / "data" / "benchmarks" / "moe"
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

    rows: List[Dict[str, Any]] = []
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
    rows.extend(run_aiter_backends(args, seq_lens, _moe_tflops, aiter_backends))

    maybe_write_json(
        json_out=args.json_out,
        device=device,
        config={
            "device": args.device,
            "hip_visible_devices": visible,
            "cpu_affinity": cpu_aff,
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
            "repeat_ms": effective_repeat_ms(args),
            "timer_trials": args.timer_trials,
            "min_replays": args.min_replays,
            "max_replays": args.max_replays,
            "pre_capture_iters": args.pre_capture_iters,
            "run_kernelbench": args.run_kernelbench,
            "run_cudaforge": args.run_cudaforge,
            "run_kernelfalcon": args.run_kernelfalcon,
            "run_ksearch": args.run_ksearch,
            "run_aiter": args.run_aiter,
            "run_aiter_asm": args.run_aiter_asm,
            "run_aiter_triton": args.run_aiter_triton,
            "aiter_backends": aiter_backends,
            "aiter_helper_script": str(args.aiter_helper_script),
        },
        rows=rows,
    )


if __name__ == "__main__":
    main()
