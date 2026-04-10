#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from attn_runtime import SharedInputs, attention_tflops, build_shared_inputs
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

try:
    import torch
except Exception:
    torch = None


def _row(*, baseline: str, kernel_path: str, seq_len: int, args: argparse.Namespace, timing) -> Dict[str, Any]:
    return {
        "baseline": baseline,
        "kernel_path": kernel_path,
        "seq_len": seq_len,
        "batch_size": args.batch_size,
        "num_q_heads": args.num_q_heads,
        "head_dim": args.head_dim,
        "causal": bool(args.causal),
        **timing_fields(
            timing,
            tflops_median=attention_tflops(
                batch_size=args.batch_size,
                seq_len=seq_len,
                num_q_heads=args.num_q_heads,
                head_dim=args.head_dim,
                causal=bool(args.causal),
                ms=timing.median_ms,
            ),
        ),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Attention benchmark with CUDA Graph timing")
    add_common_runtime_args(p)
    p.set_defaults(seed=20260314)
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--seq-lens", type=str, default="1024,2048,4096,8192,16384")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-q-heads", type=int, default=8)
    p.add_argument("--num-kv-heads", type=int, default=1)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--causal", dest="causal", action="store_true", default=True)
    p.add_argument("--non-causal", dest="causal", action="store_false")
    add_timer_args(p)

    p.add_argument("--run-aiter", action="store_true")
    p.add_argument("--run-hipkittens", action="store_true")
    p.add_argument("--run-kernelfalcon", action="store_true")
    p.add_argument("--run-ksearch", action="store_true")
    p.add_argument("--run-kernelbench", action="store_true")
    p.add_argument("--run-cudaforge", action="store_true")

    p.add_argument("--json-out", type=Path, default=None)

    args = p.parse_args()
    enable_default_flags(
        args,
        [
            "run_aiter",
            "run_hipkittens",
            "run_kernelfalcon",
            "run_ksearch",
            "run_kernelbench",
            "run_cudaforge",
        ],
    )
    return args


def main() -> None:
    args = parse_args()
    visible = apply_visible_devices(args.hip_visible_devices)
    cpu_aff = apply_cpu_affinity(args.cpu_cores)
    validate_device_local_index(args.device, visible)

    repo_root = Path(__file__).resolve().parents[3]
    attn_root = repo_root / "data" / "benchmarks" / "attn"
    baseline_defs: List[tuple[str, Path, bool]] = [
        ("aiter", attn_root / "06_aiter" / "best_kernel.py", args.run_aiter),
        ("hipkittens", attn_root / "05_HipKittens" / "best_kernel.py", args.run_hipkittens),
        ("kernelfalcon", attn_root / "03_kernelfalcon" / "best_kernel.py", args.run_kernelfalcon),
        ("ksearch", attn_root / "04_ksearch" / "best_kernel.py", args.run_ksearch),
        ("kernelbench", attn_root / "01_kernelbench" / "best_kernel.py", args.run_kernelbench),
        ("cudaforge", attn_root / "02_cudaforge" / "best_kernel.py", args.run_cudaforge),
    ]

    seq_lens = parse_int_csv(args.seq_lens, name="seq-lens")

    if torch is None:
        raise RuntimeError("torch is required")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = parse_dtype(args.dtype)
    shared = build_shared_inputs(
        seq_lens=seq_lens,
        batch_size=args.batch_size,
        num_q_heads=args.num_q_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        device=device,
        dtype=dtype,
        seed=args.seed,
    )

    rows: List[Dict[str, Any]] = []
    for name, path, on in baseline_defs:
        if not on:
            continue
        mod = load_module(path)
        fn = build_model_fn(mod, device=device, dtype=dtype)
        for s in seq_lens:
            x = shared[s]
            timing = time_call(lambda: fn(x.q_bhsd, x.k_bhsd, x.v_bhsd), device=device, args=args)
            row = _row(baseline=name, kernel_path=str(path), seq_len=s, args=args, timing=timing)
            rows.append(row)

    maybe_write_json(
        json_out=args.json_out,
        device=device,
        config={
            "device": args.device,
            "hip_visible_devices": visible,
            "cpu_affinity": cpu_aff,
            "dtype": args.dtype,
            "seq_lens": seq_lens,
            "batch_size": args.batch_size,
            "num_q_heads": args.num_q_heads,
            "num_kv_heads": args.num_kv_heads,
            "head_dim": args.head_dim,
            "causal": bool(args.causal),
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
            "run_hipkittens": args.run_hipkittens,
            "run_kernelfalcon": args.run_kernelfalcon,
            "run_ksearch": args.run_ksearch,
            "run_kernelbench": args.run_kernelbench,
            "run_cudaforge": args.run_cudaforge,
        },
        rows=rows,
    )


if __name__ == "__main__":
    main()
