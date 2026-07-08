#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from attn_runtime import attention_tflops, build_shared_inputs
from common import (
    add_common_runtime_args,
    add_timer_args,
    apply_cpu_affinity,
    apply_visible_devices,
    build_csv_row,
    build_model_fn,
    enable_default_flags,
    configure_sync_wait_mode,
    load_module,
    maybe_write_csv,
    now_utc,
    parse_dtype,
    parse_int_csv,
    time_call,
    validate_device_local_index,
)
from config import benchmark_root

try:
    import torch
except Exception:
    torch = None


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
    p.add_argument("--run-triton", action="store_true")

    p.add_argument("--csv-out", type=Path, default=None)
    p.add_argument("--run-id", type=str, default="")

    args = p.parse_args()
    if not bool(args.causal):
        raise ValueError(
            "--non-causal is not supported by the bundled attention baselines; "
            "use causal mode only"
        )
    enable_default_flags(
        args,
        [
            "run_aiter",
            "run_triton",
        ],
    )
    return args


def main() -> None:
    args = parse_args()
    visible = apply_visible_devices(args.hip_visible_devices)
    apply_cpu_affinity(args.cpu_cores)
    validate_device_local_index(args.device, visible)

    repo_root = Path(__file__).resolve().parents[3]
    attn_root = benchmark_root(repo_root) / "attn"

    seq_lens = parse_int_csv(args.seq_lens, name="seq-lens")

    if torch is None:
        raise RuntimeError("torch is required")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    configure_sync_wait_mode(device=device, mode=args.sync_wait_mode)
    dtype = parse_dtype(args.dtype)
    is_hip = getattr(torch.version, "hip", None) is not None
    if not is_hip:
        args.run_aiter = False
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
    baseline_defs: List[tuple[str, Path, bool]] = [
        ("aiter", attn_root / "06_aiter" / "best_kernel.py", args.run_aiter),
        ("triton", attn_root / "05_triton" / "best_kernel.py", args.run_triton),
    ]

    timestamp = now_utc()
    csv_rows: List[Dict[str, Any]] = []
    enabled = [(n, p) for (n, p, on) in baseline_defs if on]
    for name, path in enabled:
        mod = load_module(path)
        fn = build_model_fn(mod, device=device, dtype=dtype)
        for s in seq_lens:
            x = shared[s]
            timing = time_call(lambda: fn(x.q_bshd, x.k_bshd, x.v_bshd), device=device, args=args)
            csv_rows.append(
                build_csv_row(
                    domain="attention",
                    baseline=name,
                    workload=s,
                    mean_ms=timing.mean_ms,
                    tflops=attention_tflops(
                        batch_size=args.batch_size,
                        seq_len=s,
                        num_q_heads=args.num_q_heads,
                        head_dim=args.head_dim,
                        causal=bool(args.causal),
                        ms=timing.mean_ms,
                    ),
                    status="suspicious" if timing.suspicious else "ok",
                    kernel_entry=str(path),
                    timestamp_utc=timestamp,
                    run_id=args.run_id,
                )
            )
    maybe_write_csv(csv_out=args.csv_out, rows=csv_rows)


if __name__ == "__main__":
    main()
