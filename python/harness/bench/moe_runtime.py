#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from common import csv_from_ints

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


def build_shared_inputs(
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


def build_shared_weights(
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

    w1_q = torch.randn((experts, inter_dim * 2, dim), dtype=torch.float32, device=device, generator=g).mul_(8.0).to(input_dtype)
    w2_q = torch.randn((experts, dim, inter_dim), dtype=torch.float32, device=device, generator=g).mul_(8.0).to(input_dtype)

    fc1_scale = (
        torch.randn(
            (experts, ((inter_dim * 2) // BLOCK_N) * (dim // BLOCK_K)),
            dtype=torch.float32,
            device=device,
            generator=g,
        ).mul_(2e-3).add_(1e-2)
    ).clamp_min_(1e-8)

    fc2_scale = (
        torch.randn(
            (experts, (dim // BLOCK_N) * (inter_dim // BLOCK_K)),
            dtype=torch.float32,
            device=device,
            generator=g,
        ).mul_(2e-3).add_(1e-2)
    ).clamp_min_(1e-8)

    return {
        "w1_q": w1_q.contiguous(),
        "w2_q": w2_q.contiguous(),
        "fc1_scale": fc1_scale.contiguous(),
        "fc2_scale": fc2_scale.contiguous(),
    }


def run_model_once(fn, x: SharedInputs, w: Dict[str, "torch.Tensor"]) -> None:
    with torch.inference_mode():
        y = fn(
            x.input_q,
            w["w1_q"],
            w["w2_q"],
            x.topk_weights,
            x.topk_ids,
            x.input_scale,
            w["fc1_scale"],
            w["fc2_scale"],
        )
    if not isinstance(y, torch.Tensor):
        raise RuntimeError(f"kernel output must be torch.Tensor, got {type(y).__name__}")


def run_aiter_backends(args: Any, seq_lens: List[int], tflops_fn, backends: List[str]) -> List[Dict[str, Any]]:
    if not backends:
        return []

    helper = Path(args.aiter_helper_script)
    if not helper.exists():
        raise FileNotFoundError(f"AITER helper script missing: {helper}")

    out_dir = args.json_out.parent if args.json_out is not None else Path(tempfile.mkdtemp(prefix="moe_aiter_helper_"))
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
        csv_from_ints(seq_lens),
        "--backends",
        ",".join(backends),
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

    env = os.environ.copy()
    bench_dir = str(Path(__file__).resolve().parent)
    env["PYTHONPATH"] = f"{bench_dir}:{env.get('PYTHONPATH', '')}".rstrip(":")
    subprocess.run(cmd, check=True, env=env)

    results = sorted(out_dir.glob(f"{prefix}_*.json"))
    if not results:
        raise RuntimeError("AITER helper produced no json output")
    payload = json.loads(results[-1].read_text(encoding="utf-8"))

    out: List[Dict[str, Any]] = []
    for r in payload.get("rows", []):
        backend = str(r.get("backend", "")).lower()
        if backend == "asm":
            baseline = "AITER (asm)"
            kernel_path = "data/benchmarks/moe/05_aiter/ASM/src/moe_op.py"
        elif backend == "triton":
            baseline = "Triton (aiter backend)"
            kernel_path = "data/benchmarks/moe/05_aiter/Triton/src/moe_op.py"
        else:
            continue

        seq_len = int(r["tokens"])
        med = float(r["median_ms"])
        mean = float(r["mean_ms"])
        out.append(
            {
                "baseline": baseline,
                "kernel_path": kernel_path,
                "seq_len": seq_len,
                "dim": args.dim,
                "inter_dim": args.inter_dim,
                "experts": args.experts,
                "topk": args.topk,
                "status": "ok",
                "timing_mode": "cudagraph",
                "median_ms": med,
                "mean_ms": mean,
                "stdev_ms": float(r["stdev_ms"]),
                "min_ms": float(r.get("min_ms", med)),
                "max_ms": float(r.get("max_ms", med)),
                "p10_ms": float(r["p10_ms"]),
                "p90_ms": float(r["p90_ms"]),
                "cv": float(r["cv"]),
                "tflops_median": tflops_fn(tokens=seq_len, dim=args.dim, inter_dim=args.inter_dim, topk=args.topk, ms=med),
                "eager_probe_ms": None,
                "suspicious": str(r.get("suspicious", "False")).lower() == "true",
                "suspicious_reason": str(r.get("suspicious_reason", "")),
                "num_replays": int(r["num_replays"]),
                "graph_iters": int(r["graph_iters"]),
                "warmup_calls": int(args.warmup),
                "total_calls_per_sample": int(r["total_calls_per_sample"]),
                "trial_count": int(args.timer_trials),
                "samples_ms": [],
            }
        )
    return out
