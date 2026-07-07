#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from common import parse_int_csv
from config import (
    ATTENTION_DEFAULTS,
    ATTENTION_WORKLOADS,
    GEMM_DEFAULTS,
    GEMM_WORKLOADS,
    MOE_DEFAULTS,
    MOE_WORKLOADS,
    OUTPUT_ROOT,
    TIMER_DEFAULTS,
)
from report_table import render_csv_from_csvs, render_from_csvs

THIS_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class DomainSpec:
    script: str
    col_arg: str
    fixed_args: List[str]
    csv_name: str


DOMAIN_SPECS: Dict[str, DomainSpec] = {
    "gemm": DomainSpec(
        script="benchmark_gemm_unified_graph.py",
        col_arg="--sizes",
        fixed_args=[
            "--dtype",
            GEMM_DEFAULTS["dtype"],
            "--run-hipblaslt",
            "--run-hipkittens",
            "--run-triton",
        ],
        csv_name="gemm_raw.csv",
    ),
    "attention": DomainSpec(
        script="benchmark_attention_unified_graph.py",
        col_arg="--seq-lens",
        fixed_args=[
            "--dtype",
            ATTENTION_DEFAULTS["dtype"],
            "--batch-size",
            str(ATTENTION_DEFAULTS["batch_size"]),
            "--num-q-heads",
            str(ATTENTION_DEFAULTS["num_q_heads"]),
            "--num-kv-heads",
            str(ATTENTION_DEFAULTS["num_kv_heads"]),
            "--head-dim",
            str(ATTENTION_DEFAULTS["head_dim"]),
            "--causal" if ATTENTION_DEFAULTS["causal"] else "--non-causal",
            "--run-triton",
        ],
        csv_name="attention_raw.csv",
    ),
    "moe": DomainSpec(
        script="benchmark_moe_unified_graph.py",
        col_arg="--seq-lens",
        fixed_args=[
            "--dim",
            str(MOE_DEFAULTS["dim"]),
            "--inter-dim",
            str(MOE_DEFAULTS["inter_dim"]),
            "--experts",
            str(MOE_DEFAULTS["experts"]),
            "--topk",
            str(MOE_DEFAULTS["topk"]),
            "--input-dtype",
            MOE_DEFAULTS["input_dtype"],
        ],
        csv_name="moe_raw.csv",
    ),
}


def _run(cmd: List[str], *, env: Dict[str, str] | None = None) -> None:
    subprocess.run(cmd, check=True, env=env)


def _is_hip(python_bin: str) -> bool:
    probe = (
        "import torch; "
        "print('1' if getattr(torch.version, 'hip', None) is not None else '0')"
    )
    try:
        cp = subprocess.run(
            [python_bin, "-c", probe],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return cp.returncode == 0 and cp.stdout.strip() == "1"
    except Exception:
        return False


def _require_aiter(python_bin: str) -> None:
    probe = (
        "import importlib.util; "
        "spec = importlib.util.find_spec('aiter'); "
        "print('1' if spec else '0')"
    )
    try:
        cp = subprocess.run(
            [python_bin, "-c", probe],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        ok = cp.returncode == 0 and cp.stdout.strip() == "1"
        if not ok:
            raise RuntimeError("AITER is required but not available in this Python environment")
    except Exception:
        raise RuntimeError("AITER is required but import probe failed")


def _resolve_cols(args: argparse.Namespace) -> Tuple[List[int], List[int], List[int]]:
    common = parse_int_csv(args.workloads, name="workloads")
    gemm = parse_int_csv(args.gemm_workloads, name="gemm-workloads") or common or list(GEMM_WORKLOADS)
    attn = parse_int_csv(args.attention_workloads, name="attention-workloads") or common or list(ATTENTION_WORKLOADS)
    moe = parse_int_csv(args.moe_workloads, name="moe-workloads") or common or list(MOE_WORKLOADS)
    return gemm, attn, moe


def _timer_args(args: argparse.Namespace) -> List[str]:
    return [
        "--warmup-ms",
        str(args.warmup_ms),
        "--repeat-ms",
        str(args.repeat_ms),
        "--graph-iters",
        str(args.graph_iters),
        "--timer-trials",
        str(args.timer_trials),
        "--min-replays",
        str(args.min_replays),
        "--max-replays",
        str(args.max_replays),
    ]


def _with_optional_bindings(cmd: List[str], args: argparse.Namespace) -> List[str]:
    out = list(cmd)
    if args.hip_visible_devices.strip():
        out.extend(["--hip-visible-devices", args.hip_visible_devices.strip()])
    if args.cpu_cores.strip():
        out.extend(["--cpu-cores", args.cpu_cores.strip()])
    return out


def _run_domain(
    args: argparse.Namespace,
    out_dir: Path,
    domain: str,
    cols: List[int],
    *,
    is_hip: bool,
    run_id: str,
) -> Path:
    spec = DOMAIN_SPECS[domain]
    out = out_dir / spec.csv_name
    run_flags = [*spec.fixed_args]
    if is_hip:
        run_flags.append("--run-aiter")
    cmd = [
        args.python,
        str(THIS_DIR / spec.script),
        "--device",
        args.device,
        spec.col_arg,
        ",".join(str(x) for x in cols),
        *run_flags,
        "--csv-out",
        str(out),
        "--run-id",
        run_id,
        *_timer_args(args),
    ]
    env = os.environ.copy()
    if args.benchmark_root.strip():
        env["AE_BENCHMARK_ROOT"] = str(Path(args.benchmark_root).resolve())
    _run(_with_optional_bindings(cmd, args), env=env)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run GEMM/Attention/MoE and emit raw CSV + throughput table")
    p.add_argument("--domain", choices=["all", "gemm", "attention", "moe"], default="all")
    p.add_argument("--python", type=str, default=sys.executable)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--hip-visible-devices", type=str, default="")
    p.add_argument("--cpu-cores", type=str, default="")
    p.add_argument("--workloads", type=str, default="")
    p.add_argument("--gemm-workloads", type=str, default="")
    p.add_argument("--attention-workloads", type=str, default="")
    p.add_argument("--moe-workloads", type=str, default="")
    p.add_argument("--benchmark-root", type=str, default="")
    p.add_argument("--output-dir", type=str, default=str(OUTPUT_ROOT))
    p.add_argument("--run-dir", type=str, default="")
    p.add_argument("--run-id", type=str, default="")
    p.add_argument("--report-only", action="store_true")
    p.add_argument("--no-report", action="store_true")
    p.add_argument("--warmup-ms", type=float, default=TIMER_DEFAULTS["warmup_ms"])
    p.add_argument("--repeat-ms", type=float, default=TIMER_DEFAULTS["repeat_ms"])
    p.add_argument("--graph-iters", type=int, default=TIMER_DEFAULTS["graph_iters"])
    p.add_argument("--timer-trials", type=int, default=TIMER_DEFAULTS["timer_trials"])
    p.add_argument("--min-replays", type=int, default=TIMER_DEFAULTS["min_replays"])
    p.add_argument("--max-replays", type=int, default=TIMER_DEFAULTS["max_replays"])
    return p.parse_args()


def _write_meta(path: Path, args: argparse.Namespace, run_id: str, gemm_cols: List[int], attn_cols: List[int], moe_cols: List[int]) -> None:
    payload = {
        "run_id": run_id,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "invocation": {
            "python": args.python,
            "device": args.device,
            "hip_visible_devices": args.hip_visible_devices,
            "cpu_cores": args.cpu_cores,
            "benchmark_root": args.benchmark_root,
        },
        "workloads": {
            "gemm": gemm_cols,
            "attention": attn_cols,
            "moe": moe_cols,
        },
        "timer": {
            "warmup_ms": args.warmup_ms,
            "repeat_ms": args.repeat_ms,
            "graph_iters": args.graph_iters,
            "timer_trials": args.timer_trials,
            "min_replays": args.min_replays,
            "max_replays": args.max_replays,
        },
        "gemm": {
            "dtype": GEMM_DEFAULTS["dtype"],
        },
        "attention": {
            "batch_size": ATTENTION_DEFAULTS["batch_size"],
            "num_q_heads": ATTENTION_DEFAULTS["num_q_heads"],
            "num_kv_heads": ATTENTION_DEFAULTS["num_kv_heads"],
            "head_dim": ATTENTION_DEFAULTS["head_dim"],
            "causal": ATTENTION_DEFAULTS["causal"],
        },
        "moe": {
            "dim": MOE_DEFAULTS["dim"],
            "inter_dim": MOE_DEFAULTS["inter_dim"],
            "experts": MOE_DEFAULTS["experts"],
            "topk": MOE_DEFAULTS["topk"],
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    gemm_cols, attn_cols, moe_cols = _resolve_cols(args)
    cols_map = {"gemm": gemm_cols, "attention": attn_cols, "moe": moe_cols}
    is_hip = False
    if not args.report_only:
        is_hip = _is_hip(args.python)
        if is_hip:
            _require_aiter(args.python)

    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    out_dir = Path(args.run_dir) if args.run_dir else Path(args.output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.report_only:
        for domain in ("gemm", "attention", "moe"):
            if args.domain not in ("all", domain):
                continue
            _run_domain(
                args,
                out_dir,
                domain,
                cols_map[domain],
                is_hip=is_hip,
                run_id=run_id,
            )

        meta_path = out_dir / "run_meta.json"
        _write_meta(meta_path, args, run_id, gemm_cols, attn_cols, moe_cols)
        if args.no_report:
            return 0
    else:
        meta_path = out_dir / "run_meta.json"

    md = render_from_csvs(
        gemm_csv=out_dir / DOMAIN_SPECS["gemm"].csv_name,
        attn_csv=out_dir / DOMAIN_SPECS["attention"].csv_name,
        moe_csv=out_dir / DOMAIN_SPECS["moe"].csv_name,
        meta_path=meta_path,
    )
    table_csv = render_csv_from_csvs(
        gemm_csv=out_dir / DOMAIN_SPECS["gemm"].csv_name,
        attn_csv=out_dir / DOMAIN_SPECS["attention"].csv_name,
        moe_csv=out_dir / DOMAIN_SPECS["moe"].csv_name,
        meta_path=meta_path,
    )
    md_path = out_dir / "overall_performance.md"
    md_path.write_text(md, encoding="utf-8")
    (out_dir / "overall_performance.csv").write_text(table_csv, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
