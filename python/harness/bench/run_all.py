#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from common import parse_int_csv
from report_table import (
    ATTN_ORDER,
    GEMM_ORDER,
    MOE_ORDER,
    blank,
    parse_attention,
    parse_gemm,
    parse_moe,
    render_markdown,
)

THIS_DIR = Path(__file__).resolve().parent

GEMM_COLS = [1024, 2048, 4096, 8192, 16384]
ATTN_COLS = [1024, 2048, 4096, 8192, 16384]
MOE_COLS = [1024, 2048, 4096, 8192, 16384]


@dataclass(frozen=True)
class DomainSpec:
    script: str
    col_arg: str
    fixed_args: List[str]
    json_name: str


DOMAIN_SPECS: Dict[str, DomainSpec] = {
    "gemm": DomainSpec(
        script="benchmark_gemm_unified_graph.py",
        col_arg="--sizes",
        fixed_args=[
            "--dtype",
            "bf16",
            "--run-hipblaslt",
            "--run-hipkittens",
            "--run-kernelbench",
            "--run-cudaforge",
            "--run-kernelfalcon",
            "--run-ksearch",
            "--run-triton",
        ],
        json_name="gemm.json",
    ),
    "attention": DomainSpec(
        script="benchmark_attention_unified_graph.py",
        col_arg="--seq-lens",
        fixed_args=[
            "--dtype",
            "bf16",
            "--batch-size",
            "16",
            "--num-q-heads",
            "8",
            "--num-kv-heads",
            "1",
            "--head-dim",
            "128",
            "--causal",
            "--run-hipkittens",
            "--run-kernelfalcon",
            "--run-ksearch",
            "--run-kernelbench",
            "--run-cudaforge",
        ],
        json_name="attention.json",
    ),
    "moe": DomainSpec(
        script="benchmark_moe_unified_graph.py",
        col_arg="--seq-lens",
        fixed_args=[
            "--dim",
            "7168",
            "--inter-dim",
            "2048",
            "--experts",
            "32",
            "--topk",
            "4",
            "--input-dtype",
            "fp8",
            "--run-kernelbench",
            "--run-cudaforge",
            "--run-kernelfalcon",
            "--run-ksearch",
        ],
        json_name="moe.json",
    ),
}


def _run(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True)


def _has_aiter(python_bin: str) -> bool:
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
        return cp.returncode == 0 and cp.stdout.strip() == "1"
    except Exception:
        return False


def _resolve_cols(args: argparse.Namespace) -> Tuple[List[int], List[int], List[int]]:
    common = parse_int_csv(args.workloads, name="workloads")
    gemm = parse_int_csv(args.gemm_workloads, name="gemm-workloads") or common or list(GEMM_COLS)
    attn = parse_int_csv(args.attention_workloads, name="attention-workloads") or common or list(ATTN_COLS)
    moe = parse_int_csv(args.moe_workloads, name="moe-workloads") or common or list(MOE_COLS)
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


def _run_domain(args: argparse.Namespace, out_dir: Path, domain: str, cols: List[int], *, enable_aiter: bool) -> Path:
    spec = DOMAIN_SPECS[domain]
    out = out_dir / spec.json_name
    run_flags = list(spec.fixed_args)
    if enable_aiter and domain in ("gemm", "attention", "moe"):
        run_flags.append("--run-aiter")
    cmd = [
        args.python,
        str(THIS_DIR / spec.script),
        "--device",
        args.device,
        spec.col_arg,
        ",".join(str(x) for x in cols),
        *run_flags,
        "--json-out",
        str(out),
        *_timer_args(args),
    ]
    _run(_with_optional_bindings(cmd, args))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run GEMM/Attention/MoE and print markdown timing table")
    p.add_argument("--domain", choices=["all", "gemm", "attention", "moe"], default="all")
    p.add_argument("--python", type=str, default=sys.executable)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--hip-visible-devices", type=str, default="")
    p.add_argument("--cpu-cores", type=str, default="")
    p.add_argument("--workloads", type=str, default="")
    p.add_argument("--gemm-workloads", type=str, default="")
    p.add_argument("--attention-workloads", type=str, default="")
    p.add_argument("--moe-workloads", type=str, default="")
    p.add_argument("--warmup-ms", type=float, default=1000.0)
    p.add_argument("--repeat-ms", type=float, default=5000.0)
    p.add_argument("--graph-iters", type=int, default=10)
    p.add_argument("--timer-trials", type=int, default=9)
    p.add_argument("--min-replays", type=int, default=5)
    p.add_argument("--max-replays", type=int, default=200000)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    gemm_cols, attn_cols, moe_cols = _resolve_cols(args)
    cols_map = {"gemm": gemm_cols, "attention": attn_cols, "moe": moe_cols}
    enable_aiter = _has_aiter(args.python)
    if not enable_aiter:
        print(
            "[run_all] WARN: AITER is unavailable in this Python environment; "
            "running without AITER baselines",
            file=sys.stderr,
        )

    with tempfile.TemporaryDirectory(prefix="bench_run_all_") as tmp:
        out_dir = Path(tmp)

        ran: Dict[str, Path] = {}
        for domain in ("gemm", "attention", "moe"):
            if args.domain not in ("all", domain):
                continue
            ran[domain] = _run_domain(args, out_dir, domain, cols_map[domain], enable_aiter=enable_aiter)

        tables = {
            "gemm": blank(GEMM_ORDER, gemm_cols),
            "attention": blank(ATTN_ORDER, attn_cols),
            "moe": blank(MOE_ORDER, moe_cols),
        }
        if "gemm" in ran:
            tables["gemm"] = parse_gemm(ran["gemm"], gemm_cols)
        if "attention" in ran:
            tables["attention"] = parse_attention(ran["attention"], attn_cols)
        if "moe" in ran:
            tables["moe"] = parse_moe(ran["moe"], moe_cols)

        md = render_markdown(args.domain, tables, gemm_cols=gemm_cols, attn_cols=attn_cols, moe_cols=moe_cols)
        print(md, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
