#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage_utils import copy_file, copy_trace, newest, resolve_path, task_dir_name


def find_kernel(third_party_root: Path, run_tag: str) -> Path:
    roots = [p for p in (third_party_root / "CUDAForge").glob(f"*{run_tag}*") if p.is_dir()]
    evals = [ev for root in roots for ev in root.glob("**/evaluation/eval_*.json")]
    best_file = None
    best_score = None
    for ev in evals:
        try:
            data = json.loads(ev.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not data.get("runnable", False):
            continue
        cand = Path(str(data.get("candidate_file", "")))
        if not cand.is_file():
            continue
        score = float(data.get("score", 0.0))
        if best_score is None or score > best_score:
            best_score = score
            best_file = cand
    if best_file is not None:
        return best_file

    fallback = newest([p for root in roots for p in root.glob("**/code/kernel_*.py")])
    if fallback is None:
        raise SystemExit(f"no CUDAForge kernel found for run_tag={run_tag}")
    return fallback


def main() -> int:
    p = argparse.ArgumentParser(description="Stage a generated CUDAForge kernel into data/benchmarks.")
    p.add_argument("--task", required=True, choices=["attention", "attn", "gemm", "moe"])
    p.add_argument("--run-tag", required=True)
    p.add_argument("--trace-root", required=True, type=Path)
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--third-party-root", type=Path)
    p.add_argument("--data-root", type=Path)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    repo_root = args.repo_root.resolve()
    task_dir = task_dir_name(args.task)
    third_party_root = resolve_path(args.third_party_root, repo_root, "third_party").resolve()
    data_root = resolve_path(args.data_root, repo_root, "data/benchmarks").resolve()
    trace_root = resolve_path(args.trace_root, repo_root, "").resolve()
    dst_dir = data_root / task_dir / "02_cudaforge"

    copy_file(find_kernel(third_party_root, args.run_tag), dst_dir / "best_kernel.py", args.dry_run)
    copy_trace(trace_root, dst_dir, "02_", args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
