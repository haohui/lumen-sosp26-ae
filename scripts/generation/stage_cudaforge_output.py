#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

TASK_DIR = {"attention": "attn", "attn": "attn", "gemm": "gemm", "moe": "moe"}


def newest(paths: list[Path]) -> Path | None:
    existing = [p for p in paths if p.exists()]
    return max(existing, key=lambda p: p.stat().st_mtime) if existing else None


def resolve_path(path: Path | None, repo_root: Path, default: str) -> Path:
    path = path or Path(default)
    return path if path.is_absolute() else repo_root / path


def copy_file(src: Path, dst: Path, dry_run: bool) -> None:
    print(f"stage: {src} -> {dst}")
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_trace(trace_root: Path, dst_dir: Path, trace_prefix: str, dry_run: bool) -> None:
    trace_dir = newest([p for p in trace_root.glob(f"{trace_prefix}*") if p.is_dir()])
    if trace_dir is None:
        return
    trace = trace_dir / "traffic.jsonl"
    if trace.is_file():
        copy_file(trace, dst_dir / "traffic.json", dry_run)


def find_kernel(third_party_root: Path, subproc_id: int) -> Path:
    kernel = third_party_root / "CUDAForge" / f"test_kernel_{subproc_id}.py"
    if not kernel.is_file():
        raise SystemExit(f"no CUDAForge best kernel found: {kernel}")
    return kernel


def main() -> int:
    p = argparse.ArgumentParser(description="Stage a generated CUDAForge kernel into data/benchmarks.")
    p.add_argument("--task", required=True, choices=["attention", "attn", "gemm", "moe"])
    p.add_argument("--run-tag", required=True)
    p.add_argument("--trace-root", required=True, type=Path)
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--third-party-root", type=Path)
    p.add_argument("--data-root", type=Path)
    p.add_argument("--subproc-id", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    repo_root = args.repo_root.resolve()
    task_dir = TASK_DIR[args.task]
    third_party_root = resolve_path(args.third_party_root, repo_root, "third_party").resolve()
    data_root = resolve_path(args.data_root, repo_root, "data/benchmarks").resolve()
    trace_root = resolve_path(args.trace_root, repo_root, "").resolve()
    dst_dir = data_root / task_dir / "02_cudaforge"

    copy_file(find_kernel(third_party_root, args.subproc_id), dst_dir / "best_kernel.py", args.dry_run)
    copy_trace(trace_root, dst_dir, "02_", args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
