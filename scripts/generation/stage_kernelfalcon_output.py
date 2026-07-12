#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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


def write_text(dst: Path, content: str, src_label: str, dry_run: bool) -> None:
    print(f"stage: {src_label} -> {dst}")
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(content, encoding="utf-8")


def copy_trace(trace_root: Path, dst_dir: Path, trace_prefix: str, dry_run: bool) -> None:
    trace_dir = newest([p for p in trace_root.glob(f"{trace_prefix}*") if p.is_dir()])
    if trace_dir is None:
        return
    trace = trace_dir / "traffic.jsonl"
    if trace.is_file():
        copy_file(trace, dst_dir / "traffic.jsonl", dry_run)


def stage_kernel(trace_root: Path, third_party_root: Path, dst_dir: Path, dry_run: bool) -> None:
    result = newest(list(trace_root.glob("**/optimization_result_real.json")))
    if result is not None:
        data = json.loads(result.read_text(encoding="utf-8"))
        code = data.get("kernel_code")
        if code:
            write_text(dst_dir / "best_kernel.py", code, f"{result}::kernel_code", dry_run)
            return

    candidates = list(trace_root.glob("**/final_kernel.py"))
    candidates += list((third_party_root / "KernelFalcon" / "KernelAgent" / "triton_kernel_logs").glob("session_*/final_kernel.py"))
    src = newest(candidates)
    if src is None:
        raise SystemExit(f"no KernelFalcon final_kernel.py found under {trace_root}")
    copy_file(src, dst_dir / "best_kernel.py", dry_run)


def main() -> int:
    p = argparse.ArgumentParser(description="Stage a generated KernelFalcon kernel into data/benchmarks.")
    p.add_argument("--task", required=True, choices=["attention", "attn", "gemm", "moe"])
    p.add_argument("--trace-root", required=True, type=Path)
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--third-party-root", type=Path)
    p.add_argument("--data-root", type=Path)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    repo_root = args.repo_root.resolve()
    task_dir = TASK_DIR[args.task]
    third_party_root = resolve_path(args.third_party_root, repo_root, "third_party").resolve()
    data_root = resolve_path(args.data_root, repo_root, "data/benchmarks").resolve()
    trace_root = resolve_path(args.trace_root, repo_root, "").resolve()
    dst_dir = data_root / task_dir / "03_kernelfalcon"

    stage_kernel(trace_root, third_party_root, dst_dir, args.dry_run)
    copy_trace(trace_root, dst_dir, "03_", args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
