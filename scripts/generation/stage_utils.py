from __future__ import annotations

import shutil
from pathlib import Path

TASK_DIR = {"attention": "attn", "attn": "attn", "gemm": "gemm", "moe": "moe"}


def task_dir_name(task: str) -> str:
    return TASK_DIR[task]


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
        copy_file(trace, dst_dir / "traffic.json", dry_run)
