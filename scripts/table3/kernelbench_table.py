#!/usr/bin/env python3
"""Generate the KernelBench Table 3 row summary from trace artifacts."""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import tarfile
import tempfile
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any

from kb_table.config import DEFAULT_RUN_DIRS, REPO_ROOT
from kb_table.models import GenerationStats, OptimizationStats
from kb_table.runs import summarize_generation, summarize_optimization

DEFAULT_GENERATION_ARCHIVE = (
    REPO_ROOT / "data" / "traces" / "kernelbench_generation_dsv4-07-13-2026.tar.xz"
)
DEFAULT_OPTIMIZATION_ARCHIVE = (
    REPO_ROOT / "data" / "traces" / "kernelbench_optimization_dsv4-07-13-2026.tar.xz"
)


def main() -> None:
    args = parse_args()
    with trace_root_context(args) as trace_root:
        rows = build_rows(args, trace_root)
        write_rows(rows, args)


def build_rows(args: argparse.Namespace, trace_root: Path) -> list[dict[str, Any]]:
    rows = []
    strict_denominator = not args.observed_denominator

    for level in (1, 2):
        gen_with_context = summarize_generation(
            trace_root / getattr(args, f"generation_l{level}"),
            strict_denominator=strict_denominator,
        )
        gen_without_context = summarize_generation(
            trace_root / getattr(args, f"generation_no_examples_l{level}"),
            strict_denominator=strict_denominator,
        )
        opt_without_invariants = summarize_optimization(
            trace_root / getattr(args, f"optimization_no_invariants_l{level}"),
            strict_denominator=strict_denominator,
        )
        opt_with_invariants = summarize_optimization(
            trace_root / getattr(args, f"optimization_invariants_l{level}"),
            strict_denominator=strict_denominator,
        )
        rows.append(
            {
                "level": level,
                **generation_performance_columns(gen_with_context),
                **generation_ablation_columns(
                    gen_without_context,
                    gen_with_context,
                ),
                **optimization_columns(
                    "without_invariants",
                    opt_without_invariants,
                ),
                **optimization_columns("with_invariants", opt_with_invariants),
            }
        )
    return rows


def generation_performance_columns(stats: GenerationStats) -> dict[str, Any]:
    return {
        "generation_with_in_context_denominator": stats.denominator,
        "generation_with_in_context_valid_count": stats.valid_count,
        "generation_with_in_context_valid_percent": percent(
            stats.valid_count,
            stats.denominator,
        ),
        "generation_with_in_context_geomean": stats.geom,
        "generation_with_in_context_min_speedup": stats.min_speedup,
        "generation_with_in_context_max_speedup": stats.max_speedup,
        "generation_with_in_context_gt1_count": stats.gt1_count,
    }


def generation_ablation_columns(
    without_context: GenerationStats,
    with_context: GenerationStats,
) -> dict[str, Any]:
    return {
        "generation_without_in_context_pass1_count": without_context.pass1,
        "generation_without_in_context_pass1_percent": percent(
            without_context.pass1,
            without_context.denominator,
        ),
        "generation_with_in_context_pass1_count": with_context.pass1,
        "generation_with_in_context_pass1_percent": percent(
            with_context.pass1,
            with_context.denominator,
        ),
        "generation_without_in_context_pass3_count": without_context.pass3,
        "generation_without_in_context_pass3_percent": percent(
            without_context.pass3,
            without_context.denominator,
        ),
        "generation_with_in_context_pass3_count": with_context.pass3,
        "generation_with_in_context_pass3_percent": percent(
            with_context.pass3,
            with_context.denominator,
        ),
        "generation_without_in_context_avg_files_read": (
            without_context.avg_files_read
        ),
        "generation_with_in_context_avg_files_read": with_context.avg_files_read,
    }


def optimization_columns(label: str, stats: OptimizationStats) -> dict[str, Any]:
    prefix = f"optimization_{label}"
    return {
        f"{prefix}_denominator": stats.denominator,
        f"{prefix}_pass1_count": stats.pass_all_rounds,
        f"{prefix}_pass1_percent": percent(
            stats.pass_all_rounds,
            stats.denominator,
        ),
        f"{prefix}_avg_token_usage": stats.avg_token_usage,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the KernelBench Table 3 row summary from traces."
    )
    parser.add_argument(
        "--trace-root",
        type=Path,
        default=REPO_ROOT / "data" / "traces",
        help=(
            "Directory containing already extracted KernelBench run directories. "
            "Used only with --use-expanded-traces."
        ),
    )
    parser.add_argument(
        "--generation-archive",
        type=Path,
        default=DEFAULT_GENERATION_ARCHIVE,
        help="KernelBench generation trace tar.xz archive.",
    )
    parser.add_argument(
        "--optimization-archive",
        type=Path,
        default=DEFAULT_OPTIMIZATION_ARCHIVE,
        help="KernelBench optimization trace tar.xz archive.",
    )
    parser.add_argument(
        "--use-expanded-traces",
        action="store_true",
        help="Read already extracted run directories from --trace-root.",
    )
    for key, default in DEFAULT_RUN_DIRS.items():
        parser.add_argument(f"--{key.replace('_', '-')}", default=default)
    parser.add_argument(
        "--observed-denominator",
        action="store_true",
        help=(
            "Use only observed pXX directories as the denominator for partial-run "
            "monitoring."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("csv", "json"),
        default="csv",
        help="Output format. Defaults to csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write output to this file instead of stdout.",
    )
    return parser.parse_args()


def trace_root_context(args: argparse.Namespace) -> AbstractContextManager[Path]:
    if args.use_expanded_traces:
        return nullcontext(args.trace_root)
    return ExtractedTraceRoot(args.generation_archive, args.optimization_archive)


class ExtractedTraceRoot(AbstractContextManager[Path]):
    def __init__(self, generation_archive: Path, optimization_archive: Path) -> None:
        self.generation_archive = generation_archive
        self.optimization_archive = optimization_archive
        self._tempdir: tempfile.TemporaryDirectory[str] | None = None

    def __enter__(self) -> Path:
        self._tempdir = tempfile.TemporaryDirectory(prefix="lumen-table3-traces-")
        try:
            extract_root = Path(self._tempdir.name)
            extract_archive(self.generation_archive, extract_root)
            extract_archive(self.optimization_archive, extract_root)
            return discover_trace_root(extract_root)
        except BaseException:
            self._tempdir.cleanup()
            self._tempdir = None
            raise

    def __exit__(self, *args: object) -> None:
        if self._tempdir is not None:
            self._tempdir.cleanup()


def extract_archive(archive: Path, destination: Path) -> None:
    try:
        with tarfile.open(archive, mode="r:xz") as tar:
            if hasattr(tarfile, "data_filter"):
                tar.extractall(destination, filter="data")
            else:
                tar.extractall(destination)
    except tarfile.TarError as exc:
        if is_git_lfs_pointer(archive):
            raise RuntimeError(
                f"{archive} is a Git LFS pointer, not the trace archive. "
                "Run `git lfs pull` and retry."
            ) from exc
        raise


def is_git_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as file:
            prefix = file.read(80)
    except OSError:
        return False
    return prefix.startswith(b"version https://git-lfs.github.com/spec/")


def discover_trace_root(extract_root: Path) -> Path:
    expected = set(DEFAULT_RUN_DIRS.values())
    candidates = [
        extract_root,
        *(path for path in extract_root.rglob("*") if path.is_dir()),
    ]
    for candidate in candidates:
        present = {path.name for path in candidate.iterdir() if path.is_dir()}
        if expected.issubset(present):
            return candidate
    raise FileNotFoundError(
        "Could not find all KernelBench run directories in extracted archives."
    )


def write_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    if args.format == "json":
        text = json.dumps(rows, indent=2, sort_keys=True) + "\n"
    else:
        text = csv_text(rows)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)


def csv_text(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def percent(count: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return 100.0 * count / denominator


if __name__ == "__main__":
    main()
