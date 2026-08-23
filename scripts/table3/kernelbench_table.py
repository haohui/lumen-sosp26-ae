#!/usr/bin/env python3
"""Generate the KernelBench Table 3 row summary from trace artifacts."""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path
from typing import Any

from kb_table.archive import (
    ArchiveRuns,
    summarize_generation_archive,
    summarize_optimization_archive,
)
from kb_table.config import DEFAULT_RUN_DIRS, REPO_ROOT
from kb_table.models import GenerationStats, OptimizationStats

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from experiment_summary import ExperimentSummary  # noqa: E402

DEFAULT_GENERATION_ARCHIVE = (
    REPO_ROOT / "data" / "traces" / "kernelbench_generation_dsv4-07-13-2026.tar.xz"
)
DEFAULT_OPTIMIZATION_ARCHIVE = (
    REPO_ROOT / "data" / "traces" / "kernelbench_optimization_dsv4-07-13-2026.tar.xz"
)


def main() -> None:
    args = parse_args()
    with ExperimentSummary(
        "table3-summary",
        "compare the emitted CSV/JSON rows with Table 3",
    ) as summary:
        summary.add_result(args.output or "stdout")
        summary.add_result(args.generation_archive)
        summary.add_result(args.optimization_archive)
        archive_runs = ArchiveRuns.from_archives(
            [args.generation_archive, args.optimization_archive],
            run_names(args),
        )
        rows = build_rows(args, archive_runs)
        write_rows(rows, args)


def build_rows(args: argparse.Namespace, archive_runs: ArchiveRuns) -> list[dict[str, Any]]:
    rows = []
    strict_denominator = not args.observed_denominator

    for level in (1, 2):
        gen_with_context = summarize_generation_archive(
            archive_runs.run(getattr(args, f"generation_l{level}")),
            strict_denominator=strict_denominator,
        )
        gen_without_context = summarize_generation_archive(
            archive_runs.run(getattr(args, f"generation_no_examples_l{level}")),
            strict_denominator=strict_denominator,
        )
        opt_without_invariants = summarize_optimization_archive(
            archive_runs.run(getattr(args, f"optimization_no_invariants_l{level}")),
            strict_denominator=strict_denominator,
        )
        opt_with_invariants = summarize_optimization_archive(
            archive_runs.run(getattr(args, f"optimization_invariants_l{level}")),
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


def run_names(args: argparse.Namespace) -> list[str]:
    return [getattr(args, key) for key in DEFAULT_RUN_DIRS]


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
