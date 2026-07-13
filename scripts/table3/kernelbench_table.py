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

from kb_table.config import DEFAULT_RUN_DIRS, REPO_ROOT
from kb_table.runs import summarize_generation, summarize_optimization


def main() -> None:
    args = parse_args()
    rows = []
    strict_denominator = not args.observed_denominator

    for level in (1, 2):
        gen = summarize_generation(
            args.trace_root / getattr(args, f"generation_l{level}"),
            strict_denominator=strict_denominator,
        )
        gen_no_examples = summarize_generation(
            args.trace_root / getattr(args, f"generation_no_examples_l{level}"),
            strict_denominator=strict_denominator,
        )
        opt_no_inv = summarize_optimization(
            args.trace_root / getattr(args, f"optimization_no_invariants_l{level}"),
            strict_denominator=strict_denominator,
        )
        opt_inv = summarize_optimization(
            args.trace_root / getattr(args, f"optimization_invariants_l{level}"),
            strict_denominator=strict_denominator,
        )
        gen_pass1_no_examples = percent(
            gen_no_examples.pass1,
            gen_no_examples.denominator,
        )
        gen_pass3_no_examples = percent(
            gen_no_examples.pass3,
            gen_no_examples.denominator,
        )
        opt_pass1_no_inv = percent(
            opt_no_inv.pass_all_rounds,
            opt_no_inv.denominator,
        )
        opt_pass1_inv = percent(
            opt_inv.pass_all_rounds,
            opt_inv.denominator,
        )
        rows.append(
            {
                "Level": f"Level {level}",
                "Valid%": format_percent(percent(gen.valid_count, gen.denominator)),
                "GeoMean": format_number(gen.geom),
                "Min": format_number(gen.min_speedup),
                "Max": format_number(gen.max_speedup),
                "> 1x": str(gen.gt1_count),
                "Examples Pass@1%": arrow(
                    format_percent(gen_pass1_no_examples),
                    format_percent(percent(gen.pass1, gen.denominator)),
                ),
                "Examples Pass@3%": arrow(
                    format_percent(gen_pass3_no_examples),
                    format_percent(percent(gen.pass3, gen.denominator)),
                ),
                "Avg. files read": arrow(
                    format_number(gen_no_examples.avg_files_read),
                    format_number(gen.avg_files_read),
                ),
                "Invariants Pass@1%": arrow(
                    format_percent(opt_pass1_no_inv),
                    format_percent(opt_pass1_inv),
                ),
                "Avg. token usage": arrow(
                    format_millions(opt_no_inv.avg_token_usage),
                    format_millions(opt_inv.avg_token_usage),
                ),
            }
        )

    write_rows(rows, args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the KernelBench Table 3 row summary from traces."
    )
    parser.add_argument(
        "--trace-root",
        type=Path,
        default=REPO_ROOT / "data" / "traces",
        help=(
            "Directory containing KernelBench run directories. Defaults to "
            "data/traces, matching the artifact README."
        ),
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


def arrow(left: str, right: str) -> str:
    return f"{left}->{right}"


def format_percent(value: float | None) -> str:
    if value is None:
        return ""
    return f"{format_decimal(value, 1)}%"


def format_number(value: float | None) -> str:
    if value is None:
        return ""
    return format_decimal(value, 2)


def format_millions(value: float | None) -> str:
    if value is None:
        return ""
    return f"{format_decimal(value / 1_000_000, 2)}M"


def format_decimal(value: float, digits: int) -> str:
    text = f"{value:.{digits}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


if __name__ == "__main__":
    main()
