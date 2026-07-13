#!/usr/bin/env python3
"""Generate the KernelBench Table 3 row summary from trace artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from kb_table.config import DEFAULT_RUN_DIRS, REPO_ROOT
from kb_table.render import print_latex, print_markdown
from kb_table.runs import summarize_generation, summarize_optimization


def main() -> None:
    args = parse_args()
    rows = {}
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
        rows[level] = {
            "generation": gen,
            "generation_no_examples": gen_no_examples,
            "optimization_no_invariants": opt_no_inv,
            "optimization_invariants": opt_inv,
        }

    print_markdown(rows)
    print()
    print_latex(rows)


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
    return parser.parse_args()


if __name__ == "__main__":
    main()
