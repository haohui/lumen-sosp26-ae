"""CLI for the KernelBench invariant prompt experiment."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from lumen.harness.datasets.kernelbench.generation.invariant import (
    run_invariant_generation,
)
from lumen.tools.cli.kb_generation_config import load_generation_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the KernelBench invariant prompt experiment."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the TOML generation config.",
    )
    parser.add_argument(
        "--prompt-variant",
        choices=("invariants", "no-invariants"),
        default="invariants",
        help="Select the Prompt 1 invariant treatment.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level for generation progress.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper()),
        format="%(levelname)s:%(name)s:%(message)s",
        stream=sys.stderr,
    )
    run_invariant_generation(
        load_generation_config(args.config),
        prompt_variant=args.prompt_variant,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
