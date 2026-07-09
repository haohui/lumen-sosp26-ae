"""CLI for iterative KernelBench optimization."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from lumen.harness.datasets.kernelbench.generation import run_optimization
from lumen.tools.cli.kb_generation_config import (
    load_candidate_manifest,
    load_generation_config,
    load_optimization_config,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize generated KernelBench candidates."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--optimization-config", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(levelname)s:%(name)s:%(message)s",
        stream=sys.stderr,
    )
    run_optimization(
        load_generation_config(args.config),
        load_optimization_config(args.optimization_config),
        load_candidate_manifest(args.candidates),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
