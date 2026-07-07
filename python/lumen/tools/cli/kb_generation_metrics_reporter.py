"""Report metrics from KernelBench generation artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lumen.harness.datasets.kernelbench.generation.metrics import (
    collect_generation_metrics,
    metrics_to_json_dict,
)
from lumen.tools.cli.kb_generation_config import (
    load_generation_config,
    problem_ids_from_config,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report metrics for a KernelBench generation run."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the TOML generation config.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_generation_config(args.config)
    metrics = collect_generation_metrics(
        config.run_dir,
        problem_ids_from_config(config),
    )
    json.dump(metrics_to_json_dict(metrics), sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
