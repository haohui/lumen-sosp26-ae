"""Command line interface for KernelBench CUDA graph evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from lumen.harness.datasets.kernelbench.evaluator import (
    evaluate_generated_model,
    evaluate_reference_file,
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    eval_config = _load_eval_config(args.eval_config)

    if args.mode == "baseline":
        result = evaluate_reference_file(args.original)
    else:
        result = evaluate_generated_model(
            original_model_file=args.original,
            generated_model_file=args.generated,
            eval_config=eval_config,
        )

    payload = result.model_dump()
    if args.json_output is None:
        print(json.dumps(payload, indent=2))
    else:
        output_path = args.json_output.expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    return 0 if result.compiled and result.correctness else 1


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate KernelBench models with CUDA graph timing."
    )
    parser.add_argument("--mode", choices=("baseline", "generated"), required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--generated", type=Path, default=None)
    parser.add_argument("--eval-config", type=Path, default=None)
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.mode == "baseline" and args.generated is not None:
        parser.error("--generated is only valid with --mode generated")
    if args.mode == "generated" and args.generated is None:
        parser.error("--mode generated requires --generated")
    if args.mode == "baseline" and args.eval_config is not None:
        parser.error("--eval-config is only valid with --mode generated")
    return args


def _load_eval_config(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    expanded_path = path.expanduser()
    with expanded_path.open(encoding="utf-8") as config_file:
        data = json.load(config_file)
    if not isinstance(data, dict):
        raise ValueError(f"Eval config must be a JSON object: {expanded_path}")
    return data


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
