#!/usr/bin/env python3
"""Thin CLI for the KernelBench Codex optimization loop."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PYTHON_ROOT = REPO_ROOT / "python"

python_root_str = str(PYTHON_ROOT)
if python_root_str not in sys.path:
    sys.path.insert(0, python_root_str)

from lumen.harness.kernelbench_optimization_loop import (  # noqa: E402
    KernelBenchLoopConfig,
    KernelBenchOptimizationLoop,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a KernelBench-specific Codex optimization loop."
    )
    parser.add_argument("--problem-dir", type=Path, required=True)
    parser.add_argument("--prompt-template", type=Path, required=True)
    parser.add_argument("--optimization-dir-name", default="optimization_rounds")
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num-correct-trials", type=int, default=1)
    parser.add_argument("--timing-method", default="cudagraph")
    parser.add_argument(
        "--measure-performance",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--codex-bin", type=Path, default=None)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--model-provider", default=None)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--timeout-seconds", type=float, default=None)
    parser.add_argument("--codex-home", type=Path, default=None)
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        dest="config_overrides",
        help="Raw Codex config override, e.g. key=value. May be repeated.",
    )
    parser.add_argument(
        "--dangerously-bypass-approvals-and-sandbox",
        action="store_true",
        help="Pass through Codex full-access mode.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = KernelBenchOptimizationLoop().run(
        KernelBenchLoopConfig(
            repo_root=REPO_ROOT,
            problem_dir=args.problem_dir,
            prompt_template=args.prompt_template,
            optimization_dir_name=args.optimization_dir_name,
            max_rounds=args.max_rounds,
            device=args.device,
            num_correct_trials=args.num_correct_trials,
            timing_method=args.timing_method,
            measure_performance=args.measure_performance,
            codex_bin=args.codex_bin,
            profile=args.profile,
            model=args.model,
            model_provider=args.model_provider,
            reasoning_effort=args.reasoning_effort,
            timeout_seconds=args.timeout_seconds,
            codex_home=args.codex_home,
            config_overrides=tuple(args.config_overrides),
            bypass_approvals_and_sandbox=args.dangerously_bypass_approvals_and_sandbox,
        )
    )
    print(json.dumps(result, default=lambda value: value.__dict__, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
