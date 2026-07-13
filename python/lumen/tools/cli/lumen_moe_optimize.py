"""CLI for running one Codex MoE optimization workspace."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lumen.harness.datasets.lumen.moe.generation import (
    MoEOptimizationConfig,
    prepare_moe_optimization,
    run_moe_optimization,
    run_moe_optimization_sequence,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize a Lumen MoE kernel through one or more Codex rounds."
        )
    )
    parser.add_argument("--kernel", type=Path, required=True)
    parser.add_argument(
        "--prompt-file",
        type=Path,
        action="append",
        required=True,
        help="Prompt file to apply; repeat this option to run sequential rounds.",
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument(
        "--entrypoint",
        default="fused_moe_fp8_blockscale_g1u1",
    )
    parser.add_argument(
        "--gpu-id",
        type=_non_negative_int,
        default=None,
        help=(
            "Physical GPU ID exposed to Codex and benchmark commands via "
            "HIP_VISIBLE_DEVICES."
        ),
    )
    parser.add_argument("--codex-bin", type=Path, default=None)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--model-provider", default=None)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--timeout-seconds", type=float, default=3600)
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        dest="config_overrides",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Create the workspace without invoking Codex.",
    )
    args = parser.parse_args(argv)
    if args.prepare_only and len(args.prompt_file) != 1:
        parser.error("--prepare-only supports exactly one --prompt-file")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = MoEOptimizationConfig(
        repo_root=_find_repo_root(Path(__file__)),
        kernel=args.kernel,
        prompt_file=args.prompt_file[0],
        run_dir=args.run_dir,
        entrypoint=args.entrypoint,
        gpu_id=args.gpu_id,
        codex_bin=args.codex_bin,
        profile=args.profile,
        model=args.model,
        model_provider=args.model_provider,
        reasoning_effort=args.reasoning_effort,
        timeout_seconds=args.timeout_seconds,
        config_overrides=tuple(args.config_overrides),
    )
    if args.prepare_only:
        workspace = prepare_moe_optimization(config)
        print(workspace.round_dir)
        return 0

    if len(args.prompt_file) == 1:
        rounds = [run_moe_optimization(config)]
    else:
        rounds = run_moe_optimization_sequence(config, args.prompt_file)

    print(f"run: {rounds[0][0].run_dir}")
    for round_index, (workspace, result) in enumerate(rounds):
        print(f"round{round_index}: {workspace.round_dir}")
        print(f"optimized kernel: {workspace.output_model_path}")
        if not result.ok:
            print(result.error or result.status, file=sys.stderr)
            return 1
    return 0


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _find_repo_root(start: Path) -> Path:
    for parent in (start.resolve(), *start.resolve().parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError(f"cannot find repository root from {start}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
