#!/usr/bin/env python3
"""Run the KernelBench experiments needed for Table 3."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "scripts" / "table3" / "config"
TRACE_ROOT = REPO_ROOT / "data" / "traces"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runner = Runner(
        python_bin=args.python,
        codex_home=args.codex_home,
        candidate_root=args.candidate_root,
    )
    args.func(runner, args)
    return 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        argv = ["all"]

    parser = argparse.ArgumentParser(
        description="Run the KernelBench experiments needed for Table 3."
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--codex-home",
        type=Path,
        default=Path(os.environ.get("CODEX_HOME", "/root/.codex-moonbridge")),
    )
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=Path(
            os.environ.get(
                "TABLE3_CANDIDATE_ROOT",
                str(TRACE_ROOT / "kernelbench_optimization_naive_avelang_seeds"),
            )
        ),
        help="Root containing optimization seed candidate manifests.",
    )

    subparsers = parser.add_subparsers(required=True)
    generation = subparsers.add_parser("generation")
    generation.add_argument("level", choices=("level1", "level2"))
    generation.add_argument("mode", choices=("full", "no-examples"))
    generation.set_defaults(func=run_generation_command)

    optimization = subparsers.add_parser("optimization")
    optimization.add_argument("level", choices=("level1", "level2"))
    optimization.add_argument("profile", choices=("invariants", "no-invariants"))
    optimization.set_defaults(func=run_optimization_command)

    subparsers.add_parser("generation-all").set_defaults(func=run_generation_all)
    subparsers.add_parser("optimization-all").set_defaults(func=run_optimization_all)
    subparsers.add_parser("all").set_defaults(func=run_all)
    return parser.parse_args(argv)


class Runner:
    def __init__(
        self,
        *,
        python_bin: str,
        codex_home: Path,
        candidate_root: Path,
    ) -> None:
        self.python_bin = python_bin
        self.codex_home = codex_home.expanduser()
        self.candidate_root = candidate_root.expanduser()
        self.log_dir = TRACE_ROOT / "logs"

    def generation(self, level: str, mode: str) -> None:
        suffix = generation_suffix(mode)
        config = CONFIG_DIR / f"kernelbench_generation_{level}_{suffix}.toml"
        log = (
            self.log_dir
            / f"kernelbench_generation_{level}_{suffix}_{timestamp()}.log"
        )
        self.run(
            [
                self.python_bin,
                "-m",
                "lumen.tools.cli.kernelbench_generate",
                "--config",
                str(config),
                "--log-level",
                "INFO",
            ],
            log,
            title=f"KernelBench generation: {level} {mode}",
        )

    def optimization(self, level: str, profile: str) -> None:
        level_num = level.removeprefix("level")
        suffix = optimization_suffix(profile)
        config = CONFIG_DIR / f"kernelbench_optimization_{level}.toml"
        opt_config = CONFIG_DIR / f"kernelbench_optimization_{level}_{suffix}.toml"
        candidates = (
            self.candidate_root
            / "manifests"
            / f"kernelbench_level{level_num}_naive_avelang_candidates.toml"
        )
        if not candidates.is_file():
            raise FileNotFoundError(f"candidate manifest not found: {candidates}")
        log = (
            self.log_dir
            / f"kernelbench_optimization_{level}_{suffix}_{timestamp()}.log"
        )
        self.run(
            [
                self.python_bin,
                "-m",
                "lumen.tools.cli.kernelbench_optimize",
                "--config",
                str(config),
                "--optimization-config",
                str(opt_config),
                "--candidates",
                str(candidates),
                "--log-level",
                "INFO",
            ],
            log,
            title=f"KernelBench optimization: {level} {profile}",
        )

    def run(self, cmd: list[str], log: Path, *, title: str) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["CODEX_HOME"] = str(self.codex_home)
        env["PYTHONPATH"] = prepend_path(
            str(REPO_ROOT / "python"),
            env.get("PYTHONPATH"),
        )

        print(title)
        print("Command:", " ".join(cmd))
        print("Log:", log)
        with log.open("w", encoding="utf-8") as handle:
            process = subprocess.Popen(
                cmd,
                cwd=REPO_ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                handle.write(line)
            if process.wait() != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)


def run_generation_command(runner: Runner, args: argparse.Namespace) -> None:
    runner.generation(args.level, args.mode)


def run_optimization_command(runner: Runner, args: argparse.Namespace) -> None:
    runner.optimization(args.level, args.profile)


def run_generation_all(runner: Runner, _args: argparse.Namespace) -> None:
    for level in ("level1", "level2"):
        for mode in ("full", "no-examples"):
            runner.generation(level, mode)


def run_optimization_all(runner: Runner, _args: argparse.Namespace) -> None:
    for level in ("level1", "level2"):
        for profile in ("no-invariants", "invariants"):
            runner.optimization(level, profile)


def run_all(runner: Runner, args: argparse.Namespace) -> None:
    run_generation_all(runner, args)
    run_optimization_all(runner, args)


def generation_suffix(mode: str) -> str:
    return {"full": "full", "no-examples": "no_examples"}[mode]


def optimization_suffix(profile: str) -> str:
    return {"invariants": "invariants", "no-invariants": "no_invariants"}[profile]


def prepend_path(path: str, existing: str | None) -> str:
    return path if not existing else path + os.pathsep + existing


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


if __name__ == "__main__":
    raise SystemExit(main())
