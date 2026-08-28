#!/usr/bin/env python3
"""Run the KernelBench experiments needed for Table 3."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from experiment_summary import ExperimentSummary  # noqa: E402

CONFIG_DIR = REPO_ROOT / "scripts" / "table3" / "config"
DATA_ROOT = REPO_ROOT / "data"
TRACE_ROOT = REPO_ROOT / "data" / "traces"
DEFAULT_CANDIDATE_ARCHIVE = (
    DATA_ROOT / "seeds" / "kernelbench_optimization_naive_avelang_seeds.tar.xz"
)
SEED_ROOT_NAME = "kernelbench_optimization_naive_avelang_seeds"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runner = Runner(
        python_bin=args.python,
        codex_home=args.codex_home,
        candidate_root=args.candidate_root,
        candidate_archive=args.candidate_archive,
    )
    command = getattr(args, "func").__name__.removeprefix("run_").removesuffix(
        "_command"
    )
    with ExperimentSummary(
        f"table3-{command.replace('_', '-')}",
        "compare generated summary statistics from these traces with Table 3",
    ) as summary:
        summary.add_result(TRACE_ROOT)
        try:
            args.func(runner, args)
        finally:
            for log in runner.generated_logs:
                summary.add_log(log)
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
        default=codex_home_default(),
        help=(
            "Codex home to use for LLM calls. Defaults to CODEX_HOME when set; "
            "otherwise leaves CODEX_HOME unchanged so the Codex CLI uses its "
            "own default configuration."
        ),
    )
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=candidate_root_default(),
        help=(
            "Root containing already extracted optimization seed candidates. "
            "Defaults to TABLE3_CANDIDATE_ROOT when set; otherwise the seed "
            "archive is extracted automatically."
        ),
    )
    parser.add_argument(
        "--candidate-archive",
        type=Path,
        default=Path(
            os.environ.get("TABLE3_CANDIDATE_ARCHIVE", str(DEFAULT_CANDIDATE_ARCHIVE))
        ),
        help="Tar.xz archive containing optimization seed candidates.",
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
        codex_home: Path | None,
        candidate_root: Path | None,
        candidate_archive: Path,
    ) -> None:
        self.python_bin = python_bin
        self.codex_home = codex_home.expanduser() if codex_home is not None else None
        self.candidate_root = candidate_root.expanduser() if candidate_root else None
        self.candidate_archive = candidate_archive.expanduser()
        self.log_dir = TRACE_ROOT / "logs"
        self.generated_logs: list[Path] = []

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
        if self.candidate_root is not None:
            self._optimization(level, profile, self.candidate_root)
            return

        with tempfile.TemporaryDirectory(prefix="lumen-table3-seeds-") as tempdir:
            candidate_root = extract_candidate_archive(
                self.candidate_archive,
                Path(tempdir),
            )
            self._optimization(level, profile, candidate_root)

    def _optimization(self, level: str, profile: str, candidate_root: Path) -> None:
        level_num = level.removeprefix("level")
        suffix = optimization_suffix(profile)
        config = CONFIG_DIR / f"kernelbench_optimization_{level}.toml"
        opt_config = CONFIG_DIR / f"kernelbench_optimization_{level}_{suffix}.toml"
        candidates = (
            candidate_root
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
        self.generated_logs.append(log)
        env = os.environ.copy()
        if self.codex_home is not None:
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


def codex_home_default() -> Path | None:
    value = os.environ.get("CODEX_HOME")
    return Path(value) if value else None


def candidate_root_default() -> Path | None:
    value = os.environ.get("TABLE3_CANDIDATE_ROOT")
    return Path(value) if value else None


def extract_candidate_archive(archive: Path, destination: Path) -> Path:
    if not archive.is_file():
        raise FileNotFoundError(f"candidate archive not found: {archive}")
    with tarfile.open(archive, mode="r:xz") as tar:
        if hasattr(tarfile, "data_filter"):
            tar.extractall(destination, filter="data")
        else:
            tar.extractall(destination)
    candidate_root = destination / SEED_ROOT_NAME
    if not (candidate_root / "manifests").is_dir():
        raise FileNotFoundError(
            f"candidate archive does not contain {SEED_ROOT_NAME}/manifests"
        )
    return candidate_root


def prepend_path(path: str, existing: str | None) -> str:
    return path if not existing else path + os.pathsep + existing


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


if __name__ == "__main__":
    raise SystemExit(main())
