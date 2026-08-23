#!/usr/bin/env python3
"""Single entry point for the Lumen artifact experiments."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_API_URL = "http://47.79.17.216:8080/v1"
DEFAULT_API_KEY = "lumen-ae"
DEFAULT_MODEL = "deepseek-v4-flash"
CODEX_PROVIDER = "lumen-generation"

EXPERIMENTS = (
    "api-check",
    "figure1",
    "table2-generation",
    "table2-optimization",
    "table2-benchmark",
    "figure2",
    "table3-generation",
    "table3-optimization",
    "table3-summary",
)
ALL_EXPERIMENTS = tuple(name for name in EXPERIMENTS if name != "api-check")
GROUPS = {
    "all": ALL_EXPERIMENTS,
    "generation": (
        "table2-generation",
        "table2-optimization",
        "table3-generation",
        "table3-optimization",
    ),
    "evaluation": (
        "figure1",
        "table2-benchmark",
        "figure2",
        "table3-summary",
    ),
    "table2": (
        "table2-generation",
        "table2-optimization",
        "table2-benchmark",
    ),
    "table3": (
        "table3-generation",
        "table3-optimization",
        "table3-summary",
    ),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all or selected Lumen artifact experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "experiments",
        nargs="*",
        metavar="EXPERIMENT",
        help="Experiment or group to run. With no value, runs the 'all' group.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List experiment and group names, then exit.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Run outputs and logs directory.",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu-id", type=non_negative_int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue with independent experiments after a failure.",
    )
    parser.add_argument(
        "--api-url",
        default=os.environ.get("LUMEN_GENERATION_API_URL", DEFAULT_API_URL),
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("LUMEN_GENERATION_API_KEY", DEFAULT_API_KEY),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("LUMEN_GENERATION_MODEL", DEFAULT_MODEL),
    )
    parser.add_argument(
        "--table2-baseline",
        choices=("all", "kernelbench", "cudaforge", "kernelfalcon", "ksearch"),
        default="all",
    )
    parser.add_argument(
        "--table2-task",
        choices=("all", "gemm", "attention", "moe"),
        default="all",
    )
    parser.add_argument("--generation-rounds", type=positive_int, default=10)
    parser.add_argument("--warmup", type=non_negative_int, default=10)
    parser.add_argument("--benchmark-repeat", type=positive_int, default=100)
    parser.add_argument(
        "--optimizer-timeout-seconds",
        type=positive_float,
        default=3600.0,
    )
    return parser.parse_args(argv)


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def list_experiments() -> None:
    print("Experiments:")
    for name in EXPERIMENTS:
        print(f"  {name}")
    print("Groups:")
    for name, members in GROUPS.items():
        print(f"  {name}: {', '.join(members)}")


def expand_experiments(values: list[str]) -> list[str]:
    requested = values or ["all"]
    expanded: set[str] = set()
    unknown: list[str] = []
    for value in requested:
        if value in GROUPS:
            expanded.update(GROUPS[value])
        elif value in EXPERIMENTS:
            expanded.add(value)
        else:
            unknown.append(value)
    if unknown:
        choices = ", ".join((*GROUPS, *EXPERIMENTS))
        message = f"unknown experiment(s): {', '.join(unknown)}\nchoices: {choices}"
        raise SystemExit(message)
    return [name for name in EXPERIMENTS if name in expanded]


class Runner:
    def __init__(self, args: argparse.Namespace, selected: list[str]) -> None:
        self.args = args
        self.selected = selected
        run_id = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        default_output = REPO_ROOT / "runs" / f"artifact_experiments_{run_id}"
        self.output_dir = (args.output_dir or default_output).expanduser().resolve()
        self.log_dir = self.output_dir / "logs"
        self.codex_home = self.output_dir / "codex-home"
        self.env = os.environ.copy()
        self.env.update(
            {
                "LUMEN_GENERATION_API_URL": args.api_url.rstrip("/"),
                "LUMEN_GENERATION_API_KEY": args.api_key,
                "LUMEN_GENERATION_MODEL": args.model,
                "KERNELBENCH_ROOT": str(REPO_ROOT / "third_party" / "KernelBench" / "KernelBench"),
                "CUDAFORGE_ROOT": str(REPO_ROOT / "third_party" / "CUDAForge" / "CudaForge"),
                "KERNELFALCON_ROOT": str(REPO_ROOT / "third_party" / "KernelFalcon" / "KernelAgent"),
                "KSEARCH_ROOT": str(REPO_ROOT / "third_party" / "KSearch" / "K-Search"),
            }
        )
        self.env["PYTHONPATH"] = prepend_path(
            str(REPO_ROOT / "python"), self.env.get("PYTHONPATH")
        )
        self.failures: list[tuple[str, int]] = []

    def prepare(self) -> None:
        if self.args.dry_run:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        needs_codex = any(
            "generation" in item or "optimization" in item for item in self.selected
        )
        if needs_codex:
            self.write_codex_config()

    def write_codex_config(self) -> None:
        self.codex_home.mkdir(parents=True, exist_ok=True)
        config = (
            f"model = {json.dumps(self.args.model)}\n"
            f"model_provider = {json.dumps(CODEX_PROVIDER)}\n\n"
            f"[model_providers.{json.dumps(CODEX_PROVIDER)}]\n"
            'name = "Lumen generation API"\n'
            f"base_url = {json.dumps(self.args.api_url.rstrip('/'))}\n"
            'env_key = "LUMEN_GENERATION_API_KEY"\n'
            'wire_api = "responses"\n'
        )
        path = self.codex_home / "config.toml"
        path.write_text(config, encoding="utf-8")
        path.chmod(0o600)
        self.env["CODEX_HOME"] = str(self.codex_home)

    def execute(self) -> int:
        self.prepare()
        print(f"Repository: {REPO_ROOT}")
        print(f"Output: {self.output_dir}")
        print(f"Experiments: {', '.join(self.selected)}")
        for name in self.selected:
            try:
                self.run_experiment(name)
            except subprocess.CalledProcessError as exc:
                self.failures.append((name, exc.returncode))
                print(f"FAILED: {name} (exit {exc.returncode})", file=sys.stderr)
                if not self.args.keep_going:
                    break
        if self.failures:
            print("Failures:", file=sys.stderr)
            for name, code in self.failures:
                print(f"  {name}: exit {code}", file=sys.stderr)
            return 1
        return 0

    def run_experiment(self, name: str) -> None:
        method_name = "run_" + name.replace("-", "_")
        getattr(self, method_name)()

    def run_api_check(self) -> None:
        print("\n=== Generation API check ===")
        if self.args.dry_run:
            print("OpenAI Responses request: max_output_tokens=128")
            return
        from openai import OpenAI

        client = OpenAI(
            api_key=self.args.api_key,
            base_url=self.args.api_url,
            timeout=30.0,
        )
        response = client.responses.create(
            model=self.args.model,
            input="Reply with exactly LUMEN_OK and no explanation.",
            max_output_tokens=128,
        )
        text = response.output_text.strip()
        if text != "LUMEN_OK":
            raise RuntimeError(f"unexpected generation API response: {text!r}")
        print(text)

    def run_figure1(self) -> None:
        self.run_command(
            "figure1",
            [self.args.python, "scripts/figure1/validate_invariant.py"],
        )

    def run_table2_generation(self) -> None:
        self.run_command(
            "table2-generation",
            [
                self.args.python,
                "scripts/table2/agent_generate.py",
                "--baseline",
                self.args.table2_baseline,
                "--task",
                self.args.table2_task,
                "--model",
                self.args.model,
                "--rounds",
                str(self.args.generation_rounds),
                "--device",
                str(self.args.gpu_id),
                "--workspace-dir",
                str(self.output_dir / "table2" / "generation"),
            ],
        )

    def run_table2_optimization(self) -> None:
        for domain in ("gemm", "attn", "moe"):
            self.run_command(
                f"table2-optimization-{domain}",
                [
                    self.args.python,
                    "scripts/table2/optimizer.py",
                    domain,
                    "--gpu-id",
                    str(self.args.gpu_id),
                    "--run-dir",
                    str(self.output_dir / "table2" / "optimization" / domain),
                    "--model",
                    self.args.model,
                    "--model-provider",
                    CODEX_PROVIDER,
                    "--timeout-seconds",
                    str(self.args.optimizer_timeout_seconds),
                ],
            )

    def run_table2_benchmark(self) -> None:
        self.run_command(
            "table2-benchmark",
            [
                self.args.python,
                "scripts/table2/benchmark.py",
                "--workspace-dir",
                str(self.output_dir / "table2" / "benchmark"),
                "--warmup",
                str(self.args.warmup),
                "--repeat",
                str(self.args.benchmark_repeat),
            ],
        )

    def run_figure2(self) -> None:
        self.run_command(
            "figure2",
            [
                self.args.python,
                "scripts/figure2/bench_attn_ablation.py",
                "--output",
                str(self.output_dir / "figure2" / "attention_ablation.csv"),
                "--warmup",
                str(self.args.warmup),
                "--repeat",
                str(self.args.benchmark_repeat),
            ],
        )

    def run_table3_generation(self) -> None:
        self.run_table3("generation-all")

    def run_table3_optimization(self) -> None:
        self.run_table3("optimization-all")

    def run_table3(self, command: str) -> None:
        self.run_command(
            f"table3-{command}",
            [
                self.args.python,
                "scripts/table3/run_kernelbench_table3.py",
                "--codex-home",
                str(self.codex_home),
                command,
            ],
        )

    def run_table3_summary(self) -> None:
        self.run_command(
            "table3-summary",
            [
                self.args.python,
                "scripts/table3/kernelbench_table.py",
                "--format",
                "csv",
                "--output",
                str(self.output_dir / "table3" / "table3.csv"),
            ],
        )

    def run_command(self, label: str, command: list[str]) -> None:
        print(f"\n=== {label} ===")
        print("Command:", shlex.join(command))
        if self.args.dry_run:
            return
        log_path = self.log_dir / f"{label}.log"
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=self.env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                log.write(line)
            returncode = process.wait()
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)


def prepend_path(path: str, existing: str | None) -> str:
    return path if not existing else path + os.pathsep + existing


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list:
        list_experiments()
        return 0
    selected = expand_experiments(args.experiments)
    try:
        return Runner(args, selected).execute()
    except KeyboardInterrupt:
        print("Interrupted; active experiment stopped.", file=sys.stderr)
        return 130
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
