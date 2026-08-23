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
DEPENDENCY_ROOT = (REPO_ROOT.parent / "third_party").resolve()
DEFAULT_API_URL = "http://47.79.17.216:8080/v1"
DEFAULT_API_KEY = "lumen-ae"
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_DEEPSEEK_MAX_OUTPUT_TOKENS = 65536
CODEX_PROVIDER = "lumen-generation"
BLUE = "\033[34m"
GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"

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
    parser.add_argument(
        "--table2-no-stage",
        action="store_true",
        help="Keep generated Table 2 kernels in the run workspace only.",
    )
    parser.add_argument("--generation-rounds", type=positive_int, default=10)
    parser.add_argument(
        "--kernelbench-attempts",
        type=positive_int,
        default=2,
        help="Maximum one-shot API attempts for each Table 2 KernelBench task.",
    )
    parser.add_argument(
        "--generation-api-timeout-seconds",
        type=positive_float,
        default=300.0,
        help="Timeout for each direct Table 2 generation API request.",
    )
    parser.add_argument(
        "--generation-max-output-tokens",
        type=positive_int,
        default=None,
        help=(
            "Output-token limit for each direct Table 2 KernelBench request. "
            "By default, the artifact DeepSeek-V4 API uses 65536 and other "
            "APIs use agent_generate.py's default."
        ),
    )
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
                "KERNELBENCH_ROOT": str(
                    DEPENDENCY_ROOT / "KernelBench" / "KernelBench"
                ),
                "CUDAFORGE_ROOT": str(DEPENDENCY_ROOT / "CUDAForge" / "CudaForge"),
                "KERNELFALCON_ROOT": str(
                    DEPENDENCY_ROOT / "KernelFalcon" / "KernelAgent"
                ),
                "KSEARCH_ROOT": str(DEPENDENCY_ROOT / "KSearch" / "K-Search"),
                "PYTHONUNBUFFERED": "1",
            }
        )
        self.env["PYTHONPATH"] = prepend_path(
            str(REPO_ROOT / "python"), self.env.get("PYTHONPATH")
        )
        self.failures: list[tuple[str, int]] = []
        self.experiment_summaries: list[dict[str, object]] = []
        self.active_logs: list[Path] = []

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
        try:
            self.validate_environment()
            self.prepare()
        except Exception as exc:
            reason = f"environment/setup failed: {exc}"
            for name in self.selected:
                self.record_summary(name, False, reason, [])
            self.print_final_summary()
            raise
        print(f"Repository: {REPO_ROOT}")
        print(f"Output: {self.output_dir}")
        print(f"Experiments: {', '.join(self.selected)}")
        for name in self.selected:
            self.active_logs = []
            try:
                self.run_experiment(name)
            except subprocess.CalledProcessError as exc:
                self.failures.append((name, exc.returncode))
                reason = self.command_failure_reason(exc.returncode)
                self.record_summary(name, False, reason, self.active_logs)
                print(f"FAILED: {name} (exit {exc.returncode})", file=sys.stderr)
                if not self.args.keep_going:
                    break
            except Exception as exc:
                self.failures.append((name, 1))
                self.record_summary(name, False, str(exc), self.active_logs)
                print(f"FAILED: {name} ({exc})", file=sys.stderr)
                if not self.args.keep_going:
                    break
            else:
                self.record_summary(
                    name,
                    True,
                    (
                        "dry run completed; command constructed but not executed"
                        if self.args.dry_run
                        else "completed successfully"
                    ),
                    self.active_logs,
                )

        summarized = {str(item["name"]) for item in self.experiment_summaries}
        for name in self.selected:
            if name not in summarized:
                self.record_summary(
                    name,
                    False,
                    "not run because an earlier experiment failed",
                    [],
                )
        self.print_final_summary()
        if self.failures:
            print("Failures:", file=sys.stderr)
            for name, code in self.failures:
                print(f"  {name}: exit {code}", file=sys.stderr)
            return 1
        return 0

    def record_summary(
        self,
        name: str,
        passed: bool,
        reason: str,
        logs: list[Path],
    ) -> None:
        self.experiment_summaries.append(
            {
                "name": name,
                "passed": passed,
                "reason": " ".join(reason.split()),
                "results": self.result_paths(name),
                "logs": [str(path) for path in logs],
                "paper": self.paper_comparison(name),
            }
        )

    def print_final_summary(self) -> None:
        print(f"\n=== {BLUE}FINAL_SUMMARY{RESET} ===")
        for item in self.experiment_summaries:
            passed = bool(item["passed"])
            status = "PASS" if passed else "FAIL"
            status_color = GREEN if passed else RED
            print(f"\n{BLUE}Experiment{RESET}: {item['name']}")
            print(f"Status: {status_color}{status}{RESET}")
            print(f"Reason: {item['reason']}")
            print(f"Paper comparison: {item['paper']}")
            print("Result paths:")
            for path in item["results"] or ["none"]:
                print(f"  - {path}")
            print("Log-to-paper mapping:")
            for path in item["logs"] or ["none"]:
                print(f"  - {path} -> {item['paper']}")

    def result_paths(self, name: str) -> list[str]:
        paths = {
            "figure1": [
                str(
                    REPO_ROOT
                    / "datasets/inference/attention/lumen/attn_07_invariants.py"
                )
            ],
            "table2-generation": [
                str(self.output_dir / "table2/generation"),
                str(REPO_ROOT / "datasets/inference/<task>/<baseline>"),
            ],
            "table2-optimization": [
                str(self.output_dir / "table2/optimization/gemm"),
                str(self.output_dir / "table2/optimization/attn"),
                str(self.output_dir / "table2/optimization/moe"),
            ],
            "table2-benchmark": [
                str(self.output_dir / "table2/benchmark/table2.csv"),
                str(self.output_dir / "table2/benchmark/{gemm,attention,moe}.jsonl"),
            ],
            "figure2": [
                str(self.output_dir / "figure2/attention_ablation.csv")
            ],
            "table3-generation": [str(REPO_ROOT / "data/traces")],
            "table3-optimization": [str(REPO_ROOT / "data/traces")],
            "table3-summary": [str(self.output_dir / "table3/table3.csv")],
            "api-check": ["none (connectivity status is in the log/output)"],
        }
        return paths.get(name, [])

    @staticmethod
    def paper_comparison(name: str) -> str:
        return {
            "figure1": "Figure 1 invariant-validation status",
            "table2-generation": "Table 2 agent-generated baseline kernels",
            "table2-optimization": "Table 2 Lumen optimized kernels",
            "table2-benchmark": "Table 2 throughput cells (use table2.csv)",
            "figure2": "Figure 2 attention-ablation curves (use the CSV)",
            "table3-generation": "Table 3 generation and context-ablation rows",
            "table3-optimization": "Table 3 invariant-guided optimization rows",
            "table3-summary": "Table 3 reported summary rows (use table3.csv)",
            "api-check": "environment prerequisite; no paper result",
        }.get(name, "paper artifact result")

    def command_failure_reason(self, returncode: int) -> str:
        for log_path in reversed(self.active_logs):
            try:
                lines = log_path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line in reversed(lines):
                if line.startswith("Reason: "):
                    return f"exit {returncode}: {line.removeprefix('Reason: ')}"
        return f"command exited with status {returncode}; inspect the log"

    def validate_environment(self) -> None:
        gpu_experiments = {
            "figure1",
            "table2-generation",
            "table2-optimization",
            "table2-benchmark",
            "figure2",
            "table3-generation",
            "table3-optimization",
        }
        table3_compute = {"table3-generation", "table3-optimization"}
        needs_api = any(
            item == "api-check" or "generation" in item or "optimization" in item
            for item in self.selected
        )
        needs_codex = any(
            item in {"table2-optimization", *table3_compute}
            for item in self.selected
        )
        needs_hf = any(item in table3_compute for item in self.selected)
        min_gpus = 8 if any(item in table3_compute for item in self.selected) else 0
        if min_gpus == 0 and any(item in gpu_experiments for item in self.selected):
            min_gpus = 1

        command = [
            self.args.python,
            "scripts/validate_environment.py",
            "--output-dir",
            str(self.output_dir),
            "--min-gpus",
            str(min_gpus),
        ]
        if needs_api:
            command.append("--require-api")
        if needs_codex:
            command.append("--require-codex")
        if needs_hf:
            command.append("--require-hf")
        if "table2-generation" in self.selected:
            command.append("--require-agent-frameworks")
        if self.args.dry_run:
            command.append("--skip-network")

        print("=== Environment validation ===", flush=True)
        result = subprocess.run(command, cwd=REPO_ROOT, env=self.env, check=False)
        if result.returncode:
            raise RuntimeError(
                "environment validation failed; resolve the failed checks above"
            )

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
        max_output_tokens = self.args.generation_max_output_tokens
        if max_output_tokens is None and self.uses_default_deepseek_api():
            max_output_tokens = DEFAULT_DEEPSEEK_MAX_OUTPUT_TOKENS

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
                "--kernelbench-attempts",
                str(self.args.kernelbench_attempts),
                "--api-timeout-seconds",
                str(self.args.generation_api_timeout_seconds),
                *(
                    ["--max-output-tokens", str(max_output_tokens)]
                    if max_output_tokens is not None
                    else []
                ),
                "--device",
                str(self.args.gpu_id),
                "--workspace-dir",
                str(self.output_dir / "table2" / "generation"),
                *(["--no-stage"] if self.args.table2_no_stage else []),
            ],
        )

    def uses_default_deepseek_api(self) -> bool:
        return (
            self.args.api_url.rstrip("/") == DEFAULT_API_URL.rstrip("/")
            and self.args.model.casefold() == DEFAULT_MODEL.casefold()
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
        log_path = self.log_dir / f"{label}.log"
        self.active_logs.append(log_path)
        if self.args.dry_run:
            return
        command_env = self.env.copy()
        command_env["LUMEN_EXPERIMENT_LOG"] = str(log_path)
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=command_env,
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
