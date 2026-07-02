#!/usr/bin/env python3
"""Run KernelBench eval on an imported pipeline case directory."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from io import StringIO
from pathlib import Path
from typing import Any


KB_EVAL_PIPELINE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = KB_EVAL_PIPELINE_ROOT
PYTHON_ROOT = KB_EVAL_PIPELINE_ROOT

python_root_str = str(PYTHON_ROOT)
if python_root_str not in sys.path:
    sys.path.insert(0, python_root_str)


from harness.evaluator.kernelbench.eval import eval_kernel_against_ref, get_torch_dtype_from_string
from harness.evaluator.kernelbench.utils import set_gpu_arch


@dataclass(frozen=True)
class KernelBenchRunConfig:
    seed: int = 42
    device: int = 0
    num_correct_trials: int = 1
    num_perf_trials: int = 10
    measure_performance: bool = False
    timing_method: str = "cudagraph"
    verbose: bool = False


@dataclass(frozen=True)
class KernelBenchCaseSpec:
    backend: str
    precision: str
    gpu_arch: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate KernelBench imported cases listed in a text file, one case directory per line."
        )
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        required=True,
        help="Path to a text file containing one case directory per line.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to the JSONL output file.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed passed to eval_kernel_against_ref.")
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device index passed to eval_kernel_against_ref.",
    )
    parser.add_argument(
        "--num-correct-trials",
        type=int,
        default=1,
        help="Correctness trials passed to eval_kernel_against_ref.",
    )
    parser.add_argument(
        "--num-perf-trials",
        type=int,
        default=10,
        help="Performance trials passed to eval_kernel_against_ref.",
    )
    parser.add_argument(
        "--measure-performance",
        action="store_true",
        help="Enable performance measurement in eval_kernel_against_ref.",
    )
    parser.add_argument(
        "--timing-method",
        default="cudagraph",
        help="Timing method passed to eval_kernel_against_ref when performance is enabled.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose evaluator logging.",
    )
    parser.add_argument(
        "--phase",
        default="",
        help=(
            "Optional caller phase tag written to harness_events.jsonl, "
            "for example agent_debug_eval or final_eval."
        ),
    )
    return parser.parse_args()


def resolve_case_dir(case_path: Path) -> Path:
    case_path = case_path.expanduser().resolve()
    if not case_path.exists():
        raise FileNotFoundError(f"Case path does not exist: {case_path}")

    if (case_path / "input_model.py").is_file() and (case_path / "output_model_new.py").is_file():
        return case_path

    raise FileNotFoundError(
        "Case path must point to a directory containing input_model.py and output_model_new.py: "
        f"{case_path}"
    )


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_eval_config(case_dir: Path) -> dict[str, Any]:
    config_path = case_dir / "eval_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing eval config: {config_path}")
    return json.loads(read_text(config_path))


def normalize_gpu_arch(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def normalize_timing_value(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric <= 0:
        return None
    return numeric


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str))
        handle.write("\n")


def find_event_timestamp(timings_s: dict[str, Any] | None, event_type: str) -> float | None:
    if not isinstance(timings_s, dict):
        return None
    events = timings_s.get("events")
    if not isinstance(events, list):
        return None
    for event in events:
        if not isinstance(event, dict):
            continue
        if event.get("event_type") != event_type:
            continue
        timestamp = event.get("timestamp")
        if isinstance(timestamp, (int, float)):
            return float(timestamp)
    return None


def append_harness_events(
    *,
    events_path: Path,
    payload: dict[str, Any],
    phase: str,
    output_path: Path,
    input_file: Path,
) -> None:
    run_index = payload.get("run_index")
    base = {
        "phase": phase,
        "run_index": run_index,
        "case_dir": payload.get("case_dir"),
        "output_path": str(output_path),
        "input_file": str(input_file),
        "compiled": payload.get("compiled"),
        "correctness": payload.get("correctness"),
    }
    append_jsonl(
        events_path,
        {
            **base,
            "timestamp": payload.get("run_started_at"),
            "event_type": "harness_eval_start",
        },
    )

    timings_s = payload.get("timings_s")
    if isinstance(timings_s, dict):
        compile_start = find_event_timestamp(timings_s, "custom_model_import_start")
        compile_end = find_event_timestamp(timings_s, "first_candidate_invocation_end")
        if compile_start is not None:
            append_jsonl(
                events_path,
                {
                    **base,
                    "timestamp": compile_start,
                    "event_type": "compilation_start",
                },
            )
        if compile_end is not None:
            append_jsonl(
                events_path,
                {
                    **base,
                    "timestamp": compile_end,
                    "event_type": "compilation_end",
                    "duration_s": timings_s.get("compile_time_s"),
                },
            )

        execution_start = find_event_timestamp(timings_s, "candidate_profile_start")
        execution_end = find_event_timestamp(timings_s, "reference_profile_end")
        if execution_end is None:
            execution_end = find_event_timestamp(timings_s, "candidate_profile_end")
        if execution_start is not None:
            append_jsonl(
                events_path,
                {
                    **base,
                    "timestamp": execution_start,
                    "event_type": "execution_start",
                },
            )
        if execution_end is not None:
            append_jsonl(
                events_path,
                {
                    **base,
                    "timestamp": execution_end,
                    "event_type": "execution_end",
                    "duration_s": timings_s.get("profile_time_s"),
                },
            )

        for event in timings_s.get("events") or []:
            if not isinstance(event, dict):
                continue
            append_jsonl(
                events_path,
                {
                    **base,
                    **event,
                    "phase": phase,
                    "run_index": run_index,
                    "case_dir": payload.get("case_dir"),
                },
            )

    append_jsonl(
        events_path,
        {
            **base,
            "timestamp": payload.get("run_ended_at"),
            "event_type": "harness_eval_end",
            "duration_s": payload.get("run_wall_time_s"),
            "timings_s": timings_s,
            "runtime_us": payload.get("runtime_us"),
            "ref_runtime_us": payload.get("ref_runtime_us"),
            "exception_name": payload.get("exception_name"),
            "exception": payload.get("exception"),
        },
    )


class KernelBenchCaseRunner:
    def __init__(self, config: KernelBenchRunConfig, case_dir: Path):
        self.config = config
        self.case_dir = resolve_case_dir(case_dir)
        self.case_spec = self._load_case_spec()

    def _load_case_spec(self) -> KernelBenchCaseSpec:
        config = load_eval_config(self.case_dir)
        return KernelBenchCaseSpec(
            backend=config.get("backend", "cuda"),
            precision=config.get("precision", "fp32"),
            gpu_arch=normalize_gpu_arch(config.get("gpu_arch")),
        )

    def _load_case_sources(self) -> tuple[str, str]:
        # The evaluator loads Model / ModelNew into per-call exec contexts in eval.py.
        # This runner keeps the script side free of those symbols so consecutive runs
        # do not share harness-level namespaces.
        original_model_src = read_text(self.case_dir / "input_model.py")
        custom_model_src = read_text(self.case_dir / "output_model_new.py")
        return original_model_src, custom_model_src

    def run(self) -> dict[str, Any]:
        run_started_at = round(time.time(), 6)
        run_perf_start = time.perf_counter()
        if self.case_spec.gpu_arch:
            set_gpu_arch(self.case_spec.gpu_arch)

        original_model_src, custom_model_src = self._load_case_sources()
        precision = get_torch_dtype_from_string(self.case_spec.precision)

        log_buffer = StringIO()
        with redirect_stdout(log_buffer):
            result = eval_kernel_against_ref(
                original_model_src=original_model_src,
                custom_model_src=custom_model_src,
                seed_num=self.config.seed,
                num_correct_trials=self.config.num_correct_trials,
                num_perf_trials=self.config.num_perf_trials,
                measure_performance=self.config.measure_performance,
                timing_method=self.config.timing_method,
                verbose=self.config.verbose,
                device=self.config.device,
                backend=self.case_spec.backend,
                precision=precision,
            )
        logs = log_buffer.getvalue()
        if logs:
            print(logs, file=sys.stderr, end="")

        run_ended_at = round(time.time(), 6)
        run_wall_time_s = round(time.perf_counter() - run_perf_start, 6)

        if result is None:
            return {
                "event_type": "harness_run",
                "run_started_at": run_started_at,
                "run_ended_at": run_ended_at,
                "run_wall_time_s": run_wall_time_s,
                "case_dir": str(self.case_dir),
                "config": asdict(self.config),
                "backend": self.case_spec.backend,
                "precision": self.case_spec.precision,
                "gpu_arch": self.case_spec.gpu_arch,
                "result": None,
                "message": "Evaluator returned None, likely due to a retryable compilation race.",
            }

        timings_s = result.metadata.get("timings_s") if isinstance(result.metadata, dict) else None
        return {
            "event_type": "harness_run",
            "run_started_at": run_started_at,
            "run_ended_at": run_ended_at,
            "run_wall_time_s": run_wall_time_s,
            "case_dir": str(self.case_dir),
            "config": asdict(self.config),
            "backend": self.case_spec.backend,
            "precision": self.case_spec.precision,
            "gpu_arch": self.case_spec.gpu_arch,
            "compiled": result.compiled,
            "correctness": result.correctness,
            "runtime_us": normalize_timing_value(result.runtime),
            "ref_runtime_us": normalize_timing_value(result.ref_runtime),
            "metadata": result.metadata,
            "runtime_stats": result.runtime_stats,
            "ref_runtime_stats": result.ref_runtime_stats,
            "timings_s": timings_s,
        }


def load_case_dirs(case_list_path: Path) -> list[Path]:
    case_list_path = case_list_path.expanduser().resolve()
    if not case_list_path.is_file():
        raise FileNotFoundError(f"Case list file does not exist: {case_list_path}")

    case_dirs: list[Path] = []
    for line in case_list_path.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        case_dirs.append(Path(entry))
    return case_dirs


def build_run_config(args: argparse.Namespace) -> KernelBenchRunConfig:
    return KernelBenchRunConfig(
        seed=args.seed,
        device=args.device,
        num_correct_trials=args.num_correct_trials,
        num_perf_trials=args.num_perf_trials,
        measure_performance=args.measure_performance,
        timing_method=args.timing_method,
        verbose=args.verbose,
    )


def main() -> int:
    args = parse_args()
    config = build_run_config(args)
    case_dirs = load_case_dirs(args.input_file)
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    phase = args.phase or "unknown"
    harness_events_path = output_path.parent / "harness_events.jsonl"

    exit_code = 0
    with output_path.open("w", encoding="utf-8") as output_file:
        for run_index, case_dir in enumerate(case_dirs):
            run_started_at = round(time.time(), 6)
            run_perf_start = time.perf_counter()
            try:
                payload = KernelBenchCaseRunner(config, case_dir).run()
            except Exception as exc:
                run_ended_at = round(time.time(), 6)
                payload = {
                    "event_type": "harness_run",
                    "run_index": run_index,
                    "run_started_at": run_started_at,
                    "run_ended_at": run_ended_at,
                    "run_wall_time_s": round(time.perf_counter() - run_perf_start, 6),
                    "case_dir": str(Path(case_dir).expanduser()),
                    "config": asdict(config),
                    "compiled": False,
                    "correctness": False,
                    "exception_name": f"{exc.__class__.__module__}.{exc.__class__.__name__}",
                    "exception": str(exc),
                    "traceback": traceback.format_exc(),
                }
                exit_code = 1
            else:
                if "result" in payload or not (payload["compiled"] and payload["correctness"]):
                    exit_code = 1
            payload.setdefault("event_type", "harness_run")
            payload["run_index"] = run_index
            payload.setdefault("run_started_at", run_started_at)
            payload.setdefault("run_ended_at", round(time.time(), 6))
            payload.setdefault("run_wall_time_s", round(time.perf_counter() - run_perf_start, 6))
            payload["phase"] = phase

            output_file.write(json.dumps(payload, default=str))
            output_file.write("\n")
            output_file.flush()
            append_harness_events(
                events_path=harness_events_path,
                payload=payload,
                phase=phase,
                output_path=output_path,
                input_file=args.input_file.expanduser().resolve(),
            )

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
