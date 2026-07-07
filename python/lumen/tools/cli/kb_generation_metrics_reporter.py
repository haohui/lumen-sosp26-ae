"""Report metrics from KernelBench generation artifacts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from lumen.harness.datasets.kernelbench.generation import GenerationConfig
from lumen.tools.cli.kb_generation_config import load_generation_config


@dataclass(frozen=True)
class GenerationMetrics:
    run_dir: str
    correctness_rate: float | None
    geomean_speedup: float | None
    min_speedup: float | None
    max_speedup: float | None
    trace_edit_count: int


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
    problem_ids = problem_ids_from_config(config)
    validate_report_inputs(config.run_dir, problem_ids)
    metrics = collect_generation_metrics(config.run_dir, problem_ids)
    json.dump(metrics_to_json_dict(metrics), sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
    return 0


def validate_report_inputs(run_dir: str | Path, problem_ids: list[int]) -> None:
    if not problem_ids:
        raise ValueError("Metrics reporting requires at least one problem id.")

    base = Path(run_dir)
    if not base.is_dir():
        raise FileNotFoundError(
            f"Metrics run_dir does not exist or is not a directory: {base}"
        )


def problem_ids_from_config(config: GenerationConfig) -> list[int]:
    if config.dataset.problem_ids:
        return list(config.dataset.problem_ids)

    start, end = config.dataset.subset
    if start is None or end is None:
        raise ValueError(
            "Metrics reporting requires dataset.problem_ids or a bounded "
            "dataset.subset in the generation config."
        )
    return list(range(start, end + 1))


def collect_generation_metrics(
    run_dir: str | Path,
    problem_ids: list[int],
) -> GenerationMetrics:
    base = Path(run_dir)
    metas: list[dict[str, Any]] = []
    speedups: list[float] = []
    edit_count = 0

    for pid in problem_ids:
        problem_dir = base / f"p{pid:02d}"
        meta = _read_json_object(problem_dir / "meta.json")
        metas.append(meta)
        speedup = meta.get("speedup")
        if meta.get("correctness") and isinstance(speedup, (int, float)):
            if speedup > 0:
                speedups.append(float(speedup))

        for round_dir in _sorted_prefixed_dirs(problem_dir, "round"):
            edit_count += count_trace_edits(round_dir / "trace.jsonl")

    correct_count = sum(1 for meta in metas if meta.get("correctness"))
    total_count = len(problem_ids)
    geomean = None
    if speedups:
        geomean = math.exp(sum(math.log(value) for value in speedups) / len(speedups))

    return GenerationMetrics(
        run_dir=str(base),
        correctness_rate=100.0 * correct_count / total_count if total_count else None,
        geomean_speedup=geomean,
        min_speedup=min(speedups) if speedups else None,
        max_speedup=max(speedups) if speedups else None,
        trace_edit_count=edit_count,
    )


def metrics_to_json_dict(metrics: GenerationMetrics) -> dict[str, Any]:
    return asdict(metrics)


def count_trace_edits(trace_path: str | Path) -> int:
    path = Path(trace_path)
    if not path.is_file():
        return 0

    counted_tools = {"read", "write", "edit"}
    count = 0

    def visit(node: Any) -> None:
        nonlocal count
        if isinstance(node, dict):
            name = node.get("name")
            if node.get("type") == "tool_use" and isinstance(name, str):
                if name.lower() in counted_tools:
                    count += 1
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    with path.open(encoding="utf-8") as trace_file:
        for line_number, line in enumerate(trace_file, start=1):
            try:
                visit(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSON in trace file {path}:{line_number}"
                ) from exc

    return count


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Expected metrics artifact is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload


def _sorted_prefixed_dirs(directory: str | Path, prefix: str) -> list[Path]:
    base = Path(directory)
    if not base.is_dir():
        return []

    def suffix_index(path: Path) -> int:
        try:
            return int(path.name.removeprefix(prefix))
        except ValueError:
            return -1

    return sorted(
        [
            path
            for path in base.iterdir()
            if path.is_dir() and path.name.startswith(prefix)
        ],
        key=suffix_index,
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
