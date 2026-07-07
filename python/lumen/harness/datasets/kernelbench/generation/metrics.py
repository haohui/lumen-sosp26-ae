"""Metrics parsing and reporting for KernelBench generation artifacts."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ParsedEvalResult:
    compiled: bool
    correctness: bool
    runtime: float
    ref_runtime: float
    speedup: float
    metadata: dict[str, Any]


@dataclass(frozen=True)
class GenerationMetrics:
    run_dir: str
    correctness_rate: float | None
    geomean_speedup: float | None
    min_speedup: float | None
    max_speedup: float | None
    trace_edit_count: int


def parse_eval_payload(eval_payload: dict[str, Any]) -> ParsedEvalResult:
    runtime = eval_payload.get("runtime", eval_payload.get("runtime_us", -1.0)) or -1.0
    ref_runtime = (
        eval_payload.get("ref_runtime", eval_payload.get("ref_runtime_us", -1.0))
        or -1.0
    )
    speedup = (ref_runtime / runtime) if runtime > 0 and ref_runtime > 0 else -1.0
    return ParsedEvalResult(
        compiled=bool(eval_payload.get("compiled", False)),
        correctness=bool(eval_payload.get("correctness", False)),
        runtime=float(runtime),
        ref_runtime=float(ref_runtime),
        speedup=float(speedup),
        metadata=eval_payload.get("metadata", {}),
    )


def format_eval_status(eval_payload: dict[str, Any]) -> str:
    parsed = parse_eval_payload(eval_payload)
    if parsed.compiled and parsed.correctness:
        return f"compiled=true  correct=true  speedup={parsed.speedup:.3f}x"
    if parsed.compiled:
        return "compiled=true  correct=false"
    return f"compiled=false  {str(parsed.metadata)[:80]}"


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
        for line in trace_file:
            try:
                visit(json.loads(line))
            except json.JSONDecodeError:
                continue

    return count


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
        if meta is not None:
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


def _read_json_object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


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
