"""Metrics parsing and reporting for KernelBench generation artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParsedEvalResult:
    compiled: bool
    correctness: bool
    runtime: float
    ref_runtime: float
    speedup: float
    metadata: dict[str, Any]


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
