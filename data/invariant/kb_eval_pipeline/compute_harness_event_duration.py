#!/usr/bin/env python3
"""Compute evaluation/compilation durations from harness_events.jsonl files."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, TextIO


@dataclass
class HarnessSummary:
    file: str
    phase: str | None
    harness_runs: int
    compilation_events: int
    execution_events: int
    evaluation_duration_s: float
    compilation_duration_s: float
    execution_duration_s: float


def read_jsonl(lines: Iterable[str], name: str) -> list[dict]:
    events = []
    for line_no, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{name}:{line_no}: invalid JSON: {exc}") from exc
        events.append(event)
    return events


def discover_files(paths: Sequence[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(path.rglob("harness_events.jsonl")))
        elif path.is_file():
            files.append(path)
        else:
            raise SystemExit(f"{raw}: path does not exist")
    return sorted(set(files))


def numeric_duration(event: dict) -> float:
    value = event.get("duration_s")
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def event_matches_phase(event: dict, phase: str | None) -> bool:
    return phase is None or event.get("phase") == phase


def summarize_events(events: Sequence[dict], file_name: str, phase: str | None) -> HarnessSummary:
    selected = [event for event in events if event_matches_phase(event, phase)]

    eval_ends = [event for event in selected if event.get("event_type") == "harness_eval_end"]
    compilation_ends = [event for event in selected if event.get("event_type") == "compilation_end"]
    execution_ends = [event for event in selected if event.get("event_type") == "execution_end"]

    return HarnessSummary(
        file=file_name,
        phase=phase,
        harness_runs=len(eval_ends),
        compilation_events=len(compilation_ends),
        execution_events=len(execution_ends),
        evaluation_duration_s=sum(numeric_duration(event) for event in eval_ends),
        compilation_duration_s=sum(numeric_duration(event) for event in compilation_ends),
        execution_duration_s=sum(numeric_duration(event) for event in execution_ends),
    )


def summary_to_dict(summary: HarnessSummary) -> dict:
    return {
        "file": summary.file,
        "phase": summary.phase,
        "harness_runs": summary.harness_runs,
        "compilation_events": summary.compilation_events,
        "execution_events": summary.execution_events,
        "evaluation.duration_s": round(summary.evaluation_duration_s, 6),
        "compilation.duration_s": round(summary.compilation_duration_s, 6),
        "execution.duration_s": round(summary.execution_duration_s, 6),
    }


def print_jsonl(paths: Sequence[str], phase: str | None, output: TextIO) -> None:
    if paths == ["-"]:
        events = read_jsonl(sys.stdin, "<stdin>")
        print(json.dumps(summary_to_dict(summarize_events(events, "<stdin>", phase)), separators=(",", ":")), file=output)
        return

    for path in discover_files(paths):
        with path.open("r", encoding="utf-8") as handle:
            events = read_jsonl(handle, str(path))
        print(json.dumps(summary_to_dict(summarize_events(events, str(path), phase)), separators=(",", ":")), file=output)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute evaluation/compilation durations from harness_events.jsonl files."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["."],
        help="Harness event files or directories. Use '-' to read one JSONL file from stdin.",
    )
    parser.add_argument(
        "--phase",
        default="agent_debug_eval",
        help="Only include events from this phase. Use --phase all to include every phase.",
    )
    args = parser.parse_args(argv)

    phase = None if args.phase == "all" else args.phase
    print_jsonl(args.paths, phase, sys.stdout)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        raise SystemExit(0)
