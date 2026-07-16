"""Print timing summaries for Table 2 optimizer runs."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


TIME_KEYS = {
    "timestamp",
    "time",
    "created_at",
    "createdAt",
    "started_at",
    "started_at_utc",
    "finished_at",
    "finished_at_utc",
}
HEADERS = (
    "round",
    "ok",
    "codex_s",
    "trace_s",
    "eval_s",
    "run_s",
    "compile+overhead_s",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print optimizer timing summary.")
    parser.add_argument("run_dirs", type=Path, nargs="+")
    args = parser.parse_args(argv)

    rows = [row for run_dir in args.run_dirs for row in summarize(run_dir)]
    if not rows:
        print("no optimizer rounds found", file=sys.stderr)
        return 1
    print_table(rows)
    return 0


def summarize(path: Path) -> list[dict[str, Any]]:
    root = path.expanduser().resolve()
    rounds = [root] if is_round(root) else [
        child
        for child in sorted(root.iterdir(), key=lambda item: item.name)
        if is_round(child)
    ]
    return [summarize_round(round_dir) for round_dir in rounds]


def is_round(path: Path) -> bool:
    return (
        path.is_dir()
        and path.name.startswith("round")
        and (path / "run_config.json").is_file()
    )


def summarize_round(round_dir: Path) -> dict[str, Any]:
    codex = read_json(round_dir / "codex_result.json")
    evaluation = read_json(round_dir / "eval_result.json")
    eval_s = seconds(evaluation.get("started_at_utc"), evaluation.get("finished_at_utc"))
    run_s = runtime_seconds(evaluation)
    overhead_s = None if eval_s is None or run_s is None else max(0.0, eval_s - run_s)
    return {
        "round": round_dir.name,
        "ok": evaluation.get("ok"),
        "codex_s": seconds(codex.get("started_at_utc"), codex.get("finished_at_utc")),
        "trace_s": trace_seconds(round_dir / "trace.jsonl"),
        "eval_s": eval_s,
        "run_s": run_s,
        "compile+overhead_s": overhead_s,
    }


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def trace_seconds(path: Path) -> float | None:
    times = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        try:
            times.extend(find_times(json.loads(line)))
        except json.JSONDecodeError:
            pass
    return None if not times else max(0.0, (max(times) - min(times)).total_seconds())


def find_times(value: Any) -> list[datetime]:
    if isinstance(value, dict):
        direct = [parse_time(item) for key, item in value.items() if key in TIME_KEYS]
        nested = [time for item in value.values() for time in find_times(item)]
        return [time for time in direct + nested if time is not None]
    if isinstance(value, list):
        return [time for item in value for time in find_times(item)]
    return []


def runtime_seconds(payload: dict[str, Any]) -> float | None:
    mean_ms = [
        float(record["mean_ms"])
        for record in payload.get("records", [])
        if isinstance(record, dict) and isinstance(record.get("mean_ms"), int | float)
    ]
    return sum(mean_ms) / 1000.0 if mean_ms else None


def seconds(start: Any, end: Any) -> float | None:
    start_time = parse_time(start)
    end_time = parse_time(end)
    if start_time is None or end_time is None:
        return None
    return max(0.0, (end_time - start_time).total_seconds())


def parse_time(value: Any) -> datetime | None:
    if isinstance(value, int | float):
        return datetime.fromtimestamp(float(value)).astimezone()
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def print_table(rows: list[dict[str, Any]]) -> None:
    rendered = [[format_cell(row.get(key)) for key in HEADERS] for row in rows]
    widths = [
        max(len(header), *(len(row[index]) for row in rendered))
        for index, header in enumerate(HEADERS)
    ]
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(HEADERS)))
    for row in rendered:
        print("  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)))


def format_cell(value: Any) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
