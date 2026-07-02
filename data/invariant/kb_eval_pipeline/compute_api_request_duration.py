#!/usr/bin/env python3
"""Compute api_request.duration_ms from Codex agent session traces.

The traces in this directory record one explicit request start as
`event_msg`/`token_count` with `info: null`, and record each request end as
`event_msg`/`token_count` with usage in `info`. Later request starts are not
explicit, so this script infers them from the timestamp immediately after tool
outputs complete, before the next model response item.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence, TextIO


TOOL_OUTPUT_TYPES = {
    "function_call_output",
    "custom_tool_call_output",
    "local_shell_call_output",
}

FUNCTION_CALL_TYPES = {
    "function_call",
    "custom_tool_call",
    "local_shell_call",
}


@dataclass
class ApiRequest:
    file: str
    number: int
    start_ts: datetime
    end_ts: datetime
    start_source: str
    first_response_ts: datetime | None
    response_items: int
    function_calls: int
    usage: dict | None

    @property
    def duration_ms(self) -> float:
        return (self.end_ts - self.start_ts).total_seconds() * 1000.0

    @property
    def time_to_first_response_item_ms(self) -> float | None:
        if self.first_response_ts is None:
            return None
        return (self.first_response_ts - self.start_ts).total_seconds() * 1000.0

    @property
    def inferred_start(self) -> bool:
        return self.start_source != "token_count_null"


def parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def payload_type(event: dict) -> str | None:
    payload = event.get("payload")
    if not isinstance(payload, dict):
        return None
    return payload.get("type")


def is_token_count(event: dict) -> bool:
    return event.get("type") == "event_msg" and payload_type(event) == "token_count"


def is_token_count_start(event: dict) -> bool:
    return is_token_count(event) and (event.get("payload") or {}).get("info") is None


def is_token_count_end(event: dict) -> bool:
    if not is_token_count(event):
        return False
    info = (event.get("payload") or {}).get("info")
    return isinstance(info, dict) and (
        "last_token_usage" in info or "total_token_usage" in info
    )


def is_tool_output(event: dict) -> bool:
    return event.get("type") == "response_item" and payload_type(event) in TOOL_OUTPUT_TYPES


def is_model_response_item(event: dict) -> bool:
    return event.get("type") == "response_item" and payload_type(event) not in TOOL_OUTPUT_TYPES


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
        if "timestamp" not in event:
            raise SystemExit(f"{name}:{line_no}: missing timestamp")
        events.append(event)
    return events


def usage_from_end(event: dict) -> dict | None:
    info = (event.get("payload") or {}).get("info") or {}
    return info.get("last_token_usage") or info.get("total_token_usage")


def choose_start(events: Sequence[dict], begin: int, end: int) -> tuple[int, str]:
    for idx in range(end - 1, begin - 1, -1):
        if is_token_count_start(events[idx]):
            return idx, "token_count_null"

    first_response_i = None
    for idx in range(begin, end):
        if is_model_response_item(events[idx]):
            first_response_i = idx
            break

    if first_response_i is None:
        return begin, "fallback_segment_begin"

    previous_i = first_response_i - 1
    if previous_i >= begin and is_tool_output(events[previous_i]):
        return previous_i, "after_tool_output"

    return first_response_i, "first_response_item"


def compute_requests(events: Sequence[dict], file_name: str) -> list[ApiRequest]:
    requests: list[ApiRequest] = []
    previous_end_i = -1

    for end_i, event in enumerate(events):
        if not is_token_count_end(event):
            continue

        start_i, start_source = choose_start(events, previous_end_i + 1, end_i)
        first_response_ts = None
        response_items = 0
        function_calls = 0

        for idx in range(start_i, end_i):
            item = events[idx]
            if not is_model_response_item(item):
                continue
            response_items += 1
            if first_response_ts is None:
                first_response_ts = parse_ts(item["timestamp"])
            if payload_type(item) in FUNCTION_CALL_TYPES:
                function_calls += 1

        requests.append(
            ApiRequest(
                file=file_name,
                number=len(requests) + 1,
                start_ts=parse_ts(events[start_i]["timestamp"]),
                end_ts=parse_ts(event["timestamp"]),
                start_source=start_source,
                first_response_ts=first_response_ts,
                response_items=response_items,
                function_calls=function_calls,
                usage=usage_from_end(event),
            )
        )
        previous_end_i = end_i

    return requests


def discover_files(paths: Sequence[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(path.rglob("agent_trace.jsonl")))
        elif path.is_file():
            files.append(path)
        else:
            raise SystemExit(f"{raw}: path does not exist")
    return sorted(set(files))


def load_requests(paths: Sequence[str]) -> list[ApiRequest]:
    all_requests: list[ApiRequest] = []

    if paths == ["-"]:
        events = read_jsonl(sys.stdin, "<stdin>")
        return compute_requests(events, "<stdin>")

    for path in discover_files(paths):
        with path.open("r", encoding="utf-8") as handle:
            events = read_jsonl(handle, str(path))
        all_requests.extend(compute_requests(events, str(path)))

    return all_requests


def request_to_dict(request: ApiRequest) -> dict:
    tt_first = request.time_to_first_response_item_ms
    return {
        "file": request.file,
        "request": request.number,
        "start": request.start_ts.isoformat(),
        "end": request.end_ts.isoformat(),
        "api_request.duration_ms": round(request.duration_ms, 3),
        "start_source": request.start_source,
        "inferred_start": request.inferred_start,
        "time_to_first_response_item_ms": (
            None if tt_first is None else round(tt_first, 3)
        ),
        "response_items": request.response_items,
        "function_calls": request.function_calls,
        "usage": request.usage,
    }


def print_jsonl(requests: Sequence[ApiRequest], output: TextIO) -> None:
    for request in requests:
        print(json.dumps(request_to_dict(request), separators=(",", ":")), file=output)


def trace_to_dict(events: Sequence[dict], file_name: str) -> dict:
    if not events:
        return {
            "file": file_name,
            "first_event": None,
            "last_event": None,
            "wall_clock.duration_ms": 0.0,
            "events": 0,
        }

    first_ts = parse_ts(events[0]["timestamp"])
    last_ts = parse_ts(events[-1]["timestamp"])
    return {
        "file": file_name,
        "first_event": first_ts.isoformat(),
        "last_event": last_ts.isoformat(),
        "wall_clock.duration_ms": round(
            (last_ts - first_ts).total_seconds() * 1000.0, 3
        ),
        "events": len(events),
    }


def print_trace_jsonl(paths: Sequence[str], output: TextIO) -> None:
    if paths == ["-"]:
        events = read_jsonl(sys.stdin, "<stdin>")
        print(
            json.dumps(trace_to_dict(events, "<stdin>"), separators=(",", ":")),
            file=output,
        )
        return

    for path in discover_files(paths):
        with path.open("r", encoding="utf-8") as handle:
            events = read_jsonl(handle, str(path))
        print(
            json.dumps(trace_to_dict(events, str(path)), separators=(",", ":")),
            file=output,
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute api_request.duration_ms from agent_trace.jsonl files."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["."],
        help="Trace files or directories. Use '-' to read one JSONL trace from stdin.",
    )
    parser.add_argument(
        "--mode",
        choices=("jsonl", "trace-jsonl"),
        default="jsonl",
        help="Output mode: request rows or trace-level wall-clock rows.",
    )
    args = parser.parse_args(argv)

    if args.mode == "trace-jsonl":
        print_trace_jsonl(args.paths, sys.stdout)
        return 0

    requests = load_requests(args.paths)
    print_jsonl(requests, sys.stdout)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        raise SystemExit(0)
