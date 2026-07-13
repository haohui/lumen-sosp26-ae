"""Extract file-read and token statistics from Codex JSONL traces."""

from __future__ import annotations

import json
import os
import re
import shlex
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from kb_table.common import round_trace_sort_key

READ_COMMANDS = {
    "cat",
    "grep",
    "head",
    "less",
    "more",
    "nl",
    "rg",
    "sed",
    "tail",
    "wc",
}
SHELL_SEPARATORS = {"&&", "||", ";", "|"}
PATH_SUFFIXES = (
    ".py",
    ".json",
    ".jsonl",
    ".md",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
)


def files_read_for_problem(problem_dir: Path) -> int | None:
    trace_paths = trace_paths_for_problem(problem_dir)
    if not trace_paths:
        return None
    files: set[str] = set()
    for trace_path in trace_paths:
        files.update(files_read_from_trace(trace_path))
    return len(files)


def token_usage_for_problem(problem_dir: Path) -> int | None:
    values = [
        value
        for path in trace_paths_for_problem(problem_dir)
        if (value := token_usage_from_trace(path)) is not None
    ]
    return sum(values) if values else None


def trace_paths_for_problem(problem_dir: Path) -> list[Path]:
    return sorted(
        problem_dir.glob("round*/trace.jsonl"),
        key=round_trace_sort_key,
    )


def files_read_from_trace(trace_path: Path) -> set[str]:
    cwd = str(trace_path.parent)
    try:
        with trace_path.open(encoding="utf-8") as handle:
            files, _ = trace_stats_from_lines(handle, default_cwd=cwd)
    except OSError:
        return set()
    return files


def files_read_from_trace_text(text: str, *, default_cwd: str) -> set[str]:
    files, _ = trace_stats_from_lines(text.splitlines(), default_cwd=default_cwd)
    return files


def trace_stats_from_lines(
    lines: Iterable[str],
    *,
    default_cwd: str,
) -> tuple[set[str], int | None]:
    cwd = default_cwd
    files: set[str] = set()
    max_total: int | None = None
    for payload in trace_payloads_from_lines(lines):
        if payload.get("type") == "session_meta" and isinstance(
            payload.get("payload"),
            dict,
        ):
            cwd = payload["payload"].get("cwd") or cwd
            continue
        event = payload.get("payload")
        if payload.get("type") != "event_msg":
            pass
        elif isinstance(event, dict) and event.get("type") == "token_count":
            total = (
                event.get("info", {})
                .get("total_token_usage", {})
                .get("total_tokens")
            )
            if isinstance(total, int):
                max_total = total if max_total is None else max(max_total, total)

        if is_tool_call(event):
            args = parse_tool_arguments(event)
            cmd = args.get("cmd") or args.get("command")
            workdir = args.get("workdir") or cwd
            if isinstance(cmd, str):
                files.update(files_read_from_command(cmd, str(workdir)))
    return files, max_total


def token_usage_from_trace(trace_path: Path) -> int | None:
    try:
        with trace_path.open(encoding="utf-8") as handle:
            _, token_usage = trace_stats_from_lines(
                handle,
                default_cwd=str(trace_path.parent),
            )
    except OSError:
        return None
    return token_usage


def token_usage_from_trace_text(text: str) -> int | None:
    _, token_usage = trace_stats_from_lines(text.splitlines(), default_cwd="")
    return token_usage


def trace_payloads(trace_path: Path) -> list[dict[str, Any]]:
    try:
        text = trace_path.read_text(encoding="utf-8")
    except OSError:
        return []
    return trace_payloads_from_text(text)


def trace_payloads_from_text(text: str) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for item in trace_payloads_from_lines(text.splitlines()):
        payloads.append(item)
    return payloads


def trace_payloads_from_lines(lines: Iterable[str]) -> Iterable[dict[str, Any]]:
    for line in lines:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            yield item


def is_tool_call(event: Any) -> bool:
    return (
        isinstance(event, dict)
        and event.get("type") in {"function_call", "custom_tool_call"}
        and event.get("name") in {"exec_command", "shell"}
    )


def parse_tool_arguments(event: dict[str, Any]) -> dict[str, Any]:
    raw = event.get("arguments")
    if raw is None:
        raw = event.get("input")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def files_read_from_command(cmd: str, workdir: str) -> set[str]:
    try:
        tokens = shlex.split(cmd, posix=True)
    except ValueError:
        return set()

    files: set[str] = set()
    idx = 0
    while idx < len(tokens):
        name = Path(tokens[idx]).name
        if name not in READ_COMMANDS:
            idx += 1
            continue
        idx += 1
        while idx < len(tokens) and tokens[idx] not in SHELL_SEPARATORS:
            token = tokens[idx]
            if token in {"<", ">", ">>", "2>", "2>&1"}:
                idx += 2
                continue
            if looks_like_read_path(token):
                files.add(normalize_trace_path(token, workdir))
            idx += 1
    return files


def looks_like_read_path(token: str) -> bool:
    if not token or token.startswith("-"):
        return False
    if token.startswith(("http://", "https://")):
        return False
    if token.isdigit() or re.fullmatch(r"[0-9,]+[a-zA-Z]+", token):
        return False
    if any(char in token for char in "*?[]{}"):
        return False
    return "/" in token or token.endswith(PATH_SUFFIXES)


def normalize_trace_path(token: str, workdir: str) -> str:
    token = token.strip("'\"")
    if os.path.isabs(token):
        return os.path.normpath(token)
    return os.path.normpath(os.path.join(workdir, token))
