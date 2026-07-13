"""Shared parsing and math helpers."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any


def geomean(values: list[float]) -> float | None:
    positive = [value for value in values if value > 0]
    if not positive:
        return None
    return math.exp(sum(math.log(value) for value in positive) / len(positive))


def mean(values: list[int] | list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_problem_id(name: str) -> int | None:
    match = re.fullmatch(r"p(\d+)", name)
    return int(match.group(1)) if match else None


def parse_round_index(name: str) -> int | None:
    match = re.fullmatch(r"round(\d+)", name)
    return int(match.group(1)) if match else None


def round_sort_key(path: Path) -> int:
    value = parse_round_index(path.name)
    return value if value is not None else 10**9


def round_trace_sort_key(path: Path) -> int:
    return round_sort_key(path.parent)
