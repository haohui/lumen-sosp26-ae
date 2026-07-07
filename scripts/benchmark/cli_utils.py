#!/usr/bin/env python3
from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any


def select_backend(
    backends: Mapping[str, Callable[..., Any]],
    name: str,
) -> Callable[..., Any]:
    try:
        return backends[name]
    except KeyError as e:
        expected = ", ".join(sorted(backends))
        raise RuntimeError(
            f"unknown backend {name!r}; expected one of: {expected}"
        ) from e


def emit_jsonl(record: Mapping[str, Any]) -> None:
    print(json.dumps(record, separators=(",", ":")), flush=True)
