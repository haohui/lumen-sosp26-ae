#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


def resolve_repo_root(start: Path | None = None) -> Path:
    path = (start or Path(__file__)).resolve()
    if path.is_file():
        path = path.parent

    for candidate in (path, *path.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / ".git").exists():
            return candidate

    raise RuntimeError(f"cannot resolve repository root from {path}")
