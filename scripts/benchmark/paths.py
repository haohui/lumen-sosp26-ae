#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]
