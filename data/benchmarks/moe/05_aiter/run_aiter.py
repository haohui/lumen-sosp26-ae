#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    repo_root = Path(__file__).resolve().parents[4]
    bench = repo_root / "python" / "harness" / "bench" / "benchmark_moe_unified_graph.py"
    cmd = [sys.executable, str(bench), "--run-aiter", *argv]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
