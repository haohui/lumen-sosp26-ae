#!/usr/bin/env python3
"""Compatibility entry point for KernelBench generation."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from lumen_artifact.utils import ensure_source_root_on_path, sibling_path
except ModuleNotFoundError:
    source_root = Path(__file__).resolve().parent.parent
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from lumen_artifact.utils import ensure_source_root_on_path, sibling_path

ensure_source_root_on_path(__file__)

from lumen.harness.datasets.kernelbench.generation import main


if __name__ == "__main__":
    raise SystemExit(
        main(sys.argv[1:], default_config_path=sibling_path(__file__, "runner.yaml"))
    )
