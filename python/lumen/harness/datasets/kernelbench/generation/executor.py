"""Codex execution for one KernelBench generation round."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

from lumen.harness.backend.codex.runner import CodexRunner, CodexRunnerConfig
from lumen.harness.datasets.kernelbench.generation.types import CodexGenerationConfig

SANDBOX_ENV = {"IS_SANDBOX": "1"}


def run_codex(
    round_dir: str | Path,
    prompt: str,
    config: CodexGenerationConfig,
) -> Any:
    timeout_seconds = float(config.timeout_seconds) if config.timeout_seconds else None
    return CodexRunner().execute(
        CodexRunnerConfig(
            work_dir=round_dir,
            prompt=prompt,
            codex_bin=config.codex_bin,
            profile=config.profile,
            model_provider=config.model_provider,
            reasoning_effort=config.reasoning_effort,
            timeout_seconds=timeout_seconds,
            env=SANDBOX_ENV,
            config_overrides=config.config_overrides,
            bypass_approvals_and_sandbox=config.bypass_approvals_and_sandbox,
        )
    )


def write_codex_result(path: str | Path, result: Any) -> None:
    try:
        payload = asdict(result)
    except TypeError:
        payload = {
            "ok": getattr(result, "ok", None),
            "status": getattr(result, "status", None),
            "error": getattr(result, "error", None),
            "trace_path": getattr(result, "trace_path", None),
            "final_response": getattr(result, "final_response", None),
        }
    Path(path).write_text(
        json.dumps(payload, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def copy_codex_trace(result: Any, round_dir: str | Path, *, save_trace: bool) -> None:
    if not save_trace or not getattr(result, "trace_path", None):
        return
    source = Path(result.trace_path).expanduser()
    if not source.is_file():
        return
    destination = Path(round_dir) / "trace.jsonl"
    try:
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
    except OSError:
        pass
