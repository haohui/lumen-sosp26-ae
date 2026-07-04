"""CLI for exercising the Codex runner."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from lumen.harness.backend.codex import CodexRunner, CodexRunnerConfig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one prompt with the Codex runner."
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument(
        "--dangerously-bypass-approvals-and-sandbox",
        action="store_true",
        required=True,
        help="Required for this test CLI; maps to Codex CLI full-access execution.",
    )
    parser.add_argument("--codex-bin", type=Path, default=None)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--model-provider", default=None)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--timeout-seconds", type=float, default=None)
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        dest="config_overrides",
        help="Raw Codex config override, e.g. key=value. May be repeated.",
    )
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--trace-jsonl", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = _main(args)
    return 0 if result else 1


def _main(args: argparse.Namespace) -> bool:
    prompt = args.prompt_file.expanduser().read_text(encoding="utf-8")
    config = CodexRunnerConfig(
        work_dir=args.work_dir,
        prompt=prompt,
        codex_bin=args.codex_bin,
        profile=args.profile,
        model=args.model,
        model_provider=args.model_provider,
        reasoning_effort=args.reasoning_effort,
        timeout_seconds=args.timeout_seconds,
        config_overrides=tuple(args.config_overrides),
        bypass_approvals_and_sandbox=args.dangerously_bypass_approvals_and_sandbox,
    )

    result = CodexRunner().execute(config)
    if args.trace_jsonl is not None:
        result = _copy_trace_if_requested(result, args.trace_jsonl)
    payload = asdict(result)
    if args.json_output is None:
        print(json.dumps(payload, indent=2))
    else:
        _write_json(args.json_output.expanduser(), payload)
    return result.ok


def _copy_trace_if_requested(result: Any, trace_jsonl: Path) -> Any:
    if result.trace_path is None:
        return replace(
            result,
            ok=False,
            status="failed",
            error=_join_error_parts(
                result.error,
                "Trace copy requested but Codex session trace path was not found.",
            ),
        )
    destination = trace_jsonl.expanduser().resolve()
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(result.trace_path, destination)
    except OSError as exc:
        return replace(
            result,
            ok=False,
            status="failed",
            error=_join_error_parts(result.error, f"Failed to copy trace: {exc}"),
        )
    return result


def _join_error_parts(*parts: str | None) -> str | None:
    filtered = [part for part in parts if part]
    if not filtered:
        return None
    return "\n".join(filtered)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
