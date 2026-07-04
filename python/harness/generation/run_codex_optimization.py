#!/usr/bin/env python3
"""Generic Codex interaction API and thin CLI wrapper."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
CODEX_SESSION_ID_RE = re.compile(r"session id:\s*([0-9a-f-]+)", re.IGNORECASE)


@dataclass(frozen=True)
class CodexOptimizationConfig:
    work_dir: Path
    prompt: str
    model: str = ""
    effort: str = ""
    timeout_seconds: int = 7200
    agent_args: list[str] = field(default_factory=list)
    bypass_approvals_and_sandbox: bool = True
    save_trace_to: Path | None = None
    cwd: Path | None = None
    extra_dirs: list[Path] = field(default_factory=list)


@dataclass(frozen=True)
class CodexOptimizationResult:
    returncode: int
    session_id: str | None
    stdout: str
    stderr: str
    trace_path: str | None
    started_at_utc: str
    finished_at_utc: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class CodexOptimizationClient:
    def __init__(self, config: CodexOptimizationConfig) -> None:
        self.config = config

    def run(self) -> CodexOptimizationResult:
        started_at = utc_now()
        try:
            completed = subprocess.run(
                self._build_command(),
                input=self._normalized_prompt(),
                text=True,
                capture_output=True,
                cwd=self._cwd(),
                timeout=self.config.timeout_seconds,
                env=os.environ.copy(),
            )
            session_id = find_codex_session_id(completed.stderr)
            trace_path = save_trace_if_requested(session_id, self.config.save_trace_to)
            return CodexOptimizationResult(
                returncode=completed.returncode,
                session_id=session_id,
                stdout=completed.stdout,
                stderr=completed.stderr,
                trace_path=None if trace_path is None else str(trace_path),
                started_at_utc=started_at,
                finished_at_utc=utc_now(),
            )
        except subprocess.TimeoutExpired as exc:
            return CodexOptimizationResult(
                returncode=-1,
                session_id=find_codex_session_id(str(exc.stderr or "")),
                stdout=str(exc.stdout or ""),
                stderr=str(exc.stderr or f"Timed out after {self.config.timeout_seconds}s"),
                trace_path=None,
                started_at_utc=started_at,
                finished_at_utc=utc_now(),
            )
        except Exception:
            return CodexOptimizationResult(
                returncode=-1,
                session_id=None,
                stdout="",
                stderr=traceback.format_exc(),
                trace_path=None,
                started_at_utc=started_at,
                finished_at_utc=utc_now(),
            )


    def _cwd(self) -> Path:
        return self.config.work_dir if self.config.cwd is None else self.config.cwd

    def _normalized_prompt(self) -> str:
        prompt = self.config.prompt.strip()
        if not prompt:
            raise ValueError("prompt must not be empty")
        return prompt + "\n"

    def _build_command(self) -> list[str]:
        command = [
            "codex",
            "exec",
            "-C",
            str(self._cwd()),
            "--color",
            "never",
            "--add-dir",
            str(self.config.work_dir.resolve()),
        ]
        for path in self.config.extra_dirs:
            command.extend(["--add-dir", str(path.resolve())])
        if self.config.bypass_approvals_and_sandbox:
            command.append("--dangerously-bypass-approvals-and-sandbox")
        else:
            command.extend(["--sandbox", "workspace-write"])
        if self.config.model:
            command.extend(["-m", self.config.model])
        if self.config.effort:
            command.extend(["-c", f"model_reasoning_effort={json.dumps(self.config.effort)}"])
        command.extend(self.config.agent_args)
        command.append("-")
        return command


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def dump_json(path: Path, payload: dict) -> None:
    write_text(path, json.dumps(payload, indent=2) + "\n")


def find_codex_session_id(stderr_text: str) -> str | None:
    match = CODEX_SESSION_ID_RE.search(stderr_text or "")
    return match.group(1) if match else None


def find_trace_path(session_id: str | None) -> Path | None:
    if not session_id:
        return None
    sessions_root = Path.home() / ".codex" / "sessions"
    if not sessions_root.is_dir():
        return None
    matches = sorted(
        sessions_root.rglob(f"*{session_id}.jsonl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def save_trace_if_requested(session_id: str | None, save_trace_to: Path | None) -> Path | None:
    if save_trace_to is None:
        return None
    trace_path = find_trace_path(session_id)
    if trace_path is None:
        return None
    save_trace_to = save_trace_to.expanduser().resolve()
    save_trace_to.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(trace_path, save_trace_to)
    return save_trace_to


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single Codex optimization task.")
    parser.add_argument("--work-dir", type=Path, required=True, help="Directory Codex can edit.")
    parser.add_argument("--prompt-file", type=Path, default=None, help="Path to a prompt file.")
    parser.add_argument("--prompt-text", default="", help="Inline prompt text.")
    parser.add_argument("--model", default="", help="Optional Codex model override.")
    parser.add_argument("--effort", default="", help="Optional Codex reasoning effort override.")
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--agent-arg", action="append", default=[], help="Extra argument to pass to `codex exec`.")
    parser.add_argument("--save-trace-to", type=Path, default=None, help="Optional file path for a copied agent trace.")
    parser.add_argument("--json-output", type=Path, default=None, help="Optional path to save the result as JSON.")
    parser.add_argument("--add-dir", type=Path, action="append", default=[], help="Extra directory to expose to Codex.")
    parser.add_argument(
        "--codex-dangerously-bypass-approvals-and-sandbox",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def build_prompt_from_args(args: argparse.Namespace) -> str:
    parts: list[str] = []
    if args.prompt_file is not None:
        parts.append(read_text(args.prompt_file))
    if args.prompt_text:
        parts.append(args.prompt_text)
    prompt = "\n\n".join(part.strip() for part in parts if part.strip()).strip()
    if not prompt:
        raise SystemExit("Provide --prompt-file and/or --prompt-text.")
    return prompt


def build_config_from_args(args: argparse.Namespace) -> CodexOptimizationConfig:
    work_dir = args.work_dir.expanduser().resolve()
    if not work_dir.is_dir():
        raise FileNotFoundError(f"Work directory does not exist: {work_dir}")
    return CodexOptimizationConfig(
        work_dir=work_dir,
        prompt=build_prompt_from_args(args),
        model=args.model,
        effort=args.effort,
        timeout_seconds=args.timeout_seconds,
        agent_args=list(args.agent_arg),
        bypass_approvals_and_sandbox=args.codex_dangerously_bypass_approvals_and_sandbox,
        save_trace_to=args.save_trace_to,
        cwd=None,
        extra_dirs=[path.expanduser().resolve() for path in args.add_dir],
    )


def main() -> int:
    args = parse_args()
    client = CodexOptimizationClient(build_config_from_args(args))
    result = client.run()
    payload = asdict(result)
    if args.json_output is not None:
        dump_json(args.json_output.expanduser().resolve(), payload)
    else:
        print(json.dumps(payload, indent=2))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
