"""Minimal wrapper around the Codex CLI."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import traceback
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CODEX_SESSION_ID_RE = re.compile(r"session id:\s*([0-9a-f-]+)", re.IGNORECASE)


@dataclass(frozen=True)
class CodexRunnerConfig:
    work_dir: Path | str
    prompt: str
    codex_bin: Path | str | None = None
    profile: str | None = None
    model: str | None = None
    model_provider: str | None = None
    reasoning_effort: Any | None = None
    timeout_seconds: float | None = None
    env: dict[str, str] | None = None
    config_overrides: Sequence[str] = field(default_factory=tuple)
    bypass_approvals_and_sandbox: bool = False


@dataclass(frozen=True)
class CodexRunResult:
    ok: bool
    status: str
    thread_id: str | None = None
    session_id: str | None = None
    turn_id: str | None = None
    final_response: str | None = None
    trace_path: str | None = None
    error: str | None = None
    started_at_utc: str | None = None
    finished_at_utc: str | None = None


class CodexRunner:
    def execute(self, config: CodexRunnerConfig) -> CodexRunResult:
        """Run one Codex prompt through `codex exec`.

        The raw persisted Codex session trace path is returned when discoverable.
        """
        started_at_utc = _utc_now()
        validation_error = _validate_config(config)
        if validation_error is not None:
            return _result(
                ok=False,
                status="failed",
                started_at_utc=started_at_utc,
                error=validation_error,
            )

        work_dir = Path(config.work_dir).expanduser().resolve()
        env = _build_env(config.env)
        with tempfile.NamedTemporaryFile(
            prefix="lumen-codex-last-message-",
            suffix=".txt",
            delete=False,
        ) as last_message_file:
            last_message_path = Path(last_message_file.name)

        try:
            completed = subprocess.run(
                _build_command(config, work_dir, last_message_path),
                input=config.prompt.strip() + "\n",
                text=True,
                capture_output=True,
                cwd=work_dir,
                env=env,
                timeout=config.timeout_seconds,
                check=False,
            )
            session_id = find_codex_session_id(completed.stderr)
            trace_path = find_trace_path(session_id, env)
            final_response = _read_final_response(last_message_path, completed.stdout)
            if completed.returncode == 0:
                return _result(
                    ok=True,
                    status="completed",
                    started_at_utc=started_at_utc,
                    session_id=session_id,
                    trace_path=trace_path,
                    final_response=final_response,
                )
            return _result(
                ok=False,
                status="failed",
                started_at_utc=started_at_utc,
                session_id=session_id,
                trace_path=trace_path,
                final_response=final_response,
                error=completed.stderr.strip()
                or f"Codex exited with status {completed.returncode}.",
            )
        except subprocess.TimeoutExpired as exc:
            stderr = _coerce_output(exc.stderr)
            stdout = _coerce_output(exc.stdout)
            session_id = find_codex_session_id(stderr)
            trace_path = find_trace_path(session_id, env)
            error = _join_error_parts(
                f"Timed out after {config.timeout_seconds}s.",
                stderr.strip(),
            )
            return _result(
                ok=False,
                status="timed_out",
                started_at_utc=started_at_utc,
                session_id=session_id,
                trace_path=trace_path,
                final_response=_read_final_response(last_message_path, stdout),
                error=error,
            )
        except Exception:
            return _result(
                ok=False,
                status="failed",
                started_at_utc=started_at_utc,
                error=traceback.format_exc(),
            )
        finally:
            try:
                last_message_path.unlink()
            except OSError:
                pass


def _validate_config(config: CodexRunnerConfig) -> str | None:
    work_dir = Path(config.work_dir).expanduser()
    if not work_dir.is_dir():
        return f"work_dir does not exist or is not a directory: {work_dir}"
    if not config.prompt.strip():
        return "prompt must not be empty"
    return None


def _build_command(
    config: CodexRunnerConfig,
    work_dir: Path,
    last_message_path: Path,
) -> list[str]:
    command = [
        _normalize_codex_bin(config.codex_bin),
        "exec",
        "-C",
        str(work_dir),
        "--color",
        "never",
        "--output-last-message",
        str(last_message_path),
    ]
    if config.profile is not None:
        command.extend(["--profile", config.profile])
    if config.model is not None:
        command.extend(["-m", config.model])
    if config.model_provider is not None:
        command.extend(
            ["--config", f"model_provider={json.dumps(config.model_provider)}"]
        )
    if config.reasoning_effort is not None:
        command.extend(
            [
                "--config",
                f"model_reasoning_effort={json.dumps(str(config.reasoning_effort))}",
            ]
        )
    for config_override in config.config_overrides:
        command.extend(["--config", config_override])
    if config.bypass_approvals_and_sandbox:
        command.append("--dangerously-bypass-approvals-and-sandbox")
    command.append("-")
    return command


def _normalize_codex_bin(codex_bin: Path | str | None) -> str:
    if codex_bin is None:
        return "codex"
    return str(Path(codex_bin).expanduser())


def _build_env(config_env: dict[str, str] | None) -> dict[str, str]:
    env = os.environ.copy()
    if config_env:
        env.update(config_env)
    return env


def find_codex_session_id(stderr_text: str) -> str | None:
    match = CODEX_SESSION_ID_RE.search(stderr_text or "")
    return match.group(1) if match else None


def find_trace_path(
    session_id: str | None,
    env: dict[str, str] | None = None,
) -> str | None:
    if not session_id:
        return None
    sessions_root = _codex_home(env) / "sessions"
    if not sessions_root.is_dir():
        return None
    matches = sorted(
        sessions_root.rglob(f"*{session_id}.jsonl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return str(matches[0]) if matches else None


def _codex_home(env: dict[str, str] | None = None) -> Path:
    codex_home = None if env is None else env.get("CODEX_HOME")
    codex_home = codex_home or os.environ.get("CODEX_HOME") or "~/.codex"
    return Path(codex_home).expanduser()


def _read_final_response(last_message_path: Path, stdout: str) -> str | None:
    try:
        text = last_message_path.read_text(encoding="utf-8").strip()
    except OSError:
        text = ""
    if text:
        return text
    stdout = stdout.strip()
    return stdout or None


def _coerce_output(output: str | bytes | None) -> str:
    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return output


def _join_error_parts(*parts: str | None) -> str | None:
    filtered = [part for part in parts if part]
    if not filtered:
        return None
    return "\n".join(filtered)


def _result(
    *,
    ok: bool,
    status: str,
    started_at_utc: str,
    session_id: str | None = None,
    trace_path: str | None = None,
    final_response: str | None = None,
    error: str | None = None,
) -> CodexRunResult:
    return CodexRunResult(
        ok=ok,
        status=status,
        thread_id=None,
        session_id=session_id,
        turn_id=None,
        final_response=final_response,
        trace_path=trace_path,
        error=error,
        started_at_utc=started_at_utc,
        finished_at_utc=_utc_now(),
    )


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )
