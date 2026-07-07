from __future__ import annotations

import asyncio
import os
import selectors
import shutil
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Awaitable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from lumen.harness.backend.codex import CodexRunner, CodexRunnerConfig


RunMode = Literal["sync", "async"]


@dataclass
class AgentResult:

    work_dir: Path
    returncode: int
    error: str | None = None
    trace_dir: Path | None = None


class AbstractSession(ABC):
    def __init__(
        self,
        *,
        work_dir: str | Path,
        prompt: str = "",
        timeout_seconds: int | None = None,
        save_trace: bool = False,
        trace_path: str | Path | None = None,
        run_mode: RunMode = "sync",
    ):
        if run_mode not in ("sync", "async"):
            raise ValueError(f"Unsupported session run mode: {run_mode}")

        self.work_dir = Path(work_dir).expanduser().resolve()
        if not self.work_dir.is_dir():
            raise FileNotFoundError(f"work_dir does not exist: {self.work_dir}")

        self.prompt = prompt
        self.timeout_seconds = timeout_seconds
        self.save_trace = save_trace
        self.trace_path = None if trace_path is None else Path(trace_path)
        self.run_mode = run_mode
        self._llm_env: dict[str, str] = {}

    def __enter__(self) -> "AbstractSession":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def run(self, prompt: str | None = None) -> AgentResult | Awaitable[AgentResult]:
        resolved_prompt = self.prompt if prompt is None else prompt
        if not resolved_prompt:
            raise ValueError("A non-empty prompt is required.")
        if self.run_mode == "async":
            return self._run_async(resolved_prompt)
        return self._run(resolved_prompt)

    def load_llm_config(
        self,
        *,
        endpoint: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ) -> "AbstractSession":
        return self

    @abstractmethod
    def _run(self, prompt: str) -> AgentResult:
        """Run the concrete backend once."""

    async def _run_async(self, prompt: str) -> AgentResult:
        return await asyncio.to_thread(self._run, prompt)

    def _trace_path(self) -> Path:
        return self._work_path(self.trace_path or "trace.jsonl")

    def _work_path(self, path: str | Path) -> Path:
        resolved = Path(path).expanduser()
        if not resolved.is_absolute():
            resolved = self.work_dir / resolved
        return resolved


class ClaudeSession(AbstractSession):
    backend = "claude"

    def load_llm_config(
        self,
        *,
        endpoint: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ) -> "ClaudeSession":
        if endpoint is not None:
            self._llm_env["ANTHROPIC_BASE_URL"] = endpoint
        if api_key is not None:
            self._llm_env["ANTHROPIC_AUTH_TOKEN"] = api_key
        if model is not None:
            self._llm_env["ANTHROPIC_MODEL"] = model
            self._llm_env["ANTHROPIC_DEFAULT_MODEL"] = model
            self._llm_env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = model
            self._llm_env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = model
            self._llm_env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = model
        return self

    def _run(self, prompt: str) -> AgentResult:
        try:
            command = self._build_command()
            if self.save_trace:
                returncode, trace_path = self._run_stream_json(command, prompt)
            else:
                completed = subprocess.run(
                    command,
                    input=prompt,
                    text=True,
                    capture_output=True,
                    cwd=self.work_dir,
                    env=self._env(),
                    check=False,
                )
                returncode = completed.returncode
                trace_path = None

            return AgentResult(
                work_dir=self.work_dir,
                returncode=returncode,
                error=None
                if returncode == 0
                else f"claude exited with return code {returncode}",
                trace_dir=None if trace_path is None else trace_path.parent,
            )
        except Exception as exc:
            return AgentResult(
                work_dir=self.work_dir,
                returncode=-1,
                error=f"claude invocation error: {exc}",
            )

    def _build_command(self) -> list[str]:
        output_format = "stream-json" if self.save_trace else "text"
        command = [
            "claude",
            "--print",
            "--output-format",
            output_format,
            "--dangerously-skip-permissions",
            "--permission-mode",
            "bypassPermissions",
        ]
        if self.save_trace:
            command.append("--verbose")
        return command

    def _env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["IS_SANDBOX"] = "1"
        env.setdefault("CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING", "1")
        env.setdefault("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")
        env.setdefault("CLAUDE_CODE_MAX_CONTEXT_TOKENS", "1000000")
        if self.timeout_seconds is not None:
            env["API_TIMEOUT_MS"] = str(int(self.timeout_seconds * 1000))
        env.update(self._llm_env)
        return env

    def _run_stream_json(self, command: list[str], prompt: str) -> tuple[int, Path]:
        trace_path = self._trace_path()
        trace_path.parent.mkdir(parents=True, exist_ok=True)

        proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=self.work_dir,
            env=self._env(),
        )
        assert proc.stdin is not None
        assert proc.stdout is not None
        assert proc.stderr is not None
        proc.stdin.write(prompt)
        proc.stdin.close()

        selector = selectors.DefaultSelector()
        try:
            selector.register(proc.stdout, selectors.EVENT_READ)
            selector.register(proc.stderr, selectors.EVENT_READ)
            open_streams = 2
            with trace_path.open("w", encoding="utf-8") as trace_file:
                while open_streams > 0:
                    events = selector.select()
                    for key, _ in events:
                        line = key.fileobj.readline()
                        if not line:
                            selector.unregister(key.fileobj)
                            open_streams -= 1
                            continue
                        if key.fileobj is proc.stdout:
                            trace_file.write(line)
                            trace_file.flush()
        finally:
            selector.close()

        return proc.wait(), trace_path


class CodexSession(AbstractSession):
    backend = "codex"

    def load_llm_config(
        self,
        *,
        endpoint: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ) -> "CodexSession":
        if endpoint is not None:
            self._llm_env["OPENAI_BASE_URL"] = endpoint
        if api_key is not None:
            self._llm_env["OPENAI_API_KEY"] = api_key
        if model is not None:
            self._llm_env["LUMEN_CODEX_MODEL"] = model
        return self

    def _run(self, prompt: str) -> AgentResult:
        result = CodexRunner().execute(
            CodexRunnerConfig(
                work_dir=self.work_dir,
                prompt=prompt,
                codex_bin=os.environ.get("LUMEN_CODEX_BIN"),
                profile=os.environ.get("LUMEN_CODEX_PROFILE"),
                model=self._llm_env.get("LUMEN_CODEX_MODEL"),
                model_provider=os.environ.get("LUMEN_CODEX_MODEL_PROVIDER"),
                reasoning_effort=os.environ.get("LUMEN_CODEX_REASONING_EFFORT"),
                timeout_seconds=self.timeout_seconds,
                env=self._env(),
                config_overrides=tuple(_split_codex_config_overrides()),
                bypass_approvals_and_sandbox=True,
            )
        )
        trace_dir = self._save_trace(result.trace_path) if self.save_trace else None
        return AgentResult(
            work_dir=self.work_dir,
            returncode=0 if result.ok else 1,
            error=None if result.ok else result.error or result.status,
            trace_dir=trace_dir,
        )

    def _env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["IS_SANDBOX"] = "1"
        env.update(self._llm_env)
        return env

    def _save_trace(self, source: str | None) -> Path | None:
        if source is None:
            return None
        destination = self._trace_path()
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(source, destination)
        except OSError:
            return Path(source).expanduser().parent
        return destination.parent


def _split_codex_config_overrides() -> list[str]:
    value = os.environ.get("LUMEN_CODEX_CONFIG", "")
    return [item.strip() for item in value.splitlines() if item.strip()]


class Session:
    _registry: dict[str, type[AbstractSession]] = {}

    def __new__(
        cls,
        *,
        backend: str,
        work_dir: str | Path,
        prompt: str = "",
        timeout_seconds: int | None = None,
        save_trace: bool = False,
        trace_path: str | Path | None = None,
        run_mode: RunMode = "sync",
    ) -> AbstractSession:
        try:
            session_cls = cls._registry[backend]
        except KeyError as exc:
            raise ValueError(f"Unsupported session backend: {backend}") from exc
        return session_cls(
            work_dir=work_dir,
            prompt=prompt,
            timeout_seconds=timeout_seconds,
            save_trace=save_trace,
            trace_path=trace_path,
            run_mode=run_mode,
        )

    @classmethod
    def register_backend(
        cls, backend: str, session_cls: type[AbstractSession]
    ) -> None:
        cls._registry[backend] = session_cls


Session.register_backend(ClaudeSession.backend, ClaudeSession)
Session.register_backend(CodexSession.backend, CodexSession)
