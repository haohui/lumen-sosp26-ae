"""Consistent final status output for paper experiment entry points."""

from __future__ import annotations

import os
from pathlib import Path
from types import TracebackType


BLUE = "\033[34m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"


class ExperimentSummary:
    def __init__(self, name: str, paper_result: str) -> None:
        self.name = name
        self.paper_result = paper_result
        self.results: list[str] = []
        self.logs: list[str] = []
        self.failure_reason: str | None = None
        self.skip_reason: str | None = None

    def add_result(self, path: str | Path) -> None:
        self.results.append(str(path))

    def add_log(self, path: str | Path) -> None:
        self.logs.append(str(path))

    def fail(self, reason: str) -> None:
        self.failure_reason = reason

    def skip(self, reason: str) -> None:
        self.skip_reason = reason

    def __enter__(self) -> ExperimentSummary:
        runner_log = os.environ.get("LUMEN_EXPERIMENT_LOG")
        if runner_log:
            self.add_log(runner_log)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> bool:
        if exc is not None and self.failure_reason is None:
            self.failure_reason = exception_reason(exc)
        passed = (
            exc_type is None
            and self.failure_reason is None
            and self.skip_reason is None
        )
        if self.skip_reason is not None and exc_type is None:
            status = "SKIP"
            status_color = YELLOW
        else:
            status = "PASS" if passed else "FAIL"
            status_color = GREEN if passed else RED
        print(f"\n=== {BLUE}FINAL_SUMMARY{RESET} ===", flush=True)
        print(f"{BLUE}Experiment{RESET}: {self.name}", flush=True)
        print(f"Status: {status_color}{status}{RESET}", flush=True)
        print(
            f"Reason: {self.skip_reason or self.failure_reason or 'completed successfully'}",
            flush=True,
        )
        print(f"Paper comparison: {self.paper_result}", flush=True)
        print("Result paths:", flush=True)
        for path in self.results or ["none (status is reported in the log)"]:
            print(f"  - {path}", flush=True)
        print("Log-to-paper mapping:", flush=True)
        for path in self.logs or ["stdout/stderr (redirect to retain a log)"]:
            print(f"  - {path} -> {self.paper_result}", flush=True)
        return False


def exception_reason(exc: BaseException) -> str:
    if isinstance(exc, SystemExit):
        if isinstance(exc.code, int) or exc.code is None:
            return f"exited with status {exc.code}"
        return " ".join(str(exc.code).split())
    text = " ".join(str(exc).split())
    return text or type(exc).__name__
