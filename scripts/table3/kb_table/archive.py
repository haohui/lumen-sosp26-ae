"""Summarize KernelBench Table 3 traces while streaming tar archives."""

from __future__ import annotations

import io
import json
import tarfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from kb_table.common import (
    as_float,
    geomean,
    mean,
    parse_problem_id,
    parse_round_index,
)
from kb_table.models import GenerationStats, OptimizationStats, RoundRecord
from kb_table.reward_hacking import reward_hacking_reasons_from_source
from kb_table.runs import expected_problem_ids_from_config
from kb_table.trace_stats import trace_stats_from_lines

INTERESTING_FILENAMES = {
    "generation_config.json",
    "meta.json",
    "output_model_new.py",
    "trace.jsonl",
}


@dataclass
class RoundState:
    problem_id: int
    round_name: str
    meta: dict[str, Any] | None = None
    reward_hacking: tuple[str, ...] | None = None

    @property
    def round_index(self) -> int:
        return parse_round_index(self.round_name) or 0

    def to_record(self, run_name: str) -> RoundRecord | None:
        if self.meta is None:
            return None
        return RoundRecord(
            problem_id=self.problem_id,
            round_index=self.round_index,
            round_dir=Path(run_name) / f"p{self.problem_id:02d}" / self.round_name,
            correct=bool(self.meta.get("correctness")),
            speedup=as_float(self.meta.get("speedup")),
            reward_hacking=(
                self.reward_hacking
                if self.reward_hacking is not None
                else ("missing output_model_new.py",)
            ),
        )


@dataclass
class TraceState:
    files_read: set[str]
    token_usage: int | None


@dataclass
class ProblemTraceState:
    traces: dict[str, TraceState] = field(default_factory=dict)

    def files_read_count(self) -> int | None:
        if not self.traces:
            return None
        files: set[str] = set()
        for trace in self.traces.values():
            files.update(trace.files_read)
        return len(files)

    def token_usage_sum(self) -> int | None:
        values = [
            trace.token_usage
            for trace in self.traces.values()
            if trace.token_usage is not None
        ]
        return sum(values) if values else None


@dataclass
class ArchiveRun:
    name: str
    config: dict[str, Any] = field(default_factory=dict)
    rounds: dict[tuple[int, str], RoundState] = field(default_factory=dict)
    problem_traces: dict[int, ProblemTraceState] = field(default_factory=dict)

    def add_config(self, data: dict[str, Any]) -> None:
        self.config = data

    def add_meta(self, problem_id: int, round_name: str, data: dict[str, Any]) -> None:
        self.round_state(problem_id, round_name).meta = data

    def add_output(self, problem_id: int, round_name: str, source: str) -> None:
        self.round_state(problem_id, round_name).reward_hacking = tuple(
            reward_hacking_reasons_from_source(
                source,
                filename=f"{self.name}/p{problem_id:02d}/{round_name}/output_model_new.py",
            )
        )

    def add_trace(
        self,
        problem_id: int,
        relative_path: str,
        files_read: set[str],
        token_usage: int | None,
    ) -> None:
        problem = self.problem_traces.setdefault(problem_id, ProblemTraceState())
        problem.traces[relative_path] = TraceState(
            files_read=files_read,
            token_usage=token_usage,
        )

    def round_state(self, problem_id: int, round_name: str) -> RoundState:
        key = (problem_id, round_name)
        if key not in self.rounds:
            self.rounds[key] = RoundState(
                problem_id=problem_id,
                round_name=round_name,
            )
        return self.rounds[key]

    def round_records(self) -> dict[int, list[RoundRecord]]:
        records: dict[int, list[RoundRecord]] = {}
        states = sorted(
            self.rounds.values(),
            key=lambda state: record_sort_key(state.problem_id, state.round_name),
        )
        for state in states:
            record = state.to_record(self.name)
            if record is not None:
                records.setdefault(record.problem_id, []).append(record)
        return records

    def files_read_for_problem(self, problem_id: int) -> int | None:
        problem = self.problem_traces.get(problem_id)
        return problem.files_read_count() if problem is not None else None

    def token_usage_for_problem(self, problem_id: int) -> int | None:
        problem = self.problem_traces.get(problem_id)
        return problem.token_usage_sum() if problem is not None else None


class ArchiveRuns:
    def __init__(self, runs: dict[str, ArchiveRun]) -> None:
        self.runs = runs

    @classmethod
    def from_archives(
        cls,
        archives: Iterable[Path],
        run_names: Iterable[str],
    ) -> ArchiveRuns:
        runs = {name: ArchiveRun(name=name) for name in set(run_names)}
        for archive in archives:
            read_archive(archive, runs)
        return cls(runs)

    def run(self, name: str) -> ArchiveRun:
        return self.runs.get(name, ArchiveRun(name=name))


def read_archive(archive: Path, runs: dict[str, ArchiveRun]) -> None:
    with tarfile.open(archive, mode="r|xz") as tar:
        for member in tar:
            if not member.isfile() or not is_interesting_member(member.name):
                continue
            run_name, relative_parts = split_run_path(member.name, set(runs))
            if run_name is None or relative_parts is None:
                continue
            file_obj = tar.extractfile(member)
            if file_obj is None:
                continue
            with file_obj:
                process_member(runs[run_name], relative_parts, file_obj)


def process_member(
    run: ArchiveRun,
    relative_parts: tuple[str, ...],
    file_obj: io.BufferedIOBase,
) -> None:
    filename = relative_parts[-1]
    if relative_parts == ("generation_config.json",):
        run.add_config(read_json_member(file_obj))
        return
    if len(relative_parts) != 3:
        return

    problem_id = parse_problem_id(relative_parts[0])
    round_name = relative_parts[1]
    if problem_id is None or not is_round_dir_name(round_name):
        return
    if filename == "meta.json":
        run.add_meta(problem_id, round_name, read_json_member(file_obj))
    elif filename == "output_model_new.py":
        run.add_output(problem_id, round_name, read_text_member(file_obj))
    elif filename == "trace.jsonl":
        relative_path = str(PurePosixPath(*relative_parts))
        default_cwd = str(Path(run.name) / Path(relative_path).parent)
        files_read, token_usage = trace_stats_from_lines(
            text_lines_member(file_obj),
            default_cwd=default_cwd,
        )
        run.add_trace(problem_id, relative_path, files_read, token_usage)


def read_json_member(file_obj: io.BufferedIOBase) -> dict[str, Any]:
    try:
        data = json.loads(read_text_member(file_obj))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def read_text_member(file_obj: io.BufferedIOBase) -> str:
    return file_obj.read().decode("utf-8", errors="replace")


def text_lines_member(
    file_obj: io.BufferedIOBase,
    *,
    chunk_size: int = 1024 * 1024,
) -> Iterable[str]:
    pending = ""
    while chunk := file_obj.read(chunk_size):
        text = pending + chunk.decode("utf-8", errors="replace")
        lines = text.splitlines(keepends=True)
        if lines and not lines[-1].endswith(("\n", "\r")):
            pending = lines.pop()
        else:
            pending = ""
        yield from lines
    if pending:
        yield pending


def is_interesting_member(member_name: str) -> bool:
    return PurePosixPath(member_name).name in INTERESTING_FILENAMES


def split_run_path(
    member_name: str,
    run_names: set[str],
) -> tuple[str | None, tuple[str, ...] | None]:
    parts = PurePosixPath(member_name).parts
    for index, part in enumerate(parts):
        if part not in run_names:
            continue
        relative_parts = parts[index + 1 :]
        if is_interesting_relative_path(relative_parts):
            return part, relative_parts
    return None, None


def is_interesting_relative_path(parts: tuple[str, ...]) -> bool:
    if parts == ("generation_config.json",):
        return True
    if len(parts) != 3:
        return False
    return (
        parse_problem_id(parts[0]) is not None
        and is_round_dir_name(parts[1])
        and parts[2] in INTERESTING_FILENAMES
    )


def summarize_generation_archive(
    run: ArchiveRun,
    *,
    strict_denominator: bool,
) -> GenerationStats:
    records = run.round_records()
    problem_ids = expected_problem_ids_from_config(
        run.config,
        records.keys(),
        strict_denominator,
    )
    pass1 = 0
    pass3 = 0
    speedups: list[float] = []
    files_read_values: list[int] = []

    for pid in problem_ids:
        rounds = sorted(records.get(pid, []), key=lambda item: item.round_index)
        first_valid = next((item for item in rounds if item.valid), None)
        if rounds and rounds[0].valid:
            pass1 += 1
        if any(item.valid and item.round_index < 3 for item in rounds):
            pass3 += 1
        if first_valid and first_valid.speedup is not None:
            speedups.append(first_valid.speedup)

        files_read = run.files_read_for_problem(pid)
        if files_read is not None:
            files_read_values.append(files_read)

    return GenerationStats(
        denominator=len(problem_ids),
        valid_count=len(speedups),
        geom=geomean(speedups),
        min_speedup=min(speedups) if speedups else None,
        max_speedup=max(speedups) if speedups else None,
        gt1_count=sum(value > 1.0 for value in speedups),
        pass1=pass1,
        pass3=pass3,
        avg_files_read=mean(files_read_values),
    )


def summarize_optimization_archive(
    run: ArchiveRun,
    *,
    strict_denominator: bool,
) -> OptimizationStats:
    records = run.round_records()
    problem_ids = expected_problem_ids_from_config(
        run.config,
        records.keys(),
        strict_denominator,
    )
    pass_all_rounds = 0
    token_values: list[int] = []

    for pid in problem_ids:
        rounds = sorted(records.get(pid, []), key=lambda item: item.round_index)
        if rounds and all(item.clean_correct for item in rounds):
            pass_all_rounds += 1
        token_usage = run.token_usage_for_problem(pid)
        if token_usage is not None:
            token_values.append(token_usage)

    return OptimizationStats(
        denominator=len(problem_ids),
        pass_all_rounds=pass_all_rounds,
        avg_token_usage=mean(token_values),
    )


def is_round_dir_name(name: str) -> bool:
    return len(name) > len("round") and name.startswith("round") and name[5].isdigit()


def record_sort_key(problem_id: int, round_name: str) -> tuple[int, int, str]:
    round_index = parse_round_index(round_name)
    return (
        problem_id,
        round_index if round_index is not None else 10**9,
        round_name,
    )
