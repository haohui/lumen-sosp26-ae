"""Read KernelBench Table 3 trace files directly from tar archives."""

from __future__ import annotations

import json
import tarfile
from collections.abc import Iterable
from dataclasses import dataclass
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
from kb_table.trace_stats import (
    files_read_from_trace_text,
    token_usage_from_trace_text,
)

INTERESTING_FILENAMES = {
    "generation_config.json",
    "meta.json",
    "output_model_new.py",
    "trace.jsonl",
}


@dataclass(frozen=True)
class ArchiveRun:
    name: str
    files: dict[str, str]

    def read_text(self, relative_path: str) -> str | None:
        return self.files.get(relative_path)

    def read_json(self, relative_path: str) -> dict[str, Any]:
        text = self.read_text(relative_path)
        if text is None:
            return {}
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    def round_records(self) -> dict[int, list[RoundRecord]]:
        records: dict[int, list[RoundRecord]] = {}
        for relative_path in sorted(self.files, key=record_path_sort_key):
            parts = PurePosixPath(relative_path).parts
            if len(parts) != 3 or parts[2] != "meta.json":
                continue
            pid = parse_problem_id(parts[0])
            round_index = parse_record_round_index(parts[1])
            if pid is None or round_index is None:
                continue

            meta = self.read_json(relative_path)
            if not meta:
                continue
            output_path = f"{parts[0]}/{parts[1]}/output_model_new.py"
            output = self.read_text(output_path)
            if output is None:
                reward_hacking = ["missing output_model_new.py"]
            else:
                reward_hacking = reward_hacking_reasons_from_source(
                    output,
                    filename=f"{self.name}/{output_path}",
                )
            records.setdefault(pid, []).append(
                RoundRecord(
                    problem_id=pid,
                    round_index=round_index,
                    round_dir=Path(self.name) / parts[0] / parts[1],
                    correct=bool(meta.get("correctness")),
                    speedup=as_float(meta.get("speedup")),
                    reward_hacking=tuple(reward_hacking),
                )
            )
        return records

    def problem_trace_texts(self, problem_id: int) -> list[tuple[str, str]]:
        prefix = f"p{problem_id:02d}/"
        traces = [
            (relative_path, text)
            for relative_path, text in self.files.items()
            if relative_path.startswith(prefix)
            and relative_path.endswith("/trace.jsonl")
        ]
        return sorted(traces, key=lambda item: parse_trace_round(item[0]))


class ArchiveRuns:
    def __init__(self, files_by_run: dict[str, dict[str, str]]) -> None:
        self.files_by_run = files_by_run

    @classmethod
    def from_archives(
        cls,
        archives: Iterable[Path],
        run_names: Iterable[str],
    ) -> ArchiveRuns:
        run_names_set = set(run_names)
        files_by_run = {name: {} for name in run_names_set}
        for archive in archives:
            read_archive(archive, run_names_set, files_by_run)
        return cls(files_by_run)

    def run(self, name: str) -> ArchiveRun:
        return ArchiveRun(name=name, files=self.files_by_run.get(name, {}))


def read_archive(
    archive: Path,
    run_names: set[str],
    files_by_run: dict[str, dict[str, str]],
) -> None:
    with tarfile.open(archive, mode="r|xz") as tar:
        for member in tar:
            if not member.isfile() or not is_interesting_member(member.name):
                continue
            run_name, relative_path = split_run_path(member.name, run_names)
            if run_name is None or relative_path is None:
                continue
            file_obj = tar.extractfile(member)
            if file_obj is None:
                continue
            files_by_run[run_name][relative_path] = file_obj.read().decode(
                "utf-8",
                errors="replace",
            )


def is_interesting_member(member_name: str) -> bool:
    return PurePosixPath(member_name).name in INTERESTING_FILENAMES


def split_run_path(
    member_name: str,
    run_names: set[str],
) -> tuple[str | None, str | None]:
    parts = PurePosixPath(member_name).parts
    for index, part in enumerate(parts):
        if part not in run_names:
            continue
        relative_parts = parts[index + 1 :]
        if is_interesting_relative_path(relative_parts):
            return part, str(PurePosixPath(*relative_parts))
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
        run.read_json("generation_config.json"),
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

        files_read = files_read_for_archive_problem(run, pid)
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
        run.read_json("generation_config.json"),
        records.keys(),
        strict_denominator,
    )
    pass_all_rounds = 0
    token_values: list[int] = []

    for pid in problem_ids:
        rounds = sorted(records.get(pid, []), key=lambda item: item.round_index)
        if rounds and all(item.clean_correct for item in rounds):
            pass_all_rounds += 1
        token_usage = token_usage_for_archive_problem(run, pid)
        if token_usage is not None:
            token_values.append(token_usage)

    return OptimizationStats(
        denominator=len(problem_ids),
        pass_all_rounds=pass_all_rounds,
        avg_token_usage=mean(token_values),
    )


def files_read_for_archive_problem(run: ArchiveRun, problem_id: int) -> int | None:
    trace_texts = run.problem_trace_texts(problem_id)
    if not trace_texts:
        return None
    files: set[str] = set()
    for relative_path, text in trace_texts:
        files.update(
            files_read_from_trace_text(
                text,
                default_cwd=str(Path(run.name) / Path(relative_path).parent),
            )
        )
    return len(files)


def token_usage_for_archive_problem(run: ArchiveRun, problem_id: int) -> int | None:
    values = [
        value
        for _, text in run.problem_trace_texts(problem_id)
        if (value := token_usage_from_trace_text(text)) is not None
    ]
    return sum(values) if values else None


def is_round_dir_name(name: str) -> bool:
    return len(name) > len("round") and name.startswith("round") and name[5].isdigit()


def parse_record_round_index(name: str) -> int | None:
    if not is_round_dir_name(name):
        return None
    return parse_round_index(name) or 0


def parse_trace_round(relative_path: str) -> int:
    parts = PurePosixPath(relative_path).parts
    if len(parts) < 2:
        return 10**9
    round_index = parse_round_index(parts[1])
    return round_index if round_index is not None else 10**9


def record_path_sort_key(relative_path: str) -> tuple[int, str]:
    parts = PurePosixPath(relative_path).parts
    if len(parts) < 2:
        return (10**9, relative_path)
    round_index = parse_round_index(parts[1])
    return (round_index if round_index is not None else 10**9, relative_path)
