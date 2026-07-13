"""Summarize KernelBench generation and optimization run directories."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kb_table.common import (
    as_float,
    geomean,
    mean,
    parse_problem_id,
    parse_round_index,
    read_json,
    round_sort_key,
)
from kb_table.models import GenerationStats, OptimizationStats, RoundRecord
from kb_table.reward_hacking import reward_hacking_reasons
from kb_table.trace_stats import files_read_for_problem, token_usage_for_problem


def summarize_generation(run_dir: Path, *, strict_denominator: bool) -> GenerationStats:
    records = load_round_records(run_dir)
    problem_ids = expected_problem_ids(run_dir, records.keys(), strict_denominator)
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

        files_read = files_read_for_problem(run_dir / f"p{pid:02d}")
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


def summarize_optimization(
    run_dir: Path,
    *,
    strict_denominator: bool,
) -> OptimizationStats:
    records = load_round_records(run_dir)
    problem_ids = expected_problem_ids(run_dir, records.keys(), strict_denominator)
    pass_final = 0
    token_values: list[int] = []

    for pid in problem_ids:
        rounds = sorted(records.get(pid, []), key=lambda item: item.round_index)
        final = rounds[-1] if rounds else None
        if final and final.valid:
            pass_final += 1
        token_usage = token_usage_for_problem(run_dir / f"p{pid:02d}")
        if token_usage is not None:
            token_values.append(token_usage)

    return OptimizationStats(
        denominator=len(problem_ids),
        pass_final=pass_final,
        avg_token_usage=mean(token_values),
    )


def load_round_records(run_dir: Path) -> dict[int, list[RoundRecord]]:
    records: dict[int, list[RoundRecord]] = {}
    if not run_dir.is_dir():
        return records
    for problem_dir in sorted(run_dir.glob("p[0-9]*")):
        if not problem_dir.is_dir():
            continue
        pid = parse_problem_id(problem_dir.name)
        if pid is None:
            continue
        for round_dir in sorted(problem_dir.glob("round[0-9]*"), key=round_sort_key):
            meta = read_json(round_dir / "meta.json")
            if not meta:
                continue
            records.setdefault(pid, []).append(
                RoundRecord(
                    problem_id=pid,
                    round_index=parse_round_index(round_dir.name) or 0,
                    round_dir=round_dir,
                    correct=bool(meta.get("correctness")),
                    speedup=as_float(meta.get("speedup")),
                    reward_hacking=tuple(
                        reward_hacking_reasons(round_dir / "output_model_new.py")
                    ),
                )
            )
    return records


def expected_problem_ids(
    run_dir: Path,
    observed: Any,
    strict_denominator: bool,
) -> list[int]:
    observed_ids = sorted(int(pid) for pid in observed)
    config = read_json(run_dir / "generation_config.json")
    dataset = config.get("dataset") if isinstance(config, dict) else {}
    problem_ids = dataset.get("problem_ids") if isinstance(dataset, dict) else None
    if problem_ids:
        expected = sorted(int(pid) for pid in problem_ids)
    else:
        expected = expected_subset_ids(dataset)
    if strict_denominator and expected:
        return expected
    return observed_ids or expected


def expected_subset_ids(dataset: Any) -> list[int]:
    subset = dataset.get("subset") if isinstance(dataset, dict) else None
    if (
        isinstance(subset, list)
        and len(subset) == 2
        and isinstance(subset[0], int)
        and isinstance(subset[1], int)
    ):
        return list(range(subset[0], subset[1] + 1))
    return []
