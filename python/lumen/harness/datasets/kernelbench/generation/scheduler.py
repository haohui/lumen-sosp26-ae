"""ThreadPool scheduling for KernelBench generation work."""

from __future__ import annotations

import concurrent.futures
import logging
from collections.abc import Callable
from pathlib import Path

from lumen.harness.datasets.kernelbench.generation.types import WorkArgs

LOGGER = logging.getLogger(__name__)


def build_work_items(
    problem_ids: list[int],
    run_dir: str | Path,
    gpu_ids: tuple[int, ...],
    *,
    completion_subdir: str | None = None,
) -> tuple[list[WorkArgs], int]:
    gpus = gpu_ids or (0,)
    problems: list[WorkArgs] = []
    already_done = 0
    base = Path(run_dir)
    for pid in problem_ids:
        problem_dir = base / f"p{pid:02d}"
        completion_dir = (
            problem_dir / completion_subdir
            if completion_subdir is not None
            else problem_dir
        )
        if (completion_dir / "meta.json").is_file():
            already_done += 1
            continue
        gpu_id = gpus[len(problems) % len(gpus)]
        problems.append(WorkArgs(problem_id=int(pid), gpu_id=gpu_id))
    return problems, already_done


def default_worker_count(configured_workers: int, gpu_ids: tuple[int, ...]) -> int:
    if len(gpu_ids) > 1 and configured_workers <= 1:
        return len(gpu_ids)
    return max(1, configured_workers)


def run_generation_tasks(
    problems: list[WorkArgs],
    generate_one: Callable[[WorkArgs], bool],
    *,
    num_workers: int,
) -> list[bool | None]:
    if not problems:
        return []

    if num_workers <= 1:
        results: list[bool | None] = []
        for work in problems:
            try:
                results.append(generate_one(work))
            except Exception as exc:
                LOGGER.exception("p%02d failed: %s", work.problem_id, exc)
                results.append(None)
        return results

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(generate_one, work): work for work in problems}
        for future in concurrent.futures.as_completed(futures):
            work = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                LOGGER.exception("p%02d failed: %s", work.problem_id, exc)
                results.append(None)
    return results
