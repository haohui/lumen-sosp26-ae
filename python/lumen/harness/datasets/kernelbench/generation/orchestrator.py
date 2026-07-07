"""High-level orchestration for KernelBench generation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from lumen.harness.datasets.kernelbench.generation.artifacts import (
    write_generation_config,
)
from lumen.harness.datasets.kernelbench.generation.evaluation import run_eval_phase
from lumen.harness.datasets.kernelbench.generation.round_runner import generate_problem
from lumen.harness.datasets.kernelbench.generation.scheduler import (
    build_work_items,
    default_worker_count,
    run_generation_tasks,
)
from lumen.harness.datasets.kernelbench.generation.types import (
    GenerationConfig,
    KernelBenchDatasetConfig,
)

LOGGER = logging.getLogger(__name__)


def run_generation(config: GenerationConfig) -> None:
    dataset = _construct_dataset(config.dataset)
    problem_ids = select_problem_ids(dataset, config.dataset)

    run_dir = Path(config.run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    write_generation_config(run_dir / "generation_config.json", config)

    num_workers = default_worker_count(
        int(config.num_workers),
        config.evaluation.gpu_ids,
    )
    if num_workers != int(config.num_workers):
        LOGGER.info(
            "gpu_ids=%s -> auto num_workers=%d",
            list(config.evaluation.gpu_ids),
            num_workers,
        )

    problems, already_done = build_work_items(
        problem_ids,
        run_dir,
        config.evaluation.gpu_ids,
    )
    if already_done:
        LOGGER.info(
            "%d/%d already generated; skipping.",
            already_done,
            len(problem_ids),
        )
    LOGGER.info(
        "Generating %d kernel(s) for level %d (timeout %ss each)",
        len(problems),
        config.dataset.level,
        config.codex.timeout_seconds,
    )

    results = run_generation_tasks(
        problems,
        lambda work: generate_problem(work, config, dataset, run_dir),
        num_workers=num_workers,
    )
    if results:
        num_ok = sum(1 for result in results if result)
        LOGGER.info("%d/%d generated.", num_ok, len(results))
    else:
        LOGGER.info("Nothing to generate.")

    run_eval_phase(config, dataset, problem_ids)
    LOGGER.info("Results in: %s", run_dir)


def select_problem_ids(
    dataset: Any,
    config: KernelBenchDatasetConfig,
) -> list[int]:
    all_problem_ids = dataset.get_problem_ids()
    all_set = set(all_problem_ids)
    explicit = list(config.problem_ids)

    if explicit:
        unknown = [pid for pid in explicit if pid not in all_set]
        if unknown:
            LOGGER.warning("problem_ids not in dataset, ignored: %s", unknown)
        return [pid for pid in explicit if pid in all_set]

    start, end = config.subset
    if start is None and end is None:
        return list(all_problem_ids)

    start_value = min(all_problem_ids) if start is None else start
    end_value = max(all_problem_ids) if end is None else end
    return [pid for pid in all_problem_ids if start_value <= pid <= end_value]


def _construct_dataset(config: KernelBenchDatasetConfig) -> Any:
    from kernelbench.dataset import construct_kernelbench_dataset

    kwargs: dict[str, Any] = {
        "level": int(config.level),
        "source": config.source,
    }
    if config.source == "local":
        kwargs["base_path"] = config.name
    else:
        kwargs["dataset_name"] = config.name
    return construct_kernelbench_dataset(**kwargs)
