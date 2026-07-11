"""Iterative optimization workflow for previously generated KernelBench kernels."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from lumen.harness.datasets.kernelbench.generation.artifacts import (
    write_generation_config,
    write_problem_meta,
)
from lumen.harness.datasets.kernelbench.generation.evaluation import run_eval_phase
from lumen.harness.datasets.kernelbench.generation.optimization_profiles import (
    OptimizationProfile,
    get_profile,
)
from lumen.harness.datasets.kernelbench.generation.orchestrator import (
    _construct_dataset,
    select_problem_ids,
)
from lumen.harness.datasets.kernelbench.generation.round_runner import (
    run_generation_round,
)
from lumen.harness.datasets.kernelbench.generation.scheduler import (
    build_work_items,
    default_worker_count,
    run_generation_tasks,
)
from lumen.harness.datasets.kernelbench.generation.types import (
    GenerationConfig,
    OptimizationConfig,
    WorkArgs,
)
from lumen.harness.datasets.kernelbench.generation.workspace import (
    prepare_optimization_round_workspace,
)

LOGGER = logging.getLogger(__name__)


def run_optimization(
    config: GenerationConfig,
    optimization: OptimizationConfig,
    candidates: dict[int, Path],
) -> None:
    """Optimize manifest-provided candidates in an independent output root."""
    profile = get_profile(optimization.profile)
    dataset = _construct_dataset(config.dataset)
    problem_ids = select_problem_ids(dataset, config.dataset)
    _validate_candidates(problem_ids, candidates)
    run_dir = optimization.run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    write_generation_config(run_dir / "generation_config.json", config)
    (run_dir / "optimization_config.json").write_text(
        json.dumps({"profile": profile.name}, indent=2) + "\n", encoding="utf-8"
    )
    workers = default_worker_count(int(config.num_workers), config.evaluation.gpu_ids)
    work_items, already_done = build_work_items(
        problem_ids, run_dir, config.evaluation.gpu_ids
    )
    if already_done:
        LOGGER.info(
            "%d/%d already optimized; skipping.", already_done, len(problem_ids)
        )
    results = run_generation_tasks(
        work_items,
        lambda work: _optimize_problem(
            work, config, dataset, run_dir, profile, candidates[work.problem_id]
        ),
        num_workers=workers,
    )
    if results:
        LOGGER.info(
            "%d/%d optimized.", sum(bool(result) for result in results), len(results)
        )
    run_eval_phase(_config_with_run_dir(config, run_dir), dataset, problem_ids)
    LOGGER.info("Optimization results in: %s", run_dir)


def _optimize_problem(
    work: WorkArgs,
    config: GenerationConfig,
    dataset: Any,
    run_dir: Path,
    profile: OptimizationProfile,
    candidate_path: Path,
) -> bool:
    problem = dataset.get_problem_by_id(work.problem_id)
    problem_dir = run_dir / f"p{work.problem_id:02d}"
    problem_dir.mkdir(parents=True, exist_ok=True)
    candidate_src = candidate_path.read_text(encoding="utf-8")
    provenance = {
        "path": str(candidate_path.resolve()),
        "sha256": hashlib.sha256(candidate_src.encode()).hexdigest(),
        "profile": profile.name,
    }
    (problem_dir / "candidate_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    metas: list[dict[str, Any]] = []
    any_correct = False
    for round_index in range(
        min(max(1, int(config.codex.max_retries)), profile.round_count(problem.code))
    ):
        family, guidance = profile.guidance(problem.code, round_index)
        round_dir = problem_dir / f"round{round_index}"
        workspace = prepare_optimization_round_workspace(
            round_dir,
            ref_arch_src=problem.code,
            candidate_src=candidate_src,
            evaluation=config.evaluation,
            prompt_config_name=profile.prompt_config_name,
            prompt_name=profile.prompt_name,
            profile=profile.name,
            template_family=family,
            guidance=guidance,
        )
        meta, correct, should_continue = run_generation_round(
            tag=f"p{work.problem_id:02d}/round{round_index}",
            round_dir=round_dir,
            work=work,
            config=config,
            problem_name=problem.name,
            ref_arch_src=problem.code,
            prepared_workspace=workspace,
        )
        metas.append(meta)
        any_correct = any_correct or correct
        output = round_dir / "output_model_new.py"
        if not should_continue or not output.is_file():
            break
        candidate_src = output.read_text(encoding="utf-8")
        if not candidate_src.strip():
            break
    write_problem_meta(
        problem_dir,
        problem_id=work.problem_id,
        problem_name=problem.name,
        round_metas=metas,
    )
    return any_correct or bool(metas)


def _validate_candidates(problem_ids: list[int], candidates: dict[int, Path]) -> None:
    expected = set(problem_ids)
    missing = expected - set(candidates)
    extra = set(candidates) - expected
    if missing or extra:
        raise ValueError(
            "candidate manifest mismatch: "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _config_with_run_dir(config: GenerationConfig, run_dir: Path) -> GenerationConfig:
    return GenerationConfig(
        dataset=config.dataset,
        run_dir=run_dir,
        codex=config.codex,
        evaluation=config.evaluation,
        num_workers=config.num_workers,
    )
