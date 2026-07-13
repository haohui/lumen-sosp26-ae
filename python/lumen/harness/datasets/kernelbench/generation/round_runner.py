"""Per-problem KernelBench generation workflow."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from lumen.harness.datasets.kernelbench.generation.artifacts import (
    write_problem_meta,
    write_round_artifacts,
)
from lumen.harness.datasets.kernelbench.generation.evaluation import (
    evaluate_round,
    torch_cuda_available,
)
from lumen.harness.datasets.kernelbench.generation.executor import (
    copy_codex_trace,
    run_codex,
    write_codex_result,
)
from lumen.harness.datasets.kernelbench.generation.metrics import format_eval_status
from lumen.harness.datasets.kernelbench.generation.types import (
    GenerationConfig,
    WorkArgs,
)
from lumen.harness.datasets.kernelbench.generation.validation import (
    validate_generated_avelang,
)
from lumen.harness.datasets.kernelbench.generation.workspace import (
    RoundWorkspace,
    prepare_round_workspace,
    read_generated_output,
)

LOGGER = logging.getLogger(__name__)


def generate_problem(
    work: WorkArgs,
    config: GenerationConfig,
    dataset: Any,
    run_dir: Path,
) -> bool:
    problem = dataset.get_problem_by_id(work.problem_id)
    problem_name = problem.name

    problem_dir = run_dir / f"p{work.problem_id:02d}"
    problem_dir.mkdir(parents=True, exist_ok=True)

    round_metas: list[dict[str, Any]] = []
    any_correct = False
    for round_idx in range(max(1, int(config.codex.max_retries))):
        round_dir = problem_dir / f"round{round_idx}"
        tag = f"p{work.problem_id:02d}/round{round_idx}"
        round_meta, any_correct, should_continue = run_generation_round(
            tag=tag,
            round_dir=round_dir,
            work=work,
            config=config,
            problem_name=problem_name,
            ref_arch_src=problem.code,
        )
        round_metas.append(round_meta)
        if not should_continue:
            break

    write_problem_meta(
        problem_dir,
        problem_id=work.problem_id,
        problem_name=problem_name,
        round_metas=round_metas,
    )
    return any_correct or bool(round_metas)


def run_generation_round(
    *,
    tag: str,
    round_dir: Path,
    work: WorkArgs,
    config: GenerationConfig,
    problem_name: str,
    ref_arch_src: str,
    prepared_workspace: RoundWorkspace | None = None,
) -> tuple[dict[str, Any], bool, bool]:
    if prepared_workspace is None:
        workspace = prepare_round_workspace(
            round_dir,
            ref_arch_src=ref_arch_src,
            evaluation=config.evaluation,
            reference_mode=config.prompt.reference_mode,
        )
    else:
        workspace = prepared_workspace
    prompt = workspace.prompt

    result = run_codex(workspace.round_dir, prompt, config.codex)
    write_codex_result(round_dir / "codex_result.json", result)
    copy_codex_trace(
        result,
        round_dir,
        save_trace=config.codex.save_trajectory,
    )
    if not result.ok:
        log = result.error or result.status
        error = f"codex exited with error. log={log[:400]}"
        LOGGER.error("%s: %s", tag, error)
        meta = write_round_artifacts(
            round_dir,
            problem_id=work.problem_id,
            problem_name=problem_name,
            error=error,
        )
        return meta, False, result.status == "timed_out"

    custom_kernel, output_error = read_generated_output(round_dir)
    if output_error is not None or custom_kernel is None:
        LOGGER.error("%s: %s", tag, output_error)
        meta = write_round_artifacts(
            round_dir,
            problem_id=work.problem_id,
            problem_name=problem_name,
            error=str(output_error),
        )
        return meta, False, True

    static_ok, errors, warnings = validate_generated_avelang(custom_kernel)
    if not static_ok:
        error = f"Static check failed: {errors}. Warnings: {warnings}"
        LOGGER.error("%s: %s", tag, error)
        meta = write_round_artifacts(
            round_dir,
            problem_id=work.problem_id,
            problem_name=problem_name,
            error=error,
        )
        return meta, False, True
    if warnings:
        LOGGER.warning("%s: %s", tag, warnings)

    if torch_cuda_available():
        eval_payload = evaluate_round(round_dir, config.evaluation, work.gpu_id)
        LOGGER.info("%s: %s", tag, format_eval_status(eval_payload))
        any_correct = bool(eval_payload.get("correctness", False))
        meta = write_round_artifacts(
            round_dir,
            problem_id=work.problem_id,
            problem_name=problem_name,
            error="",
            eval_payload=eval_payload,
        )
        return meta, any_correct, not any_correct

    meta = write_round_artifacts(
        round_dir,
        problem_id=work.problem_id,
        problem_name=problem_name,
        error="OK: generated",
    )
    return meta, False, False
