"""Workspace preparation for KernelBench generation rounds."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lumen.harness.datasets.kernelbench.generation.types import (
    KernelBenchEvaluationConfig,
)


@dataclass(frozen=True)
class RoundWorkspace:
    round_dir: Path
    prompt: str


def prepare_round_workspace(
    round_dir: str | Path,
    *,
    ref_arch_src: str,
    evaluation: KernelBenchEvaluationConfig,
    reference_mode: str = "full",
) -> RoundWorkspace:
    paths = _write_generation_workspace(
        round_dir,
        ref_arch_src=ref_arch_src,
        precision=evaluation.precision,
        gpu_arch=evaluation.gpu_arch,
        eval_num_correct_trials=evaluation.num_correct_trials,
        eval_num_perf_trials=evaluation.num_perf_trials,
        reference_mode=reference_mode,
    )
    prompt = paths["prompt"].read_text(encoding="utf-8")
    return RoundWorkspace(round_dir=Path(round_dir), prompt=prompt)


def prepare_optimization_round_workspace(
    round_dir: str | Path,
    *,
    ref_arch_src: str,
    candidate_src: str | None,
    evaluation: KernelBenchEvaluationConfig,
    prompt_config_name: str,
    prompt_name: str,
    profile: str,
    template_family: str,
    guidance: str,
) -> RoundWorkspace:
    from lumen.harness.datasets.kernelbench.generation.prompt import (
        write_optimization_workspace,
    )

    paths = write_optimization_workspace(
        round_dir,
        ref_arch_src=ref_arch_src,
        candidate_src=candidate_src,
        precision=evaluation.precision,
        gpu_arch=evaluation.gpu_arch,
        eval_num_correct_trials=evaluation.num_correct_trials,
        eval_num_perf_trials=evaluation.num_perf_trials,
        prompt_config_name=prompt_config_name,
        prompt_name=prompt_name,
        profile=profile,
        template_family=template_family,
        guidance=guidance,
    )
    prompt = paths["prompt"].read_text(encoding="utf-8")
    return RoundWorkspace(round_dir=Path(round_dir), prompt=prompt)


def read_generated_output(round_dir: str | Path) -> tuple[str | None, str | None]:
    output_path = Path(round_dir) / "output_model_new.py"
    if not output_path.is_file():
        return None, "output_model_new.py not found"

    custom_kernel = output_path.read_text(encoding="utf-8")
    if not custom_kernel.strip():
        return None, "output_model_new.py is empty"
    return custom_kernel, None


def _write_generation_workspace(*args: object, **kwargs: object) -> dict[str, Path]:
    try:
        from lumen.harness.datasets.kernelbench.generation.prompt import (
            write_generation_workspace,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "KernelBench prompt construction is required before running generation."
        ) from exc

    return write_generation_workspace(*args, **kwargs)
