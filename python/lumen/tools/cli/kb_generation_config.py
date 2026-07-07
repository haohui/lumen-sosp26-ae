"""TOML config loading for KernelBench generation CLIs."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from lumen.harness.datasets.kernelbench.generation import (
    CodexGenerationConfig,
    GenerationConfig,
    KernelBenchDatasetConfig,
    KernelBenchEvaluationConfig,
)


def load_generation_config(
    path: str | Path,
    *,
    base_dir: str | Path | None = None,
) -> GenerationConfig:
    config_path = Path(path).expanduser()
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    return generation_config_from_mapping(
        data,
        Path.cwd() if base_dir is None else Path(base_dir),
    )


def generation_config_from_mapping(
    values: dict[str, Any],
    base_dir: Path,
) -> GenerationConfig:
    dataset_values = values["dataset"]
    codex_values = values.get("codex", {})
    evaluation_values = values.get("evaluation", {})

    dataset = KernelBenchDatasetConfig(
        source=dataset_values["source"],
        name=dataset_values.get("name", "ScalingIntelligence/KernelBench"),
        level=dataset_values["level"],
        subset=tuple(dataset_values.get("subset", (None, None))),
        problem_ids=tuple(dataset_values.get("problem_ids", ())),
    )
    if dataset.source == "local":
        dataset = KernelBenchDatasetConfig(
            source=dataset.source,
            name=str(_resolve_path(dataset.name, base_dir)),
            level=dataset.level,
            subset=dataset.subset,
            problem_ids=dataset.problem_ids,
        )

    codex = CodexGenerationConfig(
        timeout_seconds=codex_values.get("timeout_seconds", 600),
        max_retries=codex_values.get("max_retries", 3),
        codex_bin=codex_values.get("codex_bin"),
        profile=codex_values.get("profile"),
        model_provider=codex_values.get("model_provider"),
        reasoning_effort=codex_values.get("reasoning_effort"),
        config_overrides=tuple(codex_values.get("config_overrides", ())),
        bypass_approvals_and_sandbox=codex_values.get(
            "bypass_approvals_and_sandbox",
            True,
        ),
        save_trajectory=codex_values.get("save_trajectory", True),
    )
    evaluation = KernelBenchEvaluationConfig(
        gpu_ids=tuple(evaluation_values.get("gpu_ids", (0,))),
        precision=evaluation_values.get("precision", "bf16"),
        num_correct_trials=evaluation_values.get("num_correct_trials", 5),
        num_perf_trials=evaluation_values.get("num_perf_trials", 10),
        gpu_arch=evaluation_values.get("gpu_arch", "gfx942"),
    )
    return GenerationConfig(
        dataset=dataset,
        run_dir=_resolve_path(values["run_dir"], base_dir),
        codex=codex,
        evaluation=evaluation,
        num_workers=values.get("num_workers", 1),
    )


def _resolve_path(value: str | Path, base_dir: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (Path(base_dir).expanduser() / path).resolve()
