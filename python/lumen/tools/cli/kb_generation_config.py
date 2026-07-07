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
    if not isinstance(data, dict):
        raise ValueError(f"TOML config must be a table: {config_path}")
    return generation_config_from_mapping(
        data,
        Path.cwd() if base_dir is None else Path(base_dir),
    )


def generation_config_from_mapping(
    values: dict[str, Any],
    base_dir: Path,
) -> GenerationConfig:
    dataset_values = _section(values, "dataset")
    codex_values = _section(values, "codex", required=False)
    evaluation_values = _section(values, "evaluation", required=False)

    dataset = KernelBenchDatasetConfig(
        source=_required_str(dataset_values, "source"),
        name=_optional_str(dataset_values, "name")
        or "ScalingIntelligence/KernelBench",
        level=_required_int(dataset_values, "level"),
        subset=_optional_int_pair(dataset_values, "subset"),
        problem_ids=_optional_int_tuple(dataset_values, "problem_ids"),
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
        timeout_seconds=_optional_int(codex_values, "timeout_seconds", default=600),
        max_retries=_optional_int(codex_values, "max_retries", default=3),
        codex_bin=_optional_str(codex_values, "codex_bin"),
        profile=_optional_str(codex_values, "profile"),
        model_provider=_optional_str(codex_values, "model_provider"),
        reasoning_effort=_optional_str(codex_values, "reasoning_effort"),
        config_overrides=_optional_str_tuple(codex_values, "config_overrides"),
        bypass_approvals_and_sandbox=_optional_bool(
            codex_values,
            "bypass_approvals_and_sandbox",
            default=True,
        ),
        save_trajectory=_optional_bool(
            codex_values,
            "save_trajectory",
            default=True,
        ),
    )
    evaluation = KernelBenchEvaluationConfig(
        gpu_ids=_optional_int_tuple(evaluation_values, "gpu_ids", default=(0,)),
        precision=_optional_str(evaluation_values, "precision") or "bf16",
        num_correct_trials=_optional_int(
            evaluation_values,
            "num_correct_trials",
            default=5,
        ),
        num_perf_trials=_optional_int(
            evaluation_values,
            "num_perf_trials",
            default=10,
        ),
        gpu_arch=_optional_str(evaluation_values, "gpu_arch") or "gfx942",
    )
    return GenerationConfig(
        dataset=dataset,
        run_dir=_resolve_path(_required_str(values, "run_dir"), base_dir),
        codex=codex,
        evaluation=evaluation,
        num_workers=_optional_int(values, "num_workers", default=1),
    )


def problem_ids_from_config(config: GenerationConfig) -> list[int]:
    if config.dataset.problem_ids:
        return list(config.dataset.problem_ids)

    start, end = config.dataset.subset
    if start is None or end is None:
        raise ValueError(
            "Metrics reporting requires dataset.problem_ids or a bounded "
            "dataset.subset in the generation config."
        )
    return list(range(start, end + 1))


def _section(
    values: dict[str, Any],
    key: str,
    *,
    required: bool = True,
) -> dict[str, Any]:
    value = values.get(key)
    if value is None and not required:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Config section [{key}] is required")
    return value


def _required_str(values: dict[str, Any], key: str) -> str:
    if key not in values:
        raise ValueError(f"Missing required config key: {key}")
    value = values[key]
    if not isinstance(value, str) or not value:
        raise ValueError(f"Config key {key} must be a non-empty string")
    return value


def _optional_str(values: dict[str, Any], key: str) -> str | None:
    if key not in values:
        return None
    value = values[key]
    if not isinstance(value, str) or not value:
        raise ValueError(f"Config key {key} must be a non-empty string")
    return value


def _required_int(values: dict[str, Any], key: str) -> int:
    if key not in values:
        raise ValueError(f"Missing required config key: {key}")
    value = values[key]
    if not _is_int(value):
        raise ValueError(f"Config key {key} must be an integer")
    return value


def _optional_int(values: dict[str, Any], key: str, *, default: int) -> int:
    if key not in values:
        return default
    value = values[key]
    if not _is_int(value):
        raise ValueError(f"Config key {key} must be an integer")
    return value


def _optional_bool(values: dict[str, Any], key: str, *, default: bool) -> bool:
    if key not in values:
        return default
    value = values[key]
    if not isinstance(value, bool):
        raise ValueError(f"Config key {key} must be a boolean")
    return value


def _optional_int_pair(
    values: dict[str, Any],
    key: str,
) -> tuple[int | None, int | None]:
    if key not in values:
        return (None, None)
    value = values[key]
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"Config key {key} must be a two-item TOML array")
    if not all(_is_int(item) for item in value):
        raise ValueError(f"Config key {key} must contain integers")
    return value[0], value[1]


def _optional_int_tuple(
    values: dict[str, Any],
    key: str,
    *,
    default: tuple[int, ...] = (),
) -> tuple[int, ...]:
    if key not in values:
        return default
    value = values[key]
    if not isinstance(value, list):
        raise ValueError(f"Config key {key} must be a TOML array")
    if not all(_is_int(item) for item in value):
        raise ValueError(f"Config key {key} must contain integers")
    return tuple(value)


def _optional_str_tuple(values: dict[str, Any], key: str) -> tuple[str, ...]:
    if key not in values:
        return ()
    value = values[key]
    if not isinstance(value, list):
        raise ValueError(f"Config key {key} must be a TOML array")
    if not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"Config key {key} must contain non-empty strings")
    return tuple(value)


def _resolve_path(value: str | Path, base_dir: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (Path(base_dir).expanduser() / path).resolve()


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)
