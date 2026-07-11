"""Typed configuration for KernelBench generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class KernelBenchDatasetConfig:
    source: str
    level: int
    name: str = "ScalingIntelligence/KernelBench"
    subset: tuple[int | None, int | None] = (None, None)
    problem_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class CodexGenerationConfig:
    timeout_seconds: int = 600
    max_retries: int = 3
    codex_bin: str | None = None
    profile: str | None = None
    model_provider: str | None = None
    reasoning_effort: str | None = None
    config_overrides: tuple[str, ...] = ()
    bypass_approvals_and_sandbox: bool = True
    save_trajectory: bool = True


@dataclass(frozen=True)
class KernelBenchEvaluationConfig:
    gpu_ids: tuple[int, ...] = (0,)
    precision: str = "bf16"
    num_correct_trials: int = 5
    num_perf_trials: int = 10
    gpu_arch: str = "gfx942"


@dataclass(frozen=True)
class GenerationConfig:
    dataset: KernelBenchDatasetConfig
    run_dir: Path
    codex: CodexGenerationConfig = field(default_factory=CodexGenerationConfig)
    evaluation: KernelBenchEvaluationConfig = field(
        default_factory=KernelBenchEvaluationConfig
    )
    num_workers: int = 1


@dataclass(frozen=True)
class WorkArgs:
    problem_id: int
    gpu_id: int


@dataclass(frozen=True)
class OptimizationConfig:
    run_dir: Path
    profile: str
