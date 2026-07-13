"""Prepare and run an isolated Codex workspace for GEMM optimization."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from lumen.harness.backend.codex import CodexRunner, CodexRunnerConfig, CodexRunResult
from lumen.harness.datasets.lumen.optimization_runtime import (
    append_run_prompts,
    evaluate_candidate,
    evaluation_error,
    resolve_resume_point,
    reset_run_tail,
    save_codex_result,
    set_round_status,
    sha256_file,
    write_round_config,
)

GEMM_WORKLOADS = (1024, 2048, 4096, 8192, 16384)


@dataclass(frozen=True)
class GemmOptimizationConfig:
    repo_root: Path
    kernel: Path | None
    prompt_file: Path
    run_dir: Path | None = None
    entrypoint: str = "gemm_pipeline_transposed_b"
    gpu_id: int | None = None
    codex_bin: Path | None = None
    profile: str | None = None
    model: str | None = None
    model_provider: str | None = None
    reasoning_effort: str | None = None
    timeout_seconds: float | None = 3600
    config_overrides: tuple[str, ...] = ()
    bypass_approvals_and_sandbox: bool = True


@dataclass(frozen=True)
class GemmOptimizationWorkspace:
    run_dir: Path
    round_dir: Path
    input_model_path: Path
    output_model_path: Path
    prompt: str


def prepare_gemm_optimization(
    config: GemmOptimizationConfig,
) -> GemmOptimizationWorkspace:
    repo_root = config.repo_root.expanduser().resolve()
    kernel = _require_kernel(config)
    prompt_file = config.prompt_file.expanduser().resolve()
    _validate_inputs(repo_root, kernel, (prompt_file,), config.gpu_id)

    run_dir = _resolve_run_dir(repo_root, config.run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_run_config(config, run_dir, kernel, (prompt_file,))
    return _prepare_round(
        config,
        run_dir=run_dir,
        round_index=0,
        kernel=kernel,
        prompt_file=prompt_file,
    )


def _prepare_round(
    config: GemmOptimizationConfig,
    *,
    run_dir: Path,
    round_index: int,
    kernel: Path,
    prompt_file: Path,
) -> GemmOptimizationWorkspace:
    repo_root = config.repo_root.expanduser().resolve()
    prompt = prompt_file.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"prompt file is empty: {prompt_file}")

    round_dir = run_dir / f"round{round_index}"
    round_dir.mkdir()

    input_model_path = round_dir / "input_model.py"
    output_model_path = round_dir / "output_model_new.py"
    input_model_path.write_text(kernel.read_text(encoding="utf-8"), encoding="utf-8")
    output_model_path.write_text("", encoding="utf-8")

    adapter_dir = round_dir / "datasets" / "inference" / "gemm" / "lumen"
    adapter_dir.mkdir(parents=True)
    adapter_path = adapter_dir / "model.py"
    adapter_path.write_text(
        _render_benchmark_adapter(config.entrypoint),
        encoding="utf-8",
    )

    prompt_path = round_dir / "prompt.txt"
    prompt_path.write_text(prompt + "\n", encoding="utf-8")
    agents_path = round_dir / "AGENTS.md"
    agents_path.write_text(_render_agents_md(repo_root), encoding="utf-8")
    write_round_config(
        round_dir,
        domain="gemm",
        round_index=round_index,
        input_source=kernel,
        input_model=input_model_path,
        prompt_source=prompt_file,
        prompt_path=prompt_path,
        agents_path=agents_path,
        adapter_path=adapter_path,
        entrypoint=config.entrypoint,
        gpu_id=config.gpu_id,
        workloads=GEMM_WORKLOADS,
        codex_config=_codex_provenance(config),
    )
    return GemmOptimizationWorkspace(
        run_dir=run_dir,
        round_dir=round_dir,
        input_model_path=input_model_path,
        output_model_path=output_model_path,
        prompt=prompt,
    )


def run_gemm_optimization(
    config: GemmOptimizationConfig,
) -> tuple[GemmOptimizationWorkspace, CodexRunResult]:
    results = run_gemm_optimization_sequence(config, (config.prompt_file,))
    return results[0]


def run_gemm_optimization_sequence(
    config: GemmOptimizationConfig,
    prompt_files: Sequence[Path],
) -> list[tuple[GemmOptimizationWorkspace, CodexRunResult]]:
    repo_root = config.repo_root.expanduser().resolve()
    kernel = _require_kernel(config)
    prompts = tuple(path.expanduser().resolve() for path in prompt_files)
    if not prompts:
        raise ValueError("at least one prompt file is required")
    _validate_inputs(repo_root, kernel, prompts, config.gpu_id)

    run_dir = _resolve_run_dir(repo_root, config.run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_run_config(config, run_dir, kernel, prompts)

    return _run_rounds(
        config,
        prompts=prompts,
        run_dir=run_dir,
        round_kernel=kernel,
        start_index=0,
    )


def resume_gemm_optimization_sequence(
    config: GemmOptimizationConfig,
    prompt_files: Sequence[Path],
    run_dir: Path,
) -> list[tuple[GemmOptimizationWorkspace, CodexRunResult]]:
    repo_root = config.repo_root.expanduser().resolve()
    prompts = tuple(path.expanduser().resolve() for path in prompt_files)
    if not prompts:
        raise ValueError("at least one prompt file is required")

    resolved_run_dir = run_dir.expanduser().resolve()
    start_index, round_kernel = resolve_resume_point(resolved_run_dir)
    _validate_inputs(repo_root, round_kernel, prompts, config.gpu_id)
    resumed_config = replace(
        config,
        kernel=round_kernel,
        run_dir=resolved_run_dir,
    )
    reset_run_tail(resolved_run_dir, start_index)
    append_run_prompts(resolved_run_dir, prompts, start_index=start_index)
    return _run_rounds(
        resumed_config,
        prompts=prompts,
        run_dir=resolved_run_dir,
        round_kernel=round_kernel,
        start_index=start_index,
    )


def _run_rounds(
    config: GemmOptimizationConfig,
    *,
    prompts: tuple[Path, ...],
    run_dir: Path,
    round_kernel: Path,
    start_index: int,
) -> list[tuple[GemmOptimizationWorkspace, CodexRunResult]]:
    total_rounds = start_index + len(prompts)

    rounds: list[tuple[GemmOptimizationWorkspace, CodexRunResult]] = []
    for round_index, prompt_file in enumerate(prompts, start=start_index):
        workspace = _prepare_round(
            config,
            run_dir=run_dir,
            round_index=round_index,
            kernel=round_kernel,
            prompt_file=prompt_file,
        )
        result = _run_workspace(config, workspace)
        rounds.append((workspace, result))
        status = "passed" if result.ok else "failed"
        print(
            f"round {round_index + 1}/{total_rounds} {status}: "
            f"{workspace.round_dir}",
            flush=True,
        )
        if not result.ok:
            break
        round_kernel = workspace.output_model_path
    return rounds


def _run_workspace(
    config: GemmOptimizationConfig,
    workspace: GemmOptimizationWorkspace,
) -> CodexRunResult:
    set_round_status(workspace.round_dir, "codex_running")
    result = CodexRunner().execute(
        CodexRunnerConfig(
            work_dir=workspace.round_dir,
            prompt=workspace.prompt,
            codex_bin=config.codex_bin,
            profile=config.profile,
            model=config.model,
            model_provider=config.model_provider,
            reasoning_effort=config.reasoning_effort,
            timeout_seconds=config.timeout_seconds,
            env=_codex_env(config.gpu_id, workspace.round_dir),
            config_overrides=config.config_overrides,
            bypass_approvals_and_sandbox=config.bypass_approvals_and_sandbox,
        )
    )
    result = save_codex_result(
        workspace.round_dir,
        workspace.output_model_path,
        result,
    )
    if not result.ok:
        set_round_status(
            workspace.round_dir,
            "failed",
            error=result.error or result.status,
        )
        return result

    set_round_status(workspace.round_dir, "evaluating")
    evaluation = evaluate_candidate(
        workspace.round_dir,
        domain="gemm",
        gpu_id=config.gpu_id,
        benchmark_root=workspace.round_dir / "datasets" / "inference",
        workloads=GEMM_WORKLOADS,
        workload_key="matrix_size",
        benchmark_args=(
            str(config.repo_root.resolve() / "scripts" / "benchmark" / "bench_gemm.py"),
            "--backend",
            "lumen",
            "--check-correctness",
            "--matrix-sizes",
            *(str(workload) for workload in GEMM_WORKLOADS),
        ),
    )
    if not evaluation["ok"]:
        error = evaluation_error(evaluation)
        set_round_status(workspace.round_dir, "failed", error=error)
        return replace(
            result,
            ok=False,
            status="failed",
            error=_join_errors(result.error, error),
        )
    set_round_status(workspace.round_dir, "passed")
    return result


def _validate_inputs(
    repo_root: Path,
    kernel: Path,
    prompt_files: Sequence[Path],
    gpu_id: int | None,
) -> None:
    if not (repo_root / "pyproject.toml").is_file():
        raise ValueError(f"not a repository root: {repo_root}")
    for directory in (repo_root / "scripts", repo_root / "skills"):
        if not directory.is_dir():
            raise ValueError(f"required directory not found: {directory}")
    if not kernel.is_file():
        raise ValueError(f"kernel file not found: {kernel}")
    for prompt_file in prompt_files:
        if not prompt_file.is_file():
            raise ValueError(f"prompt file not found: {prompt_file}")
        if not prompt_file.read_text(encoding="utf-8").strip():
            raise ValueError(f"prompt file is empty: {prompt_file}")
    if gpu_id is not None and gpu_id < 0:
        raise ValueError(f"gpu_id must be non-negative (got {gpu_id})")


def _require_kernel(config: GemmOptimizationConfig) -> Path:
    if config.kernel is None:
        raise ValueError("kernel is required when starting a new GEMM run")
    return config.kernel.expanduser().resolve()


def _codex_env(gpu_id: int | None, round_dir: Path) -> dict[str, str]:
    env = {
        "IS_SANDBOX": "1",
        "LUMEN_BENCHMARK_ROOT": str(round_dir / "datasets" / "inference"),
    }
    if gpu_id is not None:
        env["HIP_VISIBLE_DEVICES"] = str(gpu_id)
    return env


def _write_run_config(
    config: GemmOptimizationConfig,
    run_dir: Path,
    kernel: Path,
    prompt_files: Sequence[Path],
) -> None:
    prompts = [
        {"prompt_file": str(path), "prompt_sha256": sha256_file(path)}
        for path in prompt_files
    ]
    payload = {
        "kernel": str(kernel),
        "kernel_sha256": sha256_file(kernel),
        "prompt_file": prompts[0]["prompt_file"],
        "prompt_sha256": prompts[0]["prompt_sha256"],
        "prompts": prompts,
        "round_count": len(prompts),
        "entrypoint": config.entrypoint,
        "gpu_id": config.gpu_id,
        "workloads": list(GEMM_WORKLOADS),
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def _resolve_run_dir(repo_root: Path, configured: Path | None) -> Path:
    if configured is not None:
        path = configured.expanduser()
        return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    return repo_root / "runs" / f"lumen_gemm_codex_{timestamp}"


def _render_agents_md(repo_root: Path) -> str:
    scripts_root = repo_root / "scripts"
    skills_root = repo_root / "skills"
    benchmark = scripts_root / "benchmark" / "bench_gemm.py"
    return f"""
## Goal
Use `input_model.py` as the starting implementation and write the complete optimized implementation to `output_model_new.py`.

## Notes

- Do not read the parent repository except for the exact `scripts` and `skills` paths below. Do not read git history, other runs, or the network.
- You may read `AGENTS.md`, `prompt.txt`, `input_model.py`,
  `output_model_new.py`, `{scripts_root}/**`, `{skills_root}/**`, and
  `datasets/inference/gemm/lumen/model.py`.
- You can only write `output_model_new.py`.
- Preserve the kernel's public API.
- After writing `output_model_new.py`, validate correctness and performance with:
  `python {benchmark} --backend lumen --check-correctness --matrix-sizes 1024 2048 4096 8192 16384`.
  - A benchmark command that exits nonzero or omits `"correctness":true` has failed.
- Do not add eager PyTorch or external-library fallback compute paths.
"""


def _render_benchmark_adapter(entrypoint: str) -> str:
    return f'''#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn


_KERNEL_PATH = Path(__file__).resolve().parents[4] / "output_model_new.py"


def _load_kernel():
    spec = importlib.util.spec_from_file_location(
        "lumen_gemm_candidate",
        _KERNEL_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import candidate: {{_KERNEL_PATH}}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self._kernel = _load_kernel()
        self._out_cache = {{}}

    def build_call(self, *, a_mk: torch.Tensor, b_nk: torch.Tensor):
        key = (a_mk.shape, b_nk.shape, a_mk.device, a_mk.dtype)
        out = self._out_cache.get(key)
        if out is None:
            out = torch.empty(
                (a_mk.shape[0], b_nk.shape[0]),
                device=a_mk.device,
                dtype=a_mk.dtype,
            )
            self._out_cache[key] = out
        fn = getattr(self._kernel, {entrypoint!r})
        return lambda: fn(a_mk, b_nk, out=out)

    def forward(self, a_mk: torch.Tensor, b_nk: torch.Tensor) -> torch.Tensor:
        return self.build_call(a_mk=a_mk, b_nk=b_nk)()
'''


def _codex_provenance(config: GemmOptimizationConfig) -> dict[str, object]:
    return {
        "codex_bin": str(config.codex_bin) if config.codex_bin else None,
        "profile": config.profile,
        "model": config.model,
        "model_provider": config.model_provider,
        "reasoning_effort": config.reasoning_effort,
        "timeout_seconds": config.timeout_seconds,
        "config_overrides": list(config.config_overrides),
        "bypass_approvals_and_sandbox": config.bypass_approvals_and_sandbox,
    }


def _join_errors(*errors: str | None) -> str | None:
    values = [error for error in errors if error]
    return "\n".join(values) if values else None
