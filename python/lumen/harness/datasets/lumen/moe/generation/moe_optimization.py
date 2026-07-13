"""Prepare and run an isolated Codex workspace for MoE optimization."""

from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from lumen.harness.backend.codex import CodexRunner, CodexRunnerConfig, CodexRunResult
from lumen.harness.datasets.lumen.optimization_runtime import (
    evaluate_candidate,
    evaluation_error,
    save_codex_result,
    set_round_status,
    sha256_file,
    write_round_config,
)

MOE_WORKLOADS = (1024, 2048, 4096, 8192, 16384)


@dataclass(frozen=True)
class MoEOptimizationConfig:
    repo_root: Path
    kernel: Path
    prompt_file: Path
    run_dir: Path | None = None
    entrypoint: str = "fused_moe_fp8_blockscale_g1u1"
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
class MoEOptimizationWorkspace:
    run_dir: Path
    round_dir: Path
    input_model_path: Path
    output_model_path: Path
    prompt: str


def prepare_moe_optimization(
    config: MoEOptimizationConfig,
) -> MoEOptimizationWorkspace:
    repo_root = config.repo_root.expanduser().resolve()
    kernel = config.kernel.expanduser().resolve()
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
    config: MoEOptimizationConfig,
    *,
    run_dir: Path,
    round_index: int,
    kernel: Path,
    prompt_file: Path,
) -> MoEOptimizationWorkspace:
    repo_root = config.repo_root.expanduser().resolve()
    prompt = prompt_file.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"prompt file is empty: {prompt_file}")

    round_dir = run_dir / f"round{round_index}"
    round_dir.mkdir()

    input_model_path = round_dir / "input_model.py"
    output_model_path = round_dir / "output_model_new.py"
    shutil.copy2(kernel, input_model_path)
    output_model_path.write_text("", encoding="utf-8")

    adapter_dir = round_dir / "datasets" / "inference" / "moe" / "lumen"
    adapter_dir.mkdir(parents=True)
    adapter_path = adapter_dir / "model.py"
    adapter_path.write_text(
        _render_benchmark_adapter(repo_root, config.entrypoint),
        encoding="utf-8",
    )

    prompt_path = round_dir / "prompt.txt"
    prompt_path.write_text(prompt + "\n", encoding="utf-8")
    agents_path = round_dir / "AGENTS.md"
    agents_path.write_text(
        _render_agents_md(repo_root),
        encoding="utf-8",
    )
    write_round_config(
        round_dir,
        domain="moe",
        round_index=round_index,
        input_source=kernel,
        input_model=input_model_path,
        prompt_source=prompt_file,
        prompt_path=prompt_path,
        agents_path=agents_path,
        adapter_path=adapter_path,
        entrypoint=config.entrypoint,
        gpu_id=config.gpu_id,
        workloads=MOE_WORKLOADS,
        codex_config=_codex_provenance(config),
    )
    return MoEOptimizationWorkspace(
        run_dir=run_dir,
        round_dir=round_dir,
        input_model_path=input_model_path,
        output_model_path=output_model_path,
        prompt=prompt,
    )


def run_moe_optimization(
    config: MoEOptimizationConfig,
) -> tuple[MoEOptimizationWorkspace, CodexRunResult]:
    results = run_moe_optimization_sequence(config, (config.prompt_file,))
    return results[0]


def run_moe_optimization_sequence(
    config: MoEOptimizationConfig,
    prompt_files: Sequence[Path],
) -> list[tuple[MoEOptimizationWorkspace, CodexRunResult]]:
    repo_root = config.repo_root.expanduser().resolve()
    kernel = config.kernel.expanduser().resolve()
    prompts = tuple(path.expanduser().resolve() for path in prompt_files)
    if not prompts:
        raise ValueError("at least one prompt file is required")
    _validate_inputs(repo_root, kernel, prompts, config.gpu_id)

    run_dir = _resolve_run_dir(repo_root, config.run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_run_config(config, run_dir, kernel, prompts)

    rounds: list[tuple[MoEOptimizationWorkspace, CodexRunResult]] = []
    round_kernel = kernel
    for round_index, prompt_file in enumerate(prompts):
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
            f"round {round_index + 1}/{len(prompts)} {status}: "
            f"{workspace.round_dir}",
            flush=True,
        )
        if not result.ok:
            break
        round_kernel = workspace.output_model_path
    return rounds


def _run_workspace(
    config: MoEOptimizationConfig,
    workspace: MoEOptimizationWorkspace,
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
        domain="moe",
        gpu_id=config.gpu_id,
        benchmark_root=workspace.round_dir / "datasets" / "inference",
        workloads=MOE_WORKLOADS,
        workload_key="tokens",
        benchmark_args=(
            str(config.repo_root.resolve() / "scripts" / "benchmark" / "bench_moe.py"),
            "--backend",
            "lumen",
            "--check-correctness",
            "--tokens",
            *(str(workload) for workload in MOE_WORKLOADS),
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


def _codex_env(gpu_id: int | None, round_dir: Path) -> dict[str, str]:
    env = {
        "IS_SANDBOX": "1",
        "LUMEN_BENCHMARK_ROOT": str(round_dir / "datasets" / "inference"),
    }
    if gpu_id is not None:
        env["HIP_VISIBLE_DEVICES"] = str(gpu_id)
    return env


def _write_run_config(
    config: MoEOptimizationConfig,
    run_dir: Path,
    kernel: Path,
    prompt_files: Sequence[Path],
) -> None:
    prompts = [
        {
            "prompt_file": str(path),
            "prompt_sha256": sha256_file(path),
        }
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
        "workloads": list(MOE_WORKLOADS),
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
    return repo_root / "runs" / f"lumen_moe_codex_{timestamp}"


def _render_agents_md(repo_root: Path) -> str:
    scripts_root = repo_root / "scripts"
    skills_root = repo_root / "skills"
    benchmark = scripts_root / "benchmark" / "bench_moe.py"
    return f"""
## Goal
Use `input_model.py` as the starting implementation and write the complete optimized implementation to `output_model_new.py`.

## Notes

- Do not read the parent repository except for the exact `scripts` and `skills` paths below. Do not read git history, other runs, or the network.
- You may read `AGENTS.md`, `prompt.txt`, `input_model.py`,
  `output_model_new.py`, `{scripts_root}/**`, `{skills_root}/**`, and
  `datasets/inference/moe/lumen/model.py`.
- You can only write `output_model_new.py`.
- Preserve the kernel's public API.
- After writing `output_model_new.py`, validate correctness and performance with:
  `python {benchmark} --backend lumen --check-correctness --tokens 1024 2048 4096 8192 16384`.
  - A benchmark command that exits nonzero or omits `"correctness":true` has failed.
- Do not add eager PyTorch or external-library fallback compute paths.
"""


def _render_benchmark_adapter(repo_root: Path, entrypoint: str) -> str:
    template_path = repo_root / "datasets" / "inference" / "moe" / "lumen" / "model.py"
    source = template_path.read_text(encoding="utf-8")
    module_location = (
        '_THIS_DIR = Path(__file__).resolve().parent\n'
        '_MOE_MODULE = "fused_moe.py"'
    )
    candidate_location = (
        '_THIS_DIR = Path(__file__).resolve().parents[4]\n'
        '_MOE_MODULE = "output_model_new.py"'
    )
    if module_location not in source:
        raise ValueError(f"unexpected MoE benchmark adapter layout: {template_path}")
    source = source.replace(module_location, candidate_location, 1)

    default_lookup = (
        'getattr(self._kernel_module(), "fused_moe_fp8_blockscale_g1u1")'
    )
    if default_lookup not in source:
        raise ValueError(f"MoE entrypoint lookup not found: {template_path}")
    return source.replace(
        default_lookup,
        f"getattr(self._kernel_module(), {entrypoint!r})",
        1,
    )


def _codex_provenance(config: MoEOptimizationConfig) -> dict[str, object]:
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
