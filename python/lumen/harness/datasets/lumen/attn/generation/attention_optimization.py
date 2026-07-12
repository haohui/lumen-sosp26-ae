"""Prepare and run an isolated Codex workspace for Attention optimization."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from lumen.harness.backend.codex import CodexRunner, CodexRunnerConfig, CodexRunResult

ATTENTION_WORKLOADS = (1024, 2048, 4096, 8192, 16384)


@dataclass(frozen=True)
class AttentionOptimizationConfig:
    repo_root: Path
    kernel: Path
    prompt_file: Path
    run_dir: Path | None = None
    entrypoint: str = "flash_attn"
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
class AttentionOptimizationWorkspace:
    run_dir: Path
    round_dir: Path
    input_model_path: Path
    output_model_path: Path
    prompt: str


def prepare_attention_optimization(
    config: AttentionOptimizationConfig,
) -> AttentionOptimizationWorkspace:
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
    config: AttentionOptimizationConfig,
    *,
    run_dir: Path,
    round_index: int,
    kernel: Path,
    prompt_file: Path,
) -> AttentionOptimizationWorkspace:
    repo_root = config.repo_root.expanduser().resolve()
    prompt = prompt_file.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"prompt file is empty: {prompt_file}")

    round_dir = run_dir / f"round{round_index}"
    round_dir.mkdir()

    _copy_directory(repo_root / "scripts", round_dir / "scripts")
    _copy_directory(repo_root / "skills", round_dir / "skills")
    input_model_path = round_dir / "input_model.py"
    output_model_path = round_dir / "output_model_new.py"
    shutil.copy2(kernel, input_model_path)
    output_model_path.write_text("", encoding="utf-8")

    adapter_dir = round_dir / "datasets" / "inference" / "attention" / "lumen"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "model.py").write_text(
        _render_benchmark_adapter(config.entrypoint),
        encoding="utf-8",
    )

    (round_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
    (round_dir / "AGENTS.md").write_text(
        _render_agents_md(config.gpu_id),
        encoding="utf-8",
    )
    return AttentionOptimizationWorkspace(
        run_dir=run_dir,
        round_dir=round_dir,
        input_model_path=input_model_path,
        output_model_path=output_model_path,
        prompt=prompt,
    )


def run_attention_optimization(
    config: AttentionOptimizationConfig,
) -> tuple[AttentionOptimizationWorkspace, CodexRunResult]:
    results = run_attention_optimization_sequence(config, (config.prompt_file,))
    return results[0]


def run_attention_optimization_sequence(
    config: AttentionOptimizationConfig,
    prompt_files: Sequence[Path],
) -> list[tuple[AttentionOptimizationWorkspace, CodexRunResult]]:
    repo_root = config.repo_root.expanduser().resolve()
    kernel = config.kernel.expanduser().resolve()
    prompts = tuple(path.expanduser().resolve() for path in prompt_files)
    if not prompts:
        raise ValueError("at least one prompt file is required")
    _validate_inputs(repo_root, kernel, prompts, config.gpu_id)

    run_dir = _resolve_run_dir(repo_root, config.run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_run_config(config, run_dir, kernel, prompts)

    rounds: list[tuple[AttentionOptimizationWorkspace, CodexRunResult]] = []
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
        if not result.ok:
            break
        round_kernel = workspace.output_model_path
    return rounds


def _run_workspace(
    config: AttentionOptimizationConfig,
    workspace: AttentionOptimizationWorkspace,
) -> CodexRunResult:
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
            env=_codex_env(config.gpu_id),
            config_overrides=config.config_overrides,
            bypass_approvals_and_sandbox=config.bypass_approvals_and_sandbox,
        )
    )
    if result.ok and not _is_nonempty_file(workspace.output_model_path):
        result = replace(
            result,
            ok=False,
            status="failed",
            error=_join_errors(
                result.error,
                "Codex completed without producing a non-empty output_model_new.py.",
            ),
        )
    result = _save_result(workspace, result)
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


def _codex_env(gpu_id: int | None) -> dict[str, str]:
    env = {"IS_SANDBOX": "1"}
    if gpu_id is not None:
        env["HIP_VISIBLE_DEVICES"] = str(gpu_id)
    return env


def _write_run_config(
    config: AttentionOptimizationConfig,
    run_dir: Path,
    kernel: Path,
    prompt_files: Sequence[Path],
) -> None:
    prompts = [
        {
            "prompt_file": str(path),
            "prompt_sha256": _sha256(path),
        }
        for path in prompt_files
    ]
    payload = {
        "kernel": str(kernel),
        "kernel_sha256": _sha256(kernel),
        "prompt_file": prompts[0]["prompt_file"],
        "prompt_sha256": prompts[0]["prompt_sha256"],
        "prompts": prompts,
        "round_count": len(prompts),
        "entrypoint": config.entrypoint,
        "gpu_id": config.gpu_id,
        "workloads": list(ATTENTION_WORKLOADS),
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def _is_nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _resolve_run_dir(repo_root: Path, configured: Path | None) -> Path:
    if configured is not None:
        path = configured.expanduser()
        return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    return repo_root / "runs" / f"lumen_attn_codex_{timestamp}"


def _copy_directory(source: Path, destination: Path) -> None:
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )


def _save_result(
    workspace: AttentionOptimizationWorkspace,
    result: CodexRunResult,
) -> CodexRunResult:
    if result.trace_path:
        source = Path(result.trace_path).expanduser()
        destination = workspace.round_dir / "trace.jsonl"
        try:
            if source.is_file() and source.resolve() != destination.resolve():
                shutil.copy2(source, destination)
                result = replace(result, trace_path=str(destination))
        except OSError as exc:
            result = replace(
                result,
                error=_join_errors(result.error, f"Failed to copy trace: {exc}"),
            )
    (workspace.round_dir / "codex_result.json").write_text(
        json.dumps(asdict(result), indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def _render_agents_md(gpu_id: int | None) -> str:
    gpu_prefix = f"HIP_VISIBLE_DEVICES={gpu_id} " if gpu_id is not None else ""
    gpu_note = (
        f"- Use physical GPU {gpu_id} for all GPU work. Keep "
        f"`HIP_VISIBLE_DEVICES={gpu_id}` on benchmark commands.\n"
        if gpu_id is not None
        else ""
    )
    return f"""
## Goal
Use `input_model.py` as the starting implementation and write the complete optimized implementation to `output_model_new.py`.

## Notes

- Do not read files outside of this directory. Do not read the parent repository, git history, other runs, or the network.
- You may read `AGENTS.md`, `prompt.txt`, `input_model.py`,
  `output_model_new.py`, `scripts/**`, `skills/**`, and
  `datasets/inference/attention/lumen/model.py`.
- You can only write `output_model_new.py`.
- Preserve the kernel's public API.
{gpu_note}- After writing `output_model_new.py`, validate correctness and performance with:
  `{gpu_prefix}python scripts/benchmark/bench_attn.py --backend lumen --check-correctness --seq-lens 1024 2048 4096 8192 16384`.
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
        "lumen_attention_candidate",
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

    def build_call(
        self,
        *,
        q_bshd: torch.Tensor,
        k_bshd: torch.Tensor,
        v_bshd: torch.Tensor,
    ):
        batch_size, seq_len, num_q_heads, head_dim = q_bshd.shape
        q = q_bshd.reshape(batch_size * seq_len, num_q_heads, head_dim).contiguous()
        k = k_bshd.reshape(batch_size * seq_len, k_bshd.shape[2], head_dim).contiguous()
        v = v_bshd.reshape(batch_size * seq_len, v_bshd.shape[2], head_dim).contiguous()
        seq_ptr = torch.arange(
            batch_size + 1,
            device=q.device,
            dtype=torch.int32,
        ).mul_(seq_len)
        key = (q.shape, q.device, q.dtype)
        out = self._out_cache.get(key)
        if out is None:
            out = torch.empty_like(q)
            self._out_cache[key] = out
        fn = getattr(self._kernel, {entrypoint!r})
        return lambda: fn(q, k, v, seq_ptr, seq_len, out=out)

    def forward(
        self,
        q_bshd: torch.Tensor,
        k_bshd: torch.Tensor,
        v_bshd: torch.Tensor,
    ) -> torch.Tensor:
        call = self.build_call(q_bshd=q_bshd, k_bshd=k_bshd, v_bshd=v_bshd)
        call()
        batch_size, seq_len, num_q_heads, head_dim = q_bshd.shape
        flat_shape = (batch_size * seq_len, num_q_heads, head_dim)
        out = self._out_cache[(flat_shape, q_bshd.device, q_bshd.dtype)]
        return out.reshape(batch_size, seq_len, num_q_heads, head_dim)
'''


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _join_errors(*errors: str | None) -> str | None:
    values = [error for error in errors if error]
    return "\n".join(values) if values else None
