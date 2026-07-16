"""Shared runtime support for isolated Lumen optimization rounds."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lumen.harness.backend.codex import (
    CodexRunner,
    CodexRunnerConfig,
    CodexRunResult,
)


@dataclass(frozen=True)
class OptimizationSpec:
    domain: str
    run_slug: str
    default_kernel: Path
    default_prompts: tuple[Path, ...]
    workloads: tuple[int, ...]
    workload_key: str
    workload_flag: str
    benchmark_script: str
    adapter_source: str


@dataclass(frozen=True)
class OptimizationConfig:
    repo_root: Path
    kernel: Path | None
    run_dir: Path | None = None
    gpu_id: int | None = None
    codex_bin: Path | None = None
    profile: str | None = None
    model: str | None = None
    model_provider: str | None = None
    reasoning_effort: str | None = None
    timeout_seconds: float | None = 3600
    validation_attempts: int = 1
    config_overrides: tuple[str, ...] = ()
    bypass_approvals_and_sandbox: bool = True


@dataclass(frozen=True)
class OptimizationWorkspace:
    run_dir: Path
    round_dir: Path
    input_model_path: Path
    output_model_path: Path
    prompt: str


def prepare_optimization(
    config: OptimizationConfig,
    spec: OptimizationSpec,
    prompt_file: Path,
) -> OptimizationWorkspace:
    repo_root = config.repo_root.expanduser().resolve()
    kernel = _require_kernel(config, spec)
    prompt = prompt_file.expanduser().resolve()
    _validate_inputs(
        repo_root,
        kernel,
        (prompt,),
        config.gpu_id,
        config.validation_attempts,
    )

    run_dir = _resolve_run_dir(repo_root, config.run_dir, spec.run_slug)
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_run_config(config, spec, run_dir, kernel, (prompt,))
    return _prepare_round(
        config,
        spec,
        run_dir=run_dir,
        round_index=0,
        kernel=kernel,
        prompt_file=prompt,
    )


def run_optimization_sequence(
    config: OptimizationConfig,
    spec: OptimizationSpec,
    prompt_files: Sequence[Path],
) -> list[tuple[OptimizationWorkspace, CodexRunResult]]:
    repo_root = config.repo_root.expanduser().resolve()
    kernel = _require_kernel(config, spec)
    prompts = tuple(path.expanduser().resolve() for path in prompt_files)
    if not prompts:
        raise ValueError("at least one prompt file is required")
    _validate_inputs(
        repo_root,
        kernel,
        prompts,
        config.gpu_id,
        config.validation_attempts,
    )

    run_dir = _resolve_run_dir(repo_root, config.run_dir, spec.run_slug)
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_run_config(config, spec, run_dir, kernel, prompts)
    return _run_rounds(
        config,
        spec,
        prompts=prompts,
        run_dir=run_dir,
        round_kernel=kernel,
        start_index=0,
    )


def resume_optimization_sequence(
    config: OptimizationConfig,
    spec: OptimizationSpec,
    prompt_files: Sequence[Path],
    run_dir: Path,
) -> list[tuple[OptimizationWorkspace, CodexRunResult]]:
    repo_root = config.repo_root.expanduser().resolve()
    resolved_run_dir = run_dir.expanduser().resolve()
    start_index, round_kernel = resolve_resume_point(resolved_run_dir)
    if prompt_files:
        prompts = tuple(path.expanduser().resolve() for path in prompt_files)
    else:
        prompts = remaining_run_prompts(resolved_run_dir, start_index)
    if not prompts:
        raise ValueError(
            f"run has no remaining prompts after round{start_index - 1}: "
            f"{resolved_run_dir}"
        )

    _validate_inputs(
        repo_root,
        round_kernel,
        prompts,
        config.gpu_id,
        config.validation_attempts,
    )
    resumed_config = replace(config, kernel=round_kernel, run_dir=resolved_run_dir)
    reset_run_tail(resolved_run_dir, start_index)
    append_run_prompts(resolved_run_dir, prompts, start_index=start_index)
    return _run_rounds(
        resumed_config,
        spec,
        prompts=prompts,
        run_dir=resolved_run_dir,
        round_kernel=round_kernel,
        start_index=start_index,
    )


def _prepare_round(
    config: OptimizationConfig,
    spec: OptimizationSpec,
    *,
    run_dir: Path,
    round_index: int,
    kernel: Path,
    prompt_file: Path,
) -> OptimizationWorkspace:
    prompt = prompt_file.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"prompt file is empty: {prompt_file}")

    round_dir = run_dir / f"round{round_index}"
    round_dir.mkdir()
    input_model_path = round_dir / "input_model.py"
    output_model_path = round_dir / "output_model_new.py"
    shutil.copyfile(kernel, input_model_path)
    output_model_path.write_text("", encoding="utf-8")

    adapter_dir = round_dir / "datasets" / "inference" / spec.domain / "lumen"
    adapter_dir.mkdir(parents=True)
    adapter_path = adapter_dir / "model.py"
    adapter_path.write_text(spec.adapter_source, encoding="utf-8")

    prompt_path = round_dir / "prompt.txt"
    prompt_path.write_text(prompt + "\n", encoding="utf-8")
    agents_path = round_dir / "AGENTS.md"
    agents_path.write_text(_render_agents_md(config.repo_root, spec), encoding="utf-8")
    write_round_config(
        round_dir,
        domain=spec.domain,
        round_index=round_index,
        input_source=kernel,
        input_model=input_model_path,
        prompt_source=prompt_file,
        prompt_path=prompt_path,
        agents_path=agents_path,
        adapter_path=adapter_path,
        gpu_id=config.gpu_id,
        workloads=spec.workloads,
        validation_attempts=config.validation_attempts,
        codex_config=_codex_provenance(config),
    )
    return OptimizationWorkspace(
        run_dir=run_dir,
        round_dir=round_dir,
        input_model_path=input_model_path,
        output_model_path=output_model_path,
        prompt=prompt,
    )


def _run_rounds(
    config: OptimizationConfig,
    spec: OptimizationSpec,
    *,
    prompts: tuple[Path, ...],
    run_dir: Path,
    round_kernel: Path,
    start_index: int,
) -> list[tuple[OptimizationWorkspace, CodexRunResult]]:
    total_rounds = start_index + len(prompts)
    rounds: list[tuple[OptimizationWorkspace, CodexRunResult]] = []
    for round_index, prompt_file in enumerate(prompts, start=start_index):
        workspace = _prepare_round(
            config,
            spec,
            run_dir=run_dir,
            round_index=round_index,
            kernel=round_kernel,
            prompt_file=prompt_file,
        )
        result = _run_workspace(config, spec, workspace)
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
    config: OptimizationConfig,
    spec: OptimizationSpec,
    workspace: OptimizationWorkspace,
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
            env=_codex_env(config.gpu_id),
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
    benchmark = (
        config.repo_root.resolve()
        / "scripts"
        / "benchmark"
        / spec.benchmark_script
    )
    evaluation = evaluate_candidate_pass_at_k(
        workspace.round_dir,
        domain=spec.domain,
        gpu_id=config.gpu_id,
        workloads=spec.workloads,
        workload_key=spec.workload_key,
        attempts=config.validation_attempts,
        benchmark_args=(
            str(benchmark),
            "--backend",
            "lumen",
            "--model-path",
            str(
                workspace.round_dir
                / "datasets"
                / "inference"
                / spec.domain
                / "lumen"
                / "model.py"
            ),
            "--check-correctness",
            spec.workload_flag,
            *(str(workload) for workload in spec.workloads),
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


def write_round_config(
    round_dir: Path,
    *,
    domain: str,
    round_index: int,
    input_source: Path,
    input_model: Path,
    prompt_source: Path,
    prompt_path: Path,
    agents_path: Path,
    adapter_path: Path,
    gpu_id: int | None,
    workloads: tuple[int, ...],
    validation_attempts: int,
    codex_config: dict[str, Any],
) -> None:
    payload = {
        "domain": domain,
        "round_index": round_index,
        "prepared_at_utc": _utc_now(),
        "input_source": str(input_source),
        "input_model": str(input_model),
        "input_model_sha256": sha256_file(input_model),
        "prompt_source": str(prompt_source),
        "prompt_file": str(prompt_path),
        "prompt_sha256": sha256_file(prompt_path),
        "agents_sha256": sha256_file(agents_path),
        "adapter_sha256": sha256_file(adapter_path),
        "gpu_id": gpu_id,
        "workloads": list(workloads),
        "validation_attempts": validation_attempts,
        "codex": codex_config,
    }
    _write_json(round_dir / "run_config.json", payload)
    set_round_status(round_dir, "prepared")


def set_round_status(
    round_dir: Path,
    status: str,
    *,
    error: str | None = None,
) -> None:
    path = round_dir / "round_status.json"
    payload = {
        "status": status,
        "updated_at_utc": _utc_now(),
    }
    if error:
        payload["error"] = error
    _write_json(path, payload)


def save_codex_result(
    round_dir: Path,
    output_path: Path,
    result: CodexRunResult,
) -> CodexRunResult:
    saved_result = result
    if result.trace_path:
        source = Path(result.trace_path).expanduser()
        destination = round_dir / "trace.jsonl"
        try:
            if source.is_file() and source.resolve() != destination.resolve():
                shutil.copy2(source, destination)
                saved_result = replace(result, trace_path=str(destination))
        except OSError as exc:
            saved_result = replace(
                result,
                error=_join_errors(result.error, f"Failed to copy trace: {exc}"),
            )
    _write_json(round_dir / "codex_result.json", asdict(saved_result))

    config_path = round_dir / "run_config.json"
    config = _read_json(config_path)
    if output_path.is_file():
        config["output_model"] = str(output_path)
        config["output_model_sha256"] = sha256_file(output_path)
        config["output_model_bytes"] = output_path.stat().st_size
    config["codex_result"] = {
        "status": saved_result.status,
        "session_id": saved_result.session_id,
        "started_at_utc": saved_result.started_at_utc,
        "finished_at_utc": saved_result.finished_at_utc,
    }
    _write_json(config_path, config)
    return saved_result


def evaluate_candidate(
    round_dir: Path,
    *,
    domain: str,
    gpu_id: int | None,
    workloads: tuple[int, ...],
    workload_key: str,
    benchmark_args: tuple[str, ...],
    timeout_seconds: float = 900,
    output_path: Path | None = None,
    attempt_index: int | None = None,
    attempt_count: int | None = None,
) -> dict[str, Any]:
    started_at = _utc_now()
    command = [sys.executable, *benchmark_args]
    payload: dict[str, Any] = {
        "domain": domain,
        "started_at_utc": started_at,
        "gpu_id": gpu_id,
        "command": command,
        "expected_workloads": list(workloads),
        "workload_key": workload_key,
        "formal_evaluation_ran": False,
        "records": [],
        "workloads": {},
        "exit_code": None,
        "stdout": "",
        "stderr": "",
        "correctness": False,
        "ok": False,
    }
    if attempt_index is not None:
        payload["attempt_index"] = attempt_index
    if attempt_count is not None:
        payload["attempt_count"] = attempt_count
    env = os.environ.copy()
    env["IS_SANDBOX"] = "1"
    if gpu_id is not None:
        env["HIP_VISIBLE_DEVICES"] = str(gpu_id)
    payload["formal_evaluation_ran"] = True
    try:
        completed = subprocess.run(
            command,
            cwd=round_dir,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
        payload["exit_code"] = completed.returncode
        payload["stdout"] = completed.stdout
        payload["stderr"] = completed.stderr
        records = _parse_jsonl_records(completed.stdout, domain=domain)
        by_workload = _records_by_workload(records, workload_key=workload_key)
        expected = {str(workload) for workload in workloads}
        correctness = (
            completed.returncode == 0
            and expected.issubset(by_workload)
            and all(by_workload[key].get("correctness") is True for key in expected)
        )
        payload["records"] = records
        payload["workloads"] = by_workload
        payload["correctness"] = correctness
        payload["ok"] = correctness
        if not correctness:
            missing = sorted(expected - set(by_workload), key=int)
            benchmark_error = _benchmark_error(completed.stderr)
            payload["error"] = (
                "Formal correctness evaluation failed"
                + (f"; missing workloads: {missing}" if missing else "")
                + (f"; exit code: {completed.returncode}" if completed.returncode else "")
                + (f"; benchmark error: {benchmark_error}" if benchmark_error else "")
            )
    except subprocess.TimeoutExpired as exc:
        payload["error"] = f"Formal evaluation timed out after {timeout_seconds}s."
        payload["stdout"] = _coerce_text(exc.stdout)
        payload["stderr"] = _coerce_text(exc.stderr)
    except OSError as exc:
        payload["error"] = f"Failed to run formal evaluation: {exc}"

    payload["finished_at_utc"] = _utc_now()
    _write_json(output_path or round_dir / "eval_result.json", payload)
    return payload


def evaluate_candidate_pass_at_k(
    round_dir: Path,
    *,
    domain: str,
    gpu_id: int | None,
    workloads: tuple[int, ...],
    workload_key: str,
    benchmark_args: tuple[str, ...],
    attempts: int,
    timeout_seconds: float = 900,
) -> dict[str, Any]:
    if attempts < 1:
        raise ValueError(f"validation attempts must be positive (got {attempts})")

    if attempts == 1:
        result = evaluate_candidate(
            round_dir,
            domain=domain,
            gpu_id=gpu_id,
            workloads=workloads,
            workload_key=workload_key,
            benchmark_args=benchmark_args,
            timeout_seconds=timeout_seconds,
            attempt_index=1,
            attempt_count=1,
        )
        result["pass_at_k"] = {
            "k": attempts,
            "attempts_run": 1,
            "passed_attempt": 1 if result["ok"] else None,
        }
        result["attempts"] = [
            _evaluation_attempt_summary(result, round_dir / "eval_result.json")
        ]
        _write_json(round_dir / "eval_result.json", result)
        return result

    passed: dict[str, Any] | None = None
    last_result: dict[str, Any] | None = None
    summaries: list[dict[str, Any]] = []
    for attempt_index in range(1, attempts + 1):
        output_path = round_dir / f"eval_attempt_{attempt_index:03d}.json"
        result = evaluate_candidate(
            round_dir,
            domain=domain,
            gpu_id=gpu_id,
            workloads=workloads,
            workload_key=workload_key,
            benchmark_args=benchmark_args,
            timeout_seconds=timeout_seconds,
            output_path=output_path,
            attempt_index=attempt_index,
            attempt_count=attempts,
        )
        summaries.append(_evaluation_attempt_summary(result, output_path))
        last_result = result
        if result["ok"]:
            passed = result
            break

    aggregate = dict(passed or last_result or {})
    aggregate["ok"] = passed is not None
    aggregate["pass_at_k"] = {
        "k": attempts,
        "attempts_run": len(summaries),
        "passed_attempt": passed.get("attempt_index") if passed else None,
    }
    aggregate["attempts"] = summaries
    _write_json(round_dir / "eval_result.json", aggregate)
    return aggregate


def _evaluation_attempt_summary(
    result: dict[str, Any],
    path: Path,
) -> dict[str, Any]:
    return {
        "path": str(path),
        "attempt_index": result.get("attempt_index"),
        "ok": result.get("ok") is True,
        "correctness": result.get("correctness") is True,
        "exit_code": result.get("exit_code"),
        "error": result.get("error"),
        "started_at_utc": result.get("started_at_utc"),
        "finished_at_utc": result.get("finished_at_utc"),
    }


def evaluation_error(payload: dict[str, Any]) -> str:
    error = payload.get("error")
    if error:
        return str(error)
    return "Formal evaluation failed."


def resolve_resume_point(run_dir: Path) -> tuple[int, Path]:
    if not run_dir.is_dir():
        raise ValueError(f"resume run directory not found: {run_dir}")

    passed: list[tuple[int, Path]] = []
    for path in run_dir.glob("round*"):
        suffix = path.name.removeprefix("round")
        if not path.is_dir() or not suffix.isdigit():
            continue
        status = _read_json(path / "round_status.json").get("status")
        output = path / "output_model_new.py"
        if status == "passed" and output.is_file():
            if output.read_text(encoding="utf-8").strip():
                passed.append((int(suffix), output))

    if passed:
        last_index, output = max(passed)
        return last_index + 1, output

    run_config = _read_json(run_dir / "run_config.json")
    kernel_value = run_config.get("kernel")
    if not isinstance(kernel_value, str):
        raise ValueError(f"resume run has no original kernel: {run_dir}")
    kernel = Path(kernel_value).expanduser()
    if not kernel.is_absolute():
        kernel = kernel.resolve()
    if not kernel.is_file():
        raise ValueError(f"resume run original kernel not found: {kernel}")
    return 0, kernel


def reset_run_tail(run_dir: Path, start_index: int) -> None:
    for path in run_dir.glob("round*"):
        suffix = path.name.removeprefix("round")
        if path.is_dir() and suffix.isdigit() and int(suffix) >= start_index:
            shutil.rmtree(path)


def remaining_run_prompts(run_dir: Path, start_index: int) -> tuple[Path, ...]:
    payload = _read_json(run_dir / "run_config.json")
    configured = payload.get("prompts")
    if not isinstance(configured, list):
        raise ValueError(f"resume run has no prompt sequence: {run_dir}")

    prompts: list[Path] = []
    for round_index, entry in enumerate(configured[start_index:], start=start_index):
        value = entry.get("prompt_file") if isinstance(entry, dict) else None
        if not isinstance(value, str):
            raise ValueError(
                f"resume run has an invalid prompt for round{round_index}: {run_dir}"
            )
        prompt = Path(value).expanduser()
        prompts.append(prompt.resolve())
    return tuple(prompts)


def append_run_prompts(
    run_dir: Path,
    prompt_files: tuple[Path, ...],
    *,
    start_index: int,
) -> None:
    path = run_dir / "run_config.json"
    payload = _read_json(path)
    prompts = list(payload.get("prompts", []))[:start_index]
    prompts.extend(
        {"prompt_file": str(prompt), "prompt_sha256": sha256_file(prompt)}
        for prompt in prompt_files
    )
    payload["prompts"] = prompts
    payload["round_count"] = len(prompts)
    if prompts:
        payload["prompt_file"] = prompts[0]["prompt_file"]
        payload["prompt_sha256"] = prompts[0]["prompt_sha256"]
    _write_json(path, payload)


def _validate_inputs(
    repo_root: Path,
    kernel: Path,
    prompt_files: Sequence[Path],
    gpu_id: int | None,
    validation_attempts: int,
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
    if validation_attempts < 1:
        raise ValueError(
            f"validation_attempts must be positive (got {validation_attempts})"
        )


def _require_kernel(config: OptimizationConfig, spec: OptimizationSpec) -> Path:
    if config.kernel is None:
        raise ValueError(
            f"kernel is required when starting a new {spec.domain} run"
        )
    return config.kernel.expanduser().resolve()


def _codex_env(gpu_id: int | None) -> dict[str, str]:
    env = {"IS_SANDBOX": "1"}
    if gpu_id is not None:
        env["HIP_VISIBLE_DEVICES"] = str(gpu_id)
    return env


def _write_run_config(
    config: OptimizationConfig,
    spec: OptimizationSpec,
    run_dir: Path,
    kernel: Path,
    prompt_files: Sequence[Path],
) -> None:
    prompts = [
        {"prompt_file": str(path), "prompt_sha256": sha256_file(path)}
        for path in prompt_files
    ]
    _write_json(
        run_dir / "run_config.json",
        {
            "domain": spec.domain,
            "kernel": str(kernel),
            "kernel_sha256": sha256_file(kernel),
            "prompt_file": prompts[0]["prompt_file"],
            "prompt_sha256": prompts[0]["prompt_sha256"],
            "prompts": prompts,
            "round_count": len(prompts),
            "gpu_id": config.gpu_id,
            "workloads": list(spec.workloads),
            "validation_attempts": config.validation_attempts,
        },
    )


def _resolve_run_dir(
    repo_root: Path,
    configured: Path | None,
    run_slug: str,
) -> Path:
    if configured is not None:
        path = configured.expanduser()
        return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    return repo_root / "runs" / f"lumen_{run_slug}_codex_{timestamp}"


def _render_agents_md(repo_root: Path, spec: OptimizationSpec) -> str:
    scripts_root = repo_root.resolve() / "scripts"
    skills_root = repo_root.resolve() / "skills"
    benchmark = scripts_root / "benchmark" / spec.benchmark_script
    workloads = " ".join(str(workload) for workload in spec.workloads)
    model_path = f"datasets/inference/{spec.domain}/lumen/model.py"
    return f"""
## Goal
Use `input_model.py` as the starting implementation and write the complete optimized implementation to `output_model_new.py`.

## Notes

- Do not read the parent repository except for the exact `scripts` and `skills` paths below. Do not read git history, other runs, or the network.
- You may read `AGENTS.md`, `prompt.txt`, `input_model.py`,
  `output_model_new.py`, `{scripts_root}/**`, `{skills_root}/**`, and
  `{model_path}`.
- You can only write `output_model_new.py`.
- Preserve the kernel's public API.
- After writing `output_model_new.py`, validate correctness and performance with:
  `python {benchmark} --backend lumen --model-path {model_path} --check-correctness {spec.workload_flag} {workloads}`.
  - A benchmark command that exits nonzero or omits `"correctness":true` has failed.
- Do not add eager PyTorch or external-library fallback compute paths.
"""


def _codex_provenance(config: OptimizationConfig) -> dict[str, object]:
    return {
        "codex_bin": str(config.codex_bin) if config.codex_bin else None,
        "profile": config.profile,
        "model": config.model,
        "model_provider": config.model_provider,
        "reasoning_effort": config.reasoning_effort,
        "timeout_seconds": config.timeout_seconds,
        "validation_attempts": config.validation_attempts,
        "config_overrides": list(config.config_overrides),
        "bypass_approvals_and_sandbox": config.bypass_approvals_and_sandbox,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_jsonl_records(stdout: str, *, domain: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("domain") == domain:
            records.append(value)
    return records


def _records_by_workload(
    records: list[dict[str, Any]],
    *,
    workload_key: str,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        value = record.get(workload_key)
        if isinstance(value, int):
            result[str(value)] = record
    return result


def _benchmark_error(stderr: str) -> str | None:
    for line in reversed(stderr.splitlines()):
        stripped = line.strip()
        if stripped.startswith(("AssertionError:", "RuntimeError:", "ValueError:")):
            return stripped
    return None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def _coerce_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode(errors="replace") if isinstance(value, bytes) else value


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _join_errors(*errors: str | None) -> str | None:
    values = [error for error in errors if error]
    return "\n".join(values) if values else None
