"""Shared runtime support for isolated Lumen optimization rounds."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lumen.harness.backend.codex import CodexRunResult


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
    entrypoint: str,
    gpu_id: int | None,
    workloads: tuple[int, ...],
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
        "entrypoint": entrypoint,
        "gpu_id": gpu_id,
        "workloads": list(workloads),
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
    benchmark_root: Path,
    workloads: tuple[int, ...],
    workload_key: str,
    benchmark_args: tuple[str, ...],
    timeout_seconds: float = 900,
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
    env = os.environ.copy()
    env["IS_SANDBOX"] = "1"
    env["LUMEN_BENCHMARK_ROOT"] = str(benchmark_root)
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
            payload["error"] = (
                "Formal correctness evaluation failed"
                + (f"; missing workloads: {missing}" if missing else "")
                + (f"; exit code: {completed.returncode}" if completed.returncode else "")
            )
    except subprocess.TimeoutExpired as exc:
        payload["error"] = f"Formal evaluation timed out after {timeout_seconds}s."
        payload["stdout"] = _coerce_text(exc.stdout)
        payload["stderr"] = _coerce_text(exc.stderr)
    except OSError as exc:
        payload["error"] = f"Failed to run formal evaluation: {exc}"

    payload["finished_at_utc"] = _utc_now()
    _write_json(round_dir / "eval_result.json", payload)
    return payload


def evaluation_error(payload: dict[str, Any]) -> str:
    error = payload.get("error")
    if error:
        return str(error)
    return "Formal evaluation failed."


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
