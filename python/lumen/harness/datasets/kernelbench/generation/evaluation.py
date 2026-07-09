"""Evaluation helpers for generated KernelBench kernels."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from lumen.harness.datasets.kernelbench.generation.artifacts import (
    sorted_prefixed_dirs,
    write_problem_meta,
    write_round_artifacts,
)
from lumen.harness.datasets.kernelbench.generation.metrics import format_eval_status
from lumen.harness.datasets.kernelbench.generation.types import (
    GenerationConfig,
    KernelBenchEvaluationConfig,
)

LOGGER = logging.getLogger(__name__)

def torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def evaluate_round(
    round_dir: str | Path,
    evaluation: KernelBenchEvaluationConfig,
    gpu_id: int,
) -> dict[str, Any]:
    path = Path(round_dir).expanduser().resolve()
    output_path = path / "eval_result.json"
    eval_config_path = path / "eval_config.runtime.json"
    source_config_path = path / "eval_config.json"
    if output_path.exists():
        output_path.unlink()

    eval_config: dict[str, Any] = {}
    if source_config_path.is_file():
        eval_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    eval_config["num_correct_trials"] = int(evaluation.num_correct_trials)
    eval_config["num_trials"] = int(evaluation.num_perf_trials)
    eval_config_path.write_text(
        json.dumps(eval_config, indent=2, default=str),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(gpu_id)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "lumen.tools.cli.kernelbench_graph_eval",
            "--mode",
            "generated",
            "--original",
            str(path / "input_model.py"),
            "--generated",
            str(path / "output_model_new.py"),
            "--eval-config",
            str(eval_config_path),
            "--json-output",
            str(output_path),
        ],
        cwd=path,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.stdout:
        LOGGER.info("%s", completed.stdout.rstrip())
    if completed.stderr:
        LOGGER.info("%s", completed.stderr.rstrip())

    error = "KernelBench eval subprocess failed before writing output"
    if output_path.is_file():
        try:
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            payload.setdefault("eval_exit_code", completed.returncode)
            output_path.write_text(
                json.dumps(payload, indent=2, default=str) + "\n",
                encoding="utf-8",
            )
            return payload
        except Exception as exc:
            LOGGER.warning("Failed to parse evaluation output JSON: %s", exc)
            error = f"KernelBench eval subprocess wrote invalid JSON: {exc}"

    payload = {
        "compiled": False,
        "correctness": False,
        "eval_exit_code": completed.returncode,
        "metadata": {
            "error": error,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        },
    }
    output_path.write_text(
        json.dumps(payload, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return payload


def run_eval_phase(
    config: GenerationConfig,
    dataset: Any,
    problem_ids: list[int],
    *,
    rounds_subdir: str | None = None,
    problem_meta_subdir: str | None = None,
) -> None:
    if not torch_cuda_available():
        LOGGER.warning("No CUDA/HIP device available; eval was skipped.")
        return

    run_dir = Path(config.run_dir)
    gpu_ids = config.evaluation.gpu_ids or (0,)
    to_eval: list[tuple[int, Path, int]] = []
    for pid in problem_ids:
        problem_dir = run_dir / f"p{pid:02d}"
        rounds_dir = (
            problem_dir / rounds_subdir if rounds_subdir is not None else problem_dir
        )
        problem_meta_dir = (
            problem_dir / problem_meta_subdir
            if problem_meta_subdir is not None
            else problem_dir
        )
        top_meta_path = problem_meta_dir / "meta.json"

        if top_meta_path.is_file():
            top_meta = json.loads(top_meta_path.read_text(encoding="utf-8"))
            if top_meta.get("stage") == "speedup_eval":
                continue

        for round_dir in reversed(sorted_prefixed_dirs(rounds_dir, "round")):
            if (round_dir / "output_model_new.py").is_file():
                gpu_id = gpu_ids[len(to_eval) % len(gpu_ids)]
                to_eval.append((pid, round_dir, gpu_id))
                break

    if not to_eval:
        LOGGER.info("Eval phase: nothing to evaluate.")
        return

    LOGGER.info("Evaluating %d kernel(s) on GPU(s) %s.", len(to_eval), list(gpu_ids))
    for pid, round_dir, gpu_id in to_eval:
        problem_dir = run_dir / f"p{pid:02d}"
        problem_meta_dir = (
            problem_dir / problem_meta_subdir
            if problem_meta_subdir is not None
            else problem_dir
        )
        problem = dataset.get_problem_by_id(pid)
        start_time = time.time()
        eval_payload = evaluate_round(round_dir, config.evaluation, gpu_id)
        elapsed = time.time() - start_time
        status = format_eval_status(eval_payload)
        LOGGER.info("p%02d: %s  (%.1fs) [gpu:%s]", pid, status, elapsed, gpu_id)

        write_round_artifacts(
            round_dir,
            problem_id=pid,
            problem_name=problem.name,
            error="",
            eval_payload=eval_payload,
        )
        metas = []
        for path in sorted_prefixed_dirs(round_dir.parent, "round"):
            meta_path = path / "meta.json"
            if meta_path.is_file():
                metas.append(json.loads(meta_path.read_text(encoding="utf-8")))
        write_problem_meta(
            problem_meta_dir,
            problem_id=pid,
            problem_name=problem.name,
            round_metas=metas,
        )
