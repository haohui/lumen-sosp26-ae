"""Evaluation helpers for generated KernelBench kernels."""

from __future__ import annotations

import json
import logging
import time
import traceback
from contextlib import redirect_stdout
from io import StringIO
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
    from lumen.harness.datasets.kernelbench.evaluator import evaluate_generated_model

    path = Path(round_dir)
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

    log_buffer = StringIO()
    try:
        _set_cuda_device(gpu_id)
        with redirect_stdout(log_buffer):
            result = evaluate_generated_model(
                original_model_file=path / "input_model.py",
                generated_model_file=path / "output_model_new.py",
                eval_config=eval_config,
            )
    except Exception as exc:
        logs = log_buffer.getvalue()
        if logs:
            LOGGER.info("%s", logs.rstrip())
        return {
            "compiled": False,
            "correctness": False,
            "metadata": {
                "error": str(exc),
                "error_name": f"{exc.__class__.__module__}.{exc.__class__.__name__}",
                "traceback": traceback.format_exc(),
            },
        }

    logs = log_buffer.getvalue()
    if logs:
        LOGGER.info("%s", logs.rstrip())
    payload = result.model_dump()
    exit_code = 0 if result.compiled and result.correctness else 1
    payload.setdefault("eval_exit_code", exit_code)
    output_path.write_text(
        json.dumps(payload, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return payload


def run_eval_phase(
    config: GenerationConfig,
    dataset: Any,
    problem_ids: list[int],
) -> None:
    if not torch_cuda_available():
        LOGGER.warning("No CUDA/HIP device available; eval was skipped.")
        return

    run_dir = Path(config.run_dir)
    gpu_ids = config.evaluation.gpu_ids or (0,)
    to_eval: list[tuple[int, Path, int]] = []
    for pid in problem_ids:
        problem_dir = run_dir / f"p{pid:02d}"
        top_meta_path = problem_dir / "meta.json"

        if top_meta_path.is_file():
            top_meta = json.loads(top_meta_path.read_text(encoding="utf-8"))
            if top_meta.get("stage") == "speedup_eval":
                continue

        for round_dir in reversed(sorted_prefixed_dirs(problem_dir, "round")):
            if (round_dir / "output_model_new.py").is_file():
                gpu_id = gpu_ids[len(to_eval) % len(gpu_ids)]
                to_eval.append((pid, round_dir, gpu_id))
                break

    if not to_eval:
        LOGGER.info("Eval phase: nothing to evaluate.")
        return

    LOGGER.info("Evaluating %d kernel(s) on GPU(s) %s.", len(to_eval), list(gpu_ids))
    for pid, round_dir, gpu_id in to_eval:
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
            round_dir.parent,
            problem_id=pid,
            problem_name=problem.name,
            round_metas=metas,
        )


def _set_cuda_device(gpu_id: int) -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
