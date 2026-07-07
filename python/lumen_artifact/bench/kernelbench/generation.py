from __future__ import annotations

import concurrent.futures
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lumen_artifact.session import Session
from lumen_artifact.utils import (
    apply_yaml_overrides,
    coerce_config_values,
    evaluate_round,
    format_eval_status,
    load_yaml_mapping,
    torch_cuda_available,
    write_artifacts,
    write_problem_meta,
)

from .kernel_static_checker import validate_kernel_static
from .prompt_constructor_toml import get_custom_prompt
from .prompts.claude_context import render_claude_context


PROJECT_ROOT = Path(__file__).resolve().parents[4]
PYTHON_ROOT = PROJECT_ROOT / "python"
SKILLS_ROOT = Path(__file__).resolve().parent / "prompts" / "skills"
PROMPT_BACKEND = "substrate"
PROMPT_OPTION = "one_shot"
PROMPT_KEY = "substrate_amd_full"

_WRITE_FILE_DIRECTIVE = (
    "\n\nWrite your complete final implementation directly to the file "
    "`output_model_new.py` in the current working directory using your file "
    "writing tools. Do not print the code in your response text; write it to "
    "the file only."
)


@dataclass
class GenerationConfig:
    dataset_src: str
    level: int
    subset: tuple[int | None, int | None]
    run_dir: Path

    dataset_name: str = "ScalingIntelligence/KernelBench"
    problem_ids: str = ""

    backend: str = "claude"
    timeout_seconds: int = 600
    max_retries: int = 3
    model: str = ""
    endpoint: str = ""
    api_key: str = ""

    num_workers: int = 1
    gpu_ids: str = "0"

    precision: str = "bf16"

    save_trajectory: bool = True
    use_example_skills: bool = True

    eval_num_correct_trials: int = 5
    eval_num_perf_trials: int = 10
    eval_timing_method: str = "cudagraph"
    gpu_arch: str = "gfx942"


@dataclass(frozen=True)
class WorkArgs:
    problem_id: int
    gpu_id: int


def load_generation_config(
    path: str | Path,
    overrides: list[str] | None = None,
    *,
    base_dir: str | Path = PROJECT_ROOT,
) -> GenerationConfig:
    values = load_yaml_mapping(path)
    values = apply_yaml_overrides(values, overrides)
    return GenerationConfig(**coerce_config_values(values, base_dir))


def build_prompt(config: GenerationConfig, ref_arch_src: str) -> str:
    base = get_custom_prompt(
        PROMPT_KEY,
        ref_arch_src=ref_arch_src,
        backend=PROMPT_BACKEND,
        option=PROMPT_OPTION,
        precision=config.precision,
    )
    return base.rstrip() + _WRITE_FILE_DIRECTIVE


def write_round_inputs(
    round_dir: str | Path,
    *,
    config: GenerationConfig,
    ref_arch_src: str,
    prompt: str,
    python_root: str | Path = PYTHON_ROOT,
) -> None:
    path = Path(round_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "input_model.py").write_text(ref_arch_src, encoding="utf-8")
    (path / "prompt.txt").write_text(prompt, encoding="utf-8")
    (path / "CLAUDE.md").write_text(
        render_claude_context(
            python_root=python_root,
            skills_root=SKILLS_ROOT,
            use_example_skills=config.use_example_skills,
            python_executable=sys.executable,
            eval_num_correct_trials=config.eval_num_correct_trials,
            eval_num_perf_trials=config.eval_num_perf_trials,
            eval_timing_method=config.eval_timing_method,
        ),
        encoding="utf-8",
    )
    eval_config = {
        "backend": PROMPT_BACKEND,
        "gpu_arch": config.gpu_arch,
        "precision": config.precision,
        "num_correct_trials": config.eval_num_correct_trials,
        "num_trials": config.eval_num_perf_trials,
    }
    (path / "eval_config.json").write_text(
        json.dumps(eval_config, indent=2),
        encoding="utf-8",
    )


def run_session(
    round_dir: str | Path,
    prompt: str,
    config: GenerationConfig,
) -> tuple[bool, str]:
    with Session(
        backend=config.backend,
        work_dir=round_dir,
        prompt=prompt,
        timeout_seconds=int(config.timeout_seconds),
        save_trace=config.save_trajectory,
        trace_path="trace.jsonl",
    ) as session:
        result = session.load_llm_config(
            endpoint=config.endpoint or None,
            api_key=config.api_key or None,
            model=config.model or None,
        ).run()

    if result.returncode == 0:
        return True, ""
    return False, result.error or f"{config.backend} exited with an error"


def generate_one(
    work: WorkArgs,
    config: GenerationConfig,
    dataset: Any,
    run_dir: Path,
) -> bool:
    problem = dataset.get_problem_by_id(work.problem_id)
    ref_arch_src = problem.code
    problem_name = problem.name

    problem_dir = run_dir / f"p{work.problem_id:02d}"
    problem_dir.mkdir(parents=True, exist_ok=True)

    prompt = build_prompt(config, ref_arch_src)
    max_retries = max(1, int(config.max_retries))
    round_metas: list[dict[str, Any]] = []
    any_correct = False

    for round_idx in range(max_retries):
        round_dir = problem_dir / f"round{round_idx}"
        tag = f"p{work.problem_id:02d}/round{round_idx}"
        write_round_inputs(
            round_dir,
            config=config,
            ref_arch_src=ref_arch_src,
            prompt=prompt,
        )

        success, log = run_session(round_dir, prompt, config)
        if not success:
            err = f"{config.backend} exited with error. log={log[:400]}"
            print(f"[FAIL] {tag}: {err}")
            round_metas.append(
                write_artifacts(
                    round_dir,
                    problem_id=work.problem_id,
                    problem_name=problem_name,
                    error=err,
                )
            )
            if "timed out" in log:
                continue
            break

        output_path = round_dir / "output_model_new.py"
        if not output_path.is_file():
            err = f"output_model_new.py not found. log={log[:400]}"
            print(f"[FAIL] {tag}: {err}")
            round_metas.append(
                write_artifacts(
                    round_dir,
                    problem_id=work.problem_id,
                    problem_name=problem_name,
                    error=err,
                )
            )
            continue

        custom_kernel = output_path.read_text(encoding="utf-8")
        if not custom_kernel.strip():
            err = "output_model_new.py is empty"
            print(f"[FAIL] {tag}: {err}")
            round_metas.append(
                write_artifacts(
                    round_dir,
                    problem_id=work.problem_id,
                    problem_name=problem_name,
                    error=err,
                )
            )
            continue

        static_ok, error, warnings = validate_kernel_static(
            custom_kernel,
            backend=PROMPT_BACKEND,
            precision=config.precision,
        )
        if not static_ok:
            err = f"Static check failed: {error}. Warnings: {warnings}"
            print(f"[FAIL] {tag}: {err}")
            round_metas.append(
                write_artifacts(
                    round_dir,
                    problem_id=work.problem_id,
                    problem_name=problem_name,
                    error=err,
                )
            )
            continue
        if warnings:
            print(f"[WARN] {tag}: {warnings}")

        if torch_cuda_available():
            eval_payload = evaluate_round(round_dir, config, work.gpu_id)
            status = format_eval_status(eval_payload)
            print(f"  {tag}: {status}")
            any_correct = bool(eval_payload.get("correctness", False))

            round_metas.append(
                write_artifacts(
                    round_dir,
                    problem_id=work.problem_id,
                    problem_name=problem_name,
                    error="",
                    eval_payload=eval_payload,
                )
            )
            if any_correct:
                break
        else:
            round_metas.append(
                write_artifacts(
                    round_dir,
                    problem_id=work.problem_id,
                    problem_name=problem_name,
                    error="OK: generated",
                )
            )
            break

    write_problem_meta(
        problem_dir,
        problem_id=work.problem_id,
        problem_name=problem_name,
        round_metas=round_metas,
    )
    return any_correct or bool(round_metas)


def run_generation_tasks(
    problems: list[WorkArgs],
    config: GenerationConfig,
    dataset: Any,
    run_dir: Path,
) -> list[bool | None]:
    if not problems:
        return []

    if int(config.num_workers) <= 1:
        results: list[bool | None] = []
        for work in problems:
            try:
                results.append(generate_one(work, config, dataset, run_dir))
            except Exception as exc:
                print(f"[ERROR] p{work.problem_id:02d}: {exc}")
                results.append(None)
        return results

    results = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=int(config.num_workers)
    ) as executor:
        futures = {
            executor.submit(generate_one, work, config, dataset, run_dir): work
            for work in problems
        }
        for future in concurrent.futures.as_completed(futures):
            work = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                print(f"[ERROR] p{work.problem_id:02d}: {exc}")
                results.append(None)
    return results
