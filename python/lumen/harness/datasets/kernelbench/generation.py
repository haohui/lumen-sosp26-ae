"""Codex-only KernelBench generation runner."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lumen.harness.backend.codex.runner import CodexRunner, CodexRunnerConfig
from lumen_artifact.utils import (
    apply_yaml_overrides,
    collect_generation_metrics,
    copy_codex_trace,
    coerce_config_values,
    evaluate_round,
    format_eval_status,
    load_yaml_mapping,
    parse_int_list,
    run_eval_phase,
    select_problem_ids,
    split_config_overrides,
    torch_cuda_available,
    validate_generated_avelang,
    write_artifacts,
    write_codex_result,
    write_generation_config,
    write_problem_meta,
)


@dataclass
class GenerationConfig:
    dataset_src: str
    level: int
    subset: tuple[int | None, int | None]
    run_dir: Path

    dataset_name: str = "ScalingIntelligence/KernelBench"
    problem_ids: str = ""

    timeout_seconds: int = 600
    max_retries: int = 3
    codex_bin: str = ""
    codex_profile: str = ""
    model_provider: str = ""
    reasoning_effort: str = ""
    codex_config: str = ""
    bypass_approvals_and_sandbox: bool = True

    num_workers: int = 1
    gpu_ids: str = "0"

    precision: str = "bf16"
    save_trajectory: bool = True

    eval_num_correct_trials: int = 5
    eval_num_perf_trials: int = 10
    gpu_arch: str = "gfx942"


@dataclass(frozen=True)
class WorkArgs:
    problem_id: int
    gpu_id: int


def load_generation_config(
    path: str | Path,
    overrides: list[str] | None = None,
    *,
    base_dir: str | Path | None = None,
) -> GenerationConfig:
    values = load_yaml_mapping(path)
    values = apply_yaml_overrides(values, overrides)
    if base_dir is None:
        base_dir = Path.cwd()
    return GenerationConfig(**coerce_config_values(values, base_dir))


def run_generation(config: GenerationConfig) -> dict[str, Any]:
    dataset = _construct_dataset(config)
    problem_ids = select_problem_ids(dataset, config.problem_ids, config.subset)

    run_dir = Path(config.run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    write_generation_config(run_dir / "generation_config.yaml", config)

    gpu_ids = parse_int_list(config.gpu_ids) or [0]
    if len(gpu_ids) > 1 and int(config.num_workers) <= 1:
        config.num_workers = len(gpu_ids)
        print(f"[INFO] gpu_ids={gpu_ids} -> auto num_workers={config.num_workers}")

    problems: list[WorkArgs] = []
    already_done = 0
    for pid in problem_ids:
        if (run_dir / f"p{pid:02d}" / "meta.json").is_file():
            already_done += 1
            continue
        gpu_id = gpu_ids[len(problems) % len(gpu_ids)]
        problems.append(WorkArgs(problem_id=int(pid), gpu_id=gpu_id))

    if already_done:
        print(f"[INFO] {already_done}/{len(problem_ids)} already generated; skipping.")
    print(
        f"[INFO] Generating {len(problems)} kernel(s) for level {config.level} "
        f"(timeout {config.timeout_seconds}s each)"
    )

    results = run_generation_tasks(problems, config, dataset, run_dir)
    if results:
        num_ok = sum(1 for result in results if result)
        print(f"\n[GEN]  {num_ok}/{len(results)} generated.")
    else:
        print("[INFO] Nothing to generate.")

    run_eval_phase(config, dataset, problem_ids, run_dir, gpu_ids)

    metrics = collect_generation_metrics(run_dir, problem_ids)
    print(f"\n[OK] Results in: {run_dir}")
    return metrics


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

    max_retries = max(1, int(config.max_retries))
    round_metas: list[dict[str, Any]] = []
    any_correct = False

    for round_idx in range(max_retries):
        round_dir = problem_dir / f"round{round_idx}"
        tag = f"p{work.problem_id:02d}/round{round_idx}"
        paths = _write_generation_workspace(
            round_dir,
            ref_arch_src=ref_arch_src,
            precision=config.precision,
            gpu_arch=config.gpu_arch,
            eval_num_correct_trials=config.eval_num_correct_trials,
            eval_num_perf_trials=config.eval_num_perf_trials,
        )
        prompt = paths["prompt"].read_text(encoding="utf-8")

        result = run_codex(round_dir, prompt, config)
        write_codex_result(round_dir / "codex_result.json", result)
        copy_codex_trace(result, round_dir, save_trace=config.save_trajectory)
        if not result.ok:
            log = result.error or result.status
            err = f"codex exited with error. log={log[:400]}"
            print(f"[FAIL] {tag}: {err}")
            round_metas.append(
                write_artifacts(
                    round_dir,
                    problem_id=work.problem_id,
                    problem_name=problem_name,
                    error=err,
                )
            )
            if result.status == "timed_out":
                continue
            break

        output_path = round_dir / "output_model_new.py"
        if not output_path.is_file():
            err = "output_model_new.py not found"
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

        static_ok, errors, warnings = validate_generated_avelang(custom_kernel)
        if not static_ok:
            err = f"Static check failed: {errors}. Warnings: {warnings}"
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


def run_codex(
    round_dir: str | Path,
    prompt: str,
    config: GenerationConfig,
) -> Any:
    env = {"IS_SANDBOX": "1"}

    return CodexRunner().execute(
        CodexRunnerConfig(
            work_dir=round_dir,
            prompt=prompt,
            codex_bin=config.codex_bin or None,
            profile=config.codex_profile or None,
            model_provider=config.model_provider or None,
            reasoning_effort=config.reasoning_effort or None,
            timeout_seconds=float(config.timeout_seconds)
            if config.timeout_seconds
            else None,
            env=env,
            config_overrides=tuple(split_config_overrides(config.codex_config)),
            bypass_approvals_and_sandbox=bool(config.bypass_approvals_and_sandbox),
        )
    )


def _write_generation_workspace(*args: Any, **kwargs: Any) -> dict[str, Path]:
    try:
        from lumen.harness.datasets.kernelbench.prompt import (
            write_generation_workspace,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "KernelBench prompt construction is required before running generation."
        ) from exc

    return write_generation_workspace(*args, **kwargs)


def parse_args(
    argv: list[str] | None = None,
    *,
    default_config_path: str | Path | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate KernelBench cases with Codex."
    )
    default_config = (
        Path(default_config_path).expanduser()
        if default_config_path is not None
        else None
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        required=default_config is None,
        help="Path to the YAML generation config.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override one YAML value, e.g. run_dir=runs/my_run.",
    )
    return parser.parse_args(argv)


def main(
    argv: list[str] | None = None,
    *,
    default_config_path: str | Path | None = None,
) -> int:
    args = parse_args(argv, default_config_path=default_config_path)
    with redirect_stdout(sys.stderr):
        metrics = run_generation(load_generation_config(args.config, args.set))
    json.dump(metrics, sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
    return 0


def _construct_dataset(config: GenerationConfig) -> Any:
    from kernelbench.dataset import construct_kernelbench_dataset

    kwargs: dict[str, Any] = {
        "level": int(config.level),
        "source": config.dataset_src,
    }
    if config.dataset_src == "local":
        kwargs["base_path"] = config.dataset_name
    else:
        kwargs["dataset_name"] = config.dataset_name
    return construct_kernelbench_dataset(**kwargs)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
