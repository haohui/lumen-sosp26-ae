from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

import yaml


def load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser()
    text = os.path.expandvars(config_path.read_text(encoding="utf-8"))
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping: {config_path}")
    return data


def apply_yaml_overrides(
    values: dict[str, Any],
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    merged = dict(values)
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Override must look like key=value: {override}")
        key, value = override.split("=", 1)
        merged[key.strip()] = yaml.safe_load(value)
    return merged


def write_yaml_mapping(path: str | Path, values: dict[str, Any]) -> None:
    Path(path).expanduser().write_text(
        yaml.safe_dump(values, sort_keys=False),
        encoding="utf-8",
    )


def resolve_path(value: str | Path, base_dir: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (Path(base_dir).expanduser() / path).resolve()


def coerce_config_values(
    values: dict[str, Any],
    base_dir: str | Path,
) -> dict[str, Any]:
    coerced = dict(values)

    required = {"dataset_src", "level", "subset", "run_dir"}
    missing = sorted(required - set(coerced))
    if missing:
        raise ValueError(f"Missing required config key(s): {', '.join(missing)}")

    coerced["level"] = int(coerced["level"])
    coerced["subset"] = parse_int_pair(coerced["subset"], name="subset")
    coerced["run_dir"] = resolve_path(coerced["run_dir"], base_dir)
    if coerced["dataset_src"] == "local":
        coerced["dataset_name"] = str(resolve_path(coerced["dataset_name"], base_dir))

    for key in (
        "timeout_seconds",
        "max_retries",
        "num_workers",
        "eval_num_correct_trials",
        "eval_num_perf_trials",
    ):
        if key in coerced:
            coerced[key] = int(coerced[key])

    for key, default in (
        ("save_trajectory", True),
        ("use_example_skills", True),
    ):
        if key in coerced:
            coerced[key] = parse_bool(coerced[key], default=default)

    return coerced


def parse_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def parse_int_list(value: str | list[int] | tuple[int, ...] | None) -> list[int]:
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    text = str(value).strip().strip("[]")
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_int_pair(
    value: str | list[int | None] | tuple[int | None, int | None],
    *,
    name: str = "value",
) -> tuple[int | None, int | None]:
    if isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        text = str(value).strip()
        if text.lower() in {"", "none", "(none, none)"}:
            return (None, None)
        parts = [part.strip() for part in text.strip("()[]").split(",")]

    if len(parts) != 2:
        raise ValueError(f"{name} must contain exactly two values: {value}")

    def parse_item(item: object) -> int | None:
        if item is None:
            return None
        text = str(item).strip()
        return None if text.lower() in {"", "none"} else int(text)

    return parse_item(parts[0]), parse_item(parts[1])


def select_problem_ids(
    dataset: Any,
    problem_ids: str | list[int] | tuple[int, ...] | None,
    subset: tuple[int | None, int | None],
) -> list[int]:
    all_problem_ids = dataset.get_problem_ids()
    all_set = set(all_problem_ids)
    explicit = parse_int_list(problem_ids)

    if explicit:
        unknown = [pid for pid in explicit if pid not in all_set]
        if unknown:
            print(f"[WARN] problem_ids not in dataset, ignored: {unknown}")
        return [pid for pid in explicit if pid in all_set]

    start, end = subset
    if start is None and end is None:
        return list(all_problem_ids)

    start_value = min(all_problem_ids) if start is None else start
    end_value = max(all_problem_ids) if end is None else end
    return [pid for pid in all_problem_ids if start_value <= pid <= end_value]


def torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def sorted_prefixed_dirs(directory: str | Path, prefix: str) -> list[Path]:
    base = Path(directory)
    if not base.is_dir():
        return []

    def suffix_index(path: Path) -> int:
        try:
            return int(path.name.removeprefix(prefix))
        except ValueError:
            return -1

    return sorted(
        [
            path
            for path in base.iterdir()
            if path.is_dir() and path.name.startswith(prefix)
        ],
        key=suffix_index,
    )


def parse_eval_payload(eval_payload: dict[str, Any]) -> dict[str, Any]:
    compiled = bool(eval_payload.get("compiled", False))
    correctness = bool(eval_payload.get("correctness", False))
    runtime = eval_payload.get("runtime_us", -1.0) or -1.0
    ref_runtime = eval_payload.get("ref_runtime_us", -1.0) or -1.0
    speedup = (ref_runtime / runtime) if runtime > 0 and ref_runtime > 0 else -1.0
    return {
        "compiled": compiled,
        "correctness": correctness,
        "runtime": runtime,
        "ref_runtime": ref_runtime,
        "speedup": speedup,
        "metadata": eval_payload.get("metadata", {}),
    }


def format_eval_status(eval_payload: dict[str, Any]) -> str:
    parsed = parse_eval_payload(eval_payload)
    if parsed["compiled"] and parsed["correctness"]:
        return f"compiled=true  correct=true  speedup={parsed['speedup']:.3f}x"
    if parsed["compiled"]:
        return "compiled=true  correct=false"
    return f"compiled=false  {str(parsed['metadata'])[:80]}"


def evaluate_round(round_dir: str | Path, config: Any, gpu_id: int) -> dict[str, Any]:
    from lumen_artifact.bench import run_kernelbench_cases

    path = Path(round_dir)
    output_path = path / "eval_result.jsonl"
    exit_code = run_kernelbench_cases(
        input_file=path / "eval_cases.txt",
        output=output_path,
        device=gpu_id,
        num_correct_trials=int(config.eval_num_correct_trials),
        num_perf_trials=int(config.eval_num_perf_trials),
        measure_performance=True,
        timing_method=config.eval_timing_method,
        verbose=False,
    )
    lines = output_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return {
            "compiled": False,
            "correctness": False,
            "metadata": {"error": f"eval produced no output; exit_code={exit_code}"},
        }
    payload = json.loads(lines[0])
    payload.setdefault("eval_exit_code", exit_code)
    return payload


def write_artifacts(
    round_dir: str | Path,
    *,
    problem_id: int,
    problem_name: str,
    error: str,
    eval_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path = Path(round_dir)
    meta: dict[str, Any] = {
        "problem_id": problem_id,
        "problem_name": problem_name,
        "has_prompt": (path / "prompt.txt").is_file(),
        "has_output_model_new": (path / "output_model_new.py").is_file(),
        "stage": "generated",
        "error": error,
    }

    if eval_payload is not None:
        parsed = parse_eval_payload(eval_payload)
        metadata = parsed["metadata"]
        if parsed["compiled"] and parsed["correctness"]:
            eval_error = "OK: compiled=True, correctness=True"
        elif not parsed["compiled"]:
            eval_error = f"compiled=False. {json.dumps(metadata, default=str)}"
        else:
            eval_error = (
                f"compiled=True, correctness=False. {json.dumps(metadata, default=str)}"
            )

        meta.update(
            {
                "stage": "speedup_eval",
                "compiled": parsed["compiled"],
                "correctness": parsed["correctness"],
                "speedup": parsed["speedup"],
                "ref_ms": parsed["ref_runtime"],
                "new_ms": parsed["runtime"],
                "error": eval_error,
                "eval_metadata": metadata,
                "eval_payload": eval_payload,
            }
        )

    (path / "meta.json").write_text(
        json.dumps(meta, indent=2, default=str),
        encoding="utf-8",
    )
    (path / "error.txt").write_text(meta["error"] + "\n", encoding="utf-8")
    return meta


def write_problem_meta(
    problem_dir: str | Path,
    *,
    problem_id: int,
    problem_name: str,
    round_metas: list[dict[str, Any]],
) -> None:
    if round_metas:
        best = next(
            (meta for meta in round_metas if meta.get("correctness")),
            round_metas[-1],
        )
        meta = dict(best)
        meta["rounds"] = [
            {
                "round": idx,
                "compiled": item.get("compiled"),
                "correctness": item.get("correctness"),
                "speedup": item.get("speedup"),
            }
            for idx, item in enumerate(round_metas)
        ]
    else:
        meta = {
            "problem_id": problem_id,
            "problem_name": problem_name,
            "stage": "generated",
            "error": "no rounds completed",
            "rounds": [],
        }

    path = Path(problem_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "meta.json").write_text(
        json.dumps(meta, indent=2, default=str),
        encoding="utf-8",
    )
    (path / "error.txt").write_text(
        str(meta.get("error", "")) + "\n",
        encoding="utf-8",
    )


def run_eval_phase(
    config: Any,
    dataset: Any,
    problem_ids: list[int],
    run_dir: str | Path,
    gpu_ids: list[int],
) -> None:
    if not torch_cuda_available():
        print("[WARN] No CUDA/HIP device available; eval was skipped.")
        return

    base_run_dir = Path(run_dir)
    to_eval: list[tuple[int, Path, int]] = []
    for pid in problem_ids:
        problem_dir = base_run_dir / f"p{pid:02d}"
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
        print("[INFO] Eval phase: nothing to evaluate.")
        return

    print(f"\n[EVAL] Evaluating {len(to_eval)} kernel(s) on GPU(s) {gpu_ids}.")
    for pid, round_dir, gpu_id in to_eval:
        problem = dataset.get_problem_by_id(pid)
        start_time = time.time()
        eval_payload = evaluate_round(round_dir, config, gpu_id)
        elapsed = time.time() - start_time
        status = format_eval_status(eval_payload)
        print(f"  p{pid:02d}: {status}  ({elapsed:.1f}s) [gpu:{gpu_id}]")

        write_artifacts(
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


def count_trace_edits(trace_path: str | Path) -> int:
    path = Path(trace_path)
    if not path.is_file():
        return 0

    counted_tools = {"read", "write", "edit"}
    count = 0

    def visit(node: Any) -> None:
        nonlocal count
        if isinstance(node, dict):
            name = node.get("name")
            if node.get("type") == "tool_use" and isinstance(name, str):
                if name.lower() in counted_tools:
                    count += 1
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    with path.open(encoding="utf-8") as trace_file:
        for line in trace_file:
            try:
                visit(json.loads(line))
            except json.JSONDecodeError:
                continue

    return count


def collect_generation_metrics(
    run_dir: str | Path,
    problem_ids: list[int],
) -> dict[str, Any]:
    base = Path(run_dir)
    metas: list[dict[str, Any]] = []
    speedups: list[float] = []
    edit_count = 0

    for pid in problem_ids:
        problem_dir = base / f"p{pid:02d}"
        meta_path = problem_dir / "meta.json"
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                meta = {}
            metas.append(meta)
            speedup = meta.get("speedup")
            if meta.get("correctness") and isinstance(speedup, (int, float)):
                if speedup > 0:
                    speedups.append(float(speedup))

        for round_dir in sorted_prefixed_dirs(problem_dir, "round"):
            edit_count += count_trace_edits(round_dir / "trace.jsonl")

    correct_count = sum(1 for meta in metas if meta.get("correctness"))
    total_count = len(problem_ids)
    geomean = None
    if speedups:
        geomean = math.exp(sum(math.log(value) for value in speedups) / len(speedups))

    return {
        "run_dir": str(base),
        "correctness_rate": (
            100.0 * correct_count / total_count if total_count else None
        ),
        "geomean_speedup": geomean,
        "min_speedup": min(speedups) if speedups else None,
        "max_speedup": max(speedups) if speedups else None,
        "trace_edit_count": edit_count,
    }
