#!/usr/bin/env python3
"""KernelBench generation runner."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from lumen_artifact.bench.kernelbench.generation import (
    GenerationConfig,
    WorkArgs,
    load_generation_config,
    run_generation_tasks,
)
from lumen_artifact.bench.kernelbench.dataset import construct_kernelbench_dataset
from lumen_artifact.utils import (
    collect_generation_metrics,
    parse_int_list,
    run_eval_phase,
    select_problem_ids,
    write_yaml_mapping,
)


DEFAULT_CONFIG_PATH = Path(__file__).with_name("runner.yaml")


def run_generation(config: GenerationConfig) -> dict[str, Any]:
    dataset_kwargs = {
        "level": int(config.level),
        "source": config.dataset_src,
    }
    if config.dataset_src == "local":
        dataset_kwargs["base_path"] = config.dataset_name
    else:
        dataset_kwargs["dataset_name"] = config.dataset_name
    dataset = construct_kernelbench_dataset(**dataset_kwargs)

    problem_ids = select_problem_ids(dataset, config.problem_ids, config.subset)

    run_dir = Path(config.run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    config_data = asdict(config)
    config_data["run_dir"] = str(config_data["run_dir"])
    config_data["subset"] = list(config_data["subset"])
    write_yaml_mapping(run_dir / "generation_config.yaml", config_data)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate KernelBench cases through lumen_artifact sessions."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML generation config.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override one YAML value, e.g. "
            "run_dir=runs/my_run or model=deepseek-v4-pro."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with contextlib.redirect_stdout(sys.stderr):
        metrics = run_generation(load_generation_config(args.config, args.set))
    json.dump(metrics, sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
