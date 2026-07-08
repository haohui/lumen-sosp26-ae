#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
THIS_DIR = Path(__file__).resolve().parent
PROMPTS_TOML = THIS_DIR / "resources" / "prompts" / "prompts.toml"


@dataclass(frozen=True)
class TaskSpec:
    file_name: str
    custom_prompt_key: str


TASKS = {
    "gemm": TaskSpec("1_gemm.py", "lumen_gemm"),
    "attention": TaskSpec("2_attention.py", "lumen_attention"),
    "moe": TaskSpec("3_fused_moe.py", "lumen_fused_moe"),
}
TASK_ALIASES = {"attn": "attention"}


def add_kernelbench_to_path(kb_root: str | None) -> Path:
    candidates: list[Path] = []
    if kb_root:
        candidates.append(Path(kb_root).expanduser())
    if os.environ.get("KB_ROOT"):
        candidates.append(Path(os.environ["KB_ROOT"]).expanduser())
    candidates.extend(
        [ROOT / ".third_party" / "KernelBench", ROOT / "third_party" / "KernelBench"]
    )

    for candidate in candidates:
        src = candidate / "src"
        if src.is_dir():
            sys.path.insert(0, str(src))
            return candidate
    raise FileNotFoundError("KernelBench checkout not found; pass --kernelbench-root")


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        os.environ.setdefault(key, value.strip().strip("'\""))


def load_kernelbench_helpers():
    import builtins

    sentinel = object()
    previous_torch = getattr(builtins, "torch", sentinel)
    try:
        import torch

        # Some KernelBench revisions reference torch in annotations before importing it.
        builtins.torch = torch
        from kernelbench.prompt_constructor_toml import get_custom_prompt
        from kernelbench.utils import (
            create_inference_server_from_presets,
            extract_first_code,
        )
    finally:
        if previous_torch is sentinel:
            try:
                delattr(builtins, "torch")
            except AttributeError:
                pass
        else:
            builtins.torch = previous_torch

    return get_custom_prompt, create_inference_server_from_presets, extract_first_code


def resolve_tasks(raw_task: str) -> list[str]:
    if raw_task == "all":
        return list(TASKS)
    return [TASK_ALIASES.get(raw_task, raw_task)]


def validation_command(
    *,
    kernelbench_root: Path,
    ref_path: Path,
    kernel_path: Path,
    sample_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    return [
        sys.executable,
        str(kernelbench_root / "scripts" / "run_and_check.py"),
        "ref_origin=local",
        f"ref_arch_src_path={ref_path}",
        f"kernel_src_path={kernel_path}",
        "eval_mode=local",
        "backend=hip",
        "precision=bf16",
        f"gpu_arch=[{args.gpu_arch}]",
        f"num_correct_trials={args.num_correct_trials}",
        f"num_perf_trials={args.num_perf_trials}",
        f"measure_performance={args.perf_check}",
        f"timing_method={args.timing_method}",
        f"build_dir_prefix={sample_dir / 'build'}",
        f"check_kernel={args.static_check}",
    ]


def run_validation(cmd: list[str], sample_dir: Path) -> dict[str, object]:
    cp = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
    )
    (sample_dir / "validation.stdout").write_text(cp.stdout, encoding="utf-8")
    (sample_dir / "validation.stderr").write_text(cp.stderr, encoding="utf-8")
    return {
        "returncode": cp.returncode,
        "passed": cp.returncode == 0 and "correctness=True" in cp.stdout,
        "command": cmd,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate Lumen AE kernels with KernelBench prompts."
    )
    parser.add_argument(
        "--task",
        default="all",
        choices=sorted([*TASKS, *TASK_ALIASES, "all"]),
    )
    parser.add_argument("--kernelbench-root", default=os.environ.get("KB_ROOT"))
    parser.add_argument("--run-root", default=os.environ.get("RUN_ROOT"))
    parser.add_argument(
        "--server-type",
        default=os.environ.get("SERVER_TYPE", "openai"),
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL_NAME", "gpt-5.3-codex"),
    )
    parser.add_argument(
        "--reasoning-effort",
        default=os.environ.get("REASONING_EFFORT", "high"),
    )
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.environ.get("TEMPERATURE", "0.0")),
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=int(os.environ.get("PASS_K", "1")),
    )
    parser.add_argument("--env-file", default=os.environ.get("ENV_FILE"))
    parser.add_argument(
        "--validate",
        action="store_true",
        default=os.environ.get("KB_VALIDATE") == "1",
    )
    parser.add_argument("--require-pass", action="store_true")
    parser.add_argument("--static-check", action="store_true", default=True)
    parser.add_argument("--no-static-check", dest="static_check", action="store_false")
    parser.add_argument(
        "--perf-check",
        dest="perf_check",
        action="store_true",
        default=True,
    )
    parser.add_argument("--no-perf-check", dest="perf_check", action="store_false")
    parser.add_argument("--num-correct-trials", type=int, default=5)
    parser.add_argument("--num-perf-trials", type=int, default=100)
    parser.add_argument("--timing-method", default="cuda_event")
    parser.add_argument("--gpu-arch", default=os.environ.get("GPU_ARCH", "gfx942"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=os.environ.get("DRY_RUN") == "1",
    )
    args = parser.parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be positive")

    kernelbench_root = add_kernelbench_to_path(args.kernelbench_root)
    (
        get_custom_prompt,
        create_inference_server_from_presets,
        extract_first_code,
    ) = load_kernelbench_helpers()

    if args.env_file:
        load_env_file(Path(args.env_file).expanduser())

    run_tag = os.environ.get("RUN_TAG", time.strftime("%Y%m%d_%H%M%S", time.gmtime()))
    run_root = (
        Path(args.run_root).expanduser()
        if args.run_root
        else ROOT / "logs" / "generation" / "kernelbench" / run_tag
    )
    run_root.mkdir(parents=True, exist_ok=True)

    inference_server = None
    if not args.dry_run:
        inference_server = create_inference_server_from_presets(
            server_type=args.server_type,
            model_name=args.model,
            is_reasoning_model=True,
            reasoning_effort=args.reasoning_effort,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

    summary: dict[str, object] = {
        "run_root": str(run_root),
        "samples": args.samples,
        "validate": bool(args.validate),
        "perf_check": bool(args.perf_check),
        "tasks": {},
    }

    for task in resolve_tasks(args.task):
        spec = TASKS[task]
        ref_path = THIS_DIR / spec.file_name
        ref_src = ref_path.read_text(encoding="utf-8")

        prompt = get_custom_prompt(
            spec.custom_prompt_key,
            ref_arch_src=ref_src,
            backend="hip",
            option="one_shot",
            precision="bf16",
            include_hardware=False,
            prompts_toml=str(PROMPTS_TOML),
        )

        task_run_dir = run_root / task
        task_run_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = task_run_dir / "prompt.txt"
        prompt_path.write_text(prompt, encoding="utf-8")
        task_records = []

        if args.dry_run:
            print(
                f"[DRY-RUN] task={task} ref={ref_path} "
                f"prompt={prompt_path} samples={args.samples} "
                f"validate={args.validate} perf_check={args.perf_check}"
            )
            if args.validate:
                sample_dir = task_run_dir / "sample_000"
                kernel_path = sample_dir / "kernel.py"
                print(
                    "[DRY-RUN] validation command: "
                    + " ".join(
                        validation_command(
                            kernelbench_root=kernelbench_root,
                            ref_path=ref_path,
                            kernel_path=kernel_path,
                            sample_dir=sample_dir,
                            args=args,
                        )
                    )
                )
            continue

        assert inference_server is not None
        for sample_id in range(args.samples):
            if args.samples == 1:
                sample_dir = task_run_dir
            else:
                sample_dir = task_run_dir / f"sample_{sample_id:03d}"
                sample_dir.mkdir(parents=True, exist_ok=True)
            kernel_path = sample_dir / "kernel.py"
            response_path = sample_dir / "response.txt"

            response = inference_server(prompt)
            response_path.write_text(str(response), encoding="utf-8")
            kernel = extract_first_code(response, ["python", "cpp"])
            if not kernel:
                raise RuntimeError(
                    "no code block found in KernelBench response "
                    f"for task={task} sample={sample_id}"
                )
            kernel_path.write_text(kernel, encoding="utf-8")

            record: dict[str, object] = {
                "sample_id": sample_id,
                "kernel_path": str(kernel_path),
                "response_path": str(response_path),
            }
            if args.validate:
                cmd = validation_command(
                    kernelbench_root=kernelbench_root,
                    ref_path=ref_path,
                    kernel_path=kernel_path,
                    sample_dir=sample_dir,
                    args=args,
                )
                record["validation"] = run_validation(cmd, sample_dir)
            task_records.append(record)
            print(f"generated {task} sample {sample_id}: {kernel_path}")

        passed = sum(
            1
            for record in task_records
            if isinstance(record.get("validation"), dict)
            and record["validation"].get("passed")
        )
        task_summary = {
            "prompt_path": str(prompt_path),
            "reference_path": str(ref_path),
            "samples": task_records,
            "passed": passed,
            "pass_at_k": bool(passed),
        }
        summary["tasks"][task] = task_summary
        (task_run_dir / "summary.json").write_text(
            json.dumps(task_summary, indent=2),
            encoding="utf-8",
        )
        if args.require_pass and args.validate and not passed:
            raise RuntimeError(
                f"no KernelBench sample passed validation/perf checks for task={task}"
            )

    (run_root / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
