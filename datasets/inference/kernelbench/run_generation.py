#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
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


def add_kernelbench_to_path(kb_root: str | None) -> None:
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
            return


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
    parser.add_argument("--env-file", default=os.environ.get("ENV_FILE"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=os.environ.get("DRY_RUN") == "1",
    )
    args = parser.parse_args()

    add_kernelbench_to_path(args.kernelbench_root)
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
            temperature=0.0,
            max_tokens=args.max_tokens,
        )

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
        kernel_path = task_run_dir / "kernel.py"
        prompt_path.write_text(prompt, encoding="utf-8")

        if args.dry_run:
            print(
                f"[DRY-RUN] task={task} ref={ref_path} "
                f"prompt={prompt_path} max_tokens={args.max_tokens}"
            )
            continue

        assert inference_server is not None
        response = inference_server(prompt)
        kernel = extract_first_code(response, ["python", "cpp"])
        if not kernel:
            raise RuntimeError(
                f"no code block found in KernelBench response for task={task}"
            )
        kernel_path.write_text(kernel, encoding="utf-8")
        print(f"generated {task}: {kernel_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
