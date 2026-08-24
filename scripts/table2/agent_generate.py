#!/usr/bin/env python3.12
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from experiment_summary import ExperimentSummary  # noqa: E402

TASKS = ("gemm", "attention", "moe")
BASELINES = ("kernelbench", "cudaforge", "kernelfalcon", "ksearch")
TASK_DIR = {"gemm": "gemm", "attention": "attention", "attn": "attention", "moe": "moe"}
REFERENCE_MODEL = {
    "gemm": REPO_ROOT / "prompts" / "kernelbench" / "1_gemm.py",
    "attention": REPO_ROOT / "prompts" / "kernelbench" / "2_attention.py",
    "moe": REPO_ROOT / "prompts" / "kernelbench" / "3_fused_moe.py",
}
KERNELBENCH_PROMPT_KEY = {
    "gemm": "lumen_gemm",
    "attention": "lumen_attention",
    "moe": "lumen_fused_moe",
}
KSEARCH_TASKS = {
    "attention": (
        "dense_attention_index",
        "dense_attention",
        "dense_qkv_prefill_causal_h8_kv1or8_d128",
    ),
    "gemm": ("gemm_bf16_var_mnk_7shape", "gemm", "gemm_bf16_var_mnk"),
    "moe": (
        "moe_fp8_blockscale",
        "moe",
        "moe_fp8_blockscale_g1u1_topk4_e32_h7168_i2048",
    ),
}
SENSITIVE_VALUE_FLAGS = {"--api-key"}
API_BACKEND_LIMITED_EXIT_CODE = 75


class ApiBackendLimitedError(RuntimeError):
    """The generation backend failed after all configured attempts."""


@dataclass(frozen=True)
class RunContext:
    args: argparse.Namespace
    api_base_url: str
    api_key: str

    @property
    def env(self) -> dict[str, str]:
        env = os.environ.copy()
        pythonpath = os.pathsep.join((str(REPO_ROOT / "python"), str(REPO_ROOT)))
        existing = env.get("PYTHONPATH")
        if existing:
            pythonpath = pythonpath + os.pathsep + existing
        env["PYTHONPATH"] = pythonpath
        env["OPENAI_API_KEY"] = self.api_key
        env["OPENAI_BASE_URL"] = self.api_base_url
        env["LLM_API_KEY"] = self.api_key
        if self.args.rocr_visible_devices:
            env.pop("CUDA_VISIBLE_DEVICES", None)
            env.pop("HIP_VISIBLE_DEVICES", None)
            env["ROCR_VISIBLE_DEVICES"] = self.args.rocr_visible_devices
        return env


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Regenerate Table 2 agentic baseline kernels."
    )
    p.add_argument(
        "--baseline",
        choices=("all", *BASELINES),
        default="all",
        help="Baseline to run.",
    )
    p.add_argument(
        "--task",
        choices=("all", *TASKS, "attn"),
        default="all",
        help="Task to run.",
    )
    p.add_argument(
        "--model",
        default=os.getenv("LUMEN_GENERATION_MODEL", "gpt-5.3-codex"),
    )
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument(
        "--kernelbench-attempts",
        type=int,
        default=2,
        help="Maximum API attempts for each one-shot KernelBench task.",
    )
    p.add_argument("--api-timeout-seconds", type=float, default=300.0)
    p.add_argument("--max-output-tokens", type=int, default=32768)
    p.add_argument("--device", type=int, default=0)
    p.add_argument(
        "--rocr-visible-devices",
        default=os.getenv("ROCR_VISIBLE_DEVICES", ""),
    )
    p.add_argument("--workspace-dir", type=Path, default=None)
    p.add_argument("--api-url", default=os.getenv("LUMEN_GENERATION_API_URL", ""))
    p.add_argument("--api-key", default=os.getenv("LUMEN_GENERATION_API_KEY", ""))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-stage", action="store_true")
    p.add_argument(
        "--reference-py",
        type=Path,
        default=None,
        help=(
            "Reference Model file for CUDAForge/KernelFalcon. Defaults to "
            "datasets/inference/<task>/kernelbench/model.py."
        ),
    )
    p.add_argument(
        "--kernelbench-config",
        type=Path,
        default=None,
        help="Deprecated compatibility option; KernelBench uses the HIP prompt resource.",
    )
    p.add_argument(
        "--kernelbench-root",
        type=Path,
        default=os.getenv("KERNELBENCH_ROOT"),
        help="Prepared and patched KernelBench source checkout.",
    )
    p.add_argument(
        "--cudaforge-root",
        type=Path,
        default=os.getenv("CUDAFORGE_ROOT"),
        help="Prepared and patched CUDAForge source checkout.",
    )
    p.add_argument(
        "--kernelfalcon-root",
        type=Path,
        default=os.getenv("KERNELFALCON_ROOT"),
        help="Prepared and patched KernelFalcon/KernelAgent source checkout.",
    )
    p.add_argument(
        "--ksearch-root",
        type=Path,
        default=os.getenv("KSEARCH_ROOT"),
        help="Prepared and patched K-Search source checkout.",
    )
    p.add_argument(
        "--ksearch-task-root",
        type=Path,
        default=os.getenv(
            "KSEARCH_TASK_ROOT",
            str(
                REPO_ROOT
                / "prompts"
                / "ksearch"
                / "resources"
                / "tasks"
            ),
        ),
        help="Directory containing KSearch task datasets named by task profile.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    baselines = BASELINES if args.baseline == "all" else (args.baseline,)
    tasks = TASKS if args.task == "all" else (canonical_task(args.task),)
    workspace = args.workspace_dir or default_workspace()
    api_url = normalize_api_url(args.api_url)
    ctx = RunContext(args=args, api_base_url=api_url, api_key=args.api_key)

    if args.rounds <= 0:
        raise SystemExit("--rounds must be positive")
    if args.kernelbench_attempts <= 0:
        raise SystemExit("--kernelbench-attempts must be positive")
    if args.api_timeout_seconds <= 0:
        raise SystemExit("--api-timeout-seconds must be positive")
    if args.max_output_tokens <= 0:
        raise SystemExit("--max-output-tokens must be positive")

    with ExperimentSummary(
        "table2-generation",
        "compare the staged baseline kernels and their benchmark results with Table 2",
    ) as summary:
        summary.add_result(workspace)
        if not args.no_stage:
            for task in tasks:
                for baseline in baselines:
                    summary.add_result(output_dir(ctx, task, baseline))

        if not args.api_key and not args.dry_run:
            raise SystemExit(
                "LUMEN_GENERATION_API_KEY is required for agent baseline generation"
            )
        if not api_url and not args.dry_run:
            raise SystemExit(
                "LUMEN_GENERATION_API_URL is required for agent baseline generation"
            )

        try:
            for task in tasks:
                for baseline in baselines:
                    run_one(ctx, baseline, task, workspace)
        except ApiBackendLimitedError as exc:
            summary.skip(str(exc))
            return API_BACKEND_LIMITED_EXIT_CODE
    return 0


def default_workspace() -> Path:
    run_id = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    return REPO_ROOT / "scripts" / "table2" / "workspace" / "generation" / run_id


def canonical_task(task: str) -> str:
    try:
        return TASK_DIR[task]
    except KeyError as exc:
        raise SystemExit(f"unsupported task: {task}") from exc


def normalize_api_url(value: str) -> str:
    text = str(value or "").strip().rstrip("/")
    for suffix in ("/chat/completions", "/responses"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    return text


def run_one(ctx: RunContext, baseline: str, task: str, workspace: Path) -> None:
    run_dir = workspace / task / baseline
    if baseline == "kernelbench":
        run_kernelbench(ctx, task, run_dir)
    elif baseline == "cudaforge":
        run_cudaforge(ctx, task, run_dir)
    elif baseline == "kernelfalcon":
        run_kernelfalcon(ctx, task, run_dir)
    elif baseline == "ksearch":
        run_ksearch(ctx, task, run_dir)
    else:
        raise SystemExit(f"unsupported baseline: {baseline}")


def run_kernelbench(ctx: RunContext, task: str, run_dir: Path) -> None:
    root = require_root(
        ctx.args.kernelbench_root,
        "KERNELBENCH_ROOT",
        ctx.args.dry_run,
    )
    ref = reference_path(ctx.args.reference_py, task, ctx.args.dry_run)
    prompts_toml = (
        REPO_ROOT / "prompts" / "kernelbench" / "resources" / "prompts" / "prompts.toml"
    )
    require_path(prompts_toml, ctx.args.dry_run)
    sample_dir = run_dir / "native" / task / "sample_000"
    kernel_path = sample_dir / "kernel.py"
    prompt_path = sample_dir / "prompt.txt"
    response_path = sample_dir / "response.txt"
    meta_path = sample_dir / "meta.json"

    print(
        "[agent-generate] KernelBench "
        f"task={task} backend=hip prompt={KERNELBENCH_PROMPT_KEY[task]}",
        flush=True,
    )
    if ctx.args.dry_run:
        return

    sample_dir.mkdir(parents=True, exist_ok=True)
    get_custom_prompt, extract_first_code = load_kernelbench_helpers(root)
    prompt = get_custom_prompt(
        KERNELBENCH_PROMPT_KEY[task],
        ref_arch_src=ref.read_text(encoding="utf-8"),
        backend="hip",
        option="one_shot",
        precision="bf16",
        include_hardware=False,
        prompts_toml=str(prompts_toml),
    )
    prompt_path.write_text(prompt, encoding="utf-8")
    kernel = ""
    last_error = ""
    backend_limited = False
    attempts = int(ctx.args.kernelbench_attempts)
    for attempt in range(1, attempts + 1):
        print(
            "[agent-generate] KernelBench "
            f"task={task} API attempt={attempt}/{attempts} "
            f"timeout={ctx.args.api_timeout_seconds:g}s "
            f"max_output_tokens={ctx.args.max_output_tokens}",
            flush=True,
        )
        try:
            response = call_openai_responses(ctx, prompt)
        except Exception as exc:
            last_error = f"API request failed: {type(exc).__name__}: {exc}"
            backend_limited = backend_limited or is_api_timeout(exc)
            print(
                f"[agent-generate] KernelBench task={task} {last_error}",
                flush=True,
            )
            continue
        attempt_response = sample_dir / f"response_attempt_{attempt:03d}.txt"
        attempt_response.write_text(response, encoding="utf-8")
        candidate = extract_kernelbench_candidate(response, extract_first_code)
        ok, last_error = validate_kernelbench_hip_output(candidate)
        if ok:
            response_path.write_text(response, encoding="utf-8")
            kernel = candidate
            break
        print(
            "[agent-generate] KernelBench "
            f"task={task} attempt={attempt} rejected: {last_error}; "
            f"response={attempt_response}",
            flush=True,
        )
    if not kernel:
        if backend_limited:
            raise ApiBackendLimitedError(
                "DeepSeek generation skipped: API timed out from inactivity "
                f"after {attempts} attempt(s)"
            )
        raise SystemExit(f"KernelBench did not produce a valid HIP kernel: {last_error}")
    kernel_path.write_text(kernel.rstrip() + "\n", encoding="utf-8")
    meta_path.write_text(
        json.dumps(
            {
                "task": task,
                "backend": "hip",
                "prompt_key": KERNELBENCH_PROMPT_KEY[task],
                "reference": str(ref),
                "prompt": str(prompt_path),
                "response": str(response_path),
                "kernel": str(kernel_path),
                "model": ctx.args.model,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if not ctx.args.no_stage:
        stage_kernelbench(ctx, task, kernel_path)


def load_kernelbench_helpers(root: Path):
    import builtins

    sentinel = object()
    previous_torch = getattr(builtins, "torch", sentinel)
    sys.path.insert(0, str(root / "src"))
    try:
        import torch

        builtins.torch = torch
        from kernelbench.prompt_constructor_toml import get_custom_prompt
        from kernelbench.utils import extract_first_code
    finally:
        try:
            sys.path.remove(str(root / "src"))
        except ValueError:
            pass
        if previous_torch is sentinel:
            try:
                delattr(builtins, "torch")
            except AttributeError:
                pass
        else:
            builtins.torch = previous_torch
    return get_custom_prompt, extract_first_code


def call_openai_responses(ctx: RunContext, prompt: str) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit(f"openai package is required for KernelBench generation: {exc}") from exc

    client = OpenAI(
        api_key=ctx.api_key,
        base_url=ctx.api_base_url,
        timeout=float(ctx.args.api_timeout_seconds),
        max_retries=0,
    )
    response = client.responses.create(
        model=ctx.args.model,
        input=[
            {
                "role": "developer",
                "content": (
                    "Return only one complete Python code block. Begin immediately "
                    "with ```python. Do not output analysis, planning, alternatives, "
                    "or testing code."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        max_output_tokens=int(ctx.args.max_output_tokens),
    )
    text = extract_responses_text(response)
    if not text:
        raise SystemExit("OpenAI response did not contain text")
    return text


def is_api_timeout(exc: BaseException) -> bool:
    """Recognize both stdlib and OpenAI/httpx timeout wrappers."""
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, TimeoutError) or type(current).__name__ in {
            "APITimeoutError",
            "ConnectTimeout",
            "ReadTimeout",
            "WriteTimeout",
            "PoolTimeout",
        }:
            return True
        current = current.__cause__ or current.__context__
    return False


def extract_responses_text(response: object) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text)

    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text_value = getattr(content, "text", None)
            if text_value:
                chunks.append(str(text_value))
            elif isinstance(content, dict) and content.get("text"):
                chunks.append(str(content["text"]))
    return "\n".join(chunks).strip()


def extract_first_code_block(text: str) -> str | None:
    match = re.search(r"```(?:python|py|cpp|c\+\+)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    stripped = text.strip()
    if "class ModelNew" in stripped:
        return stripped
    return None


def extract_kernelbench_candidate(text: str, helper) -> str:
    blocks = re.findall(
        r"```(?:python|py|cpp|c\+\+)?\s*\n(.*?)```",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    for block in reversed(blocks):
        ok, _reason = validate_kernelbench_hip_output(block)
        if ok:
            return block.strip()
    candidate = helper(text, ["python", "cpp"]) or extract_first_code_block(text)
    return str(candidate or "").strip()


def validate_kernelbench_hip_output(code: str) -> tuple[bool, str]:
    stripped = str(code or "").strip()
    if not stripped:
        return False, "empty code block"
    if stripped.startswith("UNSAT_"):
        return False, stripped.splitlines()[0][:120]
    if "class ModelNew" not in stripped:
        return False, "missing ModelNew"
    has_hip_kernel = "<hip/hip_runtime.h>" in stripped or "__global__" in stripped
    if not has_hip_kernel:
        return False, "missing HIP kernel signal"
    return True, ""


def run_cudaforge(ctx: RunContext, task: str, run_dir: Path) -> None:
    root = require_root(ctx.args.cudaforge_root, "CUDAFORGE_ROOT", ctx.args.dry_run)
    ref = reference_path(ctx.args.reference_py, task, ctx.args.dry_run)
    resource_root = baseline_resource_root("cudaforge")
    require_path(resource_root, ctx.args.dry_run)
    env_extra = {
        "CUDAFORGE_PROMPT_PATH_OVERRIDE": str(resource_root)
    }
    cmd = [
        sys.executable,
        "main.py",
        str(ref),
        "--backend",
        "hip",
        "--gpu",
        "MI300X",
        "--server_type",
        "openai",
        "--model_name",
        ctx.args.model,
        "--device",
        str(ctx.args.device),
        "--round",
        str(ctx.args.rounds),
        "--subproc_id",
        "0",
        "--work_dir",
        str(run_dir / "native"),
    ]
    if task == "moe":
        cmd += ["--tol", "1e-3"]
    run_command(ctx, cmd, cwd=root, run_dir=run_dir, env_extra=env_extra)
    if not ctx.args.no_stage and not ctx.args.dry_run:
        stage_cudaforge(ctx, task, root)


def run_kernelfalcon(ctx: RunContext, task: str, run_dir: Path) -> None:
    root = require_root(
        ctx.args.kernelfalcon_root,
        "KERNELFALCON_ROOT",
        ctx.args.dry_run,
    )
    ref = reference_path(ctx.args.reference_py, task, ctx.args.dry_run)
    suffix = optional_resource_path("kernelfalcon", "prompts", f"{task}.md")
    env_extra = {
        "PYTHONPATH": prepend_env_path(ctx.env.get("PYTHONPATH", ""), root),
    }
    if suffix is not None:
        env_extra["KERNELFALCON_PROMPT_SUFFIX_FILE"] = str(suffix)
    cmd = [
        sys.executable,
        "-m",
        "Fuser.auto_agent",
        "--problem",
        str(ref),
        "--ka-model",
        ctx.args.model,
        "--router-model",
        ctx.args.model,
        "--extract-model",
        ctx.args.model,
        "--dispatch-model",
        ctx.args.model,
        "--compose-model",
        ctx.args.model,
        "--ka-workers",
        "1",
        "--ka-rounds",
        str(ctx.args.rounds),
        "--target-platform",
        "rocm",
        "--no-fallback",
        "--verify",
    ]
    run_command(ctx, cmd, cwd=run_dir, run_dir=run_dir, env_extra=env_extra)
    if not ctx.args.no_stage and not ctx.args.dry_run:
        stage_kernelfalcon(ctx, task, run_dir)


def run_ksearch(ctx: RunContext, task: str, run_dir: Path) -> None:
    root = require_root(ctx.args.ksearch_root, "KSEARCH_ROOT", ctx.args.dry_run)
    task_name, kind, definition = KSEARCH_TASKS[task]
    task_path = resolve_path(ctx.args.ksearch_task_root) / task_name
    require_path(
        task_path / "definitions" / kind / f"{definition}.json",
        ctx.args.dry_run,
    )
    require_path(
        task_path / "workloads" / kind / f"{definition}.jsonl",
        ctx.args.dry_run,
    )
    artifacts = run_dir / "artifacts"
    env_extra = {
        "KSEARCH_PROMPT_SUFFIX_FILE": str(
            resource_path("ksearch", ctx.args.dry_run, "prompts", f"{task}.md")
        )
    }
    cmd = [
        sys.executable,
        "-u",
        "generate_kernels_and_eval.py",
        "--task-source",
        "flashinfer",
        "--task-path",
        str(task_path),
        "--definition",
        definition,
        "--model-name",
        ctx.args.model,
        "--base-url",
        ctx.api_base_url,
        "--language",
        "hip",
        "--target-gpu",
        "MI300X",
        "--max-opt-rounds",
        str(ctx.args.rounds),
        "--world-model",
        "--save-solutions",
        "--artifacts-dir",
        str(artifacts),
    ]
    run_command(ctx, cmd, cwd=root, run_dir=run_dir, env_extra=env_extra)
    if not ctx.args.no_stage and not ctx.args.dry_run:
        stage_ksearch(ctx, task, artifacts)


def resolve_path(path: Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def require_path(path: Path, dry_run: bool) -> None:
    if dry_run:
        return
    if not path.exists():
        raise SystemExit(f"required path does not exist: {path}")


def require_root(path: Path | None, env_name: str, dry_run: bool) -> Path:
    if path is None:
        if dry_run:
            return Path(f"${env_name}")
        raise SystemExit(f"{env_name} must point to a prepared patched checkout")
    root = resolve_path(path)
    require_path(root, dry_run)
    return root


def prepend_env_path(existing: str, *paths: Path) -> str:
    entries = [str(path) for path in paths]
    if existing:
        entries.append(existing)
    return os.pathsep.join(entries)


def reference_path(value: Path | None, task: str, dry_run: bool) -> Path:
    if value is not None:
        ref = resolve_path(value)
        require_path(ref, dry_run)
        return ref
    ref = REFERENCE_MODEL[task]
    require_path(ref, dry_run)
    return ref


def run_command(
    ctx: RunContext,
    cmd: list[str],
    *,
    cwd: Path,
    run_dir: Path,
    env_extra: dict[str, str] | None = None,
) -> None:
    printable = " ".join(shlex.quote(x) for x in redacted_command(cmd))
    print(f"[agent-generate] cwd={cwd}")
    print(f"[agent-generate] {printable}")
    if ctx.args.dry_run:
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    command_log = run_dir / "command.log"
    env = ctx.env
    if env_extra:
        env.update(env_extra)
    output_lines: list[str] = []
    with command_log.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            output_lines.append(line)
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        returncode = process.wait()
    if returncode != 0:
        detail = extract_command_failure_detail("".join(output_lines))
        raise SystemExit(
            f"subprocess failed with status {returncode}: {detail}; "
            f"command log: {command_log}; command: {printable}"
        )


def extract_command_failure_detail(output: str) -> str:
    clean = re.sub(r"\x1b\[[0-9;]*m", "", output)
    json_errors = re.findall(r'"error"\s*:\s*"((?:\\.|[^"\\])*)"', clean)
    if json_errors:
        try:
            return json.loads(f'"{json_errors[-1]}"')
        except json.JSONDecodeError:
            return json_errors[-1]

    error_lines = [
        line.strip()
        for line in clean.splitlines()
        if re.search(r"(?:error|exception|failed|traceback)", line, re.IGNORECASE)
    ]
    if error_lines:
        return " | ".join(error_lines[-3:])[-1000:]

    tail = [line.strip() for line in clean.splitlines() if line.strip()]
    if tail:
        return " | ".join(tail[-5:])[-1000:]
    return "subprocess produced no diagnostic output"


def redacted_command(cmd: list[str]) -> list[str]:
    redacted: list[str] = []
    redact_next = False
    for raw in cmd:
        item = str(raw)
        if redact_next:
            redacted.append("<redacted>")
            redact_next = False
            continue
        if item in SENSITIVE_VALUE_FLAGS:
            redacted.append(item)
            redact_next = True
            continue
        for flag in SENSITIVE_VALUE_FLAGS:
            prefix = f"{flag}="
            if item.startswith(prefix):
                item = f"{prefix}<redacted>"
                break
        redacted.append(item)
    return redacted


def output_dir(ctx: RunContext, task: str, baseline: str) -> Path:
    del ctx
    return REPO_ROOT / "datasets" / "inference" / task / baseline


def resource_path(baseline: str, dry_run: bool, *parts: str) -> Path:
    path = baseline_resource_root(baseline) / Path(*parts)
    require_path(path, dry_run)
    return path


def optional_resource_path(baseline: str, *parts: str) -> Path | None:
    path = baseline_resource_root(baseline) / Path(*parts)
    return path if path.is_file() else None


def baseline_resource_root(baseline: str) -> Path:
    prompt_root = REPO_ROOT / "prompts" / baseline / "resources"
    if prompt_root.exists():
        return prompt_root
    return REPO_ROOT / "datasets" / "inference" / baseline / "resources"


def newest(paths: list[Path]) -> Path | None:
    paths = [p for p in paths if p.exists()]
    return max(paths, key=lambda p: p.stat().st_mtime) if paths else None


def copy_file(src: Path, dst: Path, dry_run: bool) -> None:
    print(f"[agent-generate] stage {src} -> {dst}")
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def stage_kernelbench(ctx: RunContext, task: str, kernel_path: Path) -> None:
    if not kernel_path.is_file():
        raise SystemExit(f"KernelBench did not produce expected kernel: {kernel_path}")
    copy_file(
        kernel_path,
        output_dir(ctx, task, "kernelbench") / "model.py",
        ctx.args.dry_run,
    )


def stage_cudaforge(ctx: RunContext, task: str, root: Path) -> None:
    src = root / "test_kernel_0.py"
    if not src.is_file():
        raise SystemExit(f"CUDAForge did not produce expected native output: {src}")
    copy_file(
        src,
        output_dir(ctx, task, "cudaforge") / "model.py",
        ctx.args.dry_run,
    )


def stage_kernelfalcon(ctx: RunContext, task: str, run_dir: Path) -> None:
    result = newest(list(run_dir.glob("**/optimization_result_real.json")))
    dst = output_dir(ctx, task, "kernelfalcon") / "model.py"
    if result is not None:
        data = json.loads(result.read_text(encoding="utf-8"))
        code = str(data.get("kernel_code") or "").strip()
        if code:
            print(f"[agent-generate] stage {result}::kernel_code -> {dst}")
            if not ctx.args.dry_run:
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(code + "\n", encoding="utf-8")
            return
    src = newest(list(run_dir.glob("**/final_kernel.py")))
    if src is None:
        raise SystemExit(f"KernelFalcon did not produce a final kernel under {run_dir}")
    copy_file(src, dst, ctx.args.dry_run)


def stage_ksearch(ctx: RunContext, task: str, artifacts: Path) -> None:
    solution = newest(list(artifacts.glob("**/solutions/**/*.json")))
    if solution is None:
        raise SystemExit(f"KSearch did not produce a solution JSON under {artifacts}")
    data = json.loads(solution.read_text(encoding="utf-8"))
    sources = {
        s["path"]: s.get("content", "")
        for s in data.get("sources", [])
        if "path" in s
    }
    if not sources:
        raise SystemExit(f"KSearch solution has no sources: {solution}")
    dst = output_dir(ctx, task, "ksearch")

    mapping: list[tuple[str, str]]
    if task in ("attention", "gemm"):
        mapping = [
            ("kernel.cu", "best_kernel.cu"),
            ("kernel.h", "kernel.h"),
            ("main.cpp", "best_binding.cpp"),
        ]
        if task == "gemm" and "kernel.h" in sources:
            mapping.append(("kernel.h", "best_kernel.h"))
    else:
        mapping = [
            ("kernel.cu", "kernel.cu"),
            ("kernel.h", "kernel.h"),
            ("main.cpp", "main.cpp"),
        ]

    for src_name, out_name in mapping:
        content = sources.get(src_name)
        if content is None:
            continue
        out = dst / out_name
        print(f"[agent-generate] stage {solution}::{src_name} -> {out}")
        if not ctx.args.dry_run:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
