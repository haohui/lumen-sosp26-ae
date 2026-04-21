#!/usr/bin/env python3
"""Run a multi-round Codex/Claude optimization loop for a KernelBench problem."""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm


ROUND_RE = re.compile(r"^round(\d+)$", re.IGNORECASE)
HINT_HEADER_RE = re.compile(r"^##\s*(?:Hint\s*)?(\d+)\s*[:.\-]\s*(.+?)\s*$")
TABLE_MARKER = "<!-- AUTO-GENERATED HISTORY BELOW -->"
CODEX_SESSION_ID_RE = re.compile(r"session id:\s*([0-9a-f-]+)", re.IGNORECASE)

ALT_PROMPT1_NO_INVARIANTS = """## 1: Prompt 1

Optimize the substrate kernel in /workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p09/round0/output_model_new.py

Use MFMA instructions `S.amdgpu.mfma_32x32x8_bf16_f32` to do matrix multiplications and loads the data from global memory using vectorized loads S.amdgpu.raw_buffer_load_x4. The MFMA instruction computes 32x32x8 matmul cooperatively in a wave. Build larger tiles by issuing multiple MFMA instructions across K and output subtiles.

Stage A and B through LDS. Each thread loads operand fragments in 16-byte chunks. Treat each 16-byte fragment as `(4, S.u32)` and reinterpret it as `2 x (4, S.bf16)`. Feed both `(4, S.bf16)` halves into MFMA in natural order. The intended effect is a cooperative `32x32x16` accumulation from two natural MFMA steps. Do not add lane-dependent or K-dependent control flow to select halves.
  - The two `(4, S.bf16)` halves from one 16-byte LDS load collectively represent a swizzled `32x16` operand contribution with 4-column interleaving.
  - Consuming them in natural order must produce the same final C as a naive conceptual layout because operand pairings remain consistent under MFMA swizzle.

Scale the kernel from one wave to four waves without changing the MFMA per-wave invariant. Interpret the 4 warps as a 2 x 2 warp grid. Keep MFMA math identical per warp. Only add warp ownership offsets at operand fetch and output writeback.

Note:

- The current snapshot of repo has examples of the substrate DSL. /workspace/substrate/test/examples/gemm/amdgpu/test_gemm_mfma.py has an implementation of GEMM / MFMA
- If optimizing for a fused kernel, do not directly call the amdgpu gemm kernel inside `substrate_kernel`. You can adopt the changes of it but the goal is to write a standalone fused substrate kernel.
- Make the optimized path cudagraph-safe: never build descriptor / metadata device tensors inside `forward()`. Prebuild or cache them and reuse them; only rebuild if the underlying storage pointer changes.
- Do not use torch native compute anywhere in `output_model_new.py` to perform multiplication or linear algebra, including fallback branches.
- This ban includes any direct or indirect torch compute path such as `torch.matmul`, `torch.mm`, `torch.mv`, `torch.bmm`, `torch.einsum`, `torch.mul`, `torch.addmm`, `torch.chain_matmul`, `torch.nn.functional.linear`, or equivalent compositions that let torch perform the multiply/linear work.
- The optimized kernel must actually issue MFMA instructions in the substrate kernel; a solution that does not use MFMA is not acceptable.
- Write `path/to/case.txt` so it contains exactly one line: the absolute path of the current round directory.
- Run `python /workspace/kernel_benchmark/kb_eval_pipeline/harness/tools/run_kernelbench_case.py --input path/to/case.txt --output path/to/a.jsonl` to evaluate the correctness
- do not use git to try to find any old files!!!
Strict constraints:
  - Do not browse the web.
  - Do not search online for documentation, examples, repos, or references.
  - Do not use any network access at all.
  - Use only the files already present in the workspace dir.
  - Do not read any other kernel from anywhere in the space!!!
"""

ALT_CONV_PROMPT1_NO_INVARIANTS = """## 1: Prompt 1

Optimize the substrate Conv2D kernel in xxx. Replace the scalar implicit-GEMM accumulation in `_igemm_kernel` with a 4-wave MFMA kernel using `S.amdgpu.mfma_32x32x8_bf16_f32`. Keep the existing launch structure and public behavior intact. Rename the optimized entrypoint from `conv2d_asym_naive` to `conv2d_asym`, update `__all__`, and update the benchmark/test callsites to use `conv2d_asym`.

  Scope invariants:
  - This is an MFMA-only transformation.
  - Do not add LDS staging, vectorized loads, async copies, or a different tile shape as part of this change.
  - Preserve the supported behavior of the existing kernel.
  - Do not look at other commits in the repo.

  Update callsites:
  - /workspace/substrate/test/examples/conv2d/amdgpu/test_conv2d_asym.py
  - /workspace/substrate/benchmark/conv2d/bench_conv2d.py

  Validation:
  - Write `path/to/case.txt` so it contains exactly one line: the absolute path of the current round directory.
  - Run `python -m pytest /workspace/substrate/test/examples/conv2d/amdgpu/test_conv2d_asym.py`

- Make the optimized path cudagraph-safe: never build descriptor / metadata device tensors inside `forward()`. Prebuild or cache them and reuse them; only rebuild if the underlying storage pointer changes.
- Do not use torch native compute anywhere in `output_model_new.py` to perform convolution, multiplication, or linear algebra, including fallback branches.
- The optimized kernel must actually issue MFMA instructions in the substrate kernel; a solution that does not use MFMA is not acceptable.
- Do not use git to try to find any old files!!!
Strict constraints:
  - Do not browse the web.
  - Do not search online for documentation, examples, repos, or references.
  - Do not use any network access at all.
  - Use only the files already present in the workspace dir.
  - Do not read any other kernel from anywhere in the space!!!
"""

def build_allowed_read_scope(current_round_dir: Path) -> str:
    round_dir_abs = str(current_round_dir.resolve())
    return (
        "Read only files under /workspace/substrate and under " + round_dir_abs + ".\n"
        + "You may read test and evaluation output files only if they are inside " + round_dir_abs + ".\n"
        + "Do not read files outside those allowed locations."
    )


def build_claude_round_requirements() -> str:
    return textwrap.dedent(
        """\
        Additional requirements for this Claude round:
        - Follow the prompt instructions exactly and do not deviate from them.
        - This round is time-critical. Optimize for speed of execution, not for completeness of research.
        - Do not repeatedly reread the same files unless it is strictly necessary to fix a concrete error.
        - Work in a Codex-like style: read only the minimum necessary files, move to implementation quickly, and avoid long research detours.
        - Read `candidate_input.py` first and use the nearest local MFMA example as the main reference. After that, only read a very small number of additional files if they are strictly necessary.
        - Do not read more than 3 additional files under `/workspace/substrate` beyond the main local MFMA example unless a concrete compile or runtime error forces you to do so.
        - Start editing `output_model_new.py` as soon as you have the minimum information needed for a first working attempt.
        - You must begin editing `output_model_new.py` within the first 5 minutes of the round.
        - Default workflow:
          1. Read `candidate_input.py`.
          2. Read the nearest local MFMA example.
          3. Start editing `output_model_new.py`.
          4. Run validation.
          5. Make only focused fixes required by concrete errors.
        - Do not continue searching for more references before step 3.
        - Prefer a minimal working implementation over a long investigation of the perfect implementation.
        - Use the existing kernel and the nearest local example as your primary references. Do not keep searching for many more examples once you have enough to start.
        - Do not spend time reverse-engineering compiler internals, lowering rules, or unrelated runtime details unless a concrete compile error forces you to do so.
        - Do not read deep Substrate implementation files such as lowering / expr_generator / compiler internals unless you are blocked by a specific error that cannot be resolved otherwise.
        - In particular, do not read `/workspace/substrate/lib/**`, `/workspace/substrate/include/**`, MLIR files, lowering code, compiler code, or expr_generator-related files unless a concrete compile error explicitly requires it.
        - Do not inspect compiler internals just to gain confidence. If you have not yet edited `output_model_new.py`, continue implementing instead of reading more infrastructure code.
        - Do not spend multiple turns summarizing the task, rephrasing the requirements, or explaining the API to yourself. Keep internal deliberation short and move to code quickly.
        - If you are blocked, make one focused fix and continue. Do not repeatedly switch strategies without first trying the simplest implementation path.
        - Aim to produce a first executable version quickly, then iterate only if needed for correctness.
        - If you are unsure about a substrate detail, make the simplest reasonable assumption, implement it, and validate it instead of continuing to read more files.
        - If you have not edited `output_model_new.py` yet, you are not allowed to read more internal implementation files.
        - If validation fails, fix the specific failure. Do not restart the round with a new long reading phase.
        - Work quickly and aim to finish writing within 20 minutes.
        - If you cannot finish within 30 minutes, treat the round as a failure and stop.
        """
    ).strip()

REPO_ROOT = Path(__file__).resolve().parents[4]
KB_EVAL_PIPELINE_ROOT = Path(__file__).resolve().parents[1]
THIS_DIR = Path(__file__).resolve().parent
RUN_KERNELBENCH_CASE = KB_EVAL_PIPELINE_ROOT / "harness" / "tools" / "run_kernelbench_case.py"
CONV_PYTEST_TARGET = "/workspace/substrate/test/examples/conv2d/amdgpu/test_conv2d_asym.py"


def resolve_claude_cli() -> str:
    return shutil.which("claude-yolo") or shutil.which("claude") or "/root/.local/bin/claude"


def apply_claude_provider_env(args: argparse.Namespace, agent_env: dict[str, str]) -> None:
    if args.agent != "claude":
        return
    if args.claude_provider == "default":
        return
    if args.claude_provider == "zhipu":
        base_url = (os.environ.get(args.claude_zhipu_base_url_env) or "").strip()
        auth_env_name = args.claude_zhipu_auth_token_env
        if args.claude_zhipu_key_slot == 2:
            auth_env_name = args.claude_zhipu_auth_token_env_2
        elif args.claude_zhipu_key_slot == 3:
            auth_env_name = args.claude_zhipu_auth_token_env_3
        elif args.claude_zhipu_key_slot == 4:
            auth_env_name = args.claude_zhipu_auth_token_env_4
        elif args.claude_zhipu_key_slot == 5:
            auth_env_name = args.claude_zhipu_auth_token_env_5
        auth_token = (os.environ.get(auth_env_name) or "").strip()
        if not base_url:
            raise RuntimeError(
                f"--claude-provider zhipu requires env {args.claude_zhipu_base_url_env} to be set."
            )
        if not auth_token:
            raise RuntimeError(
                f"--claude-provider zhipu with --claude-zhipu-key-slot {args.claude_zhipu_key_slot} requires env {auth_env_name} to be set."
            )
        agent_env["ANTHROPIC_BASE_URL"] = base_url
        agent_env["ANTHROPIC_AUTH_TOKEN"] = auth_token
        return
    raise RuntimeError(f"Unsupported Claude provider: {args.claude_provider}")


@dataclass(frozen=True)
class HintSection:
    number: int
    title: str
    markdown: str


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    return json.loads(read_text(path))


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def dump_jsonl(path: Path, payloads: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(payload, ensure_ascii=True) for payload in payloads]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def repo_relative(path: Path) -> str:
    return os.path.relpath(path.resolve(), REPO_ROOT.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a multi-round optimization loop for a KernelBench problem directory."
    )
    parser.add_argument(
        "--problem-dir",
        type=Path,
        default=None,
        help="Problem directory containing existing roundN/ artifacts, e.g. runs/.../p01",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Optional run group under runs/, for example kb1 or kb2. Use with --problems.",
    )
    parser.add_argument(
        "--problems",
        default="",
        help="Comma-separated problem ids under the selected run id, for example p09,p12. Use with --run-id.",
    )
    parser.add_argument(
        "--parallel-devices",
        default="",
        help="Comma-separated GPU ids for parallel problem execution, for example 0,1,2,3,4,5,6,7.",
    )
    parser.add_argument(
        "--agent",
        choices=["codex", "claude"],
        default="codex",
        help="Which code agent CLI to use. claude prefers `claude-yolo` and falls back to `claude` with stdin prompt + acceptEdits; codex spawns `codex`.",
    )
    parser.add_argument(
        "--template",
        default="gemm",
        help="Template subdirectory under kb_eval_pipeline/optimization_loop.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=3,
        help="Maximum number of optimization rounds after seed round0.",
    )
    parser.add_argument(
        "--use-mfma32-prompt1",
        action="store_true",
        help="Use the alternate prompt-1 variant without MFMA invariant hints and write to a distinct optimization directory.",
    )
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Suffix appended to optimization_rounds directory (can include leading dash).",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=["sequential", "cumulative"],
        default="sequential",
        help=(
            "How numbered sections in HINTS.md are applied. "
            "`sequential` uses only the matching section for each round; "
            "`cumulative` includes all sections up to the current round."
        ),
    )
    parser.add_argument(
        "--model",
        default="",
        help="Optional model name passed to Codex or Claude.",
    )
    parser.add_argument(
        "--claude-provider",
        choices=["default", "zhipu"],
        default="default",
        help="Provider env set used for Claude. default keeps current ANTHROPIC_* env; zhipu swaps in the Zhipu-specific env vars below.",
    )
    parser.add_argument(
        "--claude-zhipu-base-url-env",
        default="ANTHROPIC_BASE_URL_ZHIPU",
        help="Env var name containing the Zhipu Claude-compatible base URL.",
    )
    parser.add_argument(
        "--claude-zhipu-auth-token-env",
        default="ANTHROPIC_AUTH_TOKEN_ZHIPU",
        help="Env var name containing the primary Zhipu Claude-compatible auth token.",
    )
    parser.add_argument(
        "--claude-zhipu-auth-token-env-2",
        default="ANTHROPIC_AUTH_TOKEN_ZHIPU_2",
        help="Env var name containing the secondary Zhipu Claude-compatible auth token.",
    )
    parser.add_argument(
        "--claude-zhipu-auth-token-env-3",
        default="ANTHROPIC_AUTH_TOKEN_ZHIPU_3",
        help="Env var name containing the tertiary Zhipu Claude-compatible auth token.",
    )
    parser.add_argument(
        "--claude-zhipu-auth-token-env-4",
        default="ANTHROPIC_AUTH_TOKEN_ZHIPU_4",
        help="Env var name containing the quaternary Zhipu Claude-compatible auth token.",
    )
    parser.add_argument(
        "--claude-zhipu-auth-token-env-5",
        default="ANTHROPIC_AUTH_TOKEN_ZHIPU_5",
        help="Env var name containing the fifth Zhipu Claude-compatible auth token.",
    )
    parser.add_argument(
        "--claude-zhipu-key-slot",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=1,
        help="Which Zhipu API key slot to use when --claude-provider zhipu is selected.",
    )
    parser.add_argument(
        "--effort",
        default="",
        help="Optional effort level for Claude. Stored in metadata for both agents.",
    )
    parser.add_argument(
        "--agent-timeout-seconds",
        type=int,
        default=7200,
        help="Timeout for a single Codex/Claude round.",
    )
    parser.add_argument(
        "--eval-timeout-seconds",
        type=int,
        default=900,
        help="Timeout for a single round evaluation subprocess.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Device index passed to run_kernelbench_case.py.",
    )
    parser.add_argument(
        "--num-correct-trials",
        type=int,
        default=1,
        help="Correctness trials passed to run_kernelbench_case.py.",
    )
    parser.add_argument(
        "--measure-performance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to request timing from run_kernelbench_case.py.",
    )
    parser.add_argument(
        "--timing-method",
        default="cudagraph",
        help="Timing method passed to run_kernelbench_case.py.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose evaluator logging.",
    )
    parser.add_argument(
        "--agent-arg",
        action="append",
        default=[],
        help="Repeatable extra CLI arg forwarded to codex/claude.",
    )
    parser.add_argument(
        "--claude-dangerously-skip-permissions",
        action="store_true",
        help="Pass --dangerously-skip-permissions to Claude Code.",
    )
    parser.add_argument(
        "--codex-dangerously-bypass-approvals-and-sandbox",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass --dangerously-bypass-approvals-and-sandbox to Codex exec. Defaults to enabled.",
    )
    return parser.parse_args()


def assert_problem_dir(problem_dir: Path) -> None:
    if not problem_dir.is_dir():
        raise FileNotFoundError(f"Problem directory does not exist: {problem_dir}")


def resolve_problem_dirs(args: argparse.Namespace) -> list[Path]:
    if args.problem_dir is not None:
        if args.run_id or args.problems:
            raise SystemExit("Use either --problem-dir or (--run-id with --problems), not both.")
        return [args.problem_dir.expanduser().resolve()]

    if not args.run_id or not args.problems:
        raise SystemExit("Provide --problem-dir, or provide both --run-id and --problems.")

    problem_names = [item.strip() for item in args.problems.split(",") if item.strip()]
    if not problem_names:
        raise SystemExit("--problems must contain at least one comma-separated problem id.")

    return [
        (KB_EVAL_PIPELINE_ROOT / "runs" / args.run_id / problem_name).resolve()
        for problem_name in problem_names
    ]


def parse_parallel_devices(parallel_devices_text: str) -> list[int]:
    if not parallel_devices_text:
        return []
    devices: list[int] = []
    for item in parallel_devices_text.split(","):
        item = item.strip()
        if not item:
            continue
        devices.append(int(item))
    if not devices:
        raise SystemExit("--parallel-devices must contain at least one GPU id.")
    return devices


def build_gpu_env(physical_device: int) -> dict[str, str]:
    env = os.environ.copy()
    device_text = str(physical_device)
    env["CUDA_VISIBLE_DEVICES"] = device_text
    env["HIP_VISIBLE_DEVICES"] = device_text
    env.pop("ROCR_VISIBLE_DEVICES", None)
    return env


def capture_gpu_environment(env: dict[str, str] | None) -> dict[str, str | None]:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    return {
        "CUDA_VISIBLE_DEVICES": merged.get("CUDA_VISIBLE_DEVICES"),
        "HIP_VISIBLE_DEVICES": merged.get("HIP_VISIBLE_DEVICES"),
        "ROCR_VISIBLE_DEVICES": merged.get("ROCR_VISIBLE_DEVICES"),
    }


def probe_gpu_visibility(env: dict[str, str] | None) -> dict[str, Any]:
    probe_command = [
        sys.executable,
        "-c",
        (
            "import json, torch; "
            "print(json.dumps({"
            "'torch_version': torch.__version__, "
            "'cuda_available': bool(torch.cuda.is_available()), "
            "'device_count': int(torch.cuda.device_count()), "
            "'torch_version_cuda': getattr(torch.version, 'cuda', None), "
            "'torch_version_hip': getattr(torch.version, 'hip', None)"
            "}))"
        ),
    ]
    try:
        completed = subprocess.run(
            probe_command,
            text=True,
            capture_output=True,
            cwd=REPO_ROOT,
            timeout=30,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "command": probe_command,
            "timed_out": True,
            "stdout": normalize_text(exc.stdout),
            "stderr": normalize_text(exc.stderr) or "Timed out after 30s",
        }
    result: dict[str, Any] = {
        "command": probe_command,
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    stdout_text = (completed.stdout or "").strip()
    if stdout_text:
        try:
            result["result"] = json.loads(stdout_text)
        except json.JSONDecodeError:
            result["parse_error"] = "Probe output was not valid JSON"
    return result


def list_round_dirs(root: Path) -> list[tuple[int, Path]]:
    items: list[tuple[int, Path]] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        match = ROUND_RE.match(child.name)
        if match:
            items.append((int(match.group(1)), child))
    return sorted(items, key=lambda item: item[0])


def find_highest_source_round(problem_dir: Path) -> tuple[int, Path]:
    rounds = list_round_dirs(problem_dir)
    if not rounds:
        raise FileNotFoundError(f"No roundN directories found under: {problem_dir}")
    return rounds[-1]


def extract_problem_identity(problem_dir: Path, round_meta: dict[str, Any]) -> tuple[str, str]:
    problem_id = str(round_meta.get("problem_id") or problem_dir.name)
    problem_name = str(round_meta.get("problem_name") or problem_dir.name)
    return problem_id, problem_name


def load_eval_config_from_round(round_dir: Path) -> dict[str, Any]:
    config_path = round_dir / "eval_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing eval_config.json in {round_dir}")
    payload = json.loads(read_text(config_path))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {config_path}")
    return payload


def render_template_text(template_text: str, context: dict[str, str]) -> str:
    rendered = template_text
    for key, value in context.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", value)
    return rendered


def copy_template_if_missing(destination: Path, template_path: Path, context: dict[str, str]) -> None:
    if destination.exists():
        return
    write_text(destination, render_template_text(read_text(template_path), context))


def maybe_override_hints_text(args: argparse.Namespace, hints_text: str) -> str:
    if not args.use_mfma32_prompt1:
        return hints_text

    _, hints = parse_hints_document(hints_text)
    replacement_source = ALT_PROMPT1_NO_INVARIANTS
    if args.template == "conv":
        replacement_source = ALT_CONV_PROMPT1_NO_INVARIANTS
    replacement_prompt1 = parse_hints_document(replacement_source)[1]
    if not replacement_prompt1:
        return hints_text
    replacement = replacement_prompt1[0]
    rewritten: list[HintSection] = []
    replaced = False
    for hint in hints:
        if hint.number == 1 and not replaced:
            rewritten.append(replacement)
            replaced = True
        else:
            rewritten.append(hint)
    if not replaced:
        rewritten.insert(0, replacement)
    return "\n\n".join(hint.markdown.strip() for hint in rewritten if hint.markdown.strip()).strip() + "\n"


def ensure_seed_round(source_round_dir: Path, round0_dir: Path) -> None:
    if not round0_dir.exists():
        shutil.copytree(source_round_dir, round0_dir)
    candidate_input_path = round0_dir / "candidate_input.py"
    output_path = round0_dir / "output_model_new.py"
    if output_path.is_file() and not candidate_input_path.is_file():
        shutil.copy2(output_path, candidate_input_path)


def normalize_runtime_metric(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or numeric <= 0:
        return None
    return numeric


def extract_seed_metrics(round_meta: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    runtime_us = normalize_runtime_metric(round_meta.get("runtime_us"))
    ref_runtime_us = normalize_runtime_metric(round_meta.get("ref_runtime_us"))
    speedup = round_meta.get("speedup")

    if runtime_us is None and round_meta.get("new_ms") is not None:
        runtime_us = normalize_runtime_metric(float(round_meta["new_ms"]) * 1000.0)
    if ref_runtime_us is None and round_meta.get("ref_ms") is not None:
        ref_runtime_us = normalize_runtime_metric(float(round_meta["ref_ms"]) * 1000.0)
    if speedup is None and runtime_us is not None and ref_runtime_us is not None:
        speedup = ref_runtime_us / runtime_us
    else:
        speedup = normalize_runtime_metric(speedup)

    return (
        runtime_us,
        ref_runtime_us,
        speedup,
    )


def augment_seed_meta(round0_dir: Path, source_round_dir: Path, agent_name: str) -> None:
    meta_path = round0_dir / "meta.json"
    meta = load_json(meta_path, {}) or {}
    runtime_us, ref_runtime_us, speedup = extract_seed_metrics(meta)
    meta["has_prompt"] = bool(meta.get("has_prompt"))
    meta["has_output_model_new"] = (round0_dir / "output_model_new.py").is_file()
    meta["runtime_us"] = runtime_us
    meta["ref_runtime_us"] = ref_runtime_us
    meta["speedup"] = speedup
    if runtime_us is not None:
        meta["new_ms"] = runtime_us / 1000.0
    if ref_runtime_us is not None:
        meta["ref_ms"] = ref_runtime_us / 1000.0
    meta["optimization_loop"] = {
        "status": "seeded",
        "round": 0,
        "source_round_name": source_round_dir.name,
        "source_round_path": repo_relative(source_round_dir),
        "candidate_input_path": "candidate_input.py",
        "prompt_path": None,
        "allowed_hint_numbers": [],
        "allowed_hint_titles": [],
        "agent": "seed-copy",
        "requested_agent": agent_name,
        "model": None,
        "effort": None,
        "agent_exit_code": None,
        "agent_message": "Seed round copied from the highest existing source round.",
        "started_at_utc": None,
        "finished_at_utc": utc_now(),
        "error": None,
    }
    dump_json(meta_path, meta)


def initialize_problem_context(problem_dir: Path, template_dir: Path, agent_name: str, optimization_dir_name: str) -> dict[str, Any]:
    highest_round_index, source_round_dir = find_highest_source_round(problem_dir)
    source_round_meta = load_json(source_round_dir / "meta.json", {}) or {}
    eval_config = load_eval_config_from_round(source_round_dir)
    problem_id, problem_name = extract_problem_identity(problem_dir, source_round_meta)

    optimization_root = problem_dir / optimization_dir_name
    optimization_root.mkdir(parents=True, exist_ok=True)

    round0_dir = optimization_root / "round0"
    ensure_seed_round(source_round_dir, round0_dir)
    augment_seed_meta(round0_dir, source_round_dir, agent_name)

    context = {
        "PROBLEM_DIR": repo_relative(problem_dir),
        "PROBLEM_ID": problem_id,
        "PROBLEM_NAME": problem_name,
        "SOURCE_ROUND_NAME": source_round_dir.name,
        "SOURCE_ROUND_PATH": repo_relative(source_round_dir),
        "OPTIMIZATION_ROOT": repo_relative(optimization_root),
        "BACKEND": str(eval_config.get("backend", "")),
        "PRECISION": str(eval_config.get("precision", "")),
        "GPU_ARCH": (
            ", ".join(eval_config.get("gpu_arch", []))
            if isinstance(eval_config.get("gpu_arch"), list)
            else str(eval_config.get("gpu_arch", ""))
        ),
        "SKILLS_ROOT": ".claude/skills",
        "TABLE_MARKER": TABLE_MARKER,
    }

    copy_template_if_missing(optimization_root / "TASK.md", template_dir / "TASK.md", context)
    copy_template_if_missing(optimization_root / "HINTS.md", template_dir / "HINTS.md", context)
    copy_template_if_missing(optimization_root / "TABLE.md", template_dir / "TABLE.md", context)

    return {
        "problem_dir": problem_dir,
        "problem_id": problem_id,
        "problem_name": problem_name,
        "optimization_root": optimization_root,
        "source_round_dir": source_round_dir,
        "source_round_index": highest_round_index,
        "template_dir": template_dir,
        "eval_config": eval_config,
    }


def parse_hints_document(hints_text: str) -> tuple[str, list[HintSection]]:
    lines = hints_text.splitlines()
    preamble_lines: list[str] = []
    hint_headers: list[tuple[int, int, str]] = []
    in_hints = False

    for index, line in enumerate(lines):
        match = HINT_HEADER_RE.match(line.strip())
        if match:
            in_hints = True
            hint_headers.append((index, int(match.group(1)), match.group(2).strip()))
        elif not in_hints:
            preamble_lines.append(line)

    if not hint_headers:
        normalized = hints_text.strip()
        if not normalized:
            return "", []
        return "", [HintSection(number=1, title="Default Prompt", markdown=normalized)]

    hints: list[HintSection] = []
    for header_index, (line_index, number, title) in enumerate(hint_headers):
        next_index = hint_headers[header_index + 1][0] if header_index + 1 < len(hint_headers) else len(lines)
        block = "\n".join(lines[line_index:next_index]).strip()
        hints.append(HintSection(number=number, title=title, markdown=block))

    return "\n".join(preamble_lines).strip(), hints


def build_round_hints_markdown(
    preamble: str,
    hints: list[HintSection],
    round_index: int,
    prompt_mode: str,
) -> tuple[str, list[int], list[str]]:
    if not hints:
        return "", [], []

    if prompt_mode == "cumulative":
        allowed_count = min(round_index, len(hints))
        allowed = hints[:allowed_count]
        markdown = "\n\n".join(hint.markdown.strip() for hint in allowed if hint.markdown.strip()).strip()
        return markdown, [hint.number for hint in allowed], [hint.title for hint in allowed]

    hint_index = min(round_index - 1, len(hints) - 1)
    current = hints[hint_index]
    sections: list[str] = []
    if hint_index > 0:
        previous = hints[hint_index - 1].markdown.strip()
        if previous:
            sections.append(previous)
            sections.append("The optimization above has been completed. Please complete the next optimization.")
    current_markdown = current.markdown.strip()
    if current_markdown:
        sections.append(current_markdown)
    markdown = "\n\n".join(section for section in sections if section).strip()
    return markdown, [current.number], [current.title]


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    if isinstance(text, bytes):
        return text.decode("utf-8", errors="replace")
    return str(text)


def summarize_text(text: Any, limit: int = 400) -> str:
    cleaned = " ".join(normalize_text(text).split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


def prepare_round_directory(previous_round_dir: Path, current_round_dir: Path) -> None:
    current_round_dir.mkdir(parents=True, exist_ok=True)
    reference_path = previous_round_dir / "input_model.py"
    previous_output_path = previous_round_dir / "output_model_new.py"
    eval_config_path = previous_round_dir / "eval_config.json"

    if not reference_path.is_file():
        raise FileNotFoundError(f"Missing input_model.py in {previous_round_dir}")
    if not previous_output_path.is_file():
        raise FileNotFoundError(f"Missing output_model_new.py in {previous_round_dir}")
    if not eval_config_path.is_file():
        raise FileNotFoundError(f"Missing eval_config.json in {previous_round_dir}")

    shutil.copy2(reference_path, current_round_dir / "input_model.py")
    shutil.copy2(eval_config_path, current_round_dir / "eval_config.json")
    shutil.copy2(previous_output_path, current_round_dir / "candidate_input.py")
    shutil.copy2(previous_output_path, current_round_dir / "output_model_new.py")


def render_round_prompt(
    *,
    problem_name: str,
    optimization_root: Path,
    current_round_dir: Path,
    previous_round_dir: Path,
    task_text: str,
    table_text: str,
    allowed_hints_text: str,
    allowed_hint_numbers: list[int],
    agent_name: str,
) -> str:
    del problem_name, optimization_root, previous_round_dir, task_text, table_text, allowed_hint_numbers

    candidate_input_abs = str((current_round_dir / "candidate_input.py").resolve())
    output_model_abs = str((current_round_dir / "output_model_new.py").resolve())
    prompt_text = allowed_hints_text
    replacements = [
        (
            r"Optimize the substrate Conv2D kernel in .*",
            f"Optimize the substrate Conv2D kernel in {output_model_abs}",
        ),
        (
            r"Optimize the substrate kernel in .*",
            f"Optimize the substrate kernel in {output_model_abs}",
        ),
        (
            r"optimize the substrate kernel in .*",
            f"optimize the substrate kernel in {output_model_abs}",
        ),
    ]
    for pattern, replacement in replacements:
        updated = re.sub(pattern, replacement, prompt_text, count=1)
        if updated != prompt_text:
            prompt_text = updated
            break
    prompt_text = prompt_text.replace(" in xxx.", f" in {output_model_abs}.").strip()
    prompt_text = (
        f"Read the starting kernel from {candidate_input_abs}.\n"
        f"Write the final optimized kernel only to {output_model_abs}.\n"
        f"Do not modify {candidate_input_abs}.\n\n"
        + prompt_text
    )
    prompt_text = prompt_text.replace(
        "/workspace/kernel_benchmark/kb_eval_pipeline",
        str(KB_EVAL_PIPELINE_ROOT.resolve()),
    )
    prompt_text = prompt_text.replace(
        "/workspace/kernel_benchmark",
        str(REPO_ROOT.resolve()),
    )
    prompt_text = prompt_text + "\n\n" + build_allowed_read_scope(current_round_dir)
    if agent_name == "claude":
        prompt_text = prompt_text + "\n\n" + build_claude_round_requirements()
    prompt_text = prompt_text + "\n\nStop as soon as you have implemented the current round's prompt requirements and obtained a correctness-passing result for this round. Do not continue iterating for extra performance within the same round."
    return prompt_text + "\n"


def get_agent_timeout_seconds(args: argparse.Namespace) -> int:
    return args.agent_timeout_seconds


def build_codex_command(args: argparse.Namespace, problem_dir: Path, *, bypass_sandbox: bool) -> list[str]:
    command = [
        "codex",
        "exec",
        "-C",
        str(REPO_ROOT),
        "--color",
        "never",
        "--add-dir",
        str(problem_dir.resolve()),
    ]
    if bypass_sandbox:
        command.append("--dangerously-bypass-approvals-and-sandbox")
    else:
        command.extend(["--sandbox", "workspace-write"])
    if args.model:
        command.extend(["-m", args.model])
    command.extend(args.agent_arg)
    command.append("-")
    return command


def build_claude_command(args: argparse.Namespace, problem_dir: Path, prompt_text: str) -> list[str]:
    del prompt_text
    command = [
        resolve_claude_cli(),
        "-p",
        "--permission-mode",
        "dontAsk",
        "--dangerously-skip-permissions",
        "--add-dir",
        str(problem_dir.resolve()),
    ]
    if args.model:
        command.extend(["--model", args.model])
    if args.effort:
        command.extend(["--effort", args.effort])
    command.extend(args.agent_arg)
    command.append("-")
    return command


def extract_codex_session_id(stderr_text: str) -> str | None:
    match = CODEX_SESSION_ID_RE.search(stderr_text or "")
    return match.group(1) if match else None


def find_codex_session_log(session_id: str) -> Path | None:
    sessions_root = Path.home() / ".codex" / "sessions"
    if not sessions_root.is_dir():
        return None
    matches = sorted(
        sessions_root.rglob(f"*{session_id}.jsonl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def extract_codex_messages(session_log_path: Path) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for raw_line in session_log_path.read_text(encoding="utf-8").splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        payload = json.loads(raw_line)
        if payload.get("type") != "response_item":
            continue
        item = payload.get("payload") or {}
        if item.get("type") != "message":
            continue
        content = item.get("content") or []
        text_parts = [
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "output_text"
        ]
        text = "".join(text_parts).strip()
        if not text:
            continue
        messages.append(
            {
                "timestamp": payload.get("timestamp"),
                "role": item.get("role") or "assistant",
                "phase": item.get("phase"),
                "text": text,
            }
        )
    return messages


def extract_codex_token_usage(session_log_path: Path) -> dict[str, Any] | None:
    last_usage: dict[str, Any] | None = None
    for raw_line in session_log_path.read_text(encoding="utf-8").splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        payload = json.loads(raw_line)
        if payload.get("type") != "event_msg":
            continue
        event_payload = payload.get("payload") or {}
        if event_payload.get("type") != "token_count":
            continue
        info = event_payload.get("info") or {}
        total_usage = info.get("total_token_usage")
        last_turn_usage = info.get("last_token_usage")
        if total_usage or last_turn_usage:
            last_usage = {
                "provider": "codex-session-log",
                "captured_at": payload.get("timestamp"),
                "total_token_usage": total_usage,
                "last_token_usage": last_turn_usage,
                "model_context_window": info.get("model_context_window"),
            }
    return last_usage


def _extract_claude_initial_prompt(session_log_path: Path) -> str | None:
    try:
        raw_lines = session_log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None

    for raw_line in raw_lines:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError:
            continue

        if payload.get("type") == "queue-operation" and payload.get("operation") == "enqueue":
            content = payload.get("content")
            if isinstance(content, str) and content.strip():
                return content.lstrip("-").strip()

        if payload.get("type") != "user":
            continue
        message = payload.get("message") or {}
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.lstrip("-").strip()

    return None


def find_claude_session_log(current_round_dir: Path, prompt_text: str) -> Path | None:
    projects_root = Path.home() / ".claude" / "projects"
    if not projects_root.is_dir():
        return None

    round_dir_path = str(current_round_dir.resolve())
    normalized_prompt = prompt_text.strip()

    files = sorted(
        (p for p in projects_root.rglob("*.jsonl") if p.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for session_path in files:
        initial_prompt = _extract_claude_initial_prompt(session_path)
        if not initial_prompt:
            continue
        normalized_initial_prompt = initial_prompt.strip()
        if round_dir_path not in normalized_initial_prompt:
            continue
        if normalized_initial_prompt == normalized_prompt:
            return session_path
        if normalized_prompt and normalized_prompt[:700] == normalized_initial_prompt[:700]:
            return session_path
    return None


def extract_claude_messages(session_log_path: Path) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for raw_line in session_log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        message = payload.get("message") or {}
        role = message.get("role")
        if role not in {"assistant", "user"}:
            continue
        content = message.get("content") or []
        text_parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "text":
                text_value = item.get("text")
                if isinstance(text_value, str) and text_value.strip():
                    text_parts.append(text_value)
            elif item_type == "thinking":
                thinking_value = item.get("thinking")
                if isinstance(thinking_value, str) and thinking_value.strip():
                    text_parts.append(thinking_value)
        text_value = "\n\n".join(part.strip() for part in text_parts if part.strip()).strip()
        if not text_value:
            continue
        messages.append(
            {
                "timestamp": payload.get("timestamp"),
                "role": role,
                "phase": payload.get("type"),
                "text": text_value,
            }
        )
    return messages


def extract_claude_token_usage(session_log_path: Path) -> dict[str, Any] | None:
    totals = {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_output_tokens": 0,
        "total_tokens": 0,
    }
    captured_at: str | None = None
    found_usage = False

    for raw_line in session_log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        usage = (payload.get("message") or {}).get("usage") or {}
        if not isinstance(usage, dict):
            continue
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        cache_read_tokens = usage.get("cache_read_input_tokens")
        if not any(isinstance(v, int) for v in [input_tokens, output_tokens, cache_read_tokens]):
            continue
        found_usage = True
        if isinstance(input_tokens, int):
            totals["input_tokens"] += input_tokens
        if isinstance(cache_read_tokens, int):
            totals["cached_input_tokens"] += cache_read_tokens
        if isinstance(output_tokens, int):
            totals["output_tokens"] += output_tokens
        captured_at = payload.get("timestamp") or captured_at

    if not found_usage:
        return None

    totals["total_tokens"] = (
        totals["input_tokens"]
        + totals["cached_input_tokens"]
        + totals["output_tokens"]
        + totals["reasoning_output_tokens"]
    )
    return {
        "provider": "claude-session-log",
        "captured_at": captured_at,
        "total_token_usage": totals,
    }


def copy_if_exists(src: str | None, dst: Path) -> str | None:
    if not src:
        return None
    src_path = Path(src)
    if not src_path.is_file():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst)
    return dst.name


def write_agent_launch_logs(current_round_dir: Path, command: list[str], stdout_text: str, stderr_text: str) -> dict[str, str]:
    command_path = current_round_dir / "agent_command.txt"
    stdout_path = current_round_dir / "agent_stdout.txt"
    stderr_path = current_round_dir / "agent_stderr.txt"
    if not command_path.exists():
        write_text(command_path, " ".join(command) + "\n")
    if stdout_text or not stdout_path.exists():
        write_text(stdout_path, stdout_text)
    if stderr_text or not stderr_path.exists():
        write_text(stderr_path, stderr_text)
    return {
        "agent_command_path": command_path.name,
        "agent_stdout_path": stdout_path.name,
        "agent_stderr_path": stderr_path.name,
    }


def collect_agent_artifacts(
    *,
    agent_name: str,
    prompt_text: str,
    stdout_text: str,
    stderr_text: str,
    current_round_dir: Path,
    command: list[str],
) -> dict[str, Any]:
    message_export = export_agent_messages(
        agent_name=agent_name,
        prompt_text=prompt_text,
        stdout_text=stdout_text,
        stderr_text=stderr_text,
        output_path=current_round_dir / "agent_messages.jsonl",
        current_round_dir=current_round_dir,
    )
    launch_log_export = write_agent_launch_logs(
        current_round_dir,
        command,
        stdout_text,
        stderr_text,
    )
    copied_session_log_name = copy_if_exists(
        message_export.get("session_log_path"),
        current_round_dir / "agent_trace.jsonl",
    )
    return {
        **message_export,
        **launch_log_export,
        "agent_trace_path": copied_session_log_name,
        "copied_agent_session_log_path": copied_session_log_name,
    }


def export_agent_messages(
    *,
    agent_name: str,
    prompt_text: str,
    stdout_text: str,
    stderr_text: str,
    output_path: Path,
    current_round_dir: Path,
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [{"role": "user", "phase": "prompt", "text": prompt_text}]
    session_id: str | None = None
    session_log_path: Path | None = None
    token_usage: dict[str, Any] | None = None

    if agent_name == "codex":
        session_id = extract_codex_session_id(stderr_text)
        if session_id:
            session_log_path = find_codex_session_log(session_id)
            if session_log_path is not None:
                messages.extend(extract_codex_messages(session_log_path))
                token_usage = extract_codex_token_usage(session_log_path)
    elif agent_name == "claude":
        session_log_path = find_claude_session_log(current_round_dir, prompt_text)
        if session_log_path is not None:
            session_id = session_log_path.stem
            messages.extend(extract_claude_messages(session_log_path))
            token_usage = extract_claude_token_usage(session_log_path)

    if len(messages) == 1 and stdout_text.strip():
        messages.append({"role": "assistant", "phase": "final", "text": stdout_text.strip()})

    dump_jsonl(output_path, messages)
    return {
        "messages_path": output_path.name,
        "session_id": session_id,
        "session_log_path": str(session_log_path) if session_log_path is not None else None,
        "token_usage": token_usage,
    }


def run_subprocess(
    command: list[str],
    *,
    prompt_text: str | None,
    timeout_seconds: int,
    env: dict[str, str] | None,
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    if stdout_path is None and stderr_path is None:
        return subprocess.run(
            command,
            input=prompt_text,
            text=True,
            capture_output=True,
            cwd=REPO_ROOT,
            timeout=timeout_seconds,
            env=env,
        )

    stdout_handle = open(stdout_path, "w", encoding="utf-8") if stdout_path is not None else open(os.devnull, "w", encoding="utf-8")
    stderr_handle = open(stderr_path, "w", encoding="utf-8") if stderr_path is not None else open(os.devnull, "w", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE if prompt_text is not None else None,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            cwd=REPO_ROOT,
            env=env,
        )
        try:
            proc.communicate(input=prompt_text, timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            proc.kill()
            proc.wait()
            stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path is not None and stdout_path.exists() else ""
            stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path is not None and stderr_path.exists() else ""
            raise subprocess.TimeoutExpired(command, timeout_seconds, output=stdout_text, stderr=stderr_text) from exc
    finally:
        stdout_handle.close()
        stderr_handle.close()

    stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path is not None and stdout_path.exists() else ""
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path is not None and stderr_path.exists() else ""
    return subprocess.CompletedProcess(command, proc.returncode, stdout_text, stderr_text)

def invoke_agent(
    args: argparse.Namespace,
    *,
    prompt_text: str,
    problem_dir: Path,
    current_round_dir: Path,
) -> dict[str, Any]:
    command_path = current_round_dir / "agent_command.txt"
    stdout_path = current_round_dir / "agent_stdout.txt"
    stderr_path = current_round_dir / "agent_stderr.txt"
    agent_env = dict(getattr(args, "subprocess_env", None) or os.environ)
    apply_claude_provider_env(args, agent_env)
    timeout_seconds = get_agent_timeout_seconds(args)

    if args.agent == "codex":
        bypass_sandbox = bool(args.codex_dangerously_bypass_approvals_and_sandbox)
        command = build_codex_command(args, problem_dir, bypass_sandbox=bypass_sandbox)
        write_text(command_path, " ".join(command) + "\n")
        completed = run_subprocess(
            command,
            prompt_text=prompt_text,
            timeout_seconds=timeout_seconds,
            env=agent_env,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
    else:
        agent_env["IS_SANDBOX"] = "1"
        command = build_claude_command(args, problem_dir, prompt_text)
        write_text(command_path, " ".join(command) + "\n")
        completed = run_subprocess(
            command,
            prompt_text=prompt_text,
            timeout_seconds=timeout_seconds,
            env=agent_env,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )

    if completed.returncode != 0:
        write_text(
            current_round_dir / "error.txt",
            textwrap.dedent(
                f"""\
                Agent command failed.
                exit_code: {completed.returncode}
                command: {' '.join(command)}

                stdout:
                {completed.stdout}

                stderr:
                {completed.stderr}
                """
            ),
        )

    agent_artifacts = collect_agent_artifacts(
        agent_name=args.agent,
        prompt_text=prompt_text,
        stdout_text=completed.stdout,
        stderr_text=completed.stderr,
        current_round_dir=current_round_dir,
        command=command,
    )

    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "stdout_summary": summarize_text(completed.stdout, limit=1200),
        "stderr_summary": summarize_text(completed.stderr, limit=800),
        "sandbox_bypassed": bool(
            args.agent == "codex" and args.codex_dangerously_bypass_approvals_and_sandbox
        ),
        "agent_messages_path": agent_artifacts["messages_path"],
        "agent_backend_session_id": agent_artifacts["session_id"],
        "agent_backend_log_path": agent_artifacts["session_log_path"],
        "agent_trace_path": agent_artifacts["agent_trace_path"],
        "agent_session_id": agent_artifacts["session_id"],
        "agent_session_log_path": agent_artifacts["session_log_path"],
        "copied_agent_session_log_path": agent_artifacts["copied_agent_session_log_path"],
        "token_usage": agent_artifacts["token_usage"],
        "agent_command_path": agent_artifacts["agent_command_path"],
        "agent_stdout_path": agent_artifacts["agent_stdout_path"],
        "agent_stderr_path": agent_artifacts["agent_stderr_path"],
    }

def evaluate_round(args: argparse.Namespace, round_dir: Path) -> tuple[dict[str, Any], int]:
    if args.template == "conv":
        return evaluate_conv_round(args, round_dir)

    subprocess_env = getattr(args, "subprocess_env", None)
    gpu_environment = capture_gpu_environment(subprocess_env)
    gpu_probe = probe_gpu_visibility(subprocess_env)

    with tempfile.NamedTemporaryFile("w+", suffix=".jsonl", delete=False) as tmp:
        output_path = Path(tmp.name)
    with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False) as tmp:
        case_list_path = Path(tmp.name)
        tmp.write(str(round_dir.resolve()) + "\n")
        tmp.flush()

    command = [
        sys.executable,
        str(RUN_KERNELBENCH_CASE),
        "--input-file",
        str(case_list_path),
        "--output",
        str(output_path),
        "--device",
        str(args.device),
        "--num-correct-trials",
        str(args.num_correct_trials),
        "--timing-method",
        args.timing_method,
    ]
    if args.measure_performance:
        command.append("--measure-performance")
    if args.verbose:
        command.append("--verbose")

    try:
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            cwd=REPO_ROOT,
            timeout=args.eval_timeout_seconds,
            env=getattr(args, "subprocess_env", None),
        )
    except subprocess.TimeoutExpired as exc:
        payload = {
            "case_dir": str(round_dir.resolve()),
            "compiled": False,
            "correctness": False,
            "exception_name": "subprocess.TimeoutExpired",
            "exception": f"Evaluation timed out after {args.eval_timeout_seconds}s",
            "_command": command,
            "_stdout": normalize_text(exc.stdout),
            "_stderr": normalize_text(exc.stderr) or f"Timed out after {args.eval_timeout_seconds}s",
            "_timed_out": True,
            "_gpu_environment": gpu_environment,
            "_gpu_probe": gpu_probe,
        }
        if output_path.is_file():
            output_path.unlink()
        if case_list_path.is_file():
            case_list_path.unlink()
        return payload, 124

    payloads: list[dict[str, Any]] = []
    if output_path.is_file():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                payloads.append(json.loads(line))
        output_path.unlink()
    if case_list_path.is_file():
        case_list_path.unlink()

    payload = payloads[0] if payloads else {}
    payload["_command"] = command
    payload["_stdout"] = completed.stdout
    payload["_stderr"] = completed.stderr
    payload["_gpu_environment"] = gpu_environment
    payload["_gpu_probe"] = gpu_probe
    return payload, completed.returncode


def evaluate_conv_round(args: argparse.Namespace, round_dir: Path) -> tuple[dict[str, Any], int]:
    subprocess_env = getattr(args, "subprocess_env", None)
    gpu_environment = capture_gpu_environment(subprocess_env)
    gpu_probe = probe_gpu_visibility(subprocess_env)

    command = [
        sys.executable,
        "-m",
        "pytest",
        CONV_PYTEST_TARGET,
    ]

    try:
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            cwd=Path("/workspace/substrate"),
            timeout=args.eval_timeout_seconds,
            env=subprocess_env,
        )
    except subprocess.TimeoutExpired as exc:
        payload = {
            "case_dir": str(round_dir.resolve()),
            "backend": "substrate-conv-pytest",
            "compiled": False,
            "correctness": False,
            "runtime_us": None,
            "ref_runtime_us": None,
            "metadata": {
                "validation_mode": "pytest",
                "pytest_target": CONV_PYTEST_TARGET,
            },
            "exception_name": "subprocess.TimeoutExpired",
            "exception": f"Evaluation timed out after {args.eval_timeout_seconds}s",
            "_command": command,
            "_stdout": normalize_text(exc.stdout),
            "_stderr": normalize_text(exc.stderr) or f"Timed out after {args.eval_timeout_seconds}s",
            "_timed_out": True,
            "_gpu_environment": gpu_environment,
            "_gpu_probe": gpu_probe,
        }
        return payload, 124

    passed = completed.returncode == 0
    payload = {
        "case_dir": str(round_dir.resolve()),
        "backend": "substrate-conv-pytest",
        "compiled": passed,
        "correctness": passed,
        "runtime_us": None,
        "ref_runtime_us": None,
        "metadata": {
            "validation_mode": "pytest",
            "pytest_target": CONV_PYTEST_TARGET,
            "pytest_exit_code": completed.returncode,
            "pytest_passed": passed,
        },
        "_command": command,
        "_stdout": completed.stdout,
        "_stderr": completed.stderr,
        "_gpu_environment": gpu_environment,
        "_gpu_probe": gpu_probe,
    }
    if not passed:
        payload["exception"] = "Conv pytest validation failed."
    return payload, completed.returncode


def compute_speedup(runtime_us: Any, ref_runtime_us: Any) -> float | None:
    try:
        runtime_value = float(runtime_us)
        ref_value = float(ref_runtime_us)
    except (TypeError, ValueError):
        return None
    if runtime_value <= 0 or ref_value <= 0:
        return None
    return ref_value / runtime_value


def build_round_meta(
    *,
    problem_id: str,
    problem_name: str,
    current_round_index: int,
    previous_round_index: int,
    allowed_hint_numbers: list[int],
    allowed_hint_titles: list[str],
    prompt_text: str,
    agent_result: dict[str, Any],
    eval_payload: dict[str, Any],
    eval_exit_code: int,
    args: argparse.Namespace,
    started_at: str,
    finished_at: str,
    eval_was_run: bool,
) -> dict[str, Any]:
    runtime_us = normalize_runtime_metric(eval_payload.get("runtime_us"))
    ref_runtime_us = normalize_runtime_metric(eval_payload.get("ref_runtime_us"))
    speedup = compute_speedup(runtime_us, ref_runtime_us)
    compiled = eval_payload.get("compiled")
    correctness = eval_payload.get("correctness")
    token_usage = agent_result.get("token_usage")

    status = "completed"
    error = "OK"
    if agent_result["returncode"] != 0:
        status = "agent_failed"
        error = f"Agent exited with code {agent_result['returncode']}"
    if eval_was_run and eval_exit_code != 0:
        status = "eval_failed" if status == "completed" else f"{status}_eval_failed"
        eval_metadata = eval_payload.get("metadata") or {}
        if eval_payload.get("correctness") is False:
            error = summarize_text(
                eval_metadata.get("correctness_issue")
                or eval_payload.get("exception")
                or eval_payload.get("_stderr")
                or "Evaluation failed correctness checks.",
                limit=500,
            )
        elif not eval_payload.get("compiled") and eval_payload.get("exception"):
            error = summarize_text(eval_payload.get("exception"), limit=500)
        elif not eval_payload and eval_exit_code is not None:
            error = f"Evaluation exited with code {eval_exit_code} before producing any result payload."
        elif not any(key in eval_payload for key in ("compiled", "correctness", "runtime_us", "metadata", "case_dir", "backend")):
            error = (
                f"Evaluation exited with code {eval_exit_code} before producing structured results. "
                f"stdout={summarize_text(eval_payload.get('_stdout'), limit=120) or '<empty>'}; "
                f"stderr={summarize_text(eval_payload.get('_stderr'), limit=120) or '<empty>'}"
            )
        else:
            error = summarize_text(
                eval_payload.get("exception")
                or eval_payload.get("_stderr")
                or f"Evaluation exited with code {eval_exit_code}.",
                limit=500,
            )

    return {
        "problem_id": problem_id,
        "problem_name": problem_name,
        "has_prompt": True,
        "has_output_model_new": True,
        "stage": "optimization_loop_eval",
        "compiled": compiled,
        "correctness": correctness,
        "runtime_us": runtime_us,
        "ref_runtime_us": ref_runtime_us,
        "speedup": speedup,
        "new_ms": (float(runtime_us) / 1000.0) if runtime_us is not None else None,
        "ref_ms": (float(ref_runtime_us) / 1000.0) if ref_runtime_us is not None else None,
        "error": error,
        "eval_result": eval_payload,
        "optimization_loop": {
            "status": status,
            "round": current_round_index,
            "source_round_name": f"round{previous_round_index}",
            "source_round_path": repo_relative((Path(args.problem_dir) / args.optimization_dir_name / f"round{previous_round_index}")),
            "candidate_input_path": "candidate_input.py",
            "prompt_path": "prompt.txt",
            "allowed_hint_numbers": allowed_hint_numbers,
            "allowed_hint_titles": allowed_hint_titles,
            "prompt_mode": args.prompt_mode,
            "agent": args.agent,
            "provider": args.claude_provider if args.agent == "claude" else None,
            "model": args.model or None,
            "effort": args.effort or None,
            "agent_exit_code": agent_result["returncode"],
            "agent_message": agent_result["stdout_summary"] or "Agent produced no textual summary.",
            "agent_stderr": agent_result["stderr_summary"] or None,
            "agent_messages_path": agent_result.get("agent_messages_path"),
            "agent_backend_session_id": agent_result.get("agent_backend_session_id"),
            "agent_backend_log_path": agent_result.get("agent_backend_log_path"),
            "agent_trace_path": agent_result.get("agent_trace_path"),
            "agent_session_id": agent_result.get("agent_session_id"),
            "agent_session_log_path": agent_result.get("agent_session_log_path"),
            "copied_agent_session_log_path": agent_result.get("copied_agent_session_log_path"),
            "agent_command_path": agent_result.get("agent_command_path"),
            "agent_stdout_path": agent_result.get("agent_stdout_path"),
            "agent_stderr_path": agent_result.get("agent_stderr_path"),
            "sandbox_bypassed": agent_result.get("sandbox_bypassed"),
            "token_usage": token_usage,
            "gpu_environment": eval_payload.get("_gpu_environment"),
            "gpu_probe": eval_payload.get("_gpu_probe"),
            "started_at_utc": started_at,
            "finished_at_utc": finished_at,
            "prompt_preview": summarize_text(prompt_text, limit=700),
            "error": None if error == "OK" else error,
        },
    }


def split_table_prefix(table_text: str) -> str:
    if TABLE_MARKER in table_text:
        return table_text.split(TABLE_MARKER, 1)[0].rstrip()
    return table_text.rstrip()


def load_round_history(optimization_root: Path) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    for round_index, round_dir in list_round_dirs(optimization_root):
        meta_path = round_dir / "meta.json"
        if not meta_path.is_file():
            continue
        meta = load_json(meta_path, {}) or {}
        runtime_us, ref_runtime_us, speedup = extract_seed_metrics(meta)
        loop_meta = meta.get("optimization_loop", {})
        error = loop_meta.get("error")
        if not error:
            meta_error = meta.get("error")
            if isinstance(meta_error, str) and not meta_error.startswith("OK"):
                error = meta_error
        history.append(
            {
                "round": round_index,
                "status": loop_meta.get("status") or meta.get("stage") or "unknown",
                "source_round_name": loop_meta.get("source_round_name"),
                "allowed_hint_numbers": loop_meta.get("allowed_hint_numbers") or [],
                "allowed_hint_titles": loop_meta.get("allowed_hint_titles") or [],
                "compiled": meta.get("compiled"),
                "correctness": meta.get("correctness"),
                "runtime_us": runtime_us,
                "ref_runtime_us": ref_runtime_us,
                "speedup": speedup,
                "agent": loop_meta.get("agent"),
                "agent_message": loop_meta.get("agent_message"),
                "token_usage": loop_meta.get("token_usage"),
                "error": error,
            }
        )
    return history


def format_number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def render_table(history: list[dict[str, Any]]) -> str:
    visible_history = history
    lines = [
        "| round | prompts | status | compiled | correctness | speedup | total_tokens | ref_us | new_us |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in visible_history:
        hints = (
            "seed baseline"
            if item["round"] == 0
            else ",".join(str(number) for number in item["allowed_hint_numbers"]) or "-"
        )
        token_usage = item.get("token_usage") or {}
        total_token_usage = token_usage.get("total_token_usage") or token_usage.get("last_token_usage") or {}
        total_tokens = total_token_usage.get("total_tokens")
        lines.append(
            "| {round} | {hints} | {status} | {compiled} | {correctness} | {speedup} | {total_tokens} | {ref_us} | {new_us} |".format(
                round=item["round"],
                hints=hints,
                status=item["status"],
                compiled=item["compiled"],
                correctness=item["correctness"],
                speedup=format_number(item["speedup"], digits=4),
                total_tokens=total_tokens if total_tokens is not None else "-",
                ref_us=format_number(item["ref_runtime_us"], digits=3),
                new_us=format_number(item["runtime_us"], digits=3),
            )
        )

    if not visible_history:
        lines.append("| - | - | pending | - | - | - | - | - | - |")

    note_lines = ["## Round Notes"]
    if not visible_history:
        note_lines.append("- No optimization rounds completed yet.")
    for item in visible_history:
        note_lines.append(f"### round{item['round']}")
        note_lines.append(f"- status: {item['status']}")
        if item["allowed_hint_numbers"]:
            note_lines.append(f"- unlocked hints: {', '.join(str(x) for x in item['allowed_hint_numbers'])}")
        if item["agent"]:
            note_lines.append(f"- agent: {item['agent']}")
        if item["agent_message"]:
            note_lines.append(f"- summary: {item['agent_message']}")
        token_usage = item.get("token_usage") or {}
        total_token_usage = token_usage.get("total_token_usage") or token_usage.get("last_token_usage") or {}
        total_tokens = total_token_usage.get("total_tokens")
        if total_tokens is not None:
            note_lines.append(f"- total_tokens: {total_tokens}")
        if item["error"] and item["error"] != "OK":
            note_lines.append(f"- error: {item['error']}")

    sections = [
        "\n".join(lines).strip(),
        "\n".join(note_lines).strip(),
    ]

    return "\n\n".join(section for section in sections if section).strip() + "\n"


def rewrite_table(optimization_root: Path) -> None:
    table_path = optimization_root / "TABLE.md"
    prefix = split_table_prefix(read_text(table_path))
    history = load_round_history(optimization_root)
    generated = render_table(history)
    final_text = f"{prefix}\n\n{TABLE_MARKER}\n\n{generated}"
    write_text(table_path, final_text.rstrip() + "\n")


def best_round_summary(history: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [
        item for item in history
        if item.get("correctness") is True and item.get("speedup") is not None
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda item: float(item["speedup"]))


def summarize_token_usage(history: list[dict[str, Any]]) -> dict[str, Any]:
    rounds: list[dict[str, Any]] = []
    totals = {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_output_tokens": 0,
        "total_tokens": 0,
    }

    for item in history:
        usage = item.get("token_usage") or {}
        usage_numbers = usage.get("total_token_usage") or usage.get("last_token_usage") or {}
        round_tokens = {
            "round": item["round"],
            "input_tokens": usage_numbers.get("input_tokens"),
            "cached_input_tokens": usage_numbers.get("cached_input_tokens"),
            "output_tokens": usage_numbers.get("output_tokens"),
            "reasoning_output_tokens": usage_numbers.get("reasoning_output_tokens"),
            "total_tokens": usage_numbers.get("total_tokens"),
            "provider": usage.get("provider"),
            "captured_at": usage.get("captured_at"),
        }
        rounds.append(round_tokens)
        for key in totals:
            value = round_tokens.get(key)
            if isinstance(value, int):
                totals[key] += value

    return {"totals": totals, "rounds": rounds}


def rewrite_optimization_meta(context: dict[str, Any], args: argparse.Namespace) -> None:
    optimization_root = context["optimization_root"]
    history = load_round_history(optimization_root)
    best = best_round_summary(history)
    token_summary = summarize_token_usage(history)
    final_round = history[-1] if history else None
    final_output_summary = {
        "round": final_round["round"] if final_round else None,
        "correctness": final_round.get("correctness") if final_round else None,
        "compiled": final_round.get("compiled") if final_round else None,
        "runtime_us": final_round.get("runtime_us") if final_round else None,
        "ref_runtime_us": final_round.get("ref_runtime_us") if final_round else None,
        "speedup": final_round.get("speedup") if final_round else None,
    }
    best_correct_round_summary = {
        "round": best["round"] if best else None,
        "runtime_us": best.get("runtime_us") if best else None,
        "ref_runtime_us": best.get("ref_runtime_us") if best else None,
        "speedup": best.get("speedup") if best else None,
    }
    payload = {
        "problem_id": context["problem_id"],
        "problem_name": context["problem_name"],
        "problem_dir": repo_relative(context["problem_dir"]),
        "optimization_root": repo_relative(optimization_root),
        "template": context["template_dir"].name,
        "source_round_dir": repo_relative(context["source_round_dir"]),
        "agent": args.agent,
        "provider": args.claude_provider if args.agent == "claude" else None,
        "model": args.model or None,
        "effort": args.effort or None,
        "prompt_mode": args.prompt_mode,
        "max_rounds_requested": args.max_rounds,
        "updated_at_utc": utc_now(),
        "final_output": final_output_summary,
        "best_round": best["round"] if best else None,
        "best_speedup": best["speedup"] if best else None,
        "best_correct_round": best_correct_round_summary,
        "token_usage_summary": token_summary,
        "rounds": history,
    }
    dump_json(optimization_root / "meta.json", payload)
    dump_json(optimization_root / "token_usage_summary.json", token_summary)
    dump_json(
        optimization_root / "final_summary.json",
        {
            "problem_id": context["problem_id"],
            "problem_name": context["problem_name"],
            "updated_at_utc": payload["updated_at_utc"],
            "final_output": final_output_summary,
            "best_correct_round": best_correct_round_summary,
        },
    )


def round_is_complete(round_dir: Path) -> bool:
    meta = load_json(round_dir / "meta.json", {}) or {}
    status = (meta.get("optimization_loop") or {}).get("status")
    return bool(status and status != "unknown")


def run_single_round(
    args: argparse.Namespace,
    context: dict[str, Any],
    round_index: int,
    allowed_hints_text: str,
    allowed_hint_numbers: list[int],
    allowed_hint_titles: list[str],
) -> None:
    optimization_root = context["optimization_root"]
    previous_round_dir = optimization_root / f"round{round_index - 1}"
    current_round_dir = optimization_root / f"round{round_index}"

    prepare_round_directory(previous_round_dir, current_round_dir)

    prompt_text = render_round_prompt(
        problem_name=context["problem_name"],
        optimization_root=optimization_root,
        current_round_dir=current_round_dir,
        previous_round_dir=previous_round_dir,
        task_text=read_text(optimization_root / "TASK.md"),
        table_text=read_text(optimization_root / "TABLE.md"),
        allowed_hints_text=allowed_hints_text,
        allowed_hint_numbers=allowed_hint_numbers,
        agent_name=args.agent,
    )
    write_text(current_round_dir / "prompt.txt", prompt_text)

    started_at = utc_now()
    effective_agent_timeout_seconds = get_agent_timeout_seconds(args)
    try:
        agent_result = invoke_agent(
            args,
            prompt_text=prompt_text,
            problem_dir=context["problem_dir"],
            current_round_dir=current_round_dir,
        )
    except subprocess.TimeoutExpired as exc:
        stdout_text = normalize_text(exc.stdout)
        stderr_text = normalize_text(exc.stderr) or f"Timed out after {effective_agent_timeout_seconds}s"
        timeout_command = list(exc.cmd) if isinstance(exc.cmd, (list, tuple)) else [str(exc.cmd)]
        timeout_artifacts = collect_agent_artifacts(
            agent_name=args.agent,
            prompt_text=prompt_text,
            stdout_text=stdout_text,
            stderr_text=stderr_text,
            current_round_dir=current_round_dir,
            command=timeout_command,
        )
        agent_result = {
            "command": timeout_command,
            "returncode": -1,
            "stdout": stdout_text,
            "stderr": stderr_text,
            "stdout_summary": summarize_text(stdout_text, limit=1200),
            "stderr_summary": summarize_text(stderr_text, limit=800),
            "sandbox_bypassed": bool(
                args.agent == "codex" and args.codex_dangerously_bypass_approvals_and_sandbox
            ),
            "agent_messages_path": timeout_artifacts["messages_path"],
            "agent_backend_session_id": timeout_artifacts["session_id"],
            "agent_backend_log_path": timeout_artifacts["session_log_path"],
            "agent_trace_path": timeout_artifacts["agent_trace_path"],
            "agent_session_id": timeout_artifacts["session_id"],
            "agent_session_log_path": timeout_artifacts["session_log_path"],
            "copied_agent_session_log_path": timeout_artifacts["copied_agent_session_log_path"],
            "token_usage": timeout_artifacts["token_usage"],
            "agent_command_path": timeout_artifacts["agent_command_path"],
            "agent_stdout_path": timeout_artifacts["agent_stdout_path"],
            "agent_stderr_path": timeout_artifacts["agent_stderr_path"],
        }
        write_text(current_round_dir / "error.txt", f"Agent timed out after {effective_agent_timeout_seconds}s.\n")
    except Exception:
        trace = traceback.format_exc()
        agent_result = {
            "command": [],
            "returncode": -1,
            "stdout": "",
            "stderr": trace,
            "stdout_summary": "",
            "stderr_summary": summarize_text(trace, limit=800),
        }
        write_text(current_round_dir / "error.txt", trace)

    if agent_result["returncode"] == 0:
        eval_payload, eval_exit_code = evaluate_round(args, current_round_dir)
        eval_was_run = True
    else:
        eval_payload = {
            "skipped": True,
            "reason": "Evaluation skipped because the agent command failed before producing a reliable round output.",
        }
        eval_exit_code = None
        eval_was_run = False
    finished_at = utc_now()

    meta = build_round_meta(
        problem_id=context["problem_id"],
        problem_name=context["problem_name"],
        current_round_index=round_index,
        previous_round_index=round_index - 1,
        allowed_hint_numbers=allowed_hint_numbers,
        allowed_hint_titles=allowed_hint_titles,
        prompt_text=prompt_text,
        agent_result=agent_result,
        eval_payload=eval_payload,
        eval_exit_code=eval_exit_code,
        args=args,
        started_at=started_at,
        finished_at=finished_at,
        eval_was_run=eval_was_run,
    )
    dump_json(current_round_dir / "meta.json", meta)
    dump_json(current_round_dir / "token_usage.json", meta["optimization_loop"].get("token_usage"))

    if meta["error"] != "OK":
        write_text(current_round_dir / "error.txt", meta["error"] + "\n")


def run_problem(args: argparse.Namespace, template_dir: Path, problem_dir: Path) -> int:
    problem_dir = problem_dir.expanduser().resolve()
    args.problem_dir = problem_dir
    assert_problem_dir(problem_dir)
    assigned_device = getattr(args, "assigned_physical_device", None)
    if assigned_device is not None:
        print(f"[{problem_dir.name}] Assigned physical GPU {assigned_device} (worker device 0)")

    if args.output_suffix:
        optimization_dir_name = f"optimization_rounds{args.output_suffix}"
    else:
        optimization_dir_name = "optimization_rounds"
        if args.use_mfma32_prompt1:
            optimization_dir_name = "optimization_rounds_no_invariants"
        if args.agent == "claude":
            optimization_dir_name = f"{optimization_dir_name}_claude"
    args.optimization_dir_name = optimization_dir_name
    context = initialize_problem_context(problem_dir, template_dir, args.agent, optimization_dir_name)
    hints_path = context["optimization_root"] / "HINTS.md"
    hints_text = maybe_override_hints_text(args, read_text(hints_path))
    if hints_text != read_text(hints_path):
        write_text(hints_path, hints_text)
    hints_preamble, hints = parse_hints_document(hints_text)
    planned_rounds = args.max_rounds

    rewrite_table(context["optimization_root"])
    rewrite_optimization_meta(context, args)

    round_progress = tqdm(
        range(1, planned_rounds + 1),
        desc=f"Optimization rounds [{problem_dir.name}]",
        unit="round",
        disable=(planned_rounds == 0),
    )
    for round_index in round_progress:
        round_progress.set_postfix_str(f"round{round_index}")
        round_dir = context["optimization_root"] / f"round{round_index}"
        if round_is_complete(round_dir):
            round_progress.set_postfix_str(f"round{round_index} (skip)")
            continue
        allowed_hints_text, allowed_hint_numbers, allowed_hint_titles = build_round_hints_markdown(
            hints_preamble,
            hints,
            round_index,
            args.prompt_mode,
        )
        round_progress.set_postfix_str(
            f"round{round_index} hints={','.join(str(x) for x in allowed_hint_numbers) or '-'}"
        )
        run_single_round(
            args,
            context,
            round_index,
            allowed_hints_text,
            allowed_hint_numbers,
            allowed_hint_titles,
        )
        rewrite_table(context["optimization_root"])
        rewrite_optimization_meta(context, args)
    round_progress.close()

    rewrite_table(context["optimization_root"])
    rewrite_optimization_meta(context, args)

    best = best_round_summary(load_round_history(context["optimization_root"]))
    if best:
        print(
            f"[{problem_dir.name}] Best round: round{best['round']} speedup={format_number(best['speedup'], digits=4)} "
            f"compiled={best['compiled']} correctness={best['correctness']}"
        )
    else:
        print(f"[{problem_dir.name}] No correctness-passing round with a valid speedup was found.")
    print(f"[{problem_dir.name}] Optimization root: {repo_relative(context['optimization_root'])}")
    return 0


def main() -> int:
    args = parse_args()
    if args.max_rounds < 0:
        raise SystemExit("--max-rounds must be >= 0")

    parallel_devices = parse_parallel_devices(args.parallel_devices)

    template_dir = (THIS_DIR / args.template).resolve()
    if not template_dir.is_dir():
        raise FileNotFoundError(f"Template directory does not exist: {template_dir}")
    for required_name in ("TASK.md", "HINTS.md", "TABLE.md"):
        if not (template_dir / required_name).is_file():
            raise FileNotFoundError(f"Missing template file: {template_dir / required_name}")

    problem_dirs = resolve_problem_dirs(args)
    failures: list[str] = []

    def make_problem_args(problem_dir: Path, physical_device: int | None) -> argparse.Namespace:
        problem_args = argparse.Namespace(**vars(args))
        if physical_device is not None:
            problem_args.assigned_physical_device = physical_device
            problem_args.subprocess_env = build_gpu_env(physical_device)
            problem_args.device = 0
        else:
            problem_args.assigned_physical_device = None
            problem_args.subprocess_env = None
        return problem_args

    if parallel_devices and len(problem_dirs) > 1:
        max_workers = min(len(parallel_devices), len(problem_dirs))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {}
            for index, problem_dir in enumerate(problem_dirs):
                physical_device = parallel_devices[index % len(parallel_devices)]
                future = executor.submit(
                    run_problem,
                    make_problem_args(problem_dir, physical_device),
                    template_dir,
                    problem_dir,
                )
                future_map[future] = problem_dir
            for future in as_completed(future_map):
                problem_dir = future_map[future]
                try:
                    future.result()
                except Exception as exc:
                    failures.append(f"{problem_dir}: {exc}")
                    print(f"[{problem_dir.name}] Failed: {exc}", file=sys.stderr)
    else:
        for problem_dir in problem_dirs:
            problem_args = make_problem_args(problem_dir, parallel_devices[0] if parallel_devices else None)
            try:
                run_problem(problem_args, template_dir, problem_dir)
            except Exception as exc:
                failures.append(f"{problem_dir}: {exc}")
                print(f"[{problem_dir.name}] Failed: {exc}", file=sys.stderr)

    if failures:
        print("Serial optimization completed with failures:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
