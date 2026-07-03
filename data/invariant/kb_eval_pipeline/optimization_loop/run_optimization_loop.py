#!/usr/bin/env python3
"""Minimal Codex optimization loop for KernelBench pipeline runs."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import textwrap
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm


ROUND_RE = re.compile(r"^round(\d+)$", re.IGNORECASE)
HINT_HEADER_RE = re.compile(r"^##\s*(?:Hint\s*)?(\d+)\s*[:.\-]\s*(.+?)\s*$")
CODEX_SESSION_ID_RE = re.compile(r"session id:\s*([0-9a-f-]+)", re.IGNORECASE)
TABLE_MARKER = "<!-- AUTO-GENERATED HISTORY BELOW -->"

KB_EVAL_PIPELINE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = KB_EVAL_PIPELINE_ROOT
THIS_DIR = Path(__file__).resolve().parent
RUN_KERNELBENCH_CASE = KB_EVAL_PIPELINE_ROOT / "harness" / "tools" / "run_kernelbench_case.py"
CONV_PYTEST_TARGET = "/workspace/substrate/test/examples/conv2d/amdgpu/test_conv2d_asym.py"


@dataclass(frozen=True)
class HintSection:
    number: int
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


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small Codex-only KernelBench optimization loop.")
    parser.add_argument("--problem-dir", type=Path, default=None)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--problems", default="")
    parser.add_argument("--parallel-devices", default="")
    parser.add_argument("--agent", choices=["codex"], default="codex")
    parser.add_argument("--template", default="gemm")
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--no-invariant", action="store_true")
    parser.add_argument(
        "--optimization-dir-name",
        default="",
        help="Override the per-problem optimization output directory name.",
    )
    parser.add_argument("--model", default="")
    parser.add_argument("--effort", default="")
    parser.add_argument("--agent-timeout-seconds", type=int, default=7200)
    parser.add_argument("--eval-timeout-seconds", type=int, default=900)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num-correct-trials", type=int, default=1)
    parser.add_argument("--timing-method", default="cudagraph")
    parser.add_argument("--measure-performance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--agent-arg", action="append", default=[])
    parser.add_argument(
        "--codex-dangerously-bypass-approvals-and-sandbox",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def resolve_problem_dirs(args: argparse.Namespace) -> list[Path]:
    if args.problem_dir is not None:
        if args.run_id or args.problems:
            raise SystemExit("Use either --problem-dir or (--run-id with --problems), not both.")
        return [args.problem_dir.expanduser().resolve()]
    if not args.run_id or not args.problems:
        raise SystemExit("Provide --problem-dir, or provide both --run-id and --problems.")
    return [
        (KB_EVAL_PIPELINE_ROOT / "runs" / args.run_id / name.strip()).resolve()
        for name in args.problems.split(",")
        if name.strip()
    ]


def parse_parallel_devices(text: str) -> list[int]:
    if not text:
        return []
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def build_gpu_env(device: int) -> dict[str, str]:
    env = os.environ.copy()
    value = str(device)
    env["CUDA_VISIBLE_DEVICES"] = value
    env["HIP_VISIBLE_DEVICES"] = value
    env.pop("ROCR_VISIBLE_DEVICES", None)
    return env


def list_round_dirs(root: Path) -> list[tuple[int, Path]]:
    rounds: list[tuple[int, Path]] = []
    if not root.is_dir():
        return rounds
    for child in root.iterdir():
        if not child.is_dir():
            continue
        match = ROUND_RE.match(child.name)
        if match:
            rounds.append((int(match.group(1)), child))
    return sorted(rounds)


def highest_source_round(problem_dir: Path) -> Path:
    rounds = list_round_dirs(problem_dir)
    if not rounds:
        raise FileNotFoundError(f"No roundN directories found under: {problem_dir}")
    return rounds[-1][1]


def copy_table_template(dst: Path, src: Path) -> None:
    if dst.is_file():
        return
    write_text(dst, read_text(src).replace("{{TABLE_MARKER}}", TABLE_MARKER))


def parse_hints_document(text: str) -> tuple[str, list[HintSection]]:
    lines = text.splitlines()
    preamble: list[str] = []
    sections: list[HintSection] = []
    current_number: int | None = None
    current_lines: list[str] = []

    for line in lines:
        match = HINT_HEADER_RE.match(line.strip())
        if match:
            if current_number is None:
                preamble = current_lines
            else:
                sections.append(HintSection(current_number, "\n".join(current_lines).strip()))
            current_number = int(match.group(1))
            current_lines = [line]
            continue
        current_lines.append(line)

    if current_number is None:
        preamble = current_lines
    else:
        sections.append(HintSection(current_number, "\n".join(current_lines).strip()))

    return "\n".join(preamble).strip(), sections


def maybe_apply_no_invariants_prompt(template_dir: Path, hints_text: str, enabled: bool) -> str:
    if not enabled:
        return hints_text
    replacement_path = template_dir / "prompt1_no_invariants.md"
    if not replacement_path.is_file():
        raise FileNotFoundError(f"Missing no-invariants prompt file: {replacement_path}")
    _, replacements = parse_hints_document(read_text(replacement_path))
    if not replacements:
        raise ValueError(f"No numbered prompt section found in {replacement_path}")
    replacement = replacements[0]
    preamble, hints = parse_hints_document(hints_text)
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
    pieces = [preamble] if preamble else []
    pieces.extend(hint.markdown for hint in rewritten)
    return "\n\n".join(piece.strip() for piece in pieces if piece.strip()) + "\n"


def round_hints(
    preamble: str,
    hints: list[HintSection],
    round_index: int,
) -> tuple[str, list[int]]:
    chosen = [hint for hint in hints if hint.number == round_index]
    if not chosen and hints:
        chosen = [hints[min(round_index - 1, len(hints) - 1)]]
    pieces = [preamble] if preamble else []
    pieces.extend(hint.markdown for hint in chosen)
    return (
        "\n\n".join(piece.strip() for piece in pieces if piece.strip()).strip(),
        [hint.number for hint in chosen],
    )


def ensure_seed_round(problem_dir: Path, optimization_root: Path) -> Path:
    source = highest_source_round(problem_dir)
    seed = optimization_root / "round0"
    if not seed.exists():
        shutil.copytree(source, seed)
    candidate = seed / "candidate_input.py"
    output = seed / "output_model_new.py"
    if output.is_file() and not candidate.is_file():
        shutil.copy2(output, candidate)
    if not (seed / "input_model.py").is_file() or not output.is_file():
        raise FileNotFoundError(f"Seed round must contain input_model.py and output_model_new.py: {seed}")
    return source


def initialize_problem(problem_dir: Path, template_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    if not problem_dir.is_dir():
        raise FileNotFoundError(f"Problem directory does not exist: {problem_dir}")

    optimization_name = args.optimization_dir_name or (
        "optimization_rounds_no_invariants" if args.no_invariant else "optimization_rounds"
    )
    optimization_root = problem_dir / optimization_name
    optimization_root.mkdir(parents=True, exist_ok=True)
    ensure_seed_round(problem_dir, optimization_root)
    copy_table_template(optimization_root / "TABLE.md", template_dir / "TABLE.md")
    hints_text = maybe_apply_no_invariants_prompt(
        template_dir,
        read_text(template_dir / "HINTS.md"),
        args.no_invariant,
    )
    return {
        "problem_dir": problem_dir,
        "problem_id": problem_dir.name,
        "optimization_root": optimization_root,
        "template_dir": template_dir,
        "hints_text": hints_text,
    }


def uses_kernelbench_case_eval(args: argparse.Namespace) -> bool:
    return args.template == "gemm"


def uses_conv_pytest_eval(args: argparse.Namespace) -> bool:
    return args.template == "conv"


def prepare_round(previous_round: Path, current_round: Path, *, create_case_file: bool) -> None:
    current_round.mkdir(parents=True, exist_ok=True)
    for name in ("input_model.py", "eval_config.json"):
        src = previous_round / name
        if not src.is_file():
            raise FileNotFoundError(f"Missing {name} in {previous_round}")
        shutil.copy2(src, current_round / name)
    previous_output = previous_round / "output_model_new.py"
    if not previous_output.is_file():
        raise FileNotFoundError(f"Missing output_model_new.py in {previous_round}")
    shutil.copy2(previous_output, current_round / "candidate_input.py")
    shutil.copy2(previous_output, current_round / "output_model_new.py")
    if create_case_file:
        write_text(current_round / "case.txt", str(current_round.resolve()) + "\n")


def build_eval_command(args: argparse.Namespace, round_dir: Path) -> list[str]:
    command = [
        sys.executable,
        str(RUN_KERNELBENCH_CASE),
        "--input-file",
        str((round_dir / "case.txt").resolve()),
        "--output",
        str((round_dir / "debug_eval.jsonl").resolve()),
        "--device",
        str(args.device),
        "--num-correct-trials",
        str(args.num_correct_trials),
        "--timing-method",
        args.timing_method,
        "--phase",
        "agent_debug_eval",
    ]
    if args.measure_performance:
        command.append("--measure-performance")
    return command


def strict_constraints() -> str:
    return textwrap.dedent(
        """\
        Strict constraints:
        - Do not browse the web.
        - Do not search online for documentation, examples, repos, or references.
        - Do not use any network access at all.
        - Use only the files already present in the workspace dir.
        - Do not read any other kernel from anywhere in the space!!!
        """
    ).strip()


def render_prompt(args: argparse.Namespace, round_dir: Path, hints_text: str) -> str:
    candidate = (round_dir / "candidate_input.py").resolve()
    output = (round_dir / "output_model_new.py").resolve()

    prompt = hints_text
    prompt = re.sub(r"Optimize the substrate Conv2D kernel in .*", f"Optimize the substrate Conv2D kernel in {output}", prompt, count=1)
    prompt = re.sub(r"Optimize the substrate kernel in .*", f"Optimize the substrate kernel in {output}", prompt, count=1)
    prompt = re.sub(r"optimize the substrate kernel in .*", f"optimize the substrate kernel in {output}", prompt, count=1)
    prompt = prompt.replace(" in xxx.", f" in {output}.")
    prompt = prompt.replace("/workspace/kernel_benchmark/kb_eval_pipeline", str(KB_EVAL_PIPELINE_ROOT.resolve()))
    prompt = prompt.replace("/workspace/kb_eval_pipeline", str(KB_EVAL_PIPELINE_ROOT.resolve()))

    prompt_body = (
        f"Read the starting kernel from {candidate}.\n"
        f"Write the final optimized kernel only to {output}.\n"
        f"Do not modify {candidate}.\n\n"
        f"{prompt.strip()}\n\n"
    )

    if uses_kernelbench_case_eval(args):
        case_path = (round_dir / "case.txt").resolve()
        debug_eval = (round_dir / "debug_eval.jsonl").resolve()
        command = " ".join(build_eval_command(args, round_dir))
        prompt_body += (
        f"Evaluation command for this round:\n"
        f"- The fixed round-local case list is {case_path}. It contains exactly one line: {round_dir.resolve()}.\n"
        f"- Do not edit, overwrite, append to, move, or recreate {case_path}.\n"
        f"- If you manually run correctness/debug evaluation, use exactly this command and no other correctness/eval command:\n"
        f"  {command}\n"
        f"- Write debug evaluation output only to {debug_eval}.\n"
        f"- Do not create, read, or write any shared `case.txt`, `path/to/case.txt`, or other case-list file outside the current round directory.\n\n"
        f"{strict_constraints()}\n\n"
        f"Read only files under /workspace/substrate and under {round_dir.resolve()}.\n"
        f"You may read test and evaluation output files only if they are inside {round_dir.resolve()}.\n"
        f"Do not read files outside those allowed locations.\n\n"
        )

    prompt_body += (
        "Stop when this round's prompt requirement is implemented and correctness passes. "
        "Do not continue iterating on unrelated micro-optimizations.\n"
    )
    return prompt_body


def build_codex_command(args: argparse.Namespace, problem_dir: Path) -> list[str]:
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
    if args.codex_dangerously_bypass_approvals_and_sandbox:
        command.append("--dangerously-bypass-approvals-and-sandbox")
    else:
        command.extend(["--sandbox", "workspace-write"])
    if args.model:
        command.extend(["-m", args.model])
    if args.effort:
        command.extend(["-c", f"model_reasoning_effort={json.dumps(args.effort)}"])
    command.extend(args.agent_arg)
    command.append("-")
    return command


def find_codex_trace(stderr_text: str) -> tuple[str | None, Path | None]:
    match = CODEX_SESSION_ID_RE.search(stderr_text or "")
    if not match:
        return None, None
    session_id = match.group(1)
    sessions_root = Path.home() / ".codex" / "sessions"
    if not sessions_root.is_dir():
        return session_id, None
    matches = sorted(
        sessions_root.rglob(f"*{session_id}.jsonl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return session_id, matches[0] if matches else None


def invoke_codex(args: argparse.Namespace, problem_dir: Path, round_dir: Path, prompt: str) -> dict[str, Any]:
    command = build_codex_command(args, problem_dir)
    completed = subprocess.run(
        command,
        input=prompt,
        text=True,
        capture_output=True,
        cwd=REPO_ROOT,
        timeout=args.agent_timeout_seconds,
        env=getattr(args, "subprocess_env", None) or os.environ.copy(),
    )
    session_id, trace_path = find_codex_trace(completed.stderr)
    copied_trace = None
    if trace_path is not None:
        copied_trace = "agent_trace.jsonl"
        shutil.copy2(trace_path, round_dir / copied_trace)
    if completed.returncode != 0:
        write_text(
            round_dir / "error.txt",
            "Agent command failed.\n"
            f"exit_code: {completed.returncode}\n"
            f"command: {' '.join(command)}\n\n"
            f"stdout:\n{completed.stdout}\n\n"
            f"stderr:\n{completed.stderr}\n",
        )
    return {
        "returncode": completed.returncode,
        "session_id": session_id,
        "agent_trace_path": copied_trace,
    }


def read_last_jsonl(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    last: dict[str, Any] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            last = json.loads(raw)
        except json.JSONDecodeError:
            continue
    return last


def run_eval(args: argparse.Namespace, round_dir: Path) -> tuple[dict[str, Any], int]:
    command = build_eval_command(args, round_dir)
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        cwd=REPO_ROOT,
        timeout=args.eval_timeout_seconds,
        env=getattr(args, "subprocess_env", None) or os.environ.copy(),
    )
    payload = read_last_jsonl(round_dir / "debug_eval.jsonl")
    if completed.returncode != 0:
        payload["_stderr"] = completed.stderr
    debug_eval = round_dir / "debug_eval.jsonl"
    if debug_eval.is_file():
        debug_eval.unlink()
    return payload, completed.returncode


def run_conv_eval(args: argparse.Namespace, round_dir: Path) -> tuple[dict[str, Any], int]:
    command = [sys.executable, "-m", "pytest", CONV_PYTEST_TARGET]
    try:
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            cwd=Path("/workspace/substrate"),
            timeout=args.eval_timeout_seconds,
            env=getattr(args, "subprocess_env", None) or os.environ.copy(),
        )
    except subprocess.TimeoutExpired as exc:
        return (
            {
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
                "exception": f"Evaluation timed out after {args.eval_timeout_seconds}s",
                "_stdout": str(exc.stdout or ""),
                "_stderr": str(exc.stderr or ""),
            },
            124,
        )

    passed = completed.returncode == 0
    return (
        {
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
            "_stdout": completed.stdout,
            "_stderr": completed.stderr,
        },
        completed.returncode,
    )


def normalize_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or numeric <= 0:
        return None
    return numeric


def speedup(runtime_us: Any, ref_runtime_us: Any) -> float | None:
    runtime = normalize_float(runtime_us)
    ref = normalize_float(ref_runtime_us)
    if runtime is None or ref is None:
        return None
    return ref / runtime


def extract_error(agent: dict[str, Any], eval_payload: dict[str, Any], eval_code: int | None) -> tuple[str, str]:
    if agent["returncode"] != 0:
        return "agent_failed", f"Agent exited with code {agent['returncode']}"
    if eval_code not in (0, None):
        error = eval_payload.get("exception") or eval_payload.get("_stderr") or f"Evaluation exited with code {eval_code}"
        return "eval_failed", str(error)[:500]
    return "completed", "OK"


def write_round_meta(
    *,
    context: dict[str, Any],
    round_index: int,
    allowed_numbers: list[int],
    agent: dict[str, Any],
    eval_payload: dict[str, Any],
    eval_code: int | None,
    started_at: str,
    finished_at: str,
) -> dict[str, Any]:
    runtime_us = normalize_float(eval_payload.get("runtime_us"))
    ref_runtime_us = normalize_float(eval_payload.get("ref_runtime_us"))
    status, error = extract_error(agent, eval_payload, eval_code)
    meta = {
        "problem_id": context["problem_id"],
        "stage": "optimization_loop_eval",
        "error": error,
        "compiled": eval_payload.get("compiled"),
        "correctness": eval_payload.get("correctness"),
        "runtime_us": runtime_us,
        "ref_runtime_us": ref_runtime_us,
        "speedup": speedup(runtime_us, ref_runtime_us),
        "optimization_loop": {
            "status": status,
            "round": round_index,
            "allowed_hint_numbers": allowed_numbers,
            "agent_exit_code": agent["returncode"],
            "eval_exit_code": eval_code,
            "agent_trace_path": agent.get("agent_trace_path"),
            "agent_session_id": agent.get("session_id"),
            "started_at_utc": started_at,
            "finished_at_utc": finished_at,
            "error": None if error == "OK" else error,
        },
    }
    dump_json(context["optimization_root"] / f"round{round_index}" / "meta.json", meta)
    if error != "OK":
        write_text(context["optimization_root"] / f"round{round_index}" / "error.txt", error + "\n")
    return meta


def round_complete(round_dir: Path) -> bool:
    meta = load_json(round_dir / "meta.json", {}) or {}
    status = (meta.get("optimization_loop") or {}).get("status")
    return bool(status)


def load_history(optimization_root: Path) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    for index, round_dir in list_round_dirs(optimization_root):
        meta = load_json(round_dir / "meta.json", {}) or {}
        loop = meta.get("optimization_loop") or {}
        runtime_us = normalize_float(meta.get("runtime_us"))
        ref_runtime_us = normalize_float(meta.get("ref_runtime_us"))
        history.append(
            {
                "round": index,
                "status": loop.get("status") or meta.get("stage") or "unknown",
                "allowed_hint_numbers": loop.get("allowed_hint_numbers") or [],
                "compiled": meta.get("compiled"),
                "correctness": meta.get("correctness"),
                "runtime_us": runtime_us,
                "ref_runtime_us": ref_runtime_us,
                "speedup": normalize_float(meta.get("speedup")) or speedup(runtime_us, ref_runtime_us),
                "error": loop.get("error") or (None if meta.get("error") == "OK" else meta.get("error")),
            }
        )
    return history


def update_table(optimization_root: Path) -> None:
    table = optimization_root / "TABLE.md"
    prefix = read_text(table).split(TABLE_MARKER, 1)[0].rstrip() if table.is_file() else ""
    lines = [
        "| round | prompts | status | compiled | correctness | speedup | ref_us | new_us |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in load_history(optimization_root):
        hints = "seed baseline" if item["round"] == 0 else ",".join(map(str, item["allowed_hint_numbers"])) or "-"
        lines.append(
            "| {round} | {hints} | {status} | {compiled} | {correctness} | {speedup} | {ref_us} | {runtime_us} |".format(
                round=item["round"],
                hints=hints,
                status=item["status"],
                compiled=item["compiled"],
                correctness=item["correctness"],
                speedup="-" if item["speedup"] is None else f"{float(item['speedup']):.4f}",
                ref_us="-" if item["ref_runtime_us"] is None else f"{float(item['ref_runtime_us']):.3f}",
                runtime_us="-" if item["runtime_us"] is None else f"{float(item['runtime_us']):.3f}",
            )
        )
    write_text(table, f"{prefix}\n\n{TABLE_MARKER}\n\n" + "\n".join(lines).strip() + "\n")


def run_round(
    args: argparse.Namespace,
    context: dict[str, Any],
    round_index: int,
    hints_text: str,
    allowed_numbers: list[int],
) -> None:
    root = context["optimization_root"]
    previous = root / f"round{round_index - 1}"
    current = root / f"round{round_index}"
    prepare_round(previous, current, create_case_file=uses_kernelbench_case_eval(args))
    prompt = render_prompt(args, current, hints_text)
    write_text(current / "prompt.txt", prompt)

    started_at = utc_now()
    try:
        agent = invoke_codex(args, context["problem_dir"], current, prompt)
    except subprocess.TimeoutExpired as exc:
        stderr = str(exc.stderr or f"Timed out after {args.agent_timeout_seconds}s")
        session_id, trace = find_codex_trace(stderr)
        copied_trace = None
        if trace is not None:
            copied_trace = "agent_trace.jsonl"
            shutil.copy2(trace, current / copied_trace)
        agent = {
            "returncode": -1,
            "session_id": session_id,
            "agent_trace_path": copied_trace,
        }
        write_text(current / "error.txt", f"Agent timed out after {args.agent_timeout_seconds}s.\n")
    except Exception:
        trace = traceback.format_exc()
        agent = {
            "returncode": -1,
            "session_id": None,
            "agent_trace_path": None,
        }
        write_text(current / "error.txt", trace)

    if agent["returncode"] == 0 and uses_kernelbench_case_eval(args):
        eval_payload, eval_code = run_eval(args, current)
    elif agent["returncode"] == 0 and uses_conv_pytest_eval(args):
        eval_payload, eval_code = run_conv_eval(args, current)
    else:
        eval_payload, eval_code = {}, None

    write_round_meta(
        context=context,
        round_index=round_index,
        allowed_numbers=allowed_numbers,
        agent=agent,
        eval_payload=eval_payload,
        eval_code=eval_code,
        started_at=started_at,
        finished_at=utc_now(),
    )


def run_problem(args: argparse.Namespace, template_dir: Path, problem_dir: Path) -> int:
    if getattr(args, "assigned_physical_device", None) is not None:
        print(f"[{problem_dir.name}] Assigned physical GPU {args.assigned_physical_device} (worker device 0)")
    context = initialize_problem(problem_dir, template_dir, args)
    preamble, hints = parse_hints_document(context["hints_text"])

    update_table(context["optimization_root"])

    progress = tqdm(range(1, args.max_rounds + 1), desc=f"Optimization rounds [{problem_dir.name}]", unit="round")
    for round_index in progress:
        round_dir = context["optimization_root"] / f"round{round_index}"
        if round_complete(round_dir):
            progress.set_postfix_str(f"round{round_index} (skip)")
            continue
        hints_text, allowed_numbers = round_hints(preamble, hints, round_index)
        progress.set_postfix_str(f"round{round_index} hints={','.join(map(str, allowed_numbers)) or '-'}")
        run_round(args, context, round_index, hints_text, allowed_numbers)
        update_table(context["optimization_root"])
    progress.close()

    update_table(context["optimization_root"])
    best = max(
        [item for item in load_history(context["optimization_root"]) if item.get("correctness") is True and item.get("speedup") is not None],
        key=lambda item: float(item["speedup"]),
        default=None,
    )
    if best:
        print(f"[{problem_dir.name}] Best round: round{best['round']} speedup={float(best['speedup']):.4f}")
    else:
        print(f"[{problem_dir.name}] No correctness-passing round with a valid speedup was found.")
    print(f"[{problem_dir.name}] Optimization root: {context['optimization_root']}")
    return 0


def main() -> int:
    args = parse_args()
    if args.max_rounds < 0:
        raise SystemExit("--max-rounds must be >= 0")

    template_dir = (THIS_DIR / args.template).resolve()
    if not template_dir.is_dir():
        raise FileNotFoundError(f"Template directory does not exist: {template_dir}")
    for name in ("HINTS.md", "TABLE.md"):
        if not (template_dir / name).is_file():
            raise FileNotFoundError(f"Missing template file: {template_dir / name}")

    problem_dirs = resolve_problem_dirs(args)
    devices = parse_parallel_devices(args.parallel_devices)
    failures: list[str] = []

    if devices and len(problem_dirs) > 1:
        def worker(index_and_problem: tuple[int, Path]) -> str | None:
            index, problem = index_and_problem
            problem_args = argparse.Namespace(**vars(args))
            physical_device = devices[index % len(devices)]
            problem_args.assigned_physical_device = physical_device
            problem_args.subprocess_env = build_gpu_env(physical_device)
            problem_args.device = 0
            try:
                run_problem(problem_args, template_dir, problem)
            except Exception as exc:
                return f"{problem}: {exc}"
            return None

        with ThreadPoolExecutor(max_workers=len(devices)) as executor:
            futures = [executor.submit(worker, item) for item in enumerate(problem_dirs)]
            for future in as_completed(futures):
                failure = future.result()
                if failure:
                    failures.append(failure)
    else:
        for problem in problem_dirs:
            try:
                run_problem(args, template_dir, problem)
            except Exception as exc:
                failures.append(f"{problem}: {exc}")

    if failures:
        print("Optimization completed with failures:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
