"""KernelBench-specific optimization loop built on top of the Codex backend."""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import sys
import textwrap
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lumen.harness.backend.codex import CodexRunResult, CodexRunner, CodexRunnerConfig

ROUND_RE = re.compile(r"^round(\d+)$", re.IGNORECASE)
HINT_HEADER_RE = re.compile(r"^##\s*(?:Hint\s*)?(\d+)\s*[:.\-]\s*(.+?)\s*$")
TABLE_MARKER = "<!-- AUTO-GENERATED HISTORY BELOW -->"


@dataclass(frozen=True)
class HintSection:
    number: int
    markdown: str


@dataclass(frozen=True)
class KernelBenchLoopConfig:
    repo_root: Path | str
    problem_dir: Path | str
    prompt_template: Path | str
    optimization_dir_name: str = "optimization_rounds"
    max_rounds: int = 3
    device: int = 0
    num_correct_trials: int = 1
    timing_method: str = "cudagraph"
    measure_performance: bool = True
    codex_bin: Path | str | None = None
    profile: str | None = None
    model: str | None = None
    model_provider: str | None = None
    reasoning_effort: Any | None = None
    timeout_seconds: float | None = None
    codex_home: Path | str | None = None
    config_overrides: Sequence[str] = field(default_factory=tuple)
    bypass_approvals_and_sandbox: bool = False


@dataclass(frozen=True)
class KernelBenchRoundResult:
    round_index: int
    round_dir: str
    prompt_path: str
    hint_numbers: list[int]
    agent: CodexRunResult


@dataclass(frozen=True)
class KernelBenchLoopResult:
    problem_dir: str
    optimization_root: str
    rounds: list[KernelBenchRoundResult]


class KernelBenchOptimizationLoop:
    def __init__(self, codex_runner: CodexRunner | None = None) -> None:
        self._codex_runner = codex_runner or CodexRunner()

    def run(self, config: KernelBenchLoopConfig) -> KernelBenchLoopResult:
        resolved = _resolve_config(config)
        template_text = resolved.prompt_template.read_text(encoding="utf-8")
        preamble, hints = _parse_hints_document(template_text)

        optimization_root = resolved.problem_dir / resolved.optimization_dir_name
        optimization_root.mkdir(parents=True, exist_ok=True)
        _ensure_seed_round(resolved.problem_dir, optimization_root)
        _ensure_table(
            optimization_root=optimization_root,
            prompt_template=resolved.prompt_template,
        )
        _update_table(optimization_root)

        results: list[KernelBenchRoundResult] = []
        for round_index in range(1, resolved.max_rounds + 1):
            previous_round = optimization_root / f"round{round_index - 1}"
            current_round = optimization_root / f"round{round_index}"
            _prepare_round(
                previous_round,
                current_round,
                repo_root=resolved.repo_root,
                create_case_file=True,
            )

            hint_text, hint_numbers = _round_prompt_text(
                preamble=preamble,
                hints=hints,
                round_index=round_index,
            )
            prompt = _render_prompt(
                repo_root=resolved.repo_root,
                problem_dir=resolved.problem_dir,
                round_dir=current_round,
                template_text=hint_text,
                config=resolved,
            )
            prompt_path = current_round / "prompt.txt"
            prompt_path.write_text(prompt, encoding="utf-8")

            agent = self._codex_runner.execute(
                CodexRunnerConfig(
                    work_dir=resolved.repo_root,
                    prompt=prompt,
                    codex_bin=resolved.codex_bin,
                    profile=resolved.profile,
                    model=resolved.model,
                    model_provider=resolved.model_provider,
                    reasoning_effort=resolved.reasoning_effort,
                    timeout_seconds=resolved.timeout_seconds,
                    env=_runner_env(resolved.codex_home),
                    config_overrides=resolved.config_overrides,
                    bypass_approvals_and_sandbox=resolved.bypass_approvals_and_sandbox,
                )
            )

            if agent.trace_path is not None:
                shutil.copy2(agent.trace_path, current_round / "agent_trace.jsonl")

            if agent.ok:
                eval_payload, eval_code = _run_eval(resolved, current_round)
            else:
                eval_payload, eval_code = {}, None

            round_result = KernelBenchRoundResult(
                round_index=round_index,
                round_dir=str(current_round),
                prompt_path=str(prompt_path),
                hint_numbers=hint_numbers,
                agent=agent,
            )
            _write_round_meta(
                current_round,
                round_result,
                eval_payload=eval_payload,
                eval_code=eval_code,
            )
            results.append(round_result)
            _update_table(optimization_root)

            if not agent.ok:
                break

        return KernelBenchLoopResult(
            problem_dir=str(resolved.problem_dir),
            optimization_root=str(optimization_root),
            rounds=results,
        )


def _resolve_config(config: KernelBenchLoopConfig) -> KernelBenchLoopConfig:
    repo_root = Path(config.repo_root).expanduser().resolve()
    problem_dir = Path(config.problem_dir).expanduser().resolve()
    prompt_template = Path(config.prompt_template).expanduser().resolve()
    codex_home = None
    if config.codex_home is not None:
        codex_home = Path(config.codex_home).expanduser().resolve()

    if not repo_root.is_dir():
        raise FileNotFoundError(f"repo root does not exist: {repo_root}")
    if not problem_dir.is_dir():
        raise FileNotFoundError(f"problem directory does not exist: {problem_dir}")
    if not prompt_template.is_file():
        raise FileNotFoundError(f"prompt template does not exist: {prompt_template}")
    if codex_home is not None and not codex_home.is_dir():
        raise FileNotFoundError(f"codex home does not exist: {codex_home}")

    return replace(
        config,
        repo_root=repo_root,
        problem_dir=problem_dir,
        prompt_template=prompt_template,
        codex_home=codex_home,
        config_overrides=tuple(config.config_overrides),
    )


def _list_round_dirs(root: Path) -> list[tuple[int, Path]]:
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


def _highest_source_round(problem_dir: Path) -> Path:
    rounds = _list_round_dirs(problem_dir)
    if not rounds:
        raise FileNotFoundError(f"no roundN directories found under: {problem_dir}")
    return rounds[-1][1]


def _ensure_seed_round(problem_dir: Path, optimization_root: Path) -> None:
    seed_round = optimization_root / "round0"
    if seed_round.exists():
        return
    shutil.copytree(_highest_source_round(problem_dir), seed_round)
    output_path = seed_round / "output_model_new.py"
    candidate_path = seed_round / "candidate_input.py"
    if output_path.is_file() and not candidate_path.exists():
        shutil.copy2(output_path, candidate_path)


def _ensure_table(*, optimization_root: Path, prompt_template: Path) -> None:
    table_path = optimization_root / "TABLE.md"
    if table_path.exists():
        return
    table_path.write_text(_render_table_template(prompt_template), encoding="utf-8")


def _render_table_template(prompt_template: Path) -> str:
    template_path = prompt_template.with_name("TABLE.md")
    if template_path.is_file():
        return template_path.read_text(encoding="utf-8").replace(
            "{{TABLE_MARKER}}",
            TABLE_MARKER,
        )
    return (
        "# Optimization History\n\n"
        "This file is shared state between optimization rounds.\n"
        "The orchestrator rewrites the history section after each round while preserving this intro.\n\n"
        f"{TABLE_MARKER}\n"
    )


def _default_table_prefix() -> str:
    return _render_table_template(Path("TABLE.md")).split(TABLE_MARKER, 1)[0].rstrip()


def _prepare_round(
    previous_round: Path,
    current_round: Path,
    *,
    repo_root: Path,
    create_case_file: bool,
) -> None:
    current_round.mkdir(parents=True, exist_ok=True)
    for name in ("input_model.py", "eval_config.json"):
        source = previous_round / name
        if source.is_file():
            shutil.copy2(source, current_round / name)

    previous_output = previous_round / "output_model_new.py"
    if not previous_output.is_file():
        raise FileNotFoundError(f"missing output_model_new.py in {previous_round}")
    shutil.copy2(previous_output, current_round / "candidate_input.py")
    shutil.copy2(previous_output, current_round / "output_model_new.py")
    if create_case_file:
        (current_round / "case.txt").write_text(
            str(current_round.relative_to(repo_root)) + "\n",
            encoding="utf-8",
        )


def _parse_hints_document(text: str) -> tuple[str, list[HintSection]]:
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
                sections.append(
                    HintSection(current_number, "\n".join(current_lines).strip())
                )
            current_number = int(match.group(1))
            current_lines = [line]
            continue
        current_lines.append(line)

    if current_number is None:
        preamble = current_lines
    else:
        sections.append(HintSection(current_number, "\n".join(current_lines).strip()))

    return "\n".join(preamble).strip(), sections


def _round_prompt_text(
    *,
    preamble: str,
    hints: list[HintSection],
    round_index: int,
) -> tuple[str, list[int]]:
    if not hints:
        return preamble.strip(), []

    selected = [hint for hint in hints if hint.number == round_index]
    if not selected:
        selected = [hints[min(round_index - 1, len(hints) - 1)]]

    pieces = [preamble] if preamble else []
    pieces.extend(hint.markdown for hint in selected)
    return (
        "\n\n".join(piece.strip() for piece in pieces if piece.strip()).strip(),
        [hint.number for hint in selected],
    )


def _render_prompt(
    *,
    repo_root: Path,
    problem_dir: Path,
    round_dir: Path,
    template_text: str,
    config: KernelBenchLoopConfig,
) -> str:
    candidate_path = _display_path(round_dir / "candidate_input.py", repo_root)
    output_path = _display_path(round_dir / "output_model_new.py", repo_root)
    round_path = _display_path(round_dir, repo_root)
    problem_path = _display_path(problem_dir, repo_root)

    prompt = template_text
    prompt = re.sub(
        r"optimize the substrate kernel in .*",
        f"Optimize the substrate kernel in {output_path}",
        prompt,
        count=1,
        flags=re.IGNORECASE,
    )
    prompt = prompt.replace(" in xxx.", f" in {output_path}.")
    replacements = {
        "{{CANDIDATE_PATH}}": candidate_path,
        "{{OUTPUT_PATH}}": output_path,
        "{{ROUND_DIR}}": round_path,
        "{{PROBLEM_DIR}}": problem_path,
    }
    for key, value in replacements.items():
        prompt = prompt.replace(key, value)

    prompt_body = (
        f"KernelBench optimization round directory: {round_path}\n"
        f"Problem directory: {problem_path}\n"
        f"Read the starting kernel from {candidate_path}.\n"
        f"Write the final optimized kernel only to {output_path}.\n"
        f"Do not modify {candidate_path}.\n"
        f"\n{prompt.strip()}\n\n"
    )
    case_path = _display_path(round_dir / "case.txt", repo_root)
    debug_eval_path = _display_path(round_dir / "debug_eval.jsonl", repo_root)
    command = " ".join(_build_eval_command(config, repo_root, round_dir))
    prompt_body += (
        "Evaluation command for this round:\n"
        f"- The fixed round-local case list is {case_path}. It contains exactly one line: {round_path}.\n"
        f"- Do not edit, overwrite, append to, move, or recreate {case_path}.\n"
        "- If you manually run correctness/debug evaluation, use exactly this command and no other correctness/eval command:\n"
        f"  {command}\n"
        f"- Write debug evaluation output only to {debug_eval_path}.\n"
        "- Do not create, read, or write any shared `case.txt`, `path/to/case.txt`, or other case-list file outside the current round directory.\n\n"
        f"{_strict_constraints()}\n\n"
        f"Read only files under `../substrate` and under {round_path}.\n"
        f"You may read test and evaluation output files only if they are inside {round_path}.\n"
        "Do not read files outside those allowed locations.\n\n"
    )
    prompt_body += (
        "Stop when this round's prompt requirement is implemented and correctness passes. "
        "Do not continue iterating on unrelated micro-optimizations.\n"
    )
    return prompt_body


def _build_eval_command(
    config: KernelBenchLoopConfig,
    repo_root: Path,
    round_dir: Path,
) -> list[str]:
    case_file = (round_dir / "case.txt").relative_to(repo_root)
    debug_eval = (round_dir / "debug_eval.jsonl").relative_to(repo_root)
    runner = (
        repo_root / "python" / "harness" / "bench" / "run_kernelbench_case.py"
    ).relative_to(repo_root)
    command = [
        sys.executable,
        str(runner),
        "--input-file",
        str(case_file),
        "--output",
        str(debug_eval),
        "--device",
        str(config.device),
        "--num-correct-trials",
        str(config.num_correct_trials),
        "--timing-method",
        config.timing_method,
        "--phase",
        "agent_debug_eval",
    ]
    if config.measure_performance:
        command.append("--measure-performance")
    return command


def _strict_constraints() -> str:
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


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _runner_env(codex_home: Path | None) -> dict[str, str] | None:
    if codex_home is None:
        return None
    return {"CODEX_HOME": str(codex_home)}


def _read_last_jsonl(path: Path) -> dict[str, Any]:
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


def _normalize_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or numeric <= 0:
        return None
    return numeric


def _speedup(runtime_us: Any, ref_runtime_us: Any) -> float | None:
    runtime = _normalize_float(runtime_us)
    ref = _normalize_float(ref_runtime_us)
    if runtime is None or ref is None:
        return None
    return ref / runtime


def _run_eval(
    config: KernelBenchLoopConfig,
    round_dir: Path,
) -> tuple[dict[str, Any], int]:
    command = _build_eval_command(config, Path(config.repo_root), round_dir)
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        cwd=Path(config.repo_root),
        check=False,
    )
    payload = _read_last_jsonl(round_dir / "debug_eval.jsonl")
    if completed.returncode != 0:
        payload["_stderr"] = completed.stderr
    debug_eval = round_dir / "debug_eval.jsonl"
    if debug_eval.is_file():
        debug_eval.unlink()
    return payload, completed.returncode


def _extract_error(
    agent: CodexRunResult,
    eval_payload: dict[str, Any],
    eval_code: int | None,
) -> tuple[str, str]:
    if not agent.ok:
        return agent.status, agent.error or "Agent failed."
    if eval_code not in (0, None):
        error = (
            eval_payload.get("exception")
            or eval_payload.get("_stderr")
            or f"Evaluation exited with code {eval_code}"
        )
        return "eval_failed", str(error)[:500]
    return "completed", "OK"


def _load_history(optimization_root: Path) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = [
        {
            "round": 0,
            "prompt_numbers": "seed baseline",
            "status": "seeded",
            "compiled": None,
            "correctness": None,
            "speedup": None,
            "ref_runtime_us": None,
            "runtime_us": None,
        }
    ]
    for index, round_dir in _list_round_dirs(optimization_root):
        if index == 0:
            continue
        meta = json.loads((round_dir / "meta.json").read_text(encoding="utf-8"))
        loop = meta.get("optimization_loop") or {}
        runtime_us = _normalize_float(meta.get("runtime_us"))
        ref_runtime_us = _normalize_float(meta.get("ref_runtime_us"))
        history.append(
            {
                "round": index,
                "prompt_numbers": ",".join(
                    map(str, loop.get("allowed_hint_numbers") or meta.get("hint_numbers", []))
                )
                or "-",
                "status": loop.get("status") or meta.get("stage") or "unknown",
                "compiled": meta.get("compiled"),
                "correctness": meta.get("correctness"),
                "speedup": _normalize_float(meta.get("speedup"))
                or _speedup(runtime_us, ref_runtime_us),
                "ref_runtime_us": ref_runtime_us,
                "runtime_us": runtime_us,
            }
        )
    return history


def _update_table(optimization_root: Path) -> None:
    table_path = optimization_root / "TABLE.md"
    if table_path.is_file():
        prefix = table_path.read_text(encoding="utf-8").split(TABLE_MARKER, 1)[0].rstrip()
    else:
        prefix = _default_table_prefix()
    lines = [
        "| round | prompts | status | compiled | correctness | speedup | ref_us | new_us |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in _load_history(optimization_root):
        lines.append(
            "| {round} | {prompts} | {status} | {compiled} | {correctness} | {speedup} | {ref_us} | {new_us} |".format(
                round=item["round"],
                prompts=item["prompt_numbers"],
                status=item["status"],
                compiled=item["compiled"],
                correctness=item["correctness"],
                speedup="-" if item["speedup"] is None else f"{float(item['speedup']):.4f}",
                ref_us="-" if item["ref_runtime_us"] is None else f"{float(item['ref_runtime_us']):.3f}",
                new_us="-" if item["runtime_us"] is None else f"{float(item['runtime_us']):.3f}",
            )
        )
    table_path.write_text(
        f"{prefix}\n\n{TABLE_MARKER}\n\n" + "\n".join(lines).strip() + "\n",
        encoding="utf-8",
    )


def _write_round_meta(
    round_dir: Path,
    round_result: KernelBenchRoundResult,
    *,
    eval_payload: dict[str, Any],
    eval_code: int | None,
) -> None:
    runtime_us = _normalize_float(eval_payload.get("runtime_us"))
    ref_runtime_us = _normalize_float(eval_payload.get("ref_runtime_us"))
    status, error = _extract_error(round_result.agent, eval_payload, eval_code)
    payload = {
        "round": round_result.round_index,
        "round_dir": round_result.round_dir,
        "prompt_path": round_result.prompt_path,
        "hint_numbers": round_result.hint_numbers,
        "stage": "optimization_loop_eval",
        "error": error,
        "compiled": eval_payload.get("compiled"),
        "correctness": eval_payload.get("correctness"),
        "runtime_us": runtime_us,
        "ref_runtime_us": ref_runtime_us,
        "speedup": _speedup(runtime_us, ref_runtime_us),
        "agent": asdict(round_result.agent),
        "optimization_loop": {
            "status": status,
            "round": round_result.round_index,
            "allowed_hint_numbers": round_result.hint_numbers,
            "agent_session_id": round_result.agent.session_id,
            "started_at_utc": round_result.agent.started_at_utc,
            "finished_at_utc": round_result.agent.finished_at_utc,
            "error": None if error == "OK" else error,
            "eval_exit_code": eval_code,
        },
        "written_at_utc": _utc_now(),
    }
    (round_dir / "meta.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace(
        "+00:00",
        "Z",
    )
