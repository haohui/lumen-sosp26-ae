#!/usr/bin/env python3
"""Run a single Codex optimization task on a copied working directory.

This is a generic runner:
- input: a source directory to optimize and a prompt
- output: a copied output directory plus runner metadata/artifacts
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
CODEX_SESSION_ID_RE = r"session id:\s*([0-9a-f-]+)"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def dump_json(path: Path, payload: dict) -> None:
    write_text(path, json.dumps(payload, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single Codex optimization task on a copied working directory."
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Seed directory to copy into the output task dir.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for the copied task and runner artifacts.")
    parser.add_argument("--prompt-file", type=Path, default=None, help="Path to a prompt file.")
    parser.add_argument("--prompt-text", default="", help="Inline prompt text.")
    parser.add_argument("--model", default="", help="Optional Codex model override.")
    parser.add_argument("--effort", default="", help="Optional Codex reasoning effort override.")
    parser.add_argument("--agent-timeout-seconds", type=int, default=7200)
    parser.add_argument("--agent-arg", action="append", default=[], help="Extra argument to pass to `codex exec`.")
    parser.add_argument(
        "--codex-dangerously-bypass-approvals-and-sandbox",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def build_prompt(args: argparse.Namespace) -> str:
    parts: list[str] = []
    if args.prompt_file is not None:
        parts.append(read_text(args.prompt_file))
    if args.prompt_text:
        parts.append(args.prompt_text)
    prompt = "\n\n".join(part.strip() for part in parts if part.strip()).strip()
    if not prompt:
        raise SystemExit("Provide --prompt-file and/or --prompt-text.")
    return prompt + "\n"


def prepare_output_dir(input_dir: Path, output_dir: Path) -> None:
    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    shutil.copytree(input_dir, output_dir)


def build_codex_command(args: argparse.Namespace, output_dir: Path) -> list[str]:
    command = [
        "codex",
        "exec",
        "-C",
        str(REPO_ROOT),
        "--color",
        "never",
        "--add-dir",
        str(output_dir.resolve()),
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


def find_codex_session_id(stderr_text: str) -> str | None:
    import re

    match = re.search(CODEX_SESSION_ID_RE, stderr_text or "", flags=re.IGNORECASE)
    return match.group(1) if match else None


def find_trace_path(session_id: str | None) -> Path | None:
    if not session_id:
        return None
    sessions_root = Path.home() / ".codex" / "sessions"
    if not sessions_root.is_dir():
        return None
    matches = sorted(
        sessions_root.rglob(f"*{session_id}.jsonl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def copy_trace_if_present(output_dir: Path, trace_path: Path | None) -> str | None:
    if trace_path is None:
        return None
    target_name = "agent_trace.jsonl"
    shutil.copy2(trace_path, output_dir / target_name)
    return target_name


def invoke_codex(args: argparse.Namespace, output_dir: Path, prompt: str) -> dict:
    command = build_codex_command(args, output_dir)
    completed = subprocess.run(
        command,
        input=prompt,
        text=True,
        capture_output=True,
        cwd=REPO_ROOT,
        timeout=args.agent_timeout_seconds,
        env=os.environ.copy(),
    )
    session_id = find_codex_session_id(completed.stderr)
    trace_path = find_trace_path(session_id)
    copied_trace = copy_trace_if_present(output_dir, trace_path)
    if completed.returncode != 0:
        write_text(
            output_dir / "error.txt",
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


def write_meta(
    output_dir: Path,
    *,
    input_dir: Path,
    prompt_file: Path | None,
    agent: dict,
    started_at: str,
    finished_at: str,
) -> None:
    status = "completed" if agent["returncode"] == 0 else "agent_failed"
    meta = {
        "stage": "codex_optimization",
        "status": status,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "prompt_file": None if prompt_file is None else str(prompt_file),
        "agent_exit_code": agent["returncode"],
        "agent_session_id": agent.get("session_id"),
        "agent_trace_path": agent.get("agent_trace_path"),
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
    }
    dump_json(output_dir / "meta.json", meta)


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    prompt = build_prompt(args)

    prepare_output_dir(input_dir, output_dir)
    write_text(output_dir / "prompt.txt", prompt)

    started_at = utc_now()
    try:
        agent = invoke_codex(args, output_dir, prompt)
    except subprocess.TimeoutExpired:
        agent = {
            "returncode": -1,
            "session_id": None,
            "agent_trace_path": None,
        }
        write_text(output_dir / "error.txt", f"Agent timed out after {args.agent_timeout_seconds}s.\n")
    except Exception:
        agent = {
            "returncode": -1,
            "session_id": None,
            "agent_trace_path": None,
        }
        write_text(output_dir / "error.txt", traceback.format_exc())

    write_meta(
        output_dir,
        input_dir=input_dir,
        prompt_file=args.prompt_file,
        agent=agent,
        started_at=started_at,
        finished_at=utc_now(),
    )

    if agent["returncode"] != 0:
        print(f"Codex optimization failed. See {output_dir / 'error.txt'}", file=sys.stderr)
        return 1

    print(f"Codex optimization completed: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
