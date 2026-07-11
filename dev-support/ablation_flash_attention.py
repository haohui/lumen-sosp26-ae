#!/usr/bin/env python3
"""Run flash-attention ablation benchmarks across Avelang commits."""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

OPTIMIZATION_STEPS = [
    (
        "Naive",
        "2b4ebe74f3332284ab5c9341e96cce9a8c1daa2c",
    ),
    (
        "Transpose V",
        "20aee3f0b3d0e327f83a280b17881422a97754f6",
    ),
    (
        "+Async memcpy",
        "65858b9a3a00cfba6066324d49e4566ee45da1e6",
    ),
    (
        "+Bank conflict",
        "0ac8f508a5702b2625fc7db8109a97c4c852fef7",
    ),
    (
        "+Pipeline+WS",
        "d14857ec1fa4e1b5d8f2f3660a73be21962ce46f",
    ),
    (
        "+Inst. schedule (All)",
        "b5d465ff172aa5da63a77c42676b37077545595d",
    ),
]

SEQ_LENS = [1024, 2048, 4096, 8192, 16384]
DEFAULT_PYTHON = "/opt/venv/bin/python"
TFLOPS_RE = re.compile(r"\btflops=([0-9]+(?:\.[0-9]+)?)\b")


def run(
    cmd: list[str],
    cwd: Path,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )


def require_clean_worktree(repo: Path) -> None:
    status = run(["git", "status", "--porcelain"], cwd=repo).stdout.strip()
    if status:
        raise RuntimeError(
            f"{repo} has uncommitted changes. Commit or stash them before "
            f"running this checkout-based ablation.\n{status}"
        )


def current_ref(repo: Path) -> str:
    return run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo).stdout.strip()


def current_commit(repo: Path) -> str:
    return run(["git", "rev-parse", "HEAD"], cwd=repo).stdout.strip()


def benchmark_commit(repo: Path, python: str, seq_len: int) -> tuple[float, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo / "python")
    cmd = [
        python,
        "benchmark/attention/bench_flash_attn_amdgpu.py",
        "--batch-size",
        "16",
        "--seq-len",
        str(seq_len),
        "--q-heads",
        "8",
        "--kv-heads",
        "1",
        "--head-dim",
        "128",
    ]
    completed = run(cmd, cwd=repo, env=env)
    match = TFLOPS_RE.search(completed.stdout)
    if not match:
        raise RuntimeError(
            f"Could not parse tflops from benchmark output:\n{completed.stdout}"
        )
    return float(match.group(1)), completed.stdout


def write_chart(rows: list[dict[str, object]], output_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    step_names = [name for name, _ in OPTIMIZATION_STEPS]
    values = {
        (int(row["seq_len"]), str(row["name"])): float(row["tflops"]) for row in rows
    }

    x = np.arange(len(SEQ_LENS))
    width = 0.13
    fig, ax = plt.subplots(figsize=(13, 7))

    for idx, name in enumerate(step_names):
        offsets = x + (idx - (len(step_names) - 1) / 2) * width
        heights = [values[(seq_len, name)] for seq_len in SEQ_LENS]
        ax.bar(offsets, heights, width, label=name)

    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Performance (TFLOPS)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(seq_len) for seq_len in SEQ_LENS])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=3, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    chart_path = output_dir / f"flash_attention_ablation_{timestamp}.png"
    fig.savefig(chart_path, dpi=200)
    plt.close(fig)
    return chart_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run flash-attention ablation benchmarks across Avelang commits."
    )
    parser.add_argument(
        "path_to_avelang",
        nargs="?",
        default="/workspace/ae/avelang",
        help="Avelang repository path. Default: /workspace/ae/avelang",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default="/workspace",
        help="Directory for the result CSV and chart. Default: /workspace",
    )
    parser.add_argument(
        "--python",
        default=DEFAULT_PYTHON,
        help=f"Python executable used to run the benchmark. Default: {DEFAULT_PYTHON}",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path(args.path_to_avelang).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not repo.exists():
        raise FileNotFoundError(repo)

    require_clean_worktree(repo)
    original_branch = current_ref(repo)
    original_commit = current_commit(repo)
    restore_target = original_branch if original_branch != "HEAD" else original_commit

    result_file = tempfile.NamedTemporaryFile(
        mode="w",
        newline="",
        prefix="flash_attention_ablation_",
        suffix=".csv",
        dir=output_dir,
        delete=False,
    )
    result_path = Path(result_file.name)
    rows: list[dict[str, object]] = []

    try:
        writer = csv.DictWriter(
            result_file,
            fieldnames=["name", "commit", "seq_len", "tflops"],
        )
        writer.writeheader()
        result_file.flush()

        for name, commit in OPTIMIZATION_STEPS:
            print(f"\n=== {name}: {commit} ===", flush=True)
            run(["git", "checkout", commit], cwd=repo)
            run(["ninja", "-C", "build/"], cwd=repo)

            for seq_len in SEQ_LENS:
                tflops, raw_output = benchmark_commit(repo, args.python, seq_len)
                row = {
                    "name": name,
                    "commit": commit,
                    "seq_len": seq_len,
                    "tflops": tflops,
                }
                rows.append(row)
                writer.writerow(row)
                result_file.flush()
                print(raw_output.strip(), flush=True)

        chart_path = write_chart(rows, output_dir)
        print(f"\nResults CSV: {result_path}")
        print(f"Grouped bar chart: {chart_path}")
    finally:
        result_file.close()
        print(f"\nRestoring checkout to {restore_target}", flush=True)
        run(["git", "checkout", restore_target], cwd=repo)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(exc.stdout or "", file=sys.stderr)
        print(
            f"Command failed with exit code {exc.returncode}: {' '.join(exc.cmd)}",
            file=sys.stderr,
        )
        raise SystemExit(exc.returncode) from None
