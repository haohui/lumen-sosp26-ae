#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPO_ROOT / "scripts" / "benchmark"
WORKLOADS = [1024, 2048, 4096, 8192, 16384]
BASELINE_COLUMNS = [
    "hipblaslt",
    "hipkittens",
    "aiter",
    "triton",
    "aiter_asm",
    "aiter_triton",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reproduce Table 2 benchmark results")
    p.add_argument("--workspace-dir", type=Path, default=None)
    p.add_argument("--python", type=str, default=sys.executable)
    p.add_argument(
        "--skip-run",
        action="store_true",
        help="Process existing JSONL files in the workspace without running kernels",
    )
    return p.parse_args()


def default_workspace() -> Path:
    run_id = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    return Path(__file__).resolve().parent / "workspace" / run_id


def _env() -> dict[str, str]:
    env = os.environ.copy()
    pythonpath = str(BENCHMARK_DIR)
    existing = env.get("PYTHONPATH")
    if existing:
        pythonpath = pythonpath + os.pathsep + existing
    env["PYTHONPATH"] = pythonpath
    return env


def _run_jsonl_command(cmd: list[str], out_path: Path) -> None:
    cp = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
        env=_env(),
    )
    with out_path.open("a", encoding="utf-8") as f:
        for line in cp.stdout.splitlines():
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            json.loads(line)
            f.write(line + "\n")
    if cp.stderr:
        print(cp.stderr, file=sys.stderr, end="")


def run_benchmarks(*, python_bin: str, workspace: Path) -> None:
    for name in ("gemm", "attention", "moe"):
        (workspace / f"{name}.jsonl").write_text("", encoding="utf-8")

    for backend in ("aiter", "hipblaslt", "hipkittens", "triton"):
        _run_jsonl_command(
            [
                python_bin,
                str(BENCHMARK_DIR / "bench_gemm.py"),
                "--backend",
                backend,
                "--dtype",
                "bf16",
                "--matrix-sizes",
                *[str(x) for x in WORKLOADS],
            ],
            workspace / "gemm.jsonl",
        )

    for backend in ("aiter", "triton"):
        _run_jsonl_command(
            [
                python_bin,
                str(BENCHMARK_DIR / "bench_attn.py"),
                "--backend",
                backend,
                "--dtype",
                "bf16",
                "--seq-lens",
                *[str(x) for x in WORKLOADS],
                "--batch-size",
                "16",
                "--num-q-heads",
                "8",
                "--num-kv-heads",
                "1",
                "--head-dim",
                "128",
            ],
            workspace / "attention.jsonl",
        )

    for backend in ("aiter", "aiter_asm", "aiter_triton"):
        _run_jsonl_command(
            [
                python_bin,
                str(BENCHMARK_DIR / "bench_moe.py"),
                "--backend",
                backend,
                "--tokens",
                *[str(x) for x in WORKLOADS],
                "--dim",
                "7168",
                "--inter-dim",
                "2048",
                "--experts",
                "32",
                "--topk",
                "4",
                "--input-dtype",
                "fp8",
            ],
            workspace / "moe.jsonl",
        )


def load_records(workspace: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for name in ("gemm", "attention", "moe"):
        path = workspace / f"{name}.jsonl"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"invalid JSON in {path}:{lineno}: {e}") from e
                if not isinstance(record, dict):
                    raise ValueError(f"expected JSON object in {path}:{lineno}")
                records.append(record)
    return records


def record_workload(record: dict[str, Any]) -> int:
    domain = str(record.get("domain", ""))
    if domain == "gemm":
        return int(record["matrix_size"])
    if domain == "attention":
        return int(record["seq_len"])
    if domain == "moe":
        return int(record["tokens"])
    raise ValueError(f"unsupported domain: {domain!r}")


def record_tflops(record: dict[str, Any]) -> float:
    mean_ms = float(record["mean_ms"])
    if mean_ms <= 0.0:
        return float("nan")

    domain = str(record.get("domain", ""))
    if domain == "gemm":
        flops = 2.0 * float(record["m"]) * float(record["n"]) * float(record["k"])
    elif domain == "attention":
        flops = (
            4.0
            * float(record["batch_size"])
            * float(record["num_q_heads"])
            * float(record["seq_len"])
            * float(record["seq_len"])
            * float(record["head_dim"])
        )
        if bool(record["causal"]):
            flops /= 2.0
    elif domain == "moe":
        flops = (
            6.0
            * float(record["tokens"])
            * float(record["topk"])
            * float(record["dim"])
            * float(record["inter_dim"])
        )
    else:
        raise ValueError(f"unsupported domain: {domain!r}")
    return flops / (mean_ms * 1.0e-3) / 1.0e12


def render_csv(records: list[dict[str, Any]]) -> str:
    table: dict[str, dict[int, dict[str, float]]] = {
        "gemm": {},
        "attention": {},
        "moe": {},
    }
    for record in records:
        domain = str(record.get("domain", ""))
        if domain not in table:
            continue
        workload = record_workload(record)
        backend = str(record.get("backend", ""))
        table[domain].setdefault(workload, {})[backend] = record_tflops(record)

    out_lines: list[list[str]] = [["domain", "workload", *BASELINE_COLUMNS]]
    for domain in ("gemm", "attention", "moe"):
        for workload in WORKLOADS:
            vals = table[domain].get(workload, {})
            out_lines.append(
                [
                    domain,
                    str(workload),
                    *[
                        "-" if backend not in vals else f"{float(vals[backend]):.2f}"
                        for backend in BASELINE_COLUMNS
                    ],
                ]
            )

    import io

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerows(out_lines)
    return buf.getvalue()


def main() -> int:
    args = parse_args()
    workspace = (
        args.workspace_dir if args.workspace_dir is not None else default_workspace()
    )
    workspace.mkdir(parents=True, exist_ok=True)

    if not args.skip_run:
        run_benchmarks(python_bin=args.python, workspace=workspace)

    table_csv = render_csv(load_records(workspace))
    csv_path = workspace / "table2.csv"
    csv_path.write_text(table_csv, encoding="utf-8")
    print(f"workspace: {workspace}")
    print(f"csv: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
