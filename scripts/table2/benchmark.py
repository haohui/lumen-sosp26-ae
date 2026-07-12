#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
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
    "lumen",
    "kernelbench",
    "cudaforge",
    "kernelfalcon",
    "ksearch",
    "hipblaslt",
    "hipkittens",
    "aiter",
    "triton",
]
GEMM_BACKENDS = (
    "lumen",
    "kernelbench",
    "cudaforge",
    "kernelfalcon",
    "ksearch",
    "aiter",
    "hipblaslt",
    "hipkittens",
    "triton",
)
ATTENTION_BACKENDS = (
    "lumen",
    "kernelbench",
    "cudaforge",
    "kernelfalcon",
    "ksearch",
    "aiter",
    "triton",
)
MOE_BACKENDS = (
    "lumen",
    "kernelbench",
    "cudaforge",
    "kernelfalcon",
    "ksearch",
    "aiter",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reproduce Table 2 benchmark results")
    p.add_argument("--workspace-dir", type=Path, default=None)
    p.add_argument("--python", type=str, default=sys.executable)
    p.add_argument("--workloads", type=int, nargs="+", default=list(WORKLOADS))
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeat", type=int, default=100)
    p.add_argument("--graph-iters", type=int, default=1)
    p.add_argument(
        "--skip-run",
        action="store_true",
        help="Process existing JSONL files in the workspace without running kernels",
    )
    return p.parse_args()


def default_workspace() -> Path:
    run_id = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    return Path(__file__).resolve().parent / "workspace" / run_id


def _env(*, backend: str) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath = str(BENCHMARK_DIR)
    existing = env.get("PYTHONPATH")
    if existing:
        pythonpath = pythonpath + os.pathsep + existing

    if backend == "lumen":
        lumen_pythonpath = env.get("LUMEN_PYTHONPATH")
        if lumen_pythonpath:
            pythonpath = lumen_pythonpath + os.pathsep + pythonpath

        lumen_rocm_path = env.get("LUMEN_ROCM_PATH")
        if lumen_rocm_path:
            env["ROCM_PATH"] = lumen_rocm_path
            env["ROCM_HOME"] = lumen_rocm_path
            env["HIP_PATH"] = lumen_rocm_path
            env["PATH"] = (
                str(Path(lumen_rocm_path) / "bin")
                + os.pathsep
                + env.get("PATH", "")
            )

    env["PYTHONPATH"] = pythonpath
    env["AITER_JIT_DIR"] = str(REPO_ROOT / ".aiter" / "jit")
    return env


def _run_jsonl_command(cmd: list[str], out_path: Path, *, backend: str) -> None:
    cp = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        env=_env(backend=backend),
    )
    if cp.returncode != 0:
        print("command failed:", " ".join(cmd), file=sys.stderr)
        if cp.stdout:
            print(cp.stdout, file=sys.stderr, end="")
        if cp.stderr:
            print(cp.stderr, file=sys.stderr, end="")
        raise subprocess.CalledProcessError(cp.returncode, cmd)
    with out_path.open("a", encoding="utf-8") as f:
        for line in cp.stdout.splitlines():
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            json.loads(line)
            f.write(line + "\n")
    if cp.stderr:
        print(cp.stderr, file=sys.stderr, end="")


def _timer_args(*, warmup: int, repeat: int, graph_iters: int) -> list[str]:
    return [
        "--warmup",
        str(warmup),
        "--repeat",
        str(repeat),
        "--graph-iters",
        str(graph_iters),
    ]


def run_benchmarks(
    *,
    python_bin: str,
    workspace: Path,
    workloads: list[int],
    warmup: int,
    repeat: int,
    graph_iters: int,
) -> None:
    for name in ("gemm", "attention", "moe"):
        (workspace / f"{name}.jsonl").write_text("", encoding="utf-8")

    for backend in GEMM_BACKENDS:
        _run_jsonl_command(
            [
                python_bin,
                str(BENCHMARK_DIR / "bench_gemm.py"),
                "--backend",
                backend,
                "--dtype",
                "bf16",
                "--matrix-sizes",
                *[str(x) for x in workloads],
                *_timer_args(warmup=warmup, repeat=repeat, graph_iters=graph_iters),
            ],
            workspace / "gemm.jsonl",
            backend=backend,
        )

    for backend in ATTENTION_BACKENDS:
        _run_jsonl_command(
            [
                python_bin,
                str(BENCHMARK_DIR / "bench_attn.py"),
                "--backend",
                backend,
                "--dtype",
                "bf16",
                "--seq-lens",
                *[str(x) for x in workloads],
                "--batch-size",
                "16",
                "--num-q-heads",
                "8",
                "--num-kv-heads",
                "1",
                "--head-dim",
                "128",
                *_timer_args(warmup=warmup, repeat=repeat, graph_iters=graph_iters),
            ],
            workspace / "attention.jsonl",
            backend=backend,
        )

    for backend in MOE_BACKENDS:
        _run_jsonl_command(
            [
                python_bin,
                str(BENCHMARK_DIR / "bench_moe.py"),
                "--backend",
                backend,
                "--tokens",
                *[str(x) for x in workloads],
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
                *_timer_args(warmup=warmup, repeat=repeat, graph_iters=graph_iters),
            ],
            workspace / "moe.jsonl",
            backend=backend,
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


def render_csv(records: list[dict[str, Any]], *, workloads: list[int]) -> str:
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
        for workload in workloads:
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
        run_benchmarks(
            python_bin=args.python,
            workspace=workspace,
            workloads=args.workloads,
            warmup=args.warmup,
            repeat=args.repeat,
            graph_iters=args.graph_iters,
        )

    table_csv = render_csv(load_records(workspace), workloads=args.workloads)
    csv_path = workspace / "table2.csv"
    csv_path.write_text(table_csv, encoding="utf-8")
    print(f"workspace: {workspace}")
    print(f"csv: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
