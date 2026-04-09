#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
BENCH_ROOT = REPO_ROOT / "data" / "benchmarks"
GEMM_ROOT = BENCH_ROOT / "gemm"
ATTN_ROOT = BENCH_ROOT / "attn"
MOE_ROOT = BENCH_ROOT / "moe"
LOCKED_JSON = BENCH_ROOT / "locked_results_openai.json"

GEMM_COLS = [1024, 2048, 4096, 8192, 16384]
ATTN_COLS = [1024, 2048, 4096, 8192, 16384]
MOE_COLS = [1024, 2048, 4096, 8192, 16384]

GEMM_ORDER = [
    "AITER",
    "HipBlasLt",
    "HipKittens",
    "Triton",
    "KernelFalcon",
    "KSearch",
    "KernelBench",
    "CUDAForge",
]
ATTN_ORDER = ["AITER", "HipKittens", "KernelFalcon", "KSearch", "KernelBench", "CUDAForge"]
MOE_ORDER = [
    "AITER (asm)",
    "Triton (aiter backend)",
    "KernelFalcon",
    "KSearch",
    "KernelBench",
    "CUDAForge",
]

ATTN_PATH_SUFFIX_TO_NAME: dict[str, str] = {
    "data/benchmarks/attn/06_aiter/best_kernel.py": "AITER",
    "data/benchmarks/attn/05_HipKittens/best_kernel.py": "HipKittens",
    "data/benchmarks/attn/03_kernelfalcon/best_kernel.py": "KernelFalcon",
    "data/benchmarks/attn/04_ksearch/best_kernel.py": "KSearch",
    "data/benchmarks/attn/01_kernelbench/best_kernel.py": "KernelBench",
    "data/benchmarks/attn/02_cudaforge/best_kernel.py": "CUDAForge",
}

MOE_PATH_SUFFIX_TO_NAME: dict[str, str] = {
    "data/benchmarks/moe/01_kernelbench/best_kernel.py": "KernelBench",
    "data/benchmarks/moe/02_cudaforge/best_kernel.py": "CUDAForge",
    "data/benchmarks/moe/03_kernelfalcon/best_kernel.py": "KernelFalcon",
    "data/benchmarks/moe/04_ksearch/best_kernel.py": "KSearch",
    "data/benchmarks/moe/01_kernelbench/output/best_kernel.py": "KernelBench",
    "data/benchmarks/moe/02_cudaforge/output/best_kernel.py": "CUDAForge",
    "data/benchmarks/moe/03_kernelfalcon/output/best_kernel.py": "KernelFalcon",
    "data/benchmarks/moe/04_ksearch/output/best_kernel.py": "KSearch",
}


def _norm_path(path: str | Path) -> str:
    return str(path).replace("\\", "/").lower()


def _resolve_gemm_selected_py() -> dict[str, Path]:
    return {
        "kernelbench.py": GEMM_ROOT / "01_kernelbench" / "best_kernel.py",
        "cudaforge.py": GEMM_ROOT / "02_cudaforge" / "best_kernel.py",
        "kernelfalcon.py": GEMM_ROOT / "03_kernelfalcon" / "best_kernel.py",
        "ksearch.py": GEMM_ROOT / "04_ksearch" / "best_kernel.py",
        "triton.py": GEMM_ROOT / "07_triton" / "best_kernel.py",
    }


def _resolve_attention_baselines() -> list[tuple[str, Path]]:
    return [
        (
            "AITER",
            ATTN_ROOT / "06_aiter" / "best_kernel.py",
        ),
        (
            "HipKittens",
            ATTN_ROOT / "05_HipKittens" / "best_kernel.py",
        ),
        (
            "KernelFalcon",
            ATTN_ROOT / "03_kernelfalcon" / "best_kernel.py",
        ),
        (
            "KSearch",
            ATTN_ROOT / "04_ksearch" / "best_kernel.py",
        ),
        (
            "KernelBench",
            ATTN_ROOT / "01_kernelbench" / "best_kernel.py",
        ),
        (
            "CUDAForge",
            ATTN_ROOT / "02_cudaforge" / "best_kernel.py",
        ),
    ]


def _resolve_moe_baselines() -> list[tuple[str, Path]]:
    return [
        (
            "KernelBench",
            MOE_ROOT / "01_kernelbench" / "best_kernel.py",
        ),
        (
            "CUDAForge",
            MOE_ROOT / "02_cudaforge" / "best_kernel.py",
        ),
        (
            "KernelFalcon",
            MOE_ROOT / "03_kernelfalcon" / "best_kernel.py",
        ),
        (
            "KSearch",
            MOE_ROOT / "04_ksearch" / "best_kernel.py",
        ),
    ]


def _idx(cols: list[int]) -> dict[int, int]:
    return {int(v): i for i, v in enumerate(cols)}


def _blank(order: list[str], cols: list[int]) -> dict[str, list[float | None]]:
    return {name: [None for _ in cols] for name in order}


def _set_cell(
    table: dict[str, list[float | None]],
    index: dict[int, int],
    baseline: str,
    size: int,
    value: float,
) -> None:
    if baseline not in table:
        return
    if size not in index:
        return
    table[baseline][index[size]] = float(value)


def _fmt(v: float | None) -> str:
    if v is None:
        return ""
    return f"{v:.6f}"


def _render_domain(title: str, cols: list[int], order: list[str], rows: dict[str, list[float | None]]) -> list[str]:
    head = "| Kernel | " + " | ".join(str(c) for c in cols) + " |"
    sep = "|---|" + "|".join(["---:"] * len(cols)) + "|"
    out = [f"**{title} (mean_ms)**", "", head, sep]
    for name in order:
        vals = rows.get(name, [None for _ in cols])
        out.append("| " + name + " | " + " | ".join(_fmt(x) for x in vals) + " |")
    out.append("")
    return out


def render_markdown(
    tables: dict[str, Any],
    *,
    domain: str = "all",
    gemm_cols: list[int],
    attn_cols: list[int],
    moe_cols: list[int],
) -> str:
    lines: list[str] = ["**Retime (mean_ms)**", ""]
    if domain in ("all", "gemm"):
        lines.extend(_render_domain("GEMM", gemm_cols, GEMM_ORDER, tables["gemm"]))
    if domain in ("all", "attention"):
        lines.extend(_render_domain("Attention", attn_cols, ATTN_ORDER, tables["attention"]))
    if domain in ("all", "moe"):
        lines.extend(_render_domain("MoE", moe_cols, MOE_ORDER, tables["moe"]))
    return "\n".join(lines).rstrip() + "\n"


def _slice_locked_rows(
    src_rows: dict[str, list[float]],
    src_cols: list[int],
    dst_cols: list[int],
) -> dict[str, list[float | None]]:
    src_idx = _idx(src_cols)
    out: dict[str, list[float | None]] = {}
    for name, vals in src_rows.items():
        out[name] = [float(vals[src_idx[c]]) if c in src_idx else None for c in dst_cols]
    return out


def load_locked_tables(
    *,
    gemm_cols: list[int],
    attn_cols: list[int],
    moe_cols: list[int],
) -> dict[str, dict[str, list[float | None]]]:
    if not LOCKED_JSON.exists():
        raise FileNotFoundError(f"locked result table not found: {LOCKED_JSON}")
    payload = json.loads(LOCKED_JSON.read_text(encoding="utf-8"))
    t = payload["tables"]
    locked_gemm_cols = [int(x) for x in t["gemm"]["columns"]]
    locked_attn_cols = [int(x) for x in t["attention"]["columns"]]
    locked_moe_cols = [int(x) for x in t["moe"]["columns"]]

    gemm_src = {k: [float(x) for x in v] for k, v in t["gemm"]["rows"].items()}
    attention_src = {k: [float(x) for x in v] for k, v in t["attention"]["rows"].items()}
    moe_src = {k: [float(x) for x in v] for k, v in t["moe"]["rows"].items()}

    gemm = _slice_locked_rows(gemm_src, locked_gemm_cols, gemm_cols)
    attention = _slice_locked_rows(attention_src, locked_attn_cols, attn_cols)
    moe = _slice_locked_rows(moe_src, locked_moe_cols, moe_cols)
    return {"gemm": gemm, "attention": attention, "moe": moe}


def _canon(path: Path | str) -> str:
    return str(Path(path).resolve())


def _match_suffix_name(source: str, suffix_to_name: dict[str, str]) -> str | None:
    n = _norm_path(source)
    for suffix, name in suffix_to_name.items():
        s = _norm_path(suffix)
        if n.endswith(s) or s in n:
            return name
    return None


def _resolve_attention_name(*, baseline_raw: str, kernel_path: str) -> str | None:
    by_tag = _match_suffix_name(baseline_raw, ATTN_PATH_SUFFIX_TO_NAME)
    if by_tag is not None:
        return by_tag
    return _match_suffix_name(kernel_path, ATTN_PATH_SUFFIX_TO_NAME)


def _resolve_moe_name(*, baseline_raw: str, kernel_path: str) -> str | None:
    if baseline_raw in ("AITER (asm)", "Triton (aiter backend)"):
        return baseline_raw
    by_tag = _match_suffix_name(baseline_raw, MOE_PATH_SUFFIX_TO_NAME)
    if by_tag is not None:
        return by_tag
    return _match_suffix_name(kernel_path, MOE_PATH_SUFFIX_TO_NAME)


def _run(cmd: list[str]) -> None:
    print("[run] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _ensure_files(paths: list[Path]) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("missing required files:\n" + "\n".join(missing))


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _append_opt(cmd: list[str], key: str, value: str) -> None:
    if value.strip():
        cmd.extend([key, value.strip()])


def _parse_workload_csv(value: str, *, arg_name: str) -> list[int]:
    tokens = [x.strip() for x in value.split(",") if x.strip()]
    out: list[int] = []
    seen: set[int] = set()
    for tok in tokens:
        try:
            v = int(tok)
        except ValueError as e:
            raise ValueError(f"{arg_name}: invalid integer token '{tok}'") from e
        if v <= 0:
            raise ValueError(f"{arg_name}: workload must be positive, got {v}")
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _resolve_workload_columns(args: argparse.Namespace) -> tuple[list[int], list[int], list[int]]:
    global_workloads = _parse_workload_csv(args.workloads, arg_name="--workloads")
    gemm_workloads = _parse_workload_csv(args.gemm_workloads, arg_name="--gemm-workloads")
    attn_workloads = _parse_workload_csv(args.attention_workloads, arg_name="--attention-workloads")
    moe_workloads = _parse_workload_csv(args.moe_workloads, arg_name="--moe-workloads")

    gemm_cols = gemm_workloads or global_workloads or list(GEMM_COLS)
    attn_cols = attn_workloads or global_workloads or list(ATTN_COLS)
    moe_cols = moe_workloads or global_workloads or list(MOE_COLS)

    if args.domain in ("all", "gemm") and not gemm_cols:
        raise ValueError("No GEMM workloads selected")
    if args.domain in ("all", "attention") and not attn_cols:
        raise ValueError("No attention workloads selected")
    if args.domain in ("all", "moe") and not moe_cols:
        raise ValueError("No MoE workloads selected")
    return gemm_cols, attn_cols, moe_cols


def _run_gemm_job(args: argparse.Namespace, out_dir: Path, gemm_cols: list[int]) -> Path:
    gemm_py = THIS_DIR / "benchmark_gemm_unified_graph.py"
    gemm_selected_py = _resolve_gemm_selected_py()
    _ensure_files([gemm_py, *gemm_selected_py.values()])

    sizes_csv = ",".join(str(x) for x in gemm_cols)
    gemm_stage = out_dir / "gemm_selected_py"
    gemm_stage.mkdir(parents=True, exist_ok=True)
    for name, src in gemm_selected_py.items():
        _link_or_copy(src.resolve(), gemm_stage / name)

    gemm_json = out_dir / "gemm.json"
    cmd = [
        args.python,
        str(gemm_py),
        "--device",
        args.device,
        "--dtype",
        "bf16",
        "--sizes",
        sizes_csv,
        "--warmup-ms",
        str(args.warmup_ms),
        "--repeat-ms",
        str(args.repeat_ms),
        "--graph-iters",
        str(args.graph_iters),
        "--timer-trials",
        str(args.timer_trials),
        "--min-replays",
        str(args.min_replays),
        "--max-replays",
        str(args.max_replays),
        "--allow-suspicious-graph",
        "--run-aiter",
        "--run-hipblaslt",
        "--run-hipkittens",
        "--gemm-kernel-root",
        str(gemm_stage),
        "--json-out",
        str(gemm_json),
    ]
    _append_opt(cmd, "--hip-visible-devices", args.hip_visible_devices)
    _append_opt(cmd, "--cpu-cores", args.cpu_cores)
    _run(cmd)
    return gemm_json


def _run_attention_job(args: argparse.Namespace, out_dir: Path, attn_cols: list[int]) -> Path:
    attn_py = THIS_DIR / "benchmark_attention_unified_graph.py"
    attn_baselines = _resolve_attention_baselines()
    _ensure_files([attn_py, *[p for _, p in attn_baselines]])

    seq_csv = ",".join(str(x) for x in attn_cols)
    attn_json = out_dir / "attention.json"
    cmd = [
        args.python,
        str(attn_py),
        "--device",
        args.device,
        "--dtype",
        "bf16",
        "--seq-lens",
        seq_csv,
        "--batch-size",
        "16",
        "--num-q-heads",
        "8",
        "--num-kv-heads",
        "1",
        "--head-dim",
        "128",
        "--causal",
        "--warmup-ms",
        str(args.warmup_ms),
        "--repeat-ms",
        str(args.repeat_ms),
        "--graph-iters",
        str(args.graph_iters),
        "--timer-trials",
        str(args.timer_trials),
        "--min-replays",
        str(args.min_replays),
        "--max-replays",
        str(args.max_replays),
        "--allow-suspicious-graph",
        "--no-flashinfer",
        "--no-flashattention",
    ]
    for _, p in attn_baselines:
        cmd.extend(["--baseline-kernel", str(p.resolve())])
    cmd.extend(["--json-out", str(attn_json)])
    _append_opt(cmd, "--hip-visible-devices", args.hip_visible_devices)
    _append_opt(cmd, "--cpu-cores", args.cpu_cores)
    _run(cmd)
    return attn_json


def _run_moe_job(args: argparse.Namespace, out_dir: Path, moe_cols: list[int]) -> Path:
    moe_py = THIS_DIR / "benchmark_moe_unified_graph.py"
    moe_baselines = _resolve_moe_baselines()
    _ensure_files([moe_py, *[p for _, p in moe_baselines]])

    seq_csv = ",".join(str(x) for x in moe_cols)
    moe_json = out_dir / "moe_best.json"
    cmd = [
        args.python,
        str(moe_py),
        "--device",
        args.device,
        "--seq-lens",
        seq_csv,
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
        "--warmup-ms",
        str(args.warmup_ms),
        "--repeat-ms",
        str(args.repeat_ms),
        "--graph-iters",
        str(args.graph_iters),
        "--timer-trials",
        str(args.timer_trials),
        "--min-replays",
        str(args.min_replays),
        "--max-replays",
        str(args.max_replays),
        "--allow-suspicious-graph",
    ]
    for _, p in moe_baselines:
        cmd.extend(["--baseline-kernel", str(p.resolve())])
    cmd.extend(["--json-out", str(moe_json)])
    _append_opt(cmd, "--hip-visible-devices", args.hip_visible_devices)
    _append_opt(cmd, "--cpu-cores", args.cpu_cores)
    _run(cmd)
    return moe_json


def parse_measurements(
    gemm_json: Path | None,
    attn_json: Path | None,
    moe_json: Path | None,
    *,
    gemm_cols: list[int],
    attn_cols: list[int],
    moe_cols: list[int],
) -> dict[str, dict[str, list[float | None]]]:
    gemm = _blank(GEMM_ORDER, gemm_cols)
    attention = _blank(ATTN_ORDER, attn_cols)
    moe = _blank(MOE_ORDER, moe_cols)

    gidx = _idx(gemm_cols)
    aidx = _idx(attn_cols)
    midx = _idx(moe_cols)

    if gemm_json is not None and gemm_json.exists():
        payload = json.loads(gemm_json.read_text(encoding="utf-8"))
        py_map = {
            "kernelbench.py": "KernelBench",
            "cudaforge.py": "CUDAForge",
            "kernelfalcon.py": "KernelFalcon",
            "ksearch.py": "KSearch",
            "triton.py": "Triton",
        }
        for row in payload.get("rows", []):
            if row.get("status") != "ok":
                continue
            baseline_raw = str(row.get("baseline", ""))
            size = int(row.get("m", 0))
            mean_ms = row.get("mean_ms")
            if not isinstance(mean_ms, (int, float)):
                continue
            baseline_name: str | None = None
            if baseline_raw == "aiter":
                baseline_name = "AITER"
            elif baseline_raw == "hipblaslt-internal":
                baseline_name = "HipBlasLt"
            elif baseline_raw == "hipkittens":
                baseline_name = "HipKittens"
            elif baseline_raw.startswith("baseline::"):
                fname = Path(baseline_raw.split("baseline::", 1)[1]).name
                baseline_name = py_map.get(fname)
            if baseline_name is not None:
                _set_cell(gemm, gidx, baseline_name, size, float(mean_ms))

    if attn_json is not None and attn_json.exists():
        payload = json.loads(attn_json.read_text(encoding="utf-8"))
        attn_baselines = _resolve_attention_baselines()
        path_map = {_canon(p): name for name, p in attn_baselines}
        for row in payload.get("rows", []):
            if row.get("status") != "ok":
                continue
            seq = int(row.get("seq_len", 0))
            mean_ms = row.get("mean_ms")
            if not isinstance(mean_ms, (int, float)):
                continue
            baseline_raw = str(row.get("baseline", ""))
            raw_path = str(row.get("kernel_path", ""))
            baseline_name = _resolve_attention_name(
                baseline_raw=baseline_raw,
                kernel_path=raw_path,
            )
            if baseline_name is None:
                baseline_name = path_map.get(_canon(raw_path))
            if baseline_name is not None:
                _set_cell(attention, aidx, baseline_name, seq, float(mean_ms))

    if moe_json is not None and moe_json.exists():
        payload = json.loads(moe_json.read_text(encoding="utf-8"))
        moe_baselines = _resolve_moe_baselines()
        path_map = {_canon(p): name for name, p in moe_baselines}
        baseline_by_slot: dict[str, str] = {}
        cfg = payload.get("config", {})
        cfg_baselines = cfg.get("baseline_kernels", [])
        if isinstance(cfg_baselines, list):
            for i, p in enumerate(cfg_baselines, start=1):
                name = _match_suffix_name(str(p), MOE_PATH_SUFFIX_TO_NAME)
                if name is not None:
                    baseline_by_slot[f"baseline_{i}"] = name
        for row in payload.get("rows", []):
            if row.get("status") != "ok":
                continue
            seq = int(row.get("seq_len", 0))
            mean_ms = row.get("mean_ms")
            if not isinstance(mean_ms, (int, float)):
                continue
            baseline_raw = str(row.get("baseline", ""))
            raw_path = str(row.get("kernel_path", ""))
            baseline_name = _resolve_moe_name(
                baseline_raw=baseline_raw,
                kernel_path=raw_path,
            )
            if baseline_name is None:
                baseline_name = baseline_by_slot.get(baseline_raw)
            if baseline_name is None:
                baseline_name = path_map.get(_canon(raw_path))
            if baseline_name is not None:
                _set_cell(moe, midx, baseline_name, seq, float(mean_ms))

    return {"gemm": gemm, "attention": attention, "moe": moe}


def run_benchmarks(
    args: argparse.Namespace,
    *,
    gemm_cols: list[int],
    attn_cols: list[int],
    moe_cols: list[int],
) -> tuple[dict[str, dict[str, list[float | None]]], Path]:
    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    out_dir = args.out_dir if args.out_dir is not None else (THIS_DIR / "runs" / ts)
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    gemm_json: Path | None = None
    attn_json: Path | None = None
    moe_json: Path | None = None

    if args.domain in ("all", "gemm"):
        gemm_json = _run_gemm_job(args, out_dir, gemm_cols)
    if args.domain in ("all", "attention"):
        attn_json = _run_attention_job(args, out_dir, attn_cols)
    if args.domain in ("all", "moe"):
        moe_json = _run_moe_job(args, out_dir, moe_cols)

    tables = parse_measurements(
        gemm_json,
        attn_json,
        moe_json,
        gemm_cols=gemm_cols,
        attn_cols=attn_cols,
        moe_cols=moe_cols,
    )
    return tables, out_dir


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run OpenAI best-kernel timing (CUDA Graph) or print locked table."
    )
    p.add_argument("--mode", choices=["locked", "run"], default="locked")
    p.add_argument("--domain", choices=["all", "gemm", "attention", "moe"], default="all")
    p.add_argument("--python", type=str, default=sys.executable)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--hip-visible-devices", type=str, default="")
    p.add_argument("--cpu-cores", type=str, default="")
    p.add_argument(
        "--workloads",
        type=str,
        default="",
        help="Comma-separated workload sizes applied to all domains unless overridden.",
    )
    p.add_argument(
        "--gemm-workloads",
        type=str,
        default="",
        help="Comma-separated GEMM workloads (default: all).",
    )
    p.add_argument(
        "--attention-workloads",
        type=str,
        default="",
        help="Comma-separated attention sequence lengths (default: all).",
    )
    p.add_argument(
        "--moe-workloads",
        type=str,
        default="",
        help="Comma-separated MoE sequence lengths (default: all).",
    )
    p.add_argument("--warmup-ms", type=float, default=1000.0)
    p.add_argument("--repeat-ms", type=float, default=5000.0)
    p.add_argument("--graph-iters", type=int, default=10)
    p.add_argument("--timer-trials", type=int, default=9)
    p.add_argument("--min-replays", type=int, default=2)
    p.add_argument("--max-replays", type=int, default=3)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument(
        "--write-md",
        type=Path,
        default=None,
        help="Optional output markdown path (relative path recommended).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    gemm_cols, attn_cols, moe_cols = _resolve_workload_columns(args)

    if args.mode == "locked":
        tables = load_locked_tables(
            gemm_cols=gemm_cols,
            attn_cols=attn_cols,
            moe_cols=moe_cols,
        )
        out_dir = None
    else:
        tables, out_dir = run_benchmarks(
            args,
            gemm_cols=gemm_cols,
            attn_cols=attn_cols,
            moe_cols=moe_cols,
        )

    md = render_markdown(
        tables,
        domain=args.domain,
        gemm_cols=gemm_cols,
        attn_cols=attn_cols,
        moe_cols=moe_cols,
    )
    print(md, end="")

    if args.write_md is not None:
        out_path = args.write_md
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md, encoding="utf-8")
        print(f"[saved] {out_path}", flush=True)

    if out_dir is not None:
        print(f"[runs] {out_dir}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
