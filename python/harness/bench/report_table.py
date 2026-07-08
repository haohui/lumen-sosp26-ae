#!/usr/bin/env python3
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Dict, List

from config import (
    ATTENTION_DEFAULTS,
    ATTENTION_WORKLOADS,
    BASELINE_COLUMNS,
    GEMM_WORKLOADS,
    MOE_DEFAULTS,
    MOE_WORKLOADS,
)

GEMM_MAP = {
    "lumen": "Lumen",
    "aiter": "AITER",
    "hipblaslt": "HipBlasLt",
    "hipkittens": "HipKittens",
    "triton": "Triton",
}

ATTN_MAP = {
    "lumen": "Lumen",
    "aiter": "AITER",
    "triton": "Triton",
}

MOE_MAP = {
    "lumen": "Lumen",
    "AITER (asm)": "AITER",
    "Triton (aiter backend)": "Triton",
}


def _blank(cols: List[int]) -> Dict[str, List[float | None]]:
    return {k: [None] * len(cols) for k in BASELINE_COLUMNS}


def _col_idx(cols: List[int]) -> Dict[int, int]:
    return {v: i for i, v in enumerate(cols)}


def _set_cell(table: Dict[str, List[float | None]], idx: Dict[int, int], baseline: str, size: int, val: float) -> None:
    if baseline in table and size in idx:
        table[baseline][idx[size]] = float(val)


def _load_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_meta(path: Path) -> Dict[str, Dict[str, int | float | str | bool | List[int]]]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _meta_workloads(
    meta: Dict[str, Dict[str, int | float | str | bool | List[int]]],
    domain: str,
    default: List[int],
) -> List[int]:
    workloads = meta.get("workloads", {})
    vals = workloads.get(domain) if isinstance(workloads, dict) else None
    if not isinstance(vals, list):
        return list(default)
    out = [int(v) for v in vals]
    return out or list(default)


def _gemm_tflops(size: int, mean_ms: float) -> float:
    return (2.0 * size * size * size) / (mean_ms * 1.0e-3) / 1.0e12


def _attn_tflops(seq_len: int, mean_ms: float, meta: Dict[str, Dict[str, int | float | str | bool | List[int]]]) -> float:
    cfg = meta.get("attention", {})
    batch = int(cfg.get("batch_size", ATTENTION_DEFAULTS["batch_size"]))
    heads = int(cfg.get("num_q_heads", ATTENTION_DEFAULTS["num_q_heads"]))
    dim = int(cfg.get("head_dim", ATTENTION_DEFAULTS["head_dim"]))
    causal = bool(cfg.get("causal", ATTENTION_DEFAULTS["causal"]))
    flops = 4.0 * batch * heads * seq_len * seq_len * dim
    if causal:
        flops /= 2.0
    return flops / (mean_ms * 1.0e-3) / 1.0e12


def _moe_tflops(tokens: int, mean_ms: float, meta: Dict[str, Dict[str, int | float | str | bool | List[int]]]) -> float:
    cfg = meta.get("moe", {})
    dim = int(cfg.get("dim", MOE_DEFAULTS["dim"]))
    inter_dim = int(cfg.get("inter_dim", MOE_DEFAULTS["inter_dim"]))
    topk = int(cfg.get("topk", MOE_DEFAULTS["topk"]))
    flops = 6.0 * tokens * topk * dim * inter_dim
    return flops / (mean_ms * 1.0e-3) / 1.0e12


def parse_gemm(csv_path: Path, cols: List[int]) -> Dict[str, List[float | None]]:
    table = _blank(cols)
    idx = _col_idx(cols)
    for r in _load_rows(csv_path):
        if r.get("status") != "ok":
            continue
        name = GEMM_MAP.get(str(r.get("baseline", "")).strip())
        if name is None:
            continue
        size = int(r["workload"])
        mean_ms = float(r["mean_ms"])
        _set_cell(table, idx, name, size, _gemm_tflops(size, mean_ms))
    return table


def parse_attention(
    csv_path: Path,
    cols: List[int],
    meta: Dict[str, Dict[str, int | float | str | bool | List[int]]],
) -> Dict[str, List[float | None]]:
    table = _blank(cols)
    idx = _col_idx(cols)
    for r in _load_rows(csv_path):
        if r.get("status") != "ok":
            continue
        name = ATTN_MAP.get(str(r.get("baseline", "")).strip())
        if name is None:
            continue
        seq_len = int(r["workload"])
        mean_ms = float(r["mean_ms"])
        _set_cell(table, idx, name, seq_len, _attn_tflops(seq_len, mean_ms, meta))
    return table


def parse_moe(
    csv_path: Path,
    cols: List[int],
    meta: Dict[str, Dict[str, int | float | str | bool | List[int]]],
) -> Dict[str, List[float | None]]:
    table = _blank(cols)
    idx = _col_idx(cols)
    for r in _load_rows(csv_path):
        if r.get("status") != "ok":
            continue
        name = MOE_MAP.get(str(r.get("baseline", "")).strip())
        if name is None:
            continue
        seq_len = int(r["workload"])
        mean_ms = float(r["mean_ms"])
        _set_cell(table, idx, name, seq_len, _moe_tflops(seq_len, mean_ms, meta))
    return table


def _fmt(v: float | None) -> str:
    return "-" if v is None else f"{v:.2f}"


def _render_section(title: str, cols: List[int], rows: Dict[str, List[float | None]]) -> List[str]:
    lines = ["| **" + title + "** | " + " | ".join([""] * len(BASELINE_COLUMNS)) + " |"]
    for i, size in enumerate(cols):
        vals = [rows[name][i] for name in BASELINE_COLUMNS]
        lines.append("| " + str(size) + " | " + " | ".join(_fmt(v) for v in vals) + " |")
    return lines


def _csv_section(writer: csv.writer, title: str, cols: List[int], rows: Dict[str, List[float | None]]) -> None:
    writer.writerow([title, *[""] * (len(BASELINE_COLUMNS) + 1)])
    for i, size in enumerate(cols):
        writer.writerow(["", size, *[_fmt(rows[name][i]) for name in BASELINE_COLUMNS]])


def render_overall_markdown(
    *,
    gemm: Dict[str, List[float | None]],
    attn: Dict[str, List[float | None]],
    moe: Dict[str, List[float | None]],
    gemm_cols: List[int],
    attn_cols: List[int],
    moe_cols: List[int],
) -> str:
    cols = " | ".join(BASELINE_COLUMNS)
    lines = [
        "| Workload | " + cols + " |",
        "|---|" + "|".join(["---:"] * len(BASELINE_COLUMNS)) + "|",
    ]
    lines.extend(_render_section("BF16 Square GEMM, Matrix Size (M×N×K)", gemm_cols, gemm))
    lines.extend(_render_section("GQA Forward Flash Attention, Sequence length", attn_cols, attn))
    lines.extend(_render_section("Fused MoE, Sequence length", moe_cols, moe))
    return "\n".join(lines).rstrip() + "\n"


def render_overall_csv(
    *,
    gemm: Dict[str, List[float | None]],
    attn: Dict[str, List[float | None]],
    moe: Dict[str, List[float | None]],
    gemm_cols: List[int],
    attn_cols: List[int],
    moe_cols: List[int],
) -> str:
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["Section", "Workload", *BASELINE_COLUMNS])
    _csv_section(writer, "BF16 Square GEMM, Matrix Size (M×N×K)", gemm_cols, gemm)
    _csv_section(writer, "GQA Forward Flash Attention, Sequence length", attn_cols, attn)
    _csv_section(writer, "Fused MoE, Sequence length", moe_cols, moe)
    return out.getvalue()


def render_from_csvs(*, gemm_csv: Path, attn_csv: Path, moe_csv: Path, meta_path: Path) -> str:
    meta = _load_meta(meta_path)
    gemm_cols = _meta_workloads(meta, "gemm", GEMM_WORKLOADS)
    attn_cols = _meta_workloads(meta, "attention", ATTENTION_WORKLOADS)
    moe_cols = _meta_workloads(meta, "moe", MOE_WORKLOADS)
    gemm = parse_gemm(gemm_csv, gemm_cols)
    attn = parse_attention(attn_csv, attn_cols, meta)
    moe = parse_moe(moe_csv, moe_cols, meta)
    return render_overall_markdown(
        gemm=gemm,
        attn=attn,
        moe=moe,
        gemm_cols=gemm_cols,
        attn_cols=attn_cols,
        moe_cols=moe_cols,
    )


def render_csv_from_csvs(*, gemm_csv: Path, attn_csv: Path, moe_csv: Path, meta_path: Path) -> str:
    meta = _load_meta(meta_path)
    gemm_cols = _meta_workloads(meta, "gemm", GEMM_WORKLOADS)
    attn_cols = _meta_workloads(meta, "attention", ATTENTION_WORKLOADS)
    moe_cols = _meta_workloads(meta, "moe", MOE_WORKLOADS)
    gemm = parse_gemm(gemm_csv, gemm_cols)
    attn = parse_attention(attn_csv, attn_cols, meta)
    moe = parse_moe(moe_csv, moe_cols, meta)
    return render_overall_csv(
        gemm=gemm,
        attn=attn,
        moe=moe,
        gemm_cols=gemm_cols,
        attn_cols=attn_cols,
        moe_cols=moe_cols,
    )


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="Render the AE throughput table from benchmark CSV files")
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-md", type=Path, default=None)
    p.add_argument("--output-csv", type=Path, default=None)
    args = p.parse_args()

    md = render_from_csvs(
        gemm_csv=args.input_dir / "gemm_raw.csv",
        attn_csv=args.input_dir / "attention_raw.csv",
        moe_csv=args.input_dir / "moe_raw.csv",
        meta_path=args.input_dir / "run_meta.json",
    )
    table_csv = render_csv_from_csvs(
        gemm_csv=args.input_dir / "gemm_raw.csv",
        attn_csv=args.input_dir / "attention_raw.csv",
        moe_csv=args.input_dir / "moe_raw.csv",
        meta_path=args.input_dir / "run_meta.json",
    )
    md_path = args.output_md or (args.input_dir / "overall_performance.md")
    csv_path = args.output_csv or (args.input_dir / "overall_performance.csv")
    md_path.write_text(md, encoding="utf-8")
    csv_path.write_text(table_csv, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
