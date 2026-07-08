#!/usr/bin/env python3
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from config import (
    ATTENTION_WORKLOADS,
    BASELINE_COLUMNS,
    GEMM_WORKLOADS,
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


def _blank(cols: list[int]) -> dict[str, list[float | None]]:
    return {k: [None] * len(cols) for k in BASELINE_COLUMNS}


def _col_idx(cols: list[int]) -> dict[int, int]:
    return {v: i for i, v in enumerate(cols)}


def _set_cell(
    table: dict[str, list[float | None]],
    idx: dict[int, int],
    baseline: str,
    size: int,
    val: float,
) -> None:
    if baseline in table and size in idx:
        table[baseline][idx[size]] = float(val)


def _load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_meta(
    path: Path,
) -> dict[str, dict[str, int | float | str | bool | list[int]]]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _meta_workloads(
    meta: dict[str, dict[str, int | float | str | bool | list[int]]],
    domain: str,
    default: list[int],
) -> list[int]:
    workloads = meta.get("workloads", {})
    vals = workloads.get(domain) if isinstance(workloads, dict) else None
    if not isinstance(vals, list):
        return list(default)
    out = [int(v) for v in vals]
    return out or list(default)


def parse_gemm(csv_path: Path, cols: list[int]) -> dict[str, list[float | None]]:
    table = _blank(cols)
    idx = _col_idx(cols)
    for r in _load_rows(csv_path):
        if r.get("status") != "ok":
            continue
        name = GEMM_MAP.get(str(r.get("baseline", "")).strip())
        if name is None:
            continue
        size = int(r["workload"])
        _set_cell(table, idx, name, size, float(r["tflops"]))
    return table


def parse_attention(csv_path: Path, cols: list[int]) -> dict[str, list[float | None]]:
    table = _blank(cols)
    idx = _col_idx(cols)
    for r in _load_rows(csv_path):
        if r.get("status") != "ok":
            continue
        name = ATTN_MAP.get(str(r.get("baseline", "")).strip())
        if name is None:
            continue
        seq_len = int(r["workload"])
        _set_cell(table, idx, name, seq_len, float(r["tflops"]))
    return table


def parse_moe(csv_path: Path, cols: list[int]) -> dict[str, list[float | None]]:
    table = _blank(cols)
    idx = _col_idx(cols)
    for r in _load_rows(csv_path):
        if r.get("status") != "ok":
            continue
        name = MOE_MAP.get(str(r.get("baseline", "")).strip())
        if name is None:
            continue
        seq_len = int(r["workload"])
        _set_cell(table, idx, name, seq_len, float(r["tflops"]))
    return table


def _fmt(v: float | None) -> str:
    return "-" if v is None else f"{v:.2f}"


def _render_section(
    title: str,
    cols: list[int],
    rows: dict[str, list[float | None]],
) -> list[str]:
    lines = ["| **" + title + "** | " + " | ".join([""] * len(BASELINE_COLUMNS)) + " |"]
    for i, size in enumerate(cols):
        vals = [rows[name][i] for name in BASELINE_COLUMNS]
        lines.append(
            "| " + str(size) + " | " + " | ".join(_fmt(v) for v in vals) + " |"
        )
    return lines


def _csv_section(
    writer: csv.writer,
    title: str,
    cols: list[int],
    rows: dict[str, list[float | None]],
) -> None:
    writer.writerow([title, *[""] * (len(BASELINE_COLUMNS) + 1)])
    for i, size in enumerate(cols):
        writer.writerow(["", size, *[_fmt(rows[name][i]) for name in BASELINE_COLUMNS]])


def render_overall_markdown(
    *,
    gemm: dict[str, list[float | None]],
    attn: dict[str, list[float | None]],
    moe: dict[str, list[float | None]],
    gemm_cols: list[int],
    attn_cols: list[int],
    moe_cols: list[int],
) -> str:
    cols = " | ".join(BASELINE_COLUMNS)
    lines = [
        "| Workload | " + cols + " |",
        "|---|" + "|".join(["---:"] * len(BASELINE_COLUMNS)) + "|",
    ]
    lines.extend(
        _render_section("BF16 Square GEMM, Matrix Size (M×N×K)", gemm_cols, gemm)
    )
    lines.extend(
        _render_section(
            "GQA Forward Flash Attention, Sequence length",
            attn_cols,
            attn,
        )
    )
    lines.extend(_render_section("Fused MoE, Sequence length", moe_cols, moe))
    return "\n".join(lines).rstrip() + "\n"


def render_overall_csv(
    *,
    gemm: dict[str, list[float | None]],
    attn: dict[str, list[float | None]],
    moe: dict[str, list[float | None]],
    gemm_cols: list[int],
    attn_cols: list[int],
    moe_cols: list[int],
) -> str:
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["Section", "Workload", *BASELINE_COLUMNS])
    _csv_section(writer, "BF16 Square GEMM, Matrix Size (M×N×K)", gemm_cols, gemm)
    _csv_section(
        writer,
        "GQA Forward Flash Attention, Sequence length",
        attn_cols,
        attn,
    )
    _csv_section(writer, "Fused MoE, Sequence length", moe_cols, moe)
    return out.getvalue()


def render_from_csvs(
    *,
    gemm_csv: Path,
    attn_csv: Path,
    moe_csv: Path,
    meta_path: Path,
) -> str:
    meta = _load_meta(meta_path)
    gemm_cols = _meta_workloads(meta, "gemm", GEMM_WORKLOADS)
    attn_cols = _meta_workloads(meta, "attention", ATTENTION_WORKLOADS)
    moe_cols = _meta_workloads(meta, "moe", MOE_WORKLOADS)
    gemm = parse_gemm(gemm_csv, gemm_cols)
    attn = parse_attention(attn_csv, attn_cols)
    moe = parse_moe(moe_csv, moe_cols)
    return render_overall_markdown(
        gemm=gemm,
        attn=attn,
        moe=moe,
        gemm_cols=gemm_cols,
        attn_cols=attn_cols,
        moe_cols=moe_cols,
    )


def render_csv_from_csvs(
    *,
    gemm_csv: Path,
    attn_csv: Path,
    moe_csv: Path,
    meta_path: Path,
) -> str:
    meta = _load_meta(meta_path)
    gemm_cols = _meta_workloads(meta, "gemm", GEMM_WORKLOADS)
    attn_cols = _meta_workloads(meta, "attention", ATTENTION_WORKLOADS)
    moe_cols = _meta_workloads(meta, "moe", MOE_WORKLOADS)
    gemm = parse_gemm(gemm_csv, gemm_cols)
    attn = parse_attention(attn_csv, attn_cols)
    moe = parse_moe(moe_csv, moe_cols)
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

    p = argparse.ArgumentParser(
        description="Render the AE throughput table from benchmark CSV files"
    )
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
