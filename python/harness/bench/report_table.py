#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

GEMM_ORDER = ["AITER", "HipBlasLt", "HipKittens", "Triton", "KernelFalcon", "KSearch", "KernelBench", "CUDAForge"]
ATTN_ORDER = ["AITER", "HipKittens", "KernelFalcon", "KSearch", "KernelBench", "CUDAForge"]
MOE_ORDER = ["AITER (asm)", "Triton (aiter backend)", "KernelFalcon", "KSearch", "KernelBench", "CUDAForge"]

GEMM_MAP = {
    "aiter": "AITER",
    "hipblaslt": "HipBlasLt",
    "hipkittens": "HipKittens",
    "triton": "Triton",
    "kernelfalcon": "KernelFalcon",
    "ksearch": "KSearch",
    "kernelbench": "KernelBench",
    "cudaforge": "CUDAForge",
}

ATTN_MAP = {
    "aiter": "AITER",
    "hipkittens": "HipKittens",
    "kernelfalcon": "KernelFalcon",
    "ksearch": "KSearch",
    "kernelbench": "KernelBench",
    "cudaforge": "CUDAForge",
}

MOE_MAP = {
    "kernelfalcon": "KernelFalcon",
    "ksearch": "KSearch",
    "kernelbench": "KernelBench",
    "cudaforge": "CUDAForge",
    "AITER (asm)": "AITER (asm)",
    "Triton (aiter backend)": "Triton (aiter backend)",
}


def blank(order: List[str], cols: List[int]) -> Dict[str, List[float | None]]:
    return {k: [None] * len(cols) for k in order}


def _col_idx(cols: List[int]) -> Dict[int, int]:
    return {v: i for i, v in enumerate(cols)}


def _set_cell(table: Dict[str, List[float | None]], idx: Dict[int, int], baseline: str, size: int, val: float) -> None:
    if baseline in table and size in idx:
        table[baseline][idx[size]] = float(val)


def parse_gemm(path: Path, cols: List[int]) -> Dict[str, List[float | None]]:
    table = blank(GEMM_ORDER, cols)
    idx = _col_idx(cols)
    payload = json.loads(path.read_text(encoding="utf-8"))
    for r in payload.get("rows", []):
        if r.get("status") != "ok":
            continue
        name = GEMM_MAP.get(str(r.get("baseline", "")))
        if name is None:
            continue
        _set_cell(table, idx, name, int(r["m"]), float(r["mean_ms"]))
    return table


def parse_attention(path: Path, cols: List[int]) -> Dict[str, List[float | None]]:
    table = blank(ATTN_ORDER, cols)
    idx = _col_idx(cols)
    payload = json.loads(path.read_text(encoding="utf-8"))
    for r in payload.get("rows", []):
        if r.get("status") != "ok":
            continue
        name = ATTN_MAP.get(str(r.get("baseline", "")))
        if name is None:
            continue
        _set_cell(table, idx, name, int(r["seq_len"]), float(r["mean_ms"]))
    return table


def parse_moe(path: Path, cols: List[int]) -> Dict[str, List[float | None]]:
    table = blank(MOE_ORDER, cols)
    idx = _col_idx(cols)
    payload = json.loads(path.read_text(encoding="utf-8"))
    for r in payload.get("rows", []):
        if r.get("status") != "ok":
            continue
        name = MOE_MAP.get(str(r.get("baseline", "")))
        if name is None:
            continue
        _set_cell(table, idx, name, int(r["seq_len"]), float(r["mean_ms"]))
    return table


def _fmt(v: float | None) -> str:
    return "" if v is None else f"{v:.6f}"


def _render_block(title: str, cols: List[int], order: List[str], rows: Dict[str, List[float | None]]) -> List[str]:
    head = "| Kernel | " + " | ".join(str(c) for c in cols) + " |"
    sep = "|---|" + "|".join(["---:"] * len(cols)) + "|"
    lines = [f"**{title} (mean_ms)**", "", head, sep]
    for name in order:
        lines.append("| " + name + " | " + " | ".join(_fmt(v) for v in rows.get(name, [])) + " |")
    lines.append("")
    return lines


def render_markdown(
    domain: str,
    tables: Dict[str, Dict[str, List[float | None]]],
    *,
    gemm_cols: List[int],
    attn_cols: List[int],
    moe_cols: List[int],
) -> str:
    lines = ["**Retime (mean_ms)**", ""]
    if domain in ("all", "gemm"):
        lines.extend(_render_block("GEMM", gemm_cols, GEMM_ORDER, tables["gemm"]))
    if domain in ("all", "attention"):
        lines.extend(_render_block("Attention", attn_cols, ATTN_ORDER, tables["attention"]))
    if domain in ("all", "moe"):
        lines.extend(_render_block("MoE", moe_cols, MOE_ORDER, tables["moe"]))
    return "\n".join(lines).rstrip() + "\n"
