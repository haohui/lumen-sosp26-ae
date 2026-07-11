#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd


ROCM_METRIC_COLUMNS: List[str] = [
    "KernelName",
    "Kernel Name",
    "DurationNs",
    "Duration (ns)",
    "TotalDurationNs",
    "AverageNs",
    "Calls",
    "Percentage",
    "DispatchNs",
    "SQ_WAVES",
    "SQ_INSTS_VALU",
    "SQ_INSTS_SALU",
    "GRBM_GUI_ACTIVE",
    "GRBM_COUNT",
    "LDSBankConflict",
    "L2CacheHit",
    "L2CacheMiss",
]


def profile_bench_rocm(
    bench_py: str = "bench_ref_inputs.py",
    kernel_names: Optional[List[str]] = None,
    out_csv: Union[str, Path] = "rocprof_temp.csv",
    repeat: int = 100,
    output_dir: Union[str, Path] = "rocprof_out",
) -> Path:
    """Run ROCm profiling and return a CSV path suitable for prompt feedback."""
    csv_path = Path(out_csv).resolve()
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    start_ts = time.time()
    existing = {p.resolve() for p in out_dir.rglob("*.csv")}

    proc_stdout = ""
    proc_stderr = ""
    rocprof_compute = shutil.which("rocprof-compute")
    if rocprof_compute:
        try:
            proc_stdout, proc_stderr = _run_rocprof_compute(
                rocprof_bin=rocprof_compute,
                bench_py=bench_py,
                kernel_names=kernel_names,
                repeat=repeat,
                out_dir=out_dir,
            )
        except Exception as exc:
            print(f"[rocprof-compute] unavailable, falling back to rocprofv3: {exc}")
            proc_stdout, proc_stderr = _run_rocprofv3(
                bench_py=bench_py,
                repeat=repeat,
                out_dir=out_dir,
            )
    else:
        proc_stdout, proc_stderr = _run_rocprofv3(
            bench_py=bench_py,
            repeat=repeat,
            out_dir=out_dir,
        )

    csv_candidates = _find_profile_csv_candidates(
        out_dir=out_dir,
        existing=existing,
        start_ts=start_ts,
    )
    if not csv_candidates:
        raise FileNotFoundError(
            "ROCm profiler completed but no CSV file was found under "
            f"{out_dir}. stdout:\n{proc_stdout}\nstderr:\n{proc_stderr}"
        )

    src = csv_candidates[0]
    csv_path.write_text(src.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
    print(f"[ok] ROCm CSV written: {csv_path}")
    return csv_path


def _run_rocprof_compute(
    *,
    rocprof_bin: str,
    bench_py: str,
    kernel_names: Optional[List[str]],
    repeat: int,
    out_dir: Path,
) -> tuple[str, str]:
    run_name = "ae_rocm_profile"
    cmd = [rocprof_bin, "profile", "-n", run_name, "-p", str(out_dir)]
    if kernel_names:
        for name in sorted({k.strip() for k in kernel_names if k and k.strip()}):
            cmd.extend(["-k", name])
    cmd.extend(["--", sys.executable, bench_py, "--repeat", str(repeat)])

    print("[rocprof-compute] running:", " ".join(cmd))
    proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "rocprof-compute failed\n"
            f"returncode={proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc.stdout, proc.stderr


def _run_rocprofv3(*, bench_py: str, repeat: int, out_dir: Path) -> tuple[str, str]:
    rocprof_bin = shutil.which("rocprofv3")
    if not rocprof_bin:
        raise FileNotFoundError("Neither rocprof-compute nor rocprofv3 was found in PATH")

    cmd = [
        rocprof_bin,
        "--kernel-trace",
        "--stats",
        "-f",
        "csv",
        "-d",
        str(out_dir),
        "-o",
        "ae_rocm_profile",
        "--",
        sys.executable,
        bench_py,
        "--repeat",
        str(repeat),
    ]
    print("[rocprofv3] running:", " ".join(cmd))
    proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "rocprofv3 failed\n"
            f"returncode={proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc.stdout, proc.stderr


def _find_profile_csv_candidates(
    *, out_dir: Path, existing: set[Path], start_ts: float
) -> list[Path]:
    all_candidates = sorted(out_dir.rglob("*.csv"))
    new_candidates = [
        p
        for p in all_candidates
        if p.resolve() not in existing and p.stat().st_mtime >= start_ts - 2.0
    ]
    csv_candidates = new_candidates if new_candidates else all_candidates

    def _score_candidate(p: Path) -> float:
        score = 0.0
        name = p.name.lower()
        if "counter_collection" in name:
            score += 100.0
        elif "kernel_stats" in name:
            score += 80.0
        elif "kernel_trace" in name:
            score += 30.0
        elif "domain_stats" in name or "agent_info" in name:
            score -= 20.0

        try:
            probe = pd.read_csv(p, nrows=256, low_memory=False)
            cols = set(str(c) for c in probe.columns)
            if {"Counter_Name", "Counter_Value"}.issubset(cols):
                score += 60.0
            if _detect_kernel_col(probe):
                score += 20.0
            preferred = sum(1 for c in ROCM_METRIC_COLUMNS if c in cols)
            score += float(preferred * 5)
        except Exception:
            pass

        try:
            score += math.log10(max(1, p.stat().st_size))
        except Exception:
            pass
        return score

    return sorted(csv_candidates, key=_score_candidate, reverse=True)


def _detect_kernel_col(df: pd.DataFrame) -> Optional[str]:
    for c in ("Kernel Name", "Kernel_Name", "KernelName", "Name", "Kernel"):
        if c in df.columns:
            return c
    return None


def load_rocm_metrics(
    csv_path: Union[str, Path] = "rocprof_temp.csv",
    columns: Optional[Sequence[str]] = None,
    extra_keep: Optional[Sequence[str]] = None,
    coerce_numeric: bool = True,
    name_list: Optional[Sequence[str]] = None,
    select: str = "last",
) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False) if csv_path.stat().st_size else pd.DataFrame()
    kernel_col = _detect_kernel_col(df)
    if kernel_col and {"Counter_Name", "Counter_Value"}.issubset(df.columns):
        long_df = df[[kernel_col, "Counter_Name", "Counter_Value"]].copy()
        long_df["Counter_Value"] = pd.to_numeric(long_df["Counter_Value"], errors="coerce")
        long_df = long_df.dropna(subset=["Counter_Value"])
        if not long_df.empty:
            df = (
                long_df.pivot_table(
                    index=kernel_col,
                    columns="Counter_Name",
                    values="Counter_Value",
                    aggfunc="mean",
                )
                .reset_index()
            )
            kernel_col = _detect_kernel_col(df)

    preferred_cols = list(columns) if columns is not None else ROCM_METRIC_COLUMNS
    keep_cols: List[str] = []
    if kernel_col:
        keep_cols.append(kernel_col)
    if extra_keep:
        keep_cols.extend([c for c in extra_keep if c in df.columns and c not in keep_cols])
    keep_cols.extend([c for c in preferred_cols if c in df.columns and c not in keep_cols])
    if len(keep_cols) <= 1:
        fallback = [c for c in df.columns if c != kernel_col]
        numeric = [c for c in fallback if pd.api.types.is_numeric_dtype(df[c])]
        keep_cols.extend((numeric or fallback)[:24])
    if not keep_cols:
        keep_cols = list(df.columns[:24])

    sub = df[keep_cols].copy()
    if coerce_numeric:
        numeric_candidates = [c for c in sub.columns if c != kernel_col]
        sub[numeric_candidates] = sub[numeric_candidates].replace({",": "", "%": ""}, regex=True)
        for col in numeric_candidates:
            converted = pd.to_numeric(sub[col], errors="coerce")
            if converted.notna().any():
                sub[col] = converted

    if name_list and kernel_col:
        rows = []
        for name in name_list:
            matched = sub[sub[kernel_col].astype(str).str.contains(name, regex=False, na=False)]
            if matched.empty:
                continue
            rows.append(matched.iloc[[0 if select == "first" else -1]])
        if rows:
            return pd.concat(rows, ignore_index=True)

    return compact_profile_dataframe(sub, kernel_col=kernel_col)


def compact_profile_dataframe(
    df: pd.DataFrame,
    *,
    kernel_col: Optional[str] = None,
    max_rows: int = 12,
) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    kernel_col = kernel_col or _detect_kernel_col(df)
    priority_cols = [
        "DurationNs",
        "Duration (ns)",
        "TotalDurationNs",
        "AverageNs",
        "Calls",
        "Percentage",
        "DispatchNs",
        "VGPR_Count",
        "Accum_VGPR_Count",
        "SGPR_Count",
        "Workgroup_Size_X",
        "Workgroup_Size_Y",
        "Workgroup_Size_Z",
        "Grid_Size_X",
        "Grid_Size_Y",
        "Grid_Size_Z",
        "LDS_Block_Size",
        "Scratch_Size",
    ]
    keep = [c for c in priority_cols if c in df.columns]
    if kernel_col and kernel_col in df.columns:
        keep = [kernel_col] + [c for c in keep if c != kernel_col]
    if not keep:
        keep = list(df.columns[:20])

    out = df[keep].copy()
    if kernel_col and kernel_col in out.columns:
        numeric_cols = [
            c for c in out.columns if c != kernel_col and pd.api.types.is_numeric_dtype(out[c])
        ]
        if numeric_cols:
            out = out.groupby(kernel_col, as_index=False).agg({c: "mean" for c in numeric_cols})
        else:
            out = out.drop_duplicates(subset=[kernel_col])

    duration_col = next(
        (c for c in ("DurationNs", "Duration (ns)", "TotalDurationNs", "AverageNs", "DispatchNs") if c in out.columns),
        None,
    )
    if duration_col:
        out = out.sort_values(duration_col, ascending=False)
    return out.head(max_rows).reset_index(drop=True)


def metrics_to_prompt_rocm(
    df: pd.DataFrame,
    key_by: Optional[str] = None,
    round_digits: Optional[int] = 3,
    compact: bool = False,
    max_kernels: int = 8,
    max_metrics_per_kernel: int = 24,
) -> str:
    return metrics_to_prompt_summary(
        df,
        key_by=key_by,
        round_digits=round_digits,
        compact=compact,
        max_kernels=max_kernels,
        max_metrics_per_kernel=max_metrics_per_kernel,
        format_name="rocm_profile_summary",
    )


def metrics_to_prompt_summary(
    df: pd.DataFrame,
    *,
    key_by: Optional[str] = None,
    round_digits: Optional[int] = 3,
    compact: bool = False,
    max_kernels: int = 8,
    max_metrics_per_kernel: int = 24,
    format_name: str = "profile_summary",
) -> str:
    def _safe(v: Any) -> Any:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        if isinstance(v, np.generic):
            v = v.item()
        if isinstance(v, float) and math.isinf(v):
            return "inf" if v > 0 else "-inf"
        if isinstance(v, float) and round_digits is not None:
            return round(v, round_digits)
        return v

    if df is None or df.empty:
        return "{}"

    key_col = key_by or _detect_kernel_col(df)
    if round_digits is not None:
        num_cols = df.select_dtypes(include="number").columns
        if len(num_cols) > 0:
            df = df.copy()
            df[num_cols] = df[num_cols].round(round_digits)

    candidates = [c for c in df.columns if c != key_col]
    preferred = [c for c in ROCM_METRIC_COLUMNS if c in candidates]
    numeric = [
        c for c in candidates if c not in preferred and pd.api.types.is_numeric_dtype(df[c])
    ]
    rest = [c for c in candidates if c not in preferred and c not in numeric]
    value_cols = (preferred + numeric + rest)[: max(1, int(max_metrics_per_kernel))]

    if key_col is None or key_col not in df.columns:
        rows = [
            {k: _safe(v) for k, v in rec.items()}
            for rec in df[value_cols].head(max_kernels).to_dict(orient="records")
        ]
        payload: Dict[str, Any] = {
            "format": format_name,
            "row_count": int(len(df)),
            "metrics": value_cols,
            "rows": rows,
        }
    else:
        data: Dict[str, Any] = {}
        for rec in df[[key_col] + value_cols].head(max_kernels).to_dict(orient="records"):
            key = str(rec.pop(key_col))
            data[key] = {k: _safe(v) for k, v in rec.items()}
        payload = {
            "format": format_name,
            "kernel_count": int(df[key_col].astype(str).nunique()),
            "emitted_kernels": int(len(data)),
            "metrics": value_cols,
            "kernels": data,
        }
    return json.dumps(payload, ensure_ascii=False, indent=None if compact else 2)
