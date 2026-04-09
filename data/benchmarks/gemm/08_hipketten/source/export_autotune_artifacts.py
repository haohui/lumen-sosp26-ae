#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import torch

import hipkittens_triton_gemm_v01_remap_xcd_matmul as hk_remap


def parse_sizes(s: str) -> List[int]:
    out: List[int] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    if not out:
        raise ValueError("--sizes is empty")
    return out


def parse_seed(note: str, fallback: int) -> int:
    m = re.search(r"shared_input_seed=(\d+)", note or "")
    if m:
        return int(m.group(1))
    return fallback


def metadata_to_json_dict(md) -> dict:
    keys = [
        "hash",
        "target",
        "num_warps",
        "waves_per_eu",
        "num_stages",
        "num_ctas",
        "extern_libs",
        "cluster_dims",
        "debug",
        "sanitize_overflow",
        "arch",
        "supported_fp8_dtypes",
        "deprecated_fp8_dot_operand_dtypes",
        "default_dot_input_precision",
        "allowed_dot_input_precisions",
        "enable_fp_fusion",
        "launch_cooperative_grid",
        "matrix_instr_nonkdim",
        "kpack",
        "allow_flush_denorm",
        "max_num_imprecise_acc_default",
        "backend_name",
        "instrumentation_mode",
        "schedule_hint",
        "warp_size",
        "triton_version",
        "shared",
        "profile_scratch_size",
        "profile_scratch_align",
        "name",
    ]
    out = {}
    for k in keys:
        if hasattr(md, k):
            out[k] = getattr(md, k)
    target = out.get("target", None)
    if target is not None:
        out["target"] = {
            "backend": getattr(target, "backend", None),
            "arch": getattr(target, "arch", None),
            "warp_size": getattr(target, "warp_size", None),
        }
    if "cluster_dims" in out and isinstance(out["cluster_dims"], tuple):
        out["cluster_dims"] = list(out["cluster_dims"])
    return out


def _parse_compiled_key(key_str: str) -> Tuple[list, dict]:
    # key format:
    #   "[('...'), ...]{'waves_per_eu': 2, 'num_warps': 4, ...}"
    sig_part, meta_part = key_str.split("]{", 1)
    sig = ast.literal_eval(sig_part + "]")
    meta = ast.literal_eval("{" + meta_part)
    return sig, meta


def match_best_compiled_kernel(compiled_map: dict, best_cfg) -> Tuple[object, str]:
    want_kwargs = dict(getattr(best_cfg, "kwargs", {}) or {})
    want_warps = int(getattr(best_cfg, "num_warps", 0) or 0)
    want_stages = int(getattr(best_cfg, "num_stages", 0) or 0)

    # arg index mapping in Triton key signature list:
    # 12:M, 13:N, 14:K, 15:GROUP, 16:REMAP_XCD, 17:NUM_XCDS, 18:STAGGER_K
    constexpr_idx = {
        "BLOCK_SIZE_M": 12,
        "BLOCK_SIZE_N": 13,
        "BLOCK_SIZE_K": 14,
        "GROUP_SIZE_M": 15,
        "REMAP_XCD": 16,
        "NUM_XCDS": 17,
        "STAGGER_K": 18,
    }
    want_waves = want_kwargs.get("waves_per_eu", None)

    for key_str, kernel in compiled_map.items():
        try:
            sig, meta = _parse_compiled_key(str(key_str))
        except Exception:
            continue

        ok = True
        for name, idx in constexpr_idx.items():
            want = want_kwargs.get(name, None)
            if want is None:
                continue
            if idx >= len(sig):
                ok = False
                break
            ent = sig[idx]
            if not (isinstance(ent, tuple) and len(ent) == 2 and ent[0] == "constexpr" and ent[1] == want):
                ok = False
                break
        if not ok:
            continue
        if int(meta.get("num_warps", -1)) != want_warps:
            continue
        if int(meta.get("num_stages", -1)) != want_stages:
            continue
        if want_waves is not None and int(meta.get("waves_per_eu", -1)) != int(want_waves):
            continue
        return kernel, str(key_str)

    # fallback: pick first kernel with same launch meta
    for key_str, kernel in compiled_map.items():
        md = kernel.metadata
        if int(getattr(md, "num_warps", -1)) != want_warps:
            continue
        if int(getattr(md, "num_stages", -1)) != want_stages:
            continue
        return kernel, str(key_str)

    key_str, kernel = next(iter(compiled_map.items()))
    return kernel, str(key_str)


def find_cache_dir_by_hash(cache_root: Path, hash_value: str) -> str:
    for p in sorted(cache_root.glob("*/matmul_kernel.json")):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(obj.get("hash", "")) == str(hash_value):
            return p.parent.name
    return ""


def write_row_files(workload_dir: Path, row: Dict[str, str]) -> None:
    workload_dir.mkdir(parents=True, exist_ok=True)
    (workload_dir / "row.json").write_text(json.dumps(row, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    with (workload_dir / "row.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)


def copy_shared_so(cache_root: Path, out_shared: Path) -> None:
    out_shared.mkdir(parents=True, exist_ok=True)
    for pat in ("__triton_launcher*.so", "hip_utils*.so"):
        for p in sorted(cache_root.glob(pat)):
            (out_shared / p.name).write_bytes(p.read_bytes())


def main() -> None:
    ap = argparse.ArgumentParser(description="Export selected Triton autotune kernel artifacts per workload")
    ap.add_argument("--root-dir", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--sizes", default="1024,2048,4096,8192,16384")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed-base", type=int, default=20260307)
    args = ap.parse_args()

    root = Path(args.root_dir)
    cache_root = Path(args.cache_root)
    sizes = parse_sizes(args.sizes)

    rows: List[Dict[str, str]] = []
    with Path(args.csv).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    row_by_m = {int(r["M"]): r for r in rows}

    kernels_manifest: Dict[str, dict] = {}
    cache_manifest_map: Dict[str, dict] = {}

    for idx, m in enumerate(sizes):
        if m not in row_by_m:
            raise RuntimeError(f"Missing row for M={m} in {args.csv}")
        row = row_by_m[m]

        seed = parse_seed(row.get("note", ""), args.seed_base + idx)
        g = torch.Generator(device=args.device)
        g.manual_seed(seed)
        a = torch.randn((m, m), device=args.device, dtype=torch.bfloat16, generator=g)
        b = torch.randn((m, m), device=args.device, dtype=torch.bfloat16, generator=g)
        out = torch.empty((m, m), device=args.device, dtype=torch.bfloat16)

        hk_remap.matmul_bf16(a, b, out=out)

        best_cfg = hk_remap.matmul_kernel.best_config
        best_meta = hk_remap.get_last_launch_meta() or {}

        compiled_map = hk_remap.matmul_kernel.fn.device_caches[0][0]
        compiled_kernel, key_repr = match_best_compiled_kernel(compiled_map, best_cfg)

        md = compiled_kernel.metadata
        md_json = metadata_to_json_dict(md)
        hash_value = str(md_json.get("hash", ""))
        cache_dir = find_cache_dir_by_hash(cache_root, hash_value)

        mnk = f"mnk_{m}x{m}x{m}"
        kernel_dir = root / "kernels" / mnk
        workload_dir = root / "workloads" / mnk
        kernel_dir.mkdir(parents=True, exist_ok=True)

        out_base = kernel_dir / f"{mnk}_kernel"
        asm = compiled_kernel.asm
        ext_map = {
            "source": ".source",
            "ttir": ".ttir",
            "ttgir": ".ttgir",
            "llir": ".llir",
            "amdgcn": ".amdgcn",
            "hsaco": ".hsaco",
        }
        exported_files: List[str] = []
        for asm_key, ext in ext_map.items():
            if asm_key not in asm:
                continue
            data = asm[asm_key]
            p = Path(str(out_base) + ext)
            if isinstance(data, (bytes, bytearray)):
                p.write_bytes(bytes(data))
            else:
                p.write_text(str(data), encoding="utf-8")
            exported_files.append(p.name)

        kernel_json_path = Path(str(out_base) + ".json")
        kernel_json_path.write_text(json.dumps(md_json, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        exported_files.append(kernel_json_path.name)

        (kernel_dir / "kernel_meta.json").write_text(json.dumps(md_json, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

        child_paths = {}
        for asm_key, ext in ext_map.items():
            if asm_key in asm:
                child_paths[f"matmul_kernel{ext}"] = str((cache_root / cache_dir / f"matmul_kernel{ext}").resolve()) if cache_dir else ""
        child_paths["matmul_kernel.json"] = str((cache_root / cache_dir / "matmul_kernel.json").resolve()) if cache_dir else ""

        group_json = {"child_paths": child_paths}
        group_json_path = kernel_dir / f"{mnk}_kernel_group.json"
        group_json_path.write_text(json.dumps(group_json, ensure_ascii=True), encoding="utf-8")
        exported_files.append(group_json_path.name)

        write_row_files(workload_dir, row)

        cfg_kwargs = dict(getattr(best_cfg, "kwargs", {}) or {})
        selected_cfg = {
            "BLOCK_SIZE_M": cfg_kwargs.get("BLOCK_SIZE_M"),
            "BLOCK_SIZE_N": cfg_kwargs.get("BLOCK_SIZE_N"),
            "BLOCK_SIZE_K": cfg_kwargs.get("BLOCK_SIZE_K"),
            "GROUP_SIZE_M": cfg_kwargs.get("GROUP_SIZE_M"),
            "REMAP_XCD": cfg_kwargs.get("REMAP_XCD"),
            "NUM_XCDS": cfg_kwargs.get("NUM_XCDS"),
            "STAGGER_K": cfg_kwargs.get("STAGGER_K"),
            "num_warps": int(getattr(best_cfg, "num_warps", 0) or 0),
            "num_stages": int(getattr(best_cfg, "num_stages", 0) or 0),
            "waves_per_eu": cfg_kwargs.get("waves_per_eu"),
        }

        kernels_manifest[str(m)] = {
            "cache_dir": cache_dir,
            "hash": hash_value,
            "selected_config": selected_cfg,
            "best_config_meta": best_meta,
            "key_repr": key_repr,
            "files": sorted(exported_files),
        }

        if cache_dir and cache_dir not in cache_manifest_map:
            cache_manifest_map[cache_dir] = {
                "cache_dir": cache_dir,
                "hash": hash_value,
                "config": selected_cfg,
                "triton_version": md_json.get("triton_version", ""),
            }

    copy_shared_so(cache_root, root / "kernels" / "_shared")

    (root / "kernels_manifest.json").write_text(json.dumps(kernels_manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    cache_manifest = [cache_manifest_map[k] for k in sorted(cache_manifest_map.keys())]
    (root / "triton_cache_manifest.json").write_text(json.dumps(cache_manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    print(json.dumps({
        "workloads": len(sizes),
        "kernel_manifest": str(root / "kernels_manifest.json"),
        "cache_manifest": str(root / "triton_cache_manifest.json"),
        "cache_dirs": sorted(cache_manifest_map.keys()),
    }, ensure_ascii=True))


if __name__ == "__main__":
    main()
