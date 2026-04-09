#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch


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
    sig_part, meta_part = key_str.split("]{", 1)
    sig = ast.literal_eval(sig_part + "]")
    meta = ast.literal_eval("{" + meta_part)
    return sig, meta


def _find_best_kernel_for_config(compiled_map: dict, cfg) -> Tuple[object, str]:
    want_kwargs = dict(getattr(cfg, "kwargs", {}) or {})
    want_warps = int(getattr(cfg, "num_warps", 0) or 0)
    want_stages = int(getattr(cfg, "num_stages", 0) or 0)

    # For this kernel signature, Triton key signature constexpr indices:
    # 13: BLOCK_M, 14: BLOCK_N, 15: BLOCK_K, 16: GROUP_M
    constexpr_idx = {
        "BLOCK_M": 13,
        "BLOCK_N": 14,
        "BLOCK_K": 15,
        "GROUP_M": 16,
    }
    want_waves = want_kwargs.get("waves_per_eu", None)
    want_instr = want_kwargs.get("matrix_instr_nonkdim", None)

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
        if want_instr is not None and int(meta.get("matrix_instr_nonkdim", -1)) != int(want_instr):
            continue
        return kernel, str(key_str)

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
    for pat in ("*/_matmul_kernel.json", "*/matmul_kernel.json"):
        for p in sorted(cache_root.glob(pat)):
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if str(obj.get("hash", "")) == str(hash_value):
                return p.parent.name
    return ""


def copy_shared_so(cache_root: Path, out_shared: Path) -> None:
    out_shared.mkdir(parents=True, exist_ok=True)
    for pat in ("__triton_launcher*.so", "hip_utils*.so"):
        for p in sorted(cache_root.rglob(pat)):
            (out_shared / p.name).write_bytes(p.read_bytes())


def load_kernel_module(path: Path):
    spec = importlib.util.spec_from_file_location(f"kf_best_kernel_{os.getpid()}", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser(description="Export KernelFalcon Triton best-config artifacts")
    ap.add_argument("--root-dir", required=True)
    ap.add_argument("--kernel", default="", help="Path to best_kernel.py")
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--sizes", default="1024,2048,4096,8192,16384")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed-base", type=int, default=20260402)
    ap.add_argument("--hip-visible-devices", default="7")
    args = ap.parse_args()

    root = Path(args.root_dir).resolve()
    kernel_path = Path(args.kernel).resolve() if args.kernel else (root / "best_kernel.py")
    cache_root = Path(args.cache_root).resolve()
    sizes = parse_sizes(args.sizes)

    os.environ["HIP_VISIBLE_DEVICES"] = str(args.hip_visible_devices)
    os.environ.pop("ROCR_VISIBLE_DEVICES", None)
    os.environ["TRITON_CACHE_DIR"] = str(cache_root)
    os.environ["TRITON_KERNEL_DUMP"] = "1"

    cache_root.mkdir(parents=True, exist_ok=True)
    mod = load_kernel_module(kernel_path)

    if not hasattr(mod, "_matmul_kernel") or not hasattr(mod, "kernel_function"):
        raise RuntimeError(f"Unexpected kernel module layout: {kernel_path}")

    kernels_manifest: Dict[str, dict] = {}
    cache_manifest_map: Dict[str, dict] = {}

    for idx, m in enumerate(sizes):
        g = torch.Generator(device=args.device)
        g.manual_seed(args.seed_base + idx)
        a = torch.randn((m, m), device=args.device, dtype=torch.bfloat16, generator=g)
        b = torch.randn((m, m), device=args.device, dtype=torch.bfloat16, generator=g)

        _ = mod.kernel_function(a, b, transpose_b=True)
        torch.cuda.synchronize(device=args.device)

        key = (m, m, m, 1, "torch.bfloat16", "torch.bfloat16", "torch.bfloat16")
        cfg = mod._matmul_kernel.cache.get(key, None)
        if cfg is None:
            raise RuntimeError(f"Cannot find autotune config for key={key}")

        compiled_map = mod._matmul_kernel.fn.fn.device_caches[0][0]
        compiled_kernel, key_repr = _find_best_kernel_for_config(compiled_map, cfg)

        md = compiled_kernel.metadata
        md_json = metadata_to_json_dict(md)
        hash_value = str(md_json.get("hash", ""))
        cache_dir = find_cache_dir_by_hash(cache_root, hash_value)

        mnk = f"mnk_{m}x{m}x{m}"
        kernel_dir = root / "kernels" / mnk
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

        (kernel_dir / "kernel_meta.json").write_text(
            json.dumps(
                {
                    "shape": [m, m, m],
                    "hash": hash_value,
                    "cache_dir": cache_dir,
                    "key_repr": key_repr,
                    "selected_config": {
                        "BLOCK_M": cfg.kwargs.get("BLOCK_M"),
                        "BLOCK_N": cfg.kwargs.get("BLOCK_N"),
                        "BLOCK_K": cfg.kwargs.get("BLOCK_K"),
                        "GROUP_M": cfg.kwargs.get("GROUP_M"),
                        "matrix_instr_nonkdim": cfg.kwargs.get("matrix_instr_nonkdim"),
                        "waves_per_eu": cfg.kwargs.get("waves_per_eu"),
                        "num_warps": int(getattr(cfg, "num_warps", 0) or 0),
                        "num_stages": int(getattr(cfg, "num_stages", 0) or 0),
                    },
                    "metadata": md_json,
                },
                indent=2,
                ensure_ascii=True,
            )
            + "\n",
            encoding="utf-8",
        )

        child_paths = {}
        for asm_key, ext in ext_map.items():
            if asm_key in asm:
                if cache_dir:
                    p1 = cache_root / cache_dir / f"_matmul_kernel{ext}"
                    p2 = cache_root / cache_dir / f"matmul_kernel{ext}"
                    child_paths[f"matmul_kernel{ext}"] = str((p1 if p1.exists() else p2).resolve())
                else:
                    child_paths[f"matmul_kernel{ext}"] = ""
        if cache_dir:
            p1 = cache_root / cache_dir / "_matmul_kernel.json"
            p2 = cache_root / cache_dir / "matmul_kernel.json"
            child_paths["matmul_kernel.json"] = str((p1 if p1.exists() else p2).resolve())
        else:
            child_paths["matmul_kernel.json"] = ""
        group_json_path = kernel_dir / f"{mnk}_kernel_group.json"
        group_json_path.write_text(json.dumps({"child_paths": child_paths}, ensure_ascii=True), encoding="utf-8")
        exported_files.append(group_json_path.name)

        selected_cfg = {
            "BLOCK_M": cfg.kwargs.get("BLOCK_M"),
            "BLOCK_N": cfg.kwargs.get("BLOCK_N"),
            "BLOCK_K": cfg.kwargs.get("BLOCK_K"),
            "GROUP_M": cfg.kwargs.get("GROUP_M"),
            "matrix_instr_nonkdim": cfg.kwargs.get("matrix_instr_nonkdim"),
            "waves_per_eu": cfg.kwargs.get("waves_per_eu"),
            "num_warps": int(getattr(cfg, "num_warps", 0) or 0),
            "num_stages": int(getattr(cfg, "num_stages", 0) or 0),
        }

        kernels_manifest[str(m)] = {
            "cache_dir": cache_dir,
            "hash": hash_value,
            "selected_config": selected_cfg,
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

    (root / "kernels_manifest.json").write_text(
        json.dumps(kernels_manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    cache_manifest = [cache_manifest_map[k] for k in sorted(cache_manifest_map.keys())]
    (root / "triton_cache_manifest.json").write_text(
        json.dumps(cache_manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "workloads": len(sizes),
                "kernel_manifest": str((root / "kernels_manifest.json").resolve()),
                "cache_manifest": str((root / "triton_cache_manifest.json").resolve()),
                "cache_dirs": sorted(cache_manifest_map.keys()),
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
