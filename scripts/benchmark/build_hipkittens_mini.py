#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sysconfig
from pathlib import Path

from paths import resolve_repo_root

SIZES = (1024, 2048, 4096, 8192, 16384)


def parse_args() -> argparse.Namespace:
    repo_root = resolve_repo_root()
    p = argparse.ArgumentParser(description="Build HipKittens mini GEMM modules")
    p.add_argument(
        "--hipkittens-root",
        type=Path,
        default=repo_root / "third_party" / "HipKittens",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=repo_root
        / "datasets"
        / "inference"
        / "gemm"
        / "08_hipketten"
        / "build_hipkittens_mini",
    )
    p.add_argument("--arch", default="gfx942")
    return p.parse_args()


def hip_include_dir(hipcc: Path) -> Path:
    roots = [Path("/opt/rocm"), Path("/opt/rocm-7.1.1"), hipcc.parents[1]]
    for root in roots:
        for cand in (root / "include", root / "include" / "hip"):
            if (cand / "hip_bf16.h").exists():
                return cand
    raise RuntimeError("cannot locate HIP include directory containing hip_bf16.h")


def pybind11_includes() -> list[str]:
    out = subprocess.check_output(
        ["python3", "-m", "pybind11", "--includes"],
        text=True,
    )
    return out.strip().split()


def build_one(
    *,
    hipcc: str,
    hip_include: Path,
    hipkittens_root: Path,
    out_dir: Path,
    size: int,
    arch: str,
    ext: str,
    includes: list[str],
) -> None:
    src_in = (
        hipkittens_root
        / "analysis"
        / "bf16_gemm"
        / "mi325x"
        / f"kernel_{size}.cpp"
    )
    if not src_in.exists():
        raise FileNotFoundError(f"missing HipKittens source: {src_in}")

    src_out = out_dir / f"tk_kernel_{size}_mini__autogen.cpp"
    out_so = out_dir / f"tk_kernel_{size}_mini{ext}"
    text = src_in.read_text(encoding="utf-8")
    needle = "PYBIND11_MODULE(tk_kernel, m)"
    if needle not in text:
        raise RuntimeError(f"missing module macro in {src_in}")
    src_out.write_text(
        text.replace(needle, f"PYBIND11_MODULE(tk_kernel_{size}_mini, m)"),
        encoding="utf-8",
    )

    cmd = [
        hipcc,
        str(src_out),
        "-O3",
        "-DKITTENS_CDNA3",
        f"--offload-arch={arch}",
        "-std=c++20",
        "-w",
        f"-I{hip_include}",
        f"-I{hipkittens_root / 'include'}",
        f"-I{hipkittens_root / 'prototype'}",
        *includes,
        "-shared",
        "-fPIC",
        "-Rpass-analysis=kernel-resource-usage",
        "-lpthread",
        "-ldl",
        "-lutil",
        "-lm",
        "-o",
        str(out_so),
    ]
    subprocess.run(cmd, check=True)


def main() -> int:
    args = parse_args()
    hipcc = shutil.which("hipcc")
    if hipcc is None:
        raise RuntimeError("hipcc not found in PATH")
    if not args.hipkittens_root.exists():
        raise FileNotFoundError(
            f"missing {args.hipkittens_root}; run scripts/benchmark/prepare_sources.sh"
        )

    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ext = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    includes = pybind11_includes()
    hip_include = hip_include_dir(Path(hipcc).resolve())

    for size in SIZES:
        build_one(
            hipcc=hipcc,
            hip_include=hip_include,
            hipkittens_root=args.hipkittens_root,
            out_dir=args.out_dir,
            size=size,
            arch=args.arch,
            ext=ext,
            includes=includes,
        )

    print(f"built HipKittens mini modules under {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
