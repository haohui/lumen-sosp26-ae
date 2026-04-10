#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import List, Tuple


def resolve_hipkittens_kernel(case_n: int, kernels_dir: Path) -> Tuple[str, Path]:
    py_tag = f"cpython-{sys.version_info.major}{sys.version_info.minor}"
    candidates = [
        f"tk_kernel_unified_{case_n}",
        f"tk_kernel_{case_n}",
        f"tk_kernel_{case_n}_mini",
    ]
    for module_name in candidates:
        # Require exact module stem match to avoid prefix collisions such as:
        # tk_kernel_1024 (candidate) vs tk_kernel_1024_mini (file).
        matches = sorted(kernels_dir.glob(f"{module_name}.cpython-*.so"))
        if not matches:
            continue
        tag_matches = [p for p in matches if py_tag in p.name]
        if not tag_matches:
            continue
        return module_name, max(tag_matches, key=lambda p: p.stat().st_mtime)
    raise RuntimeError(
        f"no HipKittens module matching ABI tag '{py_tag}' for size={case_n} under {kernels_dir}; "
        f"tried stems: {', '.join(candidates)}"
    )


def load_extension_module(name: str, so_path: Path) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(so_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import extension: {so_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module


def load_hipblaslt_internal_module(gemm_root: Path):
    from torch.utils.cpp_extension import load_inline

    src = gemm_root / "06_hipblaslt" / "src" / "hipblaslt_internal_ext.cpp"
    if not src.exists():
        raise FileNotFoundError(f"missing source: {src}")

    lib_dirs = [
        Path("/opt/rocm/lib"),
        Path("/opt/rocm-7.1.1/lib"),
    ]
    lib_dirs = [d for d in lib_dirs if d.exists() and d.is_dir()]
    if not lib_dirs:
        raise FileNotFoundError("no ROCm library directory found for hipBLASLt extension build")

    ldflags: List[str] = []
    for d in lib_dirs:
        ldflags.extend([f"-L{d}", f"-Wl,-rpath,{d}"])
    ldflags.extend(["-lhipblaslt", "-lhipblas", "-lrocblas", "-lamdhip64"])

    for d in [Path("/opt/rocm/lib/hipblaslt/library"), Path("/opt/rocm-7.1.1/lib/hipblaslt/library")]:
        if (d / "TensileLibrary_lazy_gfx942.dat").exists():
            os.environ["HIPBLASLT_TENSILE_LIBPATH"] = str(d)
            break

    os.environ.setdefault("CXX", "hipcc")
    cpp_src = src.read_text(encoding="utf-8")
    return load_inline(
        name="kb_hipblaslt_internal_ext",
        cpp_sources=cpp_src,
        functions=["hipblaslt_bf16_mm_out"],
        extra_cflags=["-O3"],
        extra_ldflags=ldflags,
        with_cuda=False,
        verbose=False,
    )
