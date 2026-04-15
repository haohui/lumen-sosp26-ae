#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, List

try:
    import torch
except Exception:
    torch = None


@dataclass
class SharedInputs:
    a_mk: "torch.Tensor"
    b_nk: "torch.Tensor"


def gemm_tflops(m: int, n: int, k: int, ms: float) -> float:
    return (2.0 * m * n * k) / (ms * 1.0e-3) / 1.0e12


def build_shared_inputs(*, sizes: List[int], device: "torch.device", dtype: "torch.dtype", seed: int) -> Dict[int, SharedInputs]:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    out: Dict[int, SharedInputs] = {}
    for s in sizes:
        # Unified GEMM contract for all baselines: A:[M,K], B:[N,K], output C:[M,N] = A @ B^T.
        b_nk = torch.randn((s, s), device=device, dtype=dtype, generator=g)
        out[s] = SharedInputs(
            a_mk=torch.randn((s, s), device=device, dtype=dtype, generator=g),
            b_nk=b_nk,
        )
    return out


def load_hipblaslt_internal_module(gemm_root: Path):
    from torch.utils.cpp_extension import load_inline

    src = gemm_root / "06_hipblaslt" / "hipblaslt_internal_ext.cpp"
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
