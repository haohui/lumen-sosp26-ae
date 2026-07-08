#!/usr/bin/env python3
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    import torch
except Exception:
    torch = None


@dataclass
class SharedInputs:
    a_mk: torch.Tensor
    b_nk: torch.Tensor


def gemm_tflops(m: int, n: int, k: int, ms: float) -> float:
    return (2.0 * m * n * k) / (ms * 1.0e-3) / 1.0e12


def build_shared_inputs(
    *,
    sizes: list[int],
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> dict[int, SharedInputs]:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    out: dict[int, SharedInputs] = {}
    for s in sizes:
        # Unified contract: A:[M,K], B:[N,K], output C:[M,N] = A @ B^T.
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

    roots = [os.environ.get(k, "") for k in ("ROCM_PATH", "ROCM_HOME", "HIP_PATH")]
    lib_dirs = [Path(r) / "lib" for r in roots if r]
    for env_name in ("LD_LIBRARY_PATH", "LIBRARY_PATH"):
        lib_dirs.extend(Path(p) for p in os.environ.get(env_name, "").split(":") if p)
    lib_dirs = [d for d in dict.fromkeys(lib_dirs) if d.exists() and d.is_dir()]

    ldflags: list[str] = []
    for d in lib_dirs:
        ldflags.extend([f"-L{d}", f"-Wl,-rpath,{d}"])
    ldflags.extend(["-lhipblaslt", "-lhipblas", "-lrocblas", "-lamdhip64"])

    for d in [base / "hipblaslt" / "library" for base in lib_dirs]:
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
