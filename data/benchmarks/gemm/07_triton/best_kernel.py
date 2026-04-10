#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


def _load_source_module():
    src = (
        Path(__file__).resolve().parent
        / "source"
        / "hipkittens_triton_gemm_v01_remap_xcd_matmul.py"
    )
    spec = importlib.util.spec_from_file_location("triton_remap_xcd_autotune_src", src)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import source module: {src}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Ensure we are using the intended ABt + remap_xcd autotune implementation.
    required = ("matmul_bf16", "matmul_kernel", "remap_xcd")
    missing = [name for name in required if not hasattr(mod, name)]
    if missing:
        raise RuntimeError(
            f"unexpected Triton source module at {src}; missing symbols: {', '.join(missing)}"
        )
    return mod


_SRC = _load_source_module()
_CACHE: dict[tuple[int, int, int, str, str], tuple[torch.Tensor, torch.Tensor]] = {}


def kernel_function(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Triton baseline entry for benchmark_unified_graph.
    Keep this aligned with run_gemm_baselines_cudagraph.py:
    - convert B(K,N) to B(N,K) once per input pair
    - reuse output buffer between calls
    - call matmul_bf16(a, b_nk, out=out)
    """
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("kernel_function expects 2D tensors")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"incompatible shapes: a={tuple(a.shape)}, b={tuple(b.shape)}")
    m = int(a.shape[0])
    n = int(b.shape[1])
    key = (a.data_ptr(), b.data_ptr(), m, str(a.dtype), str(a.device))
    cached = _CACHE.get(key)
    if cached is None:
        b_nk = b.t().contiguous()
        out = torch.empty((m, n), device=a.device, dtype=a.dtype)
        _CACHE[key] = (b_nk, out)
    else:
        b_nk, out = cached
    _SRC.matmul_bf16(a, b_nk, out=out)
    return out
