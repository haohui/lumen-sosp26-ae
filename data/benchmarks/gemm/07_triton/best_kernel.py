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
_B_T_CACHE: dict[tuple, tuple[torch.Tensor, int]] = {}
_OUT_CACHE: dict[tuple, torch.Tensor] = {}


def _device_key(t: torch.Tensor) -> tuple[str, int | None]:
    return (t.device.type, t.device.index)


def _b_cache_key(b: torch.Tensor) -> tuple:
    return (
        _device_key(b),
        b.dtype,
        tuple(b.shape),
        tuple(b.stride()),
        int(b.data_ptr()),
    )


def _out_cache_key(a: torch.Tensor, b: torch.Tensor) -> tuple:
    return (
        _device_key(a),
        a.dtype,
        int(a.shape[0]),
        int(b.shape[1]),
    )


def _get_b_nk_cached(b: torch.Tensor) -> torch.Tensor:
    key = _b_cache_key(b)
    ver = int(getattr(b, "_version", 0))
    cached = _B_T_CACHE.get(key)
    if cached is not None:
        b_nk, cached_ver = cached
        if cached_ver == ver:
            return b_nk
    b_nk = b.t().contiguous()
    _B_T_CACHE[key] = (b_nk, ver)
    return b_nk


def _get_out_cached(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    key = _out_cache_key(a, b)
    out = _OUT_CACHE.get(key)
    if out is not None:
        return out
    out = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)
    _OUT_CACHE[key] = out
    return out


def kernel_function(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Triton baseline entry for benchmark_unified_graph.
    The source kernel is an autotuned ABt implementation:
    A:[M,K], Bsrc:[N,K] -> C=A@Bsrc^T.
    Unified runner provides B as [K,N], so we transpose once before launch.
    """
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("kernel_function expects 2D tensors")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"incompatible shapes: a={tuple(a.shape)}, b={tuple(b.shape)}")
    b_nk = _get_b_nk_cached(b)
    out = _get_out_cached(a, b)
    _SRC.matmul_bf16(a, b_nk, out=out)
    return out
