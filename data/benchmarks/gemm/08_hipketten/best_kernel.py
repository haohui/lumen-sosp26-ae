#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from sys import version_info
from pathlib import Path

import torch


_MODULE_CACHE: dict[int, object] = {}
_PREP_CACHE: dict[tuple[int, int, int, str, str], tuple[torch.Tensor, torch.Tensor]] = {}


def _build_dir() -> Path:
    # Lock to mini build artifacts (stream-explicit interface used by our retime flow).
    return Path(__file__).resolve().parent / "build_hipkittens_mini"


def _resolve_so_for_size(size: int) -> tuple[str, Path]:
    stem = f"tk_kernel_{size}_mini"
    build_dir = _build_dir()
    if not build_dir.exists():
        raise FileNotFoundError(
            f"HipKittens build dir missing: {build_dir}. "
            "Expected tk_kernel_<size>_mini.so artifacts under 08_hipketten/build_hipkittens_mini."
        )

    matches = sorted(build_dir.glob(f"{stem}*.so"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"HipKittens dispatch module for size={size} not found under {build_dir}")

    py_tag = f"cpython-{version_info.major}{version_info.minor}"
    tag_matches = [p for p in matches if py_tag in p.name]
    if tag_matches:
        return stem, tag_matches[0]

    available = ", ".join(p.name for p in matches)
    raise RuntimeError(
        f"HipKittens modules found for size={size}, but none match Python ABI tag '{py_tag}'. "
        f"available: {available}"
    )


def _load_so_module(base_name: str, so_path: Path):
    # Pybind extension exports fixed symbol PyInit_<base_name>, so module name
    # must match the compiled target name exactly.
    module_name = base_name
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, so_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import extension module: {so_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "dispatch_micro"):
        raise RuntimeError(
            f"unexpected HipKittens module {so_path}: missing dispatch_micro entry"
        )
    sys.modules[module_name] = mod
    return mod


def _module_for_size(size: int):
    cached = _MODULE_CACHE.get(size)
    if cached is not None:
        return cached
    base_name, so_path = _resolve_so_for_size(size)
    mod = _load_so_module(base_name, so_path)
    _MODULE_CACHE[size] = mod
    return mod


def kernel_function(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    HipKittens baseline entry for benchmark_unified_graph.
    Uses prebuilt dispatch modules (`dispatch_micro`) with ABt kernel layout.
    """
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("kernel_function expects 2D tensors")
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"incompatible shapes: a={tuple(a.shape)}, b={tuple(b.shape)}")
    m, k = a.shape
    n, kb = b.shape
    if m != n or m != k or kb != k:
        raise ValueError(
            "HipKittens dispatch baseline only supports square M=N=K workloads "
            f"(got a={tuple(a.shape)}, b={tuple(b.shape)})"
        )
    mod = _module_for_size(m)
    key = (a.data_ptr(), b.data_ptr(), m, str(a.dtype), str(a.device))
    cached = _PREP_CACHE.get(key)
    if cached is None:
        b_nk = b.contiguous()
        out = torch.empty((m, n), device=a.device, dtype=a.dtype)
        _PREP_CACHE[key] = (b_nk, out)
    else:
        b_nk, out = cached
    stream_ptr = int(torch.cuda.current_stream(device=a.device).cuda_stream)
    mod.dispatch_micro(a, b_nk, out, stream_ptr)
    return out
