# Source: https://github.com/HazyResearch/HipKittens/blob/7d58fa1026b4582a75ebdaf7ab5e45e3747a2b7b/analysis/bf16_gemm/mi325x/kernel_1024.cpp
# Adapter for per-size pybind modules built by scripts/benchmark/build_hipkittens_mini.py.
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from sys import version_info

import torch


_MODULE_CACHE: dict[int, object] = {}
_PREP_CACHE: dict[tuple[int, int, int, str, str], tuple[torch.Tensor, torch.Tensor]] = {}


def _build_dir() -> Path:
    return Path(__file__).resolve().parent / "build_hipkittens_mini"


def _resolve_so_for_size(size: int) -> tuple[str, Path]:
    stem = f"tk_kernel_{size}_mini"
    build_dir = _build_dir()
    if not build_dir.exists():
        raise FileNotFoundError(
            f"HipKittens build dir missing: {build_dir}. "
            "Expected tk_kernel_<size>_mini.so artifacts under 08_hipketten/build_hipkittens_mini."
        )

    matches = sorted(
        build_dir.glob(f"{stem}*.so"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(
            f"HipKittens dispatch module for size={size} not found under {build_dir}"
        )

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


def dispatch_micro(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, _ = a.shape
    mod = _module_for_size(m)
    key = (a.data_ptr(), b.data_ptr(), m, str(a.dtype), str(a.device))
    cached = _PREP_CACHE.get(key)
    if cached is None:
        b_nk = b.contiguous()
        out = torch.empty((m, m), device=a.device, dtype=a.dtype)
        _PREP_CACHE[key] = (b_nk, out)
    else:
        b_nk, out = cached
    stream_ptr = int(torch.cuda.current_stream(device=a.device).cuda_stream)
    try:
        mod.dispatch_micro(a, b_nk, out, stream_ptr)
    except TypeError:
        mod.dispatch_micro(a, b_nk, out)
    return out
