#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn


DIRECT_EXIT_AFTER_SUCCESS = True
_THIS_DIR = Path(__file__).resolve().parent
_MODULES = {
    1024: "amdgpu_gemm_1024.py",
    2048: "amdgpu_gemm_2048.py",
    4096: "amdgpu_gemm_4096.py",
    8192: "amdgpu_gemm_8192.py",
    16384: "amdgpu_gemm_16384.py",
}


def _load_sibling(filename: str):
    path = _THIS_DIR / filename
    module_name = f"lumen_gemm_{path.stem}_{abs(hash(str(path.resolve()))):x}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import LUMEN GEMM module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self._module_cache = {}
        self._out_cache = {}

    def _resolve(self, size: int):
        module_file = _MODULES.get(size)
        if module_file is None:
            supported = ", ".join(str(x) for x in sorted(_MODULES))
            raise ValueError(
                f"LUMEN GEMM does not provide size {size}; supported: {supported}"
            )
        if module_file not in self._module_cache:
            self._module_cache[module_file] = _load_sibling(module_file)
        mod = self._module_cache[module_file]
        fn_name = f"gemm_{size}_transposed_b"
        fn = getattr(mod, fn_name, None) or getattr(
            mod, "gemm_pipeline_transposed_b", None
        )
        if fn is None:
            raise AttributeError(
                f"LUMEN GEMM module {module_file} does not contain "
                f"{fn_name} or gemm_pipeline_transposed_b"
            )
        return fn

    def build_call(self, *, a_mk: torch.Tensor, b_nk: torch.Tensor):
        if a_mk.dtype != torch.bfloat16 or b_nk.dtype != torch.bfloat16:
            raise ValueError("LUMEN GEMM benchmark currently supports only bf16")
        if a_mk.ndim != 2 or b_nk.ndim != 2:
            raise ValueError("LUMEN GEMM expects 2D tensors")
        if a_mk.shape != b_nk.shape or a_mk.shape[0] != a_mk.shape[1]:
            raise ValueError(
                "LUMEN GEMM expects square A:[S,S], B:[S,S] for A@B^T, got "
                f"A={tuple(a_mk.shape)}, B={tuple(b_nk.shape)}"
            )

        size = int(a_mk.shape[0])
        fn = self._resolve(size)
        key = (size, a_mk.device, a_mk.dtype)
        out = self._out_cache.get(key)
        if out is None:
            out = torch.empty((size, size), device=a_mk.device, dtype=a_mk.dtype)
            self._out_cache[key] = out

        return lambda: fn(a_mk, b_nk, out)

    def forward(self, a_mk: torch.Tensor, b_nk: torch.Tensor) -> torch.Tensor:
        call = self.build_call(a_mk=a_mk, b_nk=b_nk)
        call()
        size = int(a_mk.shape[0])
        return self._out_cache[(size, a_mk.device, a_mk.dtype)]
