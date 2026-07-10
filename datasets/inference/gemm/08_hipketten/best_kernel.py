#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


_BINDING_PATH = Path(__file__).resolve().with_name("hipkittens_binding.py")
_BINDING_MODULE_NAME = f"hipkittens_binding_{abs(hash(str(_BINDING_PATH))):x}"
_LAZY_BINDING = None


def _load_binding():
    global _LAZY_BINDING
    if _LAZY_BINDING is not None:
        return _LAZY_BINDING
    cached = sys.modules.get(_BINDING_MODULE_NAME)
    if cached is not None:
        _LAZY_BINDING = cached
        return cached
    spec = importlib.util.spec_from_file_location(_BINDING_MODULE_NAME, _BINDING_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import HipKittens binding: {_BINDING_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_BINDING_MODULE_NAME] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(_BINDING_MODULE_NAME, None)
        raise
    _LAZY_BINDING = mod
    return mod


def kernel_function(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
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
    return _load_binding().dispatch_micro(a, b)
