#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


def _load_binding():
    path = Path(__file__).resolve().with_name("hipkittens_binding.py")
    module_name = f"hipkittens_binding_{abs(hash(str(path.resolve()))):x}"
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import HipKittens binding: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
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
