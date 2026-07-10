#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def load_module(path: Path) -> ModuleType:
    module_name = f"bench_dyn_{path.stem}_{abs(hash(str(path.resolve()))):x}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses may inspect sys.modules during class decoration (Python 3.12).
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


@dataclass(frozen=True)
class ModelBackend:
    entrypoint: str
    kind: str

    def supports(self, module: ModuleType) -> bool:
        return hasattr(module, self.entrypoint)

    def build(
        self,
        module: ModuleType,
        *,
        device: torch.device,
        dtype: torch.dtype | None,
        model_kwargs: dict[str, Any],
    ):
        target = getattr(module, self.entrypoint)
        if self.kind == "class":
            model = target(**model_kwargs)
            if hasattr(model, "to"):
                if dtype is None:
                    model = model.to(device=device)
                else:
                    model = model.to(device=device, dtype=dtype)
            return lambda *xs: model(*xs)
        if self.kind == "function":
            if model_kwargs:
                raise RuntimeError(
                    f"function backend {self.entrypoint} does not accept model kwargs"
                )
            del device, dtype
            return lambda *xs: target(*xs)
        raise ValueError(f"unsupported backend kind: {self.kind}")


BACKEND_MAP = {
    "model_new": ModelBackend(entrypoint="ModelNew", kind="class"),
    "model": ModelBackend(entrypoint="Model", kind="class"),
    "kernel_function": ModelBackend(entrypoint="kernel_function", kind="function"),
    "run": ModelBackend(entrypoint="run", kind="function"),
}


def build_model_fn(
    module: ModuleType,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
    model_kwargs: dict[str, Any] | None = None,
):
    model_kwargs = {} if model_kwargs is None else model_kwargs
    for backend in BACKEND_MAP.values():
        if backend.supports(module):
            return backend.build(
                module,
                device=device,
                dtype=dtype,
                model_kwargs=model_kwargs,
            )
    expected = ", ".join(
        f"{name} ({backend.entrypoint})" for name, backend in BACKEND_MAP.items()
    )
    raise RuntimeError(f"expected one of: {expected}")
