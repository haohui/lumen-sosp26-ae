#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Protocol

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


class EntrypointBackend(Protocol):
    name: str

    def supports(self, module: ModuleType) -> bool: ...

    def build(
        self,
        module: ModuleType,
        *,
        device: torch.device,
        dtype: torch.dtype | None,
    ): ...


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
class ModelClassBackend:
    name: str

    def supports(self, module: ModuleType) -> bool:
        return hasattr(module, self.name)

    def build(
        self,
        module: ModuleType,
        *,
        device: torch.device,
        dtype: torch.dtype | None,
    ):
        model = getattr(module, self.name)()
        if hasattr(model, "to"):
            if dtype is None:
                model = model.to(device=device)
            else:
                model = model.to(device=device, dtype=dtype)
        return lambda *xs: model(*xs)


@dataclass(frozen=True)
class FunctionBackend:
    name: str

    def supports(self, module: ModuleType) -> bool:
        return hasattr(module, self.name)

    def build(
        self,
        module: ModuleType,
        *,
        device: torch.device,
        dtype: torch.dtype | None,
    ):
        del device, dtype
        target = getattr(module, self.name)
        return lambda *xs: target(*xs)


ENTRYPOINT_BACKENDS: tuple[EntrypointBackend, ...] = (
    ModelClassBackend("ModelNew"),
    ModelClassBackend("Model"),
    FunctionBackend("kernel_function"),
    FunctionBackend("run"),
)


def build_model_fn(
    module: ModuleType,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
):
    for backend in ENTRYPOINT_BACKENDS:
        if backend.supports(module):
            return backend.build(module, device=device, dtype=dtype)
    expected = ", ".join(backend.name for backend in ENTRYPOINT_BACKENDS)
    raise RuntimeError(f"expected one of: {expected}")
