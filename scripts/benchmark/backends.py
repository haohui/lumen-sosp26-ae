#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from collections.abc import Callable
from typing import Any, Literal

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


@dataclass(frozen=True)
class ModelBackend:
    run: Callable[..., None]

    def __call__(self, **kwargs: Any) -> None:
        self.run(**kwargs)


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
class ModelEntrypoint:
    entrypoint: str
    kind: Literal["class", "function"]

    def supports(self, module: ModuleType) -> bool:
        return hasattr(module, self.entrypoint)

    def build(
        self,
        module: ModuleType,
        *,
        device: torch.device,
        dtype: torch.dtype | None,
    ):
        target = getattr(module, self.entrypoint)
        if self.kind == "class":
            model = target()
            if hasattr(model, "to"):
                if dtype is None:
                    model = model.to(device=device)
                else:
                    model = model.to(device=device, dtype=dtype)
            return lambda *xs: model(*xs)
        if self.kind == "function":
            del device, dtype
            return lambda *xs: target(*xs)
        raise ValueError(f"unsupported entrypoint kind: {self.kind}")


ENTRYPOINTS = (
    ModelEntrypoint(entrypoint="ModelNew", kind="class"),
    ModelEntrypoint(entrypoint="Model", kind="class"),
    ModelEntrypoint(entrypoint="kernel_function", kind="function"),
    ModelEntrypoint(entrypoint="run", kind="function"),
)


def build_model_fn(
    module: ModuleType,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
):
    for entrypoint in ENTRYPOINTS:
        if entrypoint.supports(module):
            return entrypoint.build(module, device=device, dtype=dtype)
    expected = ", ".join(entrypoint.entrypoint for entrypoint in ENTRYPOINTS)
    raise RuntimeError(f"expected one of: {expected}")
