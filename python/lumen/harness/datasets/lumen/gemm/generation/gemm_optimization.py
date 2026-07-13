"""GEMM-specific configuration for the shared Lumen optimization runtime."""

from __future__ import annotations

from pathlib import Path

from lumen.harness.datasets.lumen.optimization_runtime import OptimizationSpec


GEMM_WORKLOADS = (1024, 2048, 4096, 8192, 16384)


def gemm_optimization_spec(repo_root: Path) -> OptimizationSpec:
    return OptimizationSpec(
        domain="gemm",
        run_slug="gemm",
        workloads=GEMM_WORKLOADS,
        workload_key="matrix_size",
        workload_flag="--matrix-sizes",
        benchmark_script="bench_gemm.py",
        adapter_source=_benchmark_adapter_source(),
    )


def _benchmark_adapter_source() -> str:
    return '''#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn


_KERNEL_PATH = Path(__file__).resolve().parents[4] / "output_model_new.py"


def _load_kernel():
    spec = importlib.util.spec_from_file_location(
        "lumen_gemm_candidate",
        _KERNEL_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import candidate: {_KERNEL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self._kernel = _load_kernel()
        self._out_cache = {}

    def build_call(self, *, a_mk: torch.Tensor, b_nk: torch.Tensor):
        key = (a_mk.shape, b_nk.shape, a_mk.device, a_mk.dtype)
        out = self._out_cache.get(key)
        if out is None:
            out = torch.empty(
                (a_mk.shape[0], b_nk.shape[0]),
                device=a_mk.device,
                dtype=a_mk.dtype,
            )
            self._out_cache[key] = out
        return lambda: self._kernel.gemm_pipeline_transposed_b(
            a_mk,
            b_nk,
            out=out,
        )

    def forward(self, a_mk: torch.Tensor, b_nk: torch.Tensor) -> torch.Tensor:
        return self.build_call(a_mk=a_mk, b_nk=b_nk)()
'''
