from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType


def _load_amdgpu_gemm():
    package_dir = Path(__file__).resolve().parent
    package_name = "_lumen_gemm_backend"
    package = sys.modules.setdefault(package_name, ModuleType(package_name))
    package.__path__ = [str(package_dir)]
    return importlib.import_module(f"{package_name}.amdgpu_gemm")


kernel_function = _load_amdgpu_gemm().gemm_pipeline_transposed_b


__all__ = ["kernel_function"]
