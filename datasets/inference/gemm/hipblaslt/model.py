from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


def _load_extension():
    src = Path(__file__).with_name("hipblaslt_internal_ext.cpp")
    roots = [os.environ.get(k, "") for k in ("ROCM_PATH", "ROCM_HOME", "HIP_PATH")]
    lib_dirs = [Path(root) / "lib" for root in roots if root]
    for env_name in ("LD_LIBRARY_PATH", "LIBRARY_PATH"):
        lib_dirs.extend(Path(p) for p in os.environ.get(env_name, "").split(":") if p)
    lib_dirs = [path for path in dict.fromkeys(lib_dirs) if path.is_dir()]

    ldflags: list[str] = []
    for path in lib_dirs:
        ldflags.extend([f"-L{path}", f"-Wl,-rpath,{path}"])
    ldflags.extend(["-lhipblaslt", "-lhipblas", "-lrocblas", "-lamdhip64"])
    for path in (base / "hipblaslt" / "library" for base in lib_dirs):
        if (path / "TensileLibrary_lazy_gfx942.dat").exists():
            os.environ["HIPBLASLT_TENSILE_LIBPATH"] = str(path)
            break

    os.environ.setdefault("CXX", "hipcc")
    return load_inline(
        name="kb_hipblaslt_internal_ext",
        cpp_sources=src.read_text(encoding="utf-8"),
        functions=["hipblaslt_bf16_mm_out"],
        extra_cflags=["-O3"],
        extra_ldflags=ldflags,
        with_cuda=False,
        verbose=False,
    )


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self._extension = _load_extension()
        self._outputs: dict[tuple[int, int, str, str], torch.Tensor] = {}

    def forward(self, a_mk: torch.Tensor, b_nk: torch.Tensor) -> torch.Tensor:
        if a_mk.dtype is not torch.bfloat16 or b_nk.dtype is not torch.bfloat16:
            raise RuntimeError("hipblaslt backend only supports bf16")
        key = (a_mk.shape[0], b_nk.shape[0], str(a_mk.device), str(a_mk.dtype))
        out = self._outputs.get(key)
        if out is None:
            out = torch.empty(
                (a_mk.shape[0], b_nk.shape[0]),
                dtype=a_mk.dtype,
                device=a_mk.device,
            )
            self._outputs[key] = out
        self._extension.hipblaslt_bf16_mm_out(a_mk, b_nk, out)
        return out
