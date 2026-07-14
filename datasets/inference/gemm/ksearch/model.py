import hashlib
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load


_THIS_DIR = Path(__file__).resolve().parent
_BINDING = _THIS_DIR / "best_binding.cpp"
_KERNEL = _THIS_DIR / "best_kernel.cu"


def _build_extension():
    os.environ.setdefault("CXX", "hipcc")
    # Pin architecture for deterministic ROCm extension builds in this artifact.
    os.environ["PYTORCH_ROCM_ARCH"] = "gfx942"
    cache_key = hashlib.sha1(
        (
            str(_BINDING)
            + str(_KERNEL)
            + str(_BINDING.stat().st_mtime_ns)
            + str(_KERNEL.stat().st_mtime_ns)
        ).encode("utf-8")
    ).hexdigest()[:12]
    module_name = f"ksearch_gemm_bf16_var_mnk_ext_{cache_key}"
    return load(
        name=module_name,
        sources=[str(_BINDING), str(_KERNEL)],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-std=c++17", "--offload-arch=gfx942"],
        extra_include_paths=[str(_THIS_DIR)],
        with_cuda=True,
        verbose=False,
    )


_EXT = _build_extension() if torch.cuda.is_available() else None


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._ext = _EXT

    def forward(self, a_mk: torch.Tensor, b_nk: torch.Tensor) -> torch.Tensor:
        if self._ext is None:
            raise RuntimeError("KSearch extension is not available on this runtime")

        if a_mk.dim() != 2 or b_nk.dim() != 2:
            raise ValueError(f"Expected 2D tensors, got a_mk.dim={a_mk.dim()}, b_nk.dim={b_nk.dim()}")
        if a_mk.shape[1] != b_nk.shape[1]:
            raise ValueError(f"Incompatible shapes for A@B^T: a_mk={tuple(a_mk.shape)}, b_nk={tuple(b_nk.shape)}")

        if a_mk.dtype != torch.bfloat16:
            a_mk = a_mk.to(torch.bfloat16)
        if b_nk.dtype != torch.bfloat16:
            b_nk = b_nk.to(torch.bfloat16)

        a_mk = a_mk.contiguous()
        b_nk = b_nk.contiguous()
        return self._ext.run(a_mk, b_nk)
