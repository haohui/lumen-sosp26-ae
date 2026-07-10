import hashlib
import math
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
    os.environ["PYTORCH_ROCM_ARCH"] = "gfx942"

    cache_key = hashlib.sha1(
        (
            str(_BINDING)
            + str(_KERNEL)
            + str(_BINDING.stat().st_mtime_ns)
            + str(_KERNEL.stat().st_mtime_ns)
        ).encode("utf-8")
    ).hexdigest()[:12]

    module_name = f"ksearch_attn_dense_qkv_prefill_r6_nohack_ext_{cache_key}"
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

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if self._ext is None:
            raise RuntimeError("KSearch extension is not available on this runtime")

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        if q.dtype != torch.bfloat16:
            q = q.to(torch.bfloat16)
        if k.dtype != torch.bfloat16:
            k = k.to(torch.bfloat16)
        if v.dtype != torch.bfloat16:
            v = v.to(torch.bfloat16)

        sm_scale = 1.0 / math.sqrt(float(q.shape[-1]))
        return self._ext.run(q, k, v, sm_scale)
