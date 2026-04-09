import hashlib
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    module_name = f"ksearch_attn_dense_qkv_prefill_r8_nohack_ext_{cache_key}"
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
        if (
            self._ext is None
            or (not q.is_cuda)
            or (not k.is_cuda)
            or (not v.is_cuda)
            or q.dtype != torch.bfloat16
            or k.dtype != torch.bfloat16
            or v.dtype != torch.bfloat16
            or q.dim() != 4
            or k.dim() != 4
            or v.dim() != 4
            or q.shape[0] != k.shape[0]
            or q.shape[0] != v.shape[0]
            or q.shape[2] != k.shape[2]
            or q.shape[2] != v.shape[2]
            or q.shape[3] != 128
            or k.shape[3] != 128
            or v.shape[3] != 128
            or q.shape[1] != 8
            or k.shape[1] not in (1, 8)
            or v.shape[1] != k.shape[1]
        ):
            if k.shape[1] == 1:
                k = k.expand(-1, 8, -1, -1).contiguous()
                v = v.expand(-1, 8, -1, -1).contiguous()
            return F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)

        scale = 1.0 / math.sqrt(float(q.shape[-1]))
        return self._ext.run(q, k, v, scale)
