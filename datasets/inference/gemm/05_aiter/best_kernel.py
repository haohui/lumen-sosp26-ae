#!/usr/bin/env python3
from __future__ import annotations

import torch
import torch.nn as nn


_MODEL_CACHE: dict[str, "Model"] = {}


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # Cache output buffers per input signature to reduce allocator noise.
        self._cache: dict[tuple[int, int, int, int, str, str], torch.Tensor] = {}

    def forward(self, a_mk: torch.Tensor, b_nk: torch.Tensor) -> torch.Tensor:
        if a_mk.ndim != 2 or b_nk.ndim != 2:
            raise ValueError("AITER GEMM expects 2D tensors")
        if a_mk.shape[1] != b_nk.shape[1]:
            raise ValueError(
                f"AITER GEMM expects A:[M,K], B:[N,K] for A@B^T, got A={a_mk.shape}, B={b_nk.shape}"
            )

        try:
            import aiter
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"failed to import aiter: {type(e).__name__}: {e}") from e

        m, k = a_mk.shape
        n = b_nk.shape[0]
        key = (a_mk.data_ptr(), b_nk.data_ptr(), m, n, str(a_mk.dtype), str(a_mk.device))
        out = self._cache.get(key)
        if out is None:
            # gemm_a16w16_asm directly consumes B as [N, K] and computes A @ B^T.
            out = torch.empty((m, n), device=a_mk.device, dtype=torch.float32)
            self._cache[key] = out

        aiter.gemm_a16w16_asm(a_mk, b_nk, out)
        return out


def kernel_function(a_mk: torch.Tensor, b_nk: torch.Tensor) -> torch.Tensor:
    model = _MODEL_CACHE.get("default")
    if model is None:
        model = Model()
        _MODEL_CACHE["default"] = model
    return model.forward(a_mk, b_nk)
