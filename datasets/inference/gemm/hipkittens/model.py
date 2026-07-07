from __future__ import annotations

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self._outputs: dict[tuple[int, int, str, str], torch.Tensor] = {}

    def forward(self, a_mk: torch.Tensor, b_nk: torch.Tensor) -> torch.Tensor:
        if a_mk.dtype is not torch.bfloat16 or b_nk.dtype is not torch.bfloat16:
            raise RuntimeError("hipkittens backend only supports bf16")
        try:
            import hipkittens
        except ImportError as exc:
            raise RuntimeError(
                "hipkittens is not installed; set HIPKITTENS_ROOT and install with "
                "'uv pip install -e ./packages/hipkittens'"
            ) from exc
        key = (a_mk.shape[0], b_nk.shape[0], str(a_mk.device), str(a_mk.dtype))
        out = self._outputs.get(key)
        if out is None:
            out = torch.empty(
                (a_mk.shape[0], b_nk.shape[0]),
                dtype=a_mk.dtype,
                device=a_mk.device,
            )
            self._outputs[key] = out
        hipkittens.gemm(a_mk, b_nk, out)
        return out
