#!/usr/bin/env python3
from __future__ import annotations

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a_mk: torch.Tensor, b_nk: torch.Tensor) -> torch.Tensor:
        if a_mk.ndim != 2 or b_nk.ndim != 2:
            raise ValueError("AITER GEMM expects 2D tensors")
        if a_mk.shape[1] != b_nk.shape[1]:
            raise ValueError(
                "AITER GEMM expects A:[M,K], B:[N,K] for A@B^T, got "
                f"A={a_mk.shape}, B={b_nk.shape}"
            )

        import aiter

        out = torch.empty(
            (a_mk.shape[0], b_nk.shape[0]),
            device=a_mk.device,
            dtype=torch.float32,
        )

        aiter.gemm_a16w16_asm(a_mk, b_nk, out)
        return out
