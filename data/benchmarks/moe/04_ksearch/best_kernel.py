#!/usr/bin/env python3
import hashlib
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load


_THIS_DIR = Path(__file__).resolve().parent
_SRC_MAIN = _THIS_DIR / "main.cpp"
_SRC_CU = _THIS_DIR / "kernel.cu"
_SRC_H = _THIS_DIR / "kernel.h"

if not (_SRC_MAIN.exists() and _SRC_CU.exists() and _SRC_H.exists()):
    missing = [str(p) for p in (_SRC_MAIN, _SRC_CU, _SRC_H) if not p.exists()]
    raise FileNotFoundError(f"ksearch fused sources missing: {missing}")

_sig = hashlib.sha1()
for _p in (_SRC_MAIN, _SRC_CU, _SRC_H):
    _sig.update(_p.read_bytes())
_EXT_NAME = f"ksearch_moe_fused_{_sig.hexdigest()[:12]}"

_ksearch_ext = load(
    name=_EXT_NAME,
    sources=[str(_SRC_MAIN), str(_SRC_CU)],
    extra_include_paths=[str(_THIS_DIR)],
    extra_cflags=["-O3", "-std=c++17"],
    extra_cuda_cflags=["-O3", "-std=c++17", "--offload-arch=gfx942"],
    with_cuda=True,
    verbose=False,
)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        input_q: torch.Tensor,
        w1_q: torch.Tensor,
        w2_q: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        input_scale: torch.Tensor,
        fc1_scale: torch.Tensor,
        fc2_scale: torch.Tensor,
    ) -> torch.Tensor:
        return _ksearch_ext.run(
            input_q, w1_q, w2_q, topk_weights, topk_ids, input_scale, fc1_scale, fc2_scale
        )

