#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn


DIRECT_EXIT_AFTER_SUCCESS = True
_THIS_DIR = Path(__file__).resolve().parent
_ATTENTION_MODULE = "attn_06_inst_scheduling.py"


def _load_kernel_module():
    path = _THIS_DIR / _ATTENTION_MODULE
    module_name = f"lumen_attn_{path.stem}_{abs(hash(str(path.resolve()))):x}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import LUMEN attention module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self._mod = None
        self._out_cache = {}

    def _kernel_module(self):
        if self._mod is None:
            self._mod = _load_kernel_module()
        return self._mod

    def build_call(
        self,
        *,
        q_bshd: torch.Tensor,
        k_bshd: torch.Tensor,
        v_bshd: torch.Tensor,
    ):
        if q_bshd.dtype != torch.bfloat16:
            raise ValueError("LUMEN attention benchmark currently supports only bf16")
        if q_bshd.ndim != 4 or k_bshd.ndim != 4 or v_bshd.ndim != 4:
            raise ValueError("LUMEN attention expects q/k/v in [B,S,H,D] layout")

        batch_size, seq_len, num_q_heads, head_dim = q_bshd.shape
        if k_bshd.shape[:2] != (batch_size, seq_len):
            raise ValueError("LUMEN attention expects q/k to share batch and seq")
        if v_bshd.shape != k_bshd.shape:
            raise ValueError("LUMEN attention expects k/v to have identical shape")
        if k_bshd.shape[3] != head_dim:
            raise ValueError("LUMEN attention expects q/k/v to share head_dim")

        q = q_bshd.reshape(batch_size * seq_len, num_q_heads, head_dim).contiguous()
        k = k_bshd.reshape(
            batch_size * seq_len,
            int(k_bshd.shape[2]),
            head_dim,
        ).contiguous()
        v = v_bshd.reshape(
            batch_size * seq_len,
            int(v_bshd.shape[2]),
            head_dim,
        ).contiguous()
        seq_ptr = torch.arange(
            batch_size + 1,
            device=q.device,
            dtype=torch.int32,
        ).mul_(seq_len)

        key = (q.shape, q.device, q.dtype)
        out = self._out_cache.get(key)
        if out is None:
            out = torch.empty_like(q)
            self._out_cache[key] = out

        mod = self._kernel_module()
        return lambda: mod.flash_attn(
            q,
            k,
            v,
            seq_ptr,
            seq_len,
            out=out,
        )

    def forward(
        self,
        q_bshd: torch.Tensor,
        k_bshd: torch.Tensor,
        v_bshd: torch.Tensor,
    ) -> torch.Tensor:
        call = self.build_call(q_bshd=q_bshd, k_bshd=k_bshd, v_bshd=v_bshd)
        call()
        batch_size, seq_len, num_q_heads, head_dim = q_bshd.shape
        flat_shape = (batch_size * seq_len, num_q_heads, head_dim)
        out = self._out_cache[(flat_shape, q_bshd.device, q_bshd.dtype)]
        return out.reshape(batch_size, seq_len, num_q_heads, head_dim)
