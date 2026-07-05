#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


def _load_source_module():
    src = (
        Path(__file__).resolve().parent
        / "kernels"
        / "hipkittens_triton_baseline_v02.py"
    )
    spec = importlib.util.spec_from_file_location("attention_hipkittens_src", src)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import source module: {src}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_SRC = _load_source_module()


def kernel_function(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Unified attention entry for benchmark_attention_unified_graph.
    Expects layout [B, S, H, D] and returns output tensor [B, S, H, D].
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("kernel_function expects 4D tensors: [B,S,H,D]")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError("batch size mismatch between q/k/v")
    if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
        raise ValueError("sequence length mismatch between q/k/v")
    if q.shape[3] != k.shape[3] or q.shape[3] != v.shape[3]:
        raise ValueError("head dim mismatch between q/k/v")

    # Source kernel supports [B,S,H,D] directly.
    q_bshd = q.contiguous()
    k_bshd = k.contiguous()
    v_bshd = v.contiguous()

    metadata = _SRC.MetaData(sm_scale=q_bshd.shape[-1] ** -0.5)
    metadata.max_seqlens_q = int(q_bshd.shape[1])
    metadata.max_seqlens_k = int(k_bshd.shape[1])
    metadata.layout = "bshd"
    # run_all attention path is causal benchmark.
    metadata.need_causal()

    o = torch.empty_like(q_bshd)
    out_bshd, _, _ = _SRC.attention(q_bshd, k_bshd, v_bshd, o, metadata)
    return out_bshd
