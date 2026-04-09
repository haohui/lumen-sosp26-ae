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
    Expects layout [B, H, S, D] and returns output tensor [B, H, S, D].
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("kernel_function expects 4D tensors: [B,H,S,D]")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError("batch size mismatch between q/k/v")
    if q.shape[2] != k.shape[2] or q.shape[2] != v.shape[2]:
        raise ValueError("sequence length mismatch between q/k/v")
    if q.shape[3] != k.shape[3] or q.shape[3] != v.shape[3]:
        raise ValueError("head dim mismatch between q/k/v")

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    metadata = _SRC.MetaData(sm_scale=q.shape[-1] ** -0.5)
    metadata.max_seqlens_q = int(q.shape[2])
    metadata.max_seqlens_k = int(k.shape[2])
    metadata.layout = "bhsd"
    # run_all attention path is causal benchmark.
    metadata.need_causal()

    o = torch.empty_like(q)
    out, _, _ = _SRC.attention(q, k, v, o, metadata)
    return out

