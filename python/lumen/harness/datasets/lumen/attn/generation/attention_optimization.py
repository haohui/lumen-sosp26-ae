"""Attention-specific configuration for the shared Lumen optimization runtime."""

from __future__ import annotations

from pathlib import Path

from lumen.harness.datasets.lumen.optimization_runtime import OptimizationSpec


ATTENTION_WORKLOADS = (1024, 2048, 4096, 8192, 16384)


def attention_optimization_spec(repo_root: Path) -> OptimizationSpec:
    prompt_root = (
        repo_root
        / "python"
        / "lumen"
        / "harness"
        / "datasets"
        / "lumen"
        / "attn"
        / "prompts"
    )
    return OptimizationSpec(
        domain="attention",
        run_slug="attn",
        default_kernel=(
            repo_root
            / "datasets"
            / "inference"
            / "attention"
            / "lumen"
            / "attn_01_naive.py"
        ),
        default_prompts=tuple(
            prompt_root / f"optimization-{index:02d}.md" for index in range(2, 7)
        ),
        workloads=ATTENTION_WORKLOADS,
        workload_key="seq_len",
        workload_flag="--seq-lens",
        benchmark_script="bench_attn.py",
        adapter_source=_benchmark_adapter_source(),
    )


def _benchmark_adapter_source() -> str:
    return '''#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn


_KERNEL_PATH = Path(__file__).resolve().parents[4] / "output_model_new.py"


def _load_kernel():
    spec = importlib.util.spec_from_file_location(
        "lumen_attention_candidate",
        _KERNEL_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import candidate: {_KERNEL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self._kernel = _load_kernel()
        self._out_cache = {}

    def build_call(
        self,
        *,
        q_bshd: torch.Tensor,
        k_bshd: torch.Tensor,
        v_bshd: torch.Tensor,
    ):
        batch_size, seq_len, num_q_heads, head_dim = q_bshd.shape
        q = q_bshd.reshape(batch_size * seq_len, num_q_heads, head_dim).contiguous()
        k = k_bshd.reshape(batch_size * seq_len, k_bshd.shape[2], head_dim).contiguous()
        v = v_bshd.reshape(batch_size * seq_len, v_bshd.shape[2], head_dim).contiguous()
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
        return lambda: self._kernel.flash_attn(
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
'''
