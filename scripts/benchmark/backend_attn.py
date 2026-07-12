#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from backends import build_model_instance, exit_after_success_if_requested, load_module
from cli_utils import emit_jsonl
from cudagraph_timer import benchmark_with_cudagraph

try:
    import torch
except Exception:
    torch = None


@dataclass(frozen=True)
class BackendSpec:
    directory: str


BACKENDS = {
    name: BackendSpec(directory=name)
    for name in (
        "aiter",
        "cudaforge",
        "kernelbench",
        "kernelfalcon",
        "ksearch",
        "lumen",
        "triton",
    )
}

@dataclass
class SharedInputs:
    q_bshd: torch.Tensor
    k_bshd: torch.Tensor
    v_bshd: torch.Tensor


def parse_dtype(name: str) -> torch.dtype:
    n = name.strip().lower()
    if n == "bf16":
        return torch.bfloat16
    if n == "fp16":
        return torch.float16
    raise ValueError(f"unsupported dtype: {name}")


def build_shared_inputs(
    *,
    seq_lens: list[int],
    batch_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> dict[int, SharedInputs]:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    out: dict[int, SharedInputs] = {}
    for s in seq_lens:
        out[s] = SharedInputs(
            q_bshd=torch.randn(
                (batch_size, s, num_q_heads, head_dim),
                device=device,
                dtype=dtype,
                generator=g,
            ),
            k_bshd=torch.randn(
                (batch_size, s, num_kv_heads, head_dim),
                device=device,
                dtype=dtype,
                generator=g,
            ),
            v_bshd=torch.randn(
                (batch_size, s, num_kv_heads, head_dim),
                device=device,
                dtype=dtype,
                generator=g,
            ),
        )
    return out


def run_backend(
    *,
    backend: str,
    attn_root: Path,
    seq_lens: list[int],
    shared: dict[int, SharedInputs],
    device: torch.device,
    dtype: torch.dtype,
    dtype_name: str,
    batch_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    causal: bool,
    warmup: int,
    repeat: int,
    graph_iters: int,
) -> None:
    path = attn_root / BACKENDS[backend].directory / "model.py"
    mod = load_module(path)
    model = build_model_instance(mod, device=device, dtype=dtype)
    for s in seq_lens:
        x = shared[s]
        if hasattr(model, "build_call"):
            call = model.build_call(q_bshd=x.q_bshd, k_bshd=x.k_bshd, v_bshd=x.v_bshd)
        else:
            call = lambda x=x, model=model: model(x.q_bshd, x.k_bshd, x.v_bshd)
        timing = benchmark_with_cudagraph(
            call,
            warmup=warmup,
            repeat=repeat,
            graph_iters=graph_iters,
        )
        emit_jsonl(
            {
                "domain": "attention",
                "backend": backend,
                "seq_len": s,
                "batch_size": batch_size,
                "num_q_heads": num_q_heads,
                "num_kv_heads": num_kv_heads,
                "head_dim": head_dim,
                "causal": causal,
                "dtype": dtype_name,
                "mean_ms": timing.mean_ms,
            }
        )
    exit_after_success_if_requested(mod)
