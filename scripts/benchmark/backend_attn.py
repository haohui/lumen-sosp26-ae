#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from backends import build_model_fn, load_module
from cli_utils import emit_jsonl
from cudagraph_timer import benchmark_with_cudagraph

try:
    import torch
except Exception:
    torch = None


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


def _time_call(
    call,
    *,
    warmup: int,
    repeat: int,
    graph_iters: int,
):
    return benchmark_with_cudagraph(
        call,
        warmup=warmup,
        repeat=repeat,
        graph_iters=graph_iters,
    )


def _emit_record(
    *,
    backend: str,
    seq_len: int,
    batch_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    causal: bool,
    dtype_name: str,
    mean_ms: float,
) -> None:
    emit_jsonl(
        {
            "domain": "attention",
            "backend": backend,
            "seq_len": seq_len,
            "batch_size": batch_size,
            "num_q_heads": num_q_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "causal": causal,
            "dtype": dtype_name,
            "mean_ms": mean_ms,
        }
    )


def _run_python_backend(
    *,
    backend: str,
    path: Path,
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
    mod = load_module(path)
    fn = build_model_fn(mod, device=device, dtype=dtype)
    for s in seq_lens:
        x = shared[s]
        timing = _time_call(
            lambda x=x: fn(x.q_bshd, x.k_bshd, x.v_bshd),
            warmup=warmup,
            repeat=repeat,
            graph_iters=graph_iters,
        )
        _emit_record(
            backend=backend,
            seq_len=s,
            batch_size=batch_size,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            causal=causal,
            dtype_name=dtype_name,
            mean_ms=timing.mean_ms,
        )


def run_aiter(
    *,
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
    _run_python_backend(
        backend="aiter",
        path=attn_root / "06_aiter" / "best_kernel.py",
        seq_lens=seq_lens,
        shared=shared,
        device=device,
        dtype=dtype,
        dtype_name=dtype_name,
        batch_size=batch_size,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        causal=causal,
        warmup=warmup,
        repeat=repeat,
        graph_iters=graph_iters,
    )


def run_triton(
    *,
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
    _run_python_backend(
        backend="triton",
        path=attn_root / "05_triton" / "best_kernel.py",
        seq_lens=seq_lens,
        shared=shared,
        device=device,
        dtype=dtype,
        dtype_name=dtype_name,
        batch_size=batch_size,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        causal=causal,
        warmup=warmup,
        repeat=repeat,
        graph_iters=graph_iters,
    )


BACKEND_MAP = {
    "aiter": run_aiter,
    "triton": run_triton,
}

BACKENDS = BACKEND_MAP
