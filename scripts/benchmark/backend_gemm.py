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


@dataclass(frozen=True)
class BackendSpec:
    directory: str


BACKENDS = {
    name: BackendSpec(directory=name)
    for name in (
        "aiter",
        "cudaforge",
        "hipblaslt",
        "hipkittens",
        "kernelbench",
        "kernelfalcon",
        "ksearch",
        "triton",
    )
}


@dataclass
class SharedInputs:
    a_mk: torch.Tensor
    b_nk: torch.Tensor


def parse_dtype(name: str) -> torch.dtype:
    n = name.strip().lower()
    if n == "bf16":
        return torch.bfloat16
    if n == "fp16":
        return torch.float16
    raise ValueError(f"unsupported dtype: {name}")


def build_shared_inputs(
    *, sizes: list[int], device: torch.device, dtype: torch.dtype, seed: int
) -> dict[int, SharedInputs]:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return {
        s: SharedInputs(
            a_mk=torch.randn((s, s), device=device, dtype=dtype, generator=g),
            b_nk=torch.randn((s, s), device=device, dtype=dtype, generator=g),
        )
        for s in sizes
    }


def run_backend(
    *,
    backend: str,
    gemm_root: Path,
    matrix_sizes: list[int],
    shared: dict[int, SharedInputs],
    device: torch.device,
    dtype: torch.dtype,
    dtype_name: str,
    warmup: int,
    repeat: int,
    graph_iters: int,
) -> None:
    spec = BACKENDS[backend]
    path = gemm_root / spec.directory / "model.py"
    mod = load_module(path)
    fn = build_model_fn(mod, device=device, dtype=dtype)
    for s in matrix_sizes:
        x = shared[s]
        timing = benchmark_with_cudagraph(
            lambda x=x: fn(x.a_mk, x.b_nk),
            warmup=warmup,
            repeat=repeat,
            graph_iters=graph_iters,
        )
        emit_jsonl(
            {
                "domain": "gemm",
                "backend": backend,
                "matrix_size": s,
                "m": s,
                "n": s,
                "k": s,
                "dtype": dtype_name,
                "mean_ms": timing.mean_ms,
            }
        )
