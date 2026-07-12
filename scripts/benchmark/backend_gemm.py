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
        "lumen",
        "triton",
    )
}

LUMEN_GEMM_MODULES = {
    1024: "amdgpu_gemm_1024.py",
    2048: "amdgpu_gemm_2048.py",
    4096: "amdgpu_gemm_4096.py",
    8192: "amdgpu_gemm_8192.py",
    16384: "amdgpu_gemm_16384.py",
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
    if backend == "lumen":
        _run_lumen_backend(
            backend=backend,
            lumen_root=gemm_root / spec.directory,
            matrix_sizes=matrix_sizes,
            shared=shared,
            dtype=dtype,
            dtype_name=dtype_name,
            warmup=warmup,
            repeat=repeat,
            graph_iters=graph_iters,
        )
        return

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


def _run_lumen_backend(
    *,
    backend: str,
    lumen_root: Path,
    matrix_sizes: list[int],
    shared: dict[int, SharedInputs],
    dtype: torch.dtype,
    dtype_name: str,
    warmup: int,
    repeat: int,
    graph_iters: int,
) -> None:
    if dtype is not torch.bfloat16:
        raise ValueError("LUMEN GEMM benchmark currently supports only bf16")

    module_cache = {}
    for s in matrix_sizes:
        module_file = LUMEN_GEMM_MODULES.get(s)
        if module_file is None:
            supported = ", ".join(str(x) for x in sorted(LUMEN_GEMM_MODULES))
            raise ValueError(
                f"LUMEN GEMM does not provide size {s}; supported: {supported}"
            )
        if module_file not in module_cache:
            module_cache[module_file] = load_module(lumen_root / module_file)
        mod = module_cache[module_file]
        fn_name = f"gemm_{s}_transposed_b"
        fn = getattr(mod, fn_name, None) or getattr(
            mod, "gemm_pipeline_transposed_b", None
        )
        if fn is None:
            raise AttributeError(
                f"LUMEN GEMM module {module_file} does not contain "
                f"{fn_name} or gemm_pipeline_transposed_b"
            )
        x = shared[s]
        out = torch.empty((s, s), device=x.a_mk.device, dtype=dtype)
        timing = benchmark_with_cudagraph(
            lambda x=x, out=out, fn=fn: fn(x.a_mk, x.b_nk, out),
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
