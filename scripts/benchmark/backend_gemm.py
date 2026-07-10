#!/usr/bin/env python3
from __future__ import annotations

import os
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
    *,
    sizes: list[int],
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> dict[int, SharedInputs]:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    out: dict[int, SharedInputs] = {}
    for s in sizes:
        b_nk = torch.randn((s, s), device=device, dtype=dtype, generator=g)
        out[s] = SharedInputs(
            a_mk=torch.randn((s, s), device=device, dtype=dtype, generator=g),
            b_nk=b_nk,
        )
    return out


def load_hipblaslt_internal_module(gemm_root: Path):
    from torch.utils.cpp_extension import load_inline

    src = gemm_root / "06_hipblaslt" / "hipblaslt_internal_ext.cpp"
    if not src.exists():
        raise FileNotFoundError(f"missing source: {src}")

    roots = [os.environ.get(k, "") for k in ("ROCM_PATH", "ROCM_HOME", "HIP_PATH")]
    lib_dirs = [Path(r) / "lib" for r in roots if r]
    for env_name in ("LD_LIBRARY_PATH", "LIBRARY_PATH"):
        lib_dirs.extend(Path(p) for p in os.environ.get(env_name, "").split(":") if p)
    lib_dirs = [d for d in dict.fromkeys(lib_dirs) if d.exists() and d.is_dir()]

    ldflags: list[str] = []
    for d in lib_dirs:
        ldflags.extend([f"-L{d}", f"-Wl,-rpath,{d}"])
    ldflags.extend(["-lhipblaslt", "-lhipblas", "-lrocblas", "-lamdhip64"])

    for d in [base / "hipblaslt" / "library" for base in lib_dirs]:
        if (d / "TensileLibrary_lazy_gfx942.dat").exists():
            os.environ["HIPBLASLT_TENSILE_LIBPATH"] = str(d)
            break

    os.environ.setdefault("CXX", "hipcc")
    cpp_src = src.read_text(encoding="utf-8")
    return load_inline(
        name="kb_hipblaslt_internal_ext",
        cpp_sources=cpp_src,
        functions=["hipblaslt_bf16_mm_out"],
        extra_cflags=["-O3"],
        extra_ldflags=ldflags,
        with_cuda=False,
        verbose=False,
    )


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
    matrix_size: int,
    dtype_name: str,
    mean_ms: float,
) -> None:
    emit_jsonl(
        {
            "domain": "gemm",
            "backend": backend,
            "matrix_size": matrix_size,
            "m": matrix_size,
            "n": matrix_size,
            "k": matrix_size,
            "dtype": dtype_name,
            "mean_ms": mean_ms,
        }
    )


def _run_python_backend(
    *,
    backend: str,
    path: Path,
    matrix_sizes: list[int],
    shared: dict[int, SharedInputs],
    device: torch.device,
    dtype: torch.dtype,
    dtype_name: str,
    warmup: int,
    repeat: int,
    graph_iters: int,
) -> None:
    mod = load_module(path)
    fn = build_model_fn(mod, device=device, dtype=dtype)
    for s in matrix_sizes:
        x = shared[s]
        timing = _time_call(
            lambda x=x: fn(x.a_mk, x.b_nk),
            warmup=warmup,
            repeat=repeat,
            graph_iters=graph_iters,
        )
        _emit_record(
            backend=backend,
            matrix_size=s,
            dtype_name=dtype_name,
            mean_ms=timing.mean_ms,
        )


def run_aiter(
    *,
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
    _run_python_backend(
        backend="aiter",
        path=gemm_root / "05_aiter" / "best_kernel.py",
        matrix_sizes=matrix_sizes,
        shared=shared,
        device=device,
        dtype=dtype,
        dtype_name=dtype_name,
        warmup=warmup,
        repeat=repeat,
        graph_iters=graph_iters,
    )


def run_hipblaslt(
    *,
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
    if dtype is not torch.bfloat16:
        raise RuntimeError("hipblaslt backend only supports bf16")
    hipblaslt_mod = load_hipblaslt_internal_module(gemm_root)
    for s in matrix_sizes:
        x = shared[s]
        out = torch.empty((s, s), dtype=torch.bfloat16, device=device)
        timing = _time_call(
            lambda x=x, out=out: hipblaslt_mod.hipblaslt_bf16_mm_out(
                x.a_mk,
                x.b_nk,
                out,
            ),
            warmup=warmup,
            repeat=repeat,
            graph_iters=graph_iters,
        )
        _emit_record(
            backend="hipblaslt",
            matrix_size=s,
            dtype_name=dtype_name,
            mean_ms=timing.mean_ms,
        )


def run_hipkittens(
    *,
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
    if dtype is not torch.bfloat16:
        raise RuntimeError("hipkittens backend only supports bf16")
    try:
        import hipkittens
    except ImportError as exc:
        raise RuntimeError(
            "hipkittens is not installed; set HIPKITTENS_ROOT and install with "
            "'uv pip install -e ./packages/hipkittens'"
        ) from exc

    for s in matrix_sizes:
        x = shared[s]
        out = torch.empty((s, s), dtype=dtype, device=device)
        timing = _time_call(
            lambda x=x, out=out: hipkittens.gemm(x.a_mk, x.b_nk, out),
            warmup=warmup,
            repeat=repeat,
            graph_iters=graph_iters,
        )
        _emit_record(
            backend="hipkittens",
            matrix_size=s,
            dtype_name=dtype_name,
            mean_ms=timing.mean_ms,
        )


def run_triton(
    *,
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
    _run_python_backend(
        backend="triton",
        path=gemm_root / "07_triton" / "best_kernel.py",
        matrix_sizes=matrix_sizes,
        shared=shared,
        device=device,
        dtype=dtype,
        dtype_name=dtype_name,
        warmup=warmup,
        repeat=repeat,
        graph_iters=graph_iters,
    )


BACKEND_MAP = {
    "aiter": run_aiter,
    "hipblaslt": run_hipblaslt,
    "hipkittens": run_hipkittens,
    "triton": run_triton,
}

BACKENDS = BACKEND_MAP
