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
        "hipblaslt",
        "hipkittens",
        "kernelbench",
        "kernelfalcon",
        "ksearch",
        "lumen",
        "triton",
    )
}

@dataclass
class SharedInputs:
    a_mk: torch.Tensor
    b_nk: torch.Tensor


CORRECTNESS_TOLERANCES = {
    "bf16": {"rtol": 1e-1, "atol": 1e-1},
    "fp16": {"rtol": 1e-2, "atol": 1e-2},
}


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
    check_correctness: bool = False,
) -> None:
    spec = BACKENDS[backend]
    path = gemm_root / spec.directory / "model.py"
    mod = load_module(path)
    model = build_model_instance(mod, device=device, dtype=dtype)
    for s in matrix_sizes:
        x = shared[s]
        if check_correctness:
            _check_correctness(
                model=model,
                inputs=x,
                matrix_size=s,
                dtype_name=dtype_name,
            )
        if hasattr(model, "build_call"):
            call = model.build_call(a_mk=x.a_mk, b_nk=x.b_nk)
        else:
            call = lambda x=x, model=model: model(x.a_mk, x.b_nk)
        timing = benchmark_with_cudagraph(
            call,
            warmup=warmup,
            repeat=repeat,
            graph_iters=graph_iters,
        )
        tflops = 2 * s**3 / (timing.mean_ms / 1000) / 1e12
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
                **({"correctness": True} if check_correctness else {}),
                "tflops": tflops,
            }
        )
    exit_after_success_if_requested(mod)


def _check_correctness(
    *,
    model,
    inputs: SharedInputs,
    matrix_size: int,
    dtype_name: str,
) -> None:
    with torch.inference_mode():
        actual = model(inputs.a_mk, inputs.b_nk)
        expected = torch.matmul(inputs.a_mk, inputs.b_nk.transpose(0, 1))

    if not isinstance(actual, torch.Tensor):
        raise AssertionError(
            f"GEMM correctness failed for size {matrix_size}: "
            f"backend returned {type(actual).__name__}, expected torch.Tensor"
        )
    expected_shape = (matrix_size, matrix_size)
    if tuple(actual.shape) != expected_shape:
        raise AssertionError(
            f"GEMM correctness failed for size {matrix_size}: "
            f"output shape {tuple(actual.shape)}, expected {expected_shape}"
        )
    if actual.dtype != expected.dtype:
        raise AssertionError(
            f"GEMM correctness failed for size {matrix_size}: "
            f"output dtype {actual.dtype}, expected {expected.dtype}"
        )

    tolerances = CORRECTNESS_TOLERANCES[dtype_name]
    try:
        torch.testing.assert_close(actual, expected, **tolerances)
    except AssertionError as exc:
        raise AssertionError(
            f"GEMM correctness failed for size {matrix_size} "
            f"(rtol={tolerances['rtol']}, atol={tolerances['atol']}):\n{exc}"
        ) from exc
