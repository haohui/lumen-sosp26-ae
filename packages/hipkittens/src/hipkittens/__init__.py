"""BF16 square-GEMM bindings for the pinned HIPKittens kernels."""

from __future__ import annotations

from typing import Final

import torch

from . import _C

SUPPORTED_SIZES: Final = (1024, 2048, 4096, 8192, 16384)


def gemm(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Compute ``out = a @ b.T`` using the matching HIPKittens GEMM kernel.

    All tensors must be contiguous CUDA BF16 tensors on the same device, with a
    supported square shape. ``out`` is supplied by the caller so the operation
    performs no allocation or layout conversion while being benchmarked.
    """
    size = _validate_inputs(a, b, out)
    stream_ptr = int(torch.cuda.current_stream(device=a.device).cuda_stream)
    getattr(_C, f"gemm_{size}")(a, b, out, stream_ptr)
    return out


def _validate_inputs(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> int:
    tensors = {"a": a, "b": b, "out": out}
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.device.type != "cuda":
            raise ValueError(f"{name} must be a CUDA tensor")
        if tensor.dtype is not torch.bfloat16:
            raise ValueError(f"{name} must have dtype torch.bfloat16")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    if b.device != a.device or out.device != a.device:
        raise ValueError("a, b, and out must be on the same CUDA device")
    if a.ndim != 2 or b.ndim != 2 or out.ndim != 2:
        raise ValueError("a, b, and out must be 2D tensors")

    m, k = a.shape
    n, b_k = b.shape
    if m != n or m != k or b_k != k:
        raise ValueError(
            "HIPKittens only supports square M=N=K GEMM workloads "
            f"(got a={tuple(a.shape)}, b={tuple(b.shape)})"
        )
    if tuple(out.shape) != (m, n):
        raise ValueError(f"out must have shape {(m, n)}, got {tuple(out.shape)}")
    if m not in SUPPORTED_SIZES:
        supported = ", ".join(map(str, SUPPORTED_SIZES))
        raise ValueError(f"unsupported GEMM size {m}; supported sizes: {supported}")
    return m


__all__ = ["SUPPORTED_SIZES", "gemm"]
