#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from backends import ModelBackend, load_module
from cli_utils import emit_jsonl
from cudagraph_timer import benchmark_with_cudagraph

try:
    import torch
except Exception:
    torch = None


BLOCK_N = 128
BLOCK_K = 128


@dataclass
class SharedInputs:
    input_q: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    input_scale: torch.Tensor


def parse_dtype(name: str) -> torch.dtype:
    n = name.strip().lower()
    if n == "bf16":
        return torch.bfloat16
    if n == "fp8":
        dt = getattr(torch, "float8_e4m3fnuz", None) or getattr(
            torch,
            "float8_e4m3fn",
            None,
        )
        if dt is None:
            raise RuntimeError("torch float8 dtype is unavailable")
        return dt
    raise ValueError(f"unsupported dtype: {name}")


def validate_config(*, dim: int, inter_dim: int, experts: int, topk: int) -> None:
    if topk > experts:
        raise ValueError(f"topk ({topk}) must be <= experts ({experts})")
    if dim % BLOCK_K != 0 or dim % BLOCK_N != 0:
        raise ValueError(f"dim must be divisible by {BLOCK_N}/{BLOCK_K}, got {dim}")
    if inter_dim % BLOCK_K != 0:
        raise ValueError(f"inter_dim must be divisible by {BLOCK_K}, got {inter_dim}")


def build_shared_inputs(
    *,
    seq_lens: list[int],
    dim: int,
    experts: int,
    topk: int,
    input_dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> dict[int, SharedInputs]:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    out: dict[int, SharedInputs] = {}
    hidden_blocks = dim // BLOCK_K

    for s in seq_lens:
        input_q = torch.randn(
            (s, dim),
            dtype=torch.float32,
            device=device,
            generator=g,
        ).add_(0.1)
        input_q = input_q.to(input_dtype).contiguous()

        input_scale = (
            (
                torch.randn(
                    (s, hidden_blocks),
                    dtype=torch.float32,
                    device=device,
                    generator=g,
                )
                .mul_(2e-2)
                .add_(1e-1)
            )
            .clamp_min_(1e-8)
            .contiguous()
        )

        scores = torch.randn(
            (s, experts),
            dtype=torch.float32,
            device=device,
            generator=g,
        )
        topk_val, topk_idx = torch.topk(
            scores,
            k=topk,
            dim=-1,
            largest=True,
            sorted=True,
        )
        topk_ids = topk_idx.to(torch.int32).contiguous()
        topk_weights = torch.softmax(topk_val, dim=-1).to(torch.float32).contiguous()

        out[s] = SharedInputs(
            input_q=input_q,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            input_scale=input_scale,
        )
    return out


def build_shared_weights(
    *,
    dim: int,
    inter_dim: int,
    experts: int,
    input_dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> dict[str, torch.Tensor]:
    g = torch.Generator(device=device)
    g.manual_seed(seed + 17)

    w1_q = (
        torch.randn(
            (experts, inter_dim * 2, dim),
            dtype=torch.float32,
            device=device,
            generator=g,
        )
        .mul_(8.0)
        .to(input_dtype)
    )
    w2_q = (
        torch.randn(
            (experts, dim, inter_dim),
            dtype=torch.float32,
            device=device,
            generator=g,
        )
        .mul_(8.0)
        .to(input_dtype)
    )

    fc1_scale = (
        torch.randn(
            (experts, ((inter_dim * 2) // BLOCK_N) * (dim // BLOCK_K)),
            dtype=torch.float32,
            device=device,
            generator=g,
        )
        .mul_(2e-3)
        .add_(1e-2)
    ).clamp_min_(1e-8)

    fc2_scale = (
        torch.randn(
            (experts, (dim // BLOCK_N) * (inter_dim // BLOCK_K)),
            dtype=torch.float32,
            device=device,
            generator=g,
        )
        .mul_(2e-3)
        .add_(1e-2)
    ).clamp_min_(1e-8)

    return {
        "w1_q": w1_q.contiguous(),
        "w2_q": w2_q.contiguous(),
        "fc1_scale": fc1_scale.contiguous(),
        "fc2_scale": fc2_scale.contiguous(),
    }


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
    tokens: int,
    dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    input_dtype_name: str,
    mean_ms: float,
) -> None:
    emit_jsonl(
        {
            "domain": "moe",
            "backend": backend,
            "tokens": tokens,
            "dim": dim,
            "inter_dim": inter_dim,
            "experts": experts,
            "topk": topk,
            "input_dtype": input_dtype_name,
            "mean_ms": mean_ms,
        }
    )


def _run_aiter_variant(
    *,
    backend: str,
    moe_root: Path,
    token_counts: list[int],
    shared_inputs: dict[int, SharedInputs],
    shared_weights: dict[str, torch.Tensor],
    dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    input_dtype_name: str,
    warmup: int,
    repeat: int,
    graph_iters: int,
) -> None:
    aiter_entry = moe_root / "05_aiter" / "run_aiter.py"
    aiter_mod = load_module(aiter_entry)
    build_aiter_cases = getattr(aiter_mod, "build_cases_for_seq", None)
    resolve_aiter_backends = getattr(aiter_mod, "resolve_backends", None)
    if not callable(build_aiter_cases):
        raise RuntimeError(f"missing build_cases_for_seq() in {aiter_entry}")
    if not callable(resolve_aiter_backends):
        raise RuntimeError(f"missing resolve_backends() in {aiter_entry}")

    aiter_args = SimpleNamespace(
        run_aiter=backend == "aiter",
        run_aiter_asm=backend == "aiter_asm",
        run_aiter_triton=backend == "aiter_triton",
        dim=dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        input_dtype=input_dtype_name,
    )
    resolved_backends = resolve_aiter_backends(aiter_args)
    if not isinstance(resolved_backends, list):
        raise RuntimeError(
            "resolve_backends() must return list, got "
            f"{type(resolved_backends).__name__}"
        )

    for tokens in token_counts:
        cases = build_aiter_cases(
            args=aiter_args,
            seq_len=tokens,
            shared_input=shared_inputs[tokens],
            shared_weights=shared_weights,
            backends=[str(x) for x in resolved_backends],
        )
        for case in cases:
            timing = _time_call(
                case.fn,
                warmup=warmup,
                repeat=repeat,
                graph_iters=graph_iters,
            )
            _emit_record(
                backend=backend,
                tokens=tokens,
                dim=dim,
                inter_dim=inter_dim,
                experts=experts,
                topk=topk,
                input_dtype_name=input_dtype_name,
                mean_ms=timing.mean_ms,
            )


def run_aiter(**kwargs) -> None:
    _run_aiter_variant(backend="aiter", **kwargs)


def run_aiter_asm(**kwargs) -> None:
    _run_aiter_variant(backend="aiter_asm", **kwargs)


def run_aiter_triton(**kwargs) -> None:
    _run_aiter_variant(backend="aiter_triton", **kwargs)


BACKEND_MAP = {
    "aiter": ModelBackend(run=run_aiter),
    "aiter_asm": ModelBackend(run=run_aiter_asm),
    "aiter_triton": ModelBackend(run=run_aiter_triton),
}

BACKENDS = BACKEND_MAP
