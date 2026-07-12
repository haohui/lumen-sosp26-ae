#!/usr/bin/env python3
from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path

import torch.nn.functional as F

from backends import build_model_instance, exit_after_success_if_requested, load_module
from cli_utils import emit_jsonl
from cudagraph_timer import benchmark_with_cudagraph

try:
    import torch
except Exception:
    torch = None


BLOCK_N = 128
BLOCK_K = 128
CORRECTNESS_TOLERANCES = {
    "fp8": {"rtol": 1e-1, "atol": 1e-1},
    "bf16": {"rtol": 1e-1, "atol": 1e-1},
}


@dataclass(frozen=True)
class BackendSpec:
    directory: str
    variant: str


BACKENDS = {
    name: BackendSpec(directory=name, variant=name)
    for name in (
        "cudaforge",
        "kernelbench",
        "kernelfalcon",
        "ksearch",
        "lumen",
        "triton",
    )
}
BACKENDS.update(
    {
        name: BackendSpec(directory="aiter", variant=name)
        for name in ("aiter",)
    }
)


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


def _release_correctness_temporaries() -> None:
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


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
    check_correctness: bool,
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
            **({"correctness": True} if check_correctness else {}),
        }
    )


def _run_case_backend(
    *,
    backend: str,
    mod,
    spec: BackendSpec,
    token_counts: list[int],
    shared_inputs: dict[int, SharedInputs],
    shared_weights: dict[str, torch.Tensor],
    device: torch.device,
    input_dtype: torch.dtype,
    dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    input_dtype_name: str,
    warmup: int,
    repeat: int,
    graph_iters: int,
    check_correctness: bool,
) -> None:
    del device, input_dtype
    model_cls = getattr(mod, "Model", None)
    if model_cls is None:
        raise RuntimeError(f"missing Model for backend {backend}")
    model = model_cls(variant=spec.variant)

    for tokens in token_counts:
        cases = model.build_cases(
            seq_len=tokens,
            shared_input=shared_inputs[tokens],
            shared_weights=shared_weights,
            dim=dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            input_dtype=input_dtype_name,
        )
        for fn in cases:
            if check_correctness:
                actual = fn()
                _check_correctness(
                    actual=actual,
                    inputs=shared_inputs[tokens],
                    weights=shared_weights,
                    tokens=tokens,
                    dim=dim,
                    inter_dim=inter_dim,
                    experts=experts,
                    input_dtype_name=input_dtype_name,
                )
                del actual
                _release_correctness_temporaries()
            timing = _time_call(
                fn,
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
                check_correctness=check_correctness,
            )
        shared_inputs.pop(tokens, None)
        out_cache = getattr(model, "_out_cache", None)
        if isinstance(out_cache, dict):
            out_cache.clear()
        _release_correctness_temporaries()


def _run_python_backend(
    *,
    backend: str,
    mod,
    token_counts: list[int],
    shared_inputs: dict[int, SharedInputs],
    shared_weights: dict[str, torch.Tensor],
    device: torch.device,
    input_dtype: torch.dtype,
    input_dtype_name: str,
    dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    warmup: int,
    repeat: int,
    graph_iters: int,
    check_correctness: bool,
) -> None:
    fn = build_model_instance(mod, device=device, dtype=input_dtype)
    for tokens in token_counts:
        x = shared_inputs[tokens]
        call = lambda x=x: fn(
            x.input_q,
            shared_weights["w1_q"],
            shared_weights["w2_q"],
            x.topk_weights,
            x.topk_ids,
            x.input_scale,
            shared_weights["fc1_scale"],
            shared_weights["fc2_scale"],
        )
        if check_correctness:
            actual = call()
            _check_correctness(
                actual=actual,
                inputs=x,
                weights=shared_weights,
                tokens=tokens,
                dim=dim,
                inter_dim=inter_dim,
                experts=experts,
                input_dtype_name=input_dtype_name,
            )
            del actual
            _release_correctness_temporaries()
        timing = _time_call(
            call,
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
            check_correctness=check_correctness,
        )
        shared_inputs.pop(tokens, None)
        _release_correctness_temporaries()


def run_backend(*, backend: str, **kwargs) -> None:
    spec = BACKENDS[backend]
    model_path = kwargs.pop("model_path", None)
    mod = load_module(model_path or kwargs["moe_root"] / spec.directory / "model.py")
    model_cls = getattr(mod, "Model", None)
    if model_cls is not None and hasattr(model_cls, "build_cases"):
        case_kwargs = dict(kwargs)
        case_kwargs.pop("moe_root", None)
        _run_case_backend(backend=backend, mod=mod, spec=spec, **case_kwargs)
    else:
        python_kwargs = dict(kwargs)
        python_kwargs.pop("moe_root", None)
        _run_python_backend(backend=backend, mod=mod, **python_kwargs)
    exit_after_success_if_requested(mod)


def _check_correctness(
    *,
    actual,
    inputs: SharedInputs,
    weights: dict[str, torch.Tensor],
    tokens: int,
    dim: int,
    inter_dim: int,
    experts: int,
    input_dtype_name: str,
) -> None:
    with torch.inference_mode():
        expected = _moe_reference(
            inputs=inputs,
            weights=weights,
            dim=dim,
            inter_dim=inter_dim,
            experts=experts,
        )

    if not isinstance(actual, torch.Tensor):
        raise AssertionError(
            f"MoE correctness failed for {tokens} tokens: "
            f"backend returned {type(actual).__name__}, expected torch.Tensor"
        )
    expected_shape = (tokens, dim)
    if tuple(actual.shape) != expected_shape:
        raise AssertionError(
            f"MoE correctness failed for {tokens} tokens: "
            f"output shape {tuple(actual.shape)}, expected {expected_shape}"
        )
    if actual.dtype != expected.dtype:
        raise AssertionError(
            f"MoE correctness failed for {tokens} tokens: "
            f"output dtype {actual.dtype}, expected {expected.dtype}"
        )

    tolerances = CORRECTNESS_TOLERANCES[input_dtype_name]
    try:
        torch.testing.assert_close(actual, expected, **tolerances)
    except AssertionError as exc:
        raise AssertionError(
            f"MoE correctness failed for {tokens} tokens "
            f"(rtol={tolerances['rtol']}, atol={tolerances['atol']}):\n{exc}"
        ) from exc


def _moe_reference(
    *,
    inputs: SharedInputs,
    weights: dict[str, torch.Tensor],
    dim: int,
    inter_dim: int,
    experts: int,
) -> torch.Tensor:
    input_f = (
        inputs.input_q.to(torch.float32).view(-1, dim // BLOCK_K, BLOCK_K)
        * inputs.input_scale.to(torch.float32).unsqueeze(-1)
    ).reshape(-1, dim)
    out = torch.zeros(
        (input_f.shape[0], dim),
        dtype=torch.float32,
        device=input_f.device,
    )

    for expert in range(experts):
        token_ids, slots = torch.nonzero(inputs.topk_ids == expert, as_tuple=True)
        if token_ids.numel() == 0:
            continue
        w1 = _dequantize_weight(
            weights["w1_q"][expert],
            weights["fc1_scale"][expert],
        )
        w2 = _dequantize_weight(
            weights["w2_q"][expert],
            weights["fc2_scale"][expert],
        )
        stage1 = input_f[token_ids] @ w1.transpose(0, 1)
        gate, up = stage1.split(inter_dim, dim=-1)
        expert_out = (F.silu(gate) * up) @ w2.transpose(0, 1)
        route_weights = inputs.topk_weights[token_ids, slots].unsqueeze(1)
        out.index_add_(0, token_ids, expert_out * route_weights)
    return out.to(torch.bfloat16)


def _dequantize_weight(
    weight_q: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    rows, cols = weight_q.shape
    row_blocks = rows // BLOCK_N
    col_blocks = cols // BLOCK_K
    blocks = (
        weight_q.to(torch.float32)
        .view(row_blocks, BLOCK_N, col_blocks, BLOCK_K)
        .permute(0, 2, 1, 3)
    )
    scaled = blocks * scale.to(torch.float32).view(
        row_blocks,
        col_blocks,
        1,
        1,
    )
    return scaled.permute(0, 2, 1, 3).reshape(rows, cols)
