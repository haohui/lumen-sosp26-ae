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


BLOCK_N = 128
BLOCK_K = 128


@dataclass(frozen=True)
class BackendSpec:
    directory: str
    variant: str


BACKENDS = {
    name: BackendSpec(directory=name, variant=name)
    for name in ("cudaforge", "kernelbench", "kernelfalcon", "ksearch", "lumen")
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
) -> None:
    del device, input_dtype
    aiter_entry = moe_root / BACKENDS[backend].directory / "model.py"
    aiter_mod = load_module(aiter_entry)
    model_cls = getattr(aiter_mod, "Model", None)
    if model_cls is None:
        raise RuntimeError(f"missing Model in {aiter_entry}")
    model = model_cls(variant=BACKENDS[backend].variant)

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
            )


def _run_python_backend(
    *,
    backend: str,
    moe_root: Path,
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
) -> None:
    path = moe_root / BACKENDS[backend].directory / "model.py"
    fn = build_model_fn(load_module(path), device=device, dtype=input_dtype)
    for tokens in token_counts:
        x = shared_inputs[tokens]
        timing = _time_call(
            lambda x=x: fn(
                x.input_q,
                shared_weights["w1_q"],
                shared_weights["w2_q"],
                x.topk_weights,
                x.topk_ids,
                x.input_scale,
                shared_weights["fc1_scale"],
                shared_weights["fc2_scale"],
            ),
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


def _build_sorted_routes(
    *,
    topk_ids: "torch.Tensor",
    topk_weights: "torch.Tensor",
    experts: int,
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    tokens, topk = topk_ids.shape
    route_group_size = 32
    max_num_tokens_padded = tokens * topk + experts * route_group_size - topk
    max_num_m_blocks = (
        max_num_tokens_padded + route_group_size - 1
    ) // route_group_size

    init_val = (topk << 24) | tokens
    sorted_token_ids = torch.full(
        (max_num_tokens_padded,),
        init_val,
        dtype=torch.int32,
        device=topk_ids.device,
    )
    sorted_weights = torch.zeros(
        (max_num_tokens_padded,),
        dtype=torch.float32,
        device=topk_weights.device,
    )
    sorted_expert_ids = torch.full(
        (max_num_m_blocks,),
        -1,
        dtype=torch.int32,
        device=topk_ids.device,
    )

    sorted_ids_begin = 0
    sorted_expert_ids_begin = 0
    for expert in range(experts):
        mask = topk_ids.eq(expert)
        if not bool(mask.any()):
            continue
        token_ids, slot_ids = torch.nonzero(mask, as_tuple=True)
        tokens_num = int(token_ids.numel())

        route_ids = (slot_ids.to(torch.int32) << 24) | token_ids.to(torch.int32)
        sorted_token_ids[sorted_ids_begin : sorted_ids_begin + tokens_num] = route_ids
        sorted_weights[sorted_ids_begin : sorted_ids_begin + tokens_num] = topk_weights[
            token_ids,
            slot_ids,
        ]

        sorted_expert_ids_num = (tokens_num + route_group_size - 1) // route_group_size
        tokens_num_pad = sorted_expert_ids_num * route_group_size
        sorted_expert_ids[
            sorted_expert_ids_begin : sorted_expert_ids_begin + sorted_expert_ids_num
        ] = expert

        sorted_ids_begin += tokens_num_pad
        sorted_expert_ids_begin += sorted_expert_ids_num

    num_valid_ids = torch.empty((2,), dtype=torch.int32, device=topk_ids.device)
    num_valid_ids[0] = sorted_ids_begin
    num_valid_ids[1] = tokens
    return (
        sorted_token_ids.contiguous(),
        sorted_weights.contiguous(),
        sorted_expert_ids.contiguous(),
        num_valid_ids.contiguous(),
    )


def _run_lumen_backend(
    *,
    backend: str,
    moe_root: Path,
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
) -> None:
    del device, input_dtype
    mod = load_module(moe_root / BACKENDS[backend].directory / "fused_moe.py")
    fn = getattr(mod, "fused_moe_fp8_blockscale_g1u1")

    for tokens in token_counts:
        x = shared_inputs[tokens]
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids = (
            _build_sorted_routes(
                topk_ids=x.topk_ids,
                topk_weights=x.topk_weights,
                experts=experts,
            )
        )
        out = torch.empty((tokens, dim), dtype=torch.bfloat16, device=x.input_q.device)
        timing = _time_call(
            lambda x=x,
            sorted_ids=sorted_ids,
            sorted_weights=sorted_weights,
            sorted_expert_ids=sorted_expert_ids,
            num_valid_ids=num_valid_ids,
            out=out: fn(
                x.input_q,
                shared_weights["w1_q"],
                shared_weights["w2_q"],
                sorted_ids,
                sorted_weights,
                sorted_expert_ids,
                num_valid_ids,
                topk,
                x.input_scale,
                shared_weights["fc1_scale"],
                shared_weights["fc2_scale"],
                out=out,
            ),
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


def run_backend(*, backend: str, **kwargs) -> None:
    spec = BACKENDS[backend]
    if spec.directory == "aiter":
        _run_aiter_variant(backend=backend, **kwargs)
    elif spec.directory == "lumen":
        _run_lumen_backend(backend=backend, **kwargs)
    else:
        _run_python_backend(backend=backend, **kwargs)
