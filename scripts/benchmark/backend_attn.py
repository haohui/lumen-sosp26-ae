#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys

import torch.nn.functional as F

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


def _enable_attn_opt() -> None:
    try:
        from avelang import knobs as avelang_knobs

        avelang_knobs.amdgpu.enable_attn_opt = True
    except Exception:
        os.environ["ENABLE_ATTN_OPT"] = "1"


@dataclass
class SharedInputs:
    q_bshd: torch.Tensor
    k_bshd: torch.Tensor
    v_bshd: torch.Tensor


CORRECTNESS_TOLERANCES = {
    "bf16": {"rtol": 2e-2, "atol": 2e-2},
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
    check_correctness: bool = False,
) -> None:
    path = attn_root / BACKENDS[backend].directory / "model.py"
    if backend == "lumen":
        _enable_attn_opt()
    mod = load_module(path)
    model = build_model_instance(mod, device=device, dtype=dtype)
    for s in seq_lens:
        x = shared[s]
        if check_correctness:
            _check_correctness(
                model=model,
                inputs=x,
                seq_len=s,
                dtype_name=dtype_name,
                causal=causal,
            )
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
                **({"correctness": True} if check_correctness else {}),
            }
        )
    exit_after_success_if_requested(mod)


def _check_correctness(
    *,
    model,
    inputs: SharedInputs,
    seq_len: int,
    dtype_name: str,
    causal: bool,
) -> None:
    with torch.inference_mode():
        actual = model(inputs.q_bshd, inputs.k_bshd, inputs.v_bshd)
        expected = _attention_reference(inputs, causal=causal)

    if not isinstance(actual, torch.Tensor):
        raise AssertionError(
            f"Attention correctness failed for seq_len {seq_len}: "
            f"backend returned {type(actual).__name__}, expected torch.Tensor"
        )
    if actual.shape != expected.shape:
        raise AssertionError(
            f"Attention correctness failed for seq_len {seq_len}: "
            f"output shape {tuple(actual.shape)}, expected {tuple(expected.shape)}"
        )
    if actual.dtype != expected.dtype:
        raise AssertionError(
            f"Attention correctness failed for seq_len {seq_len}: "
            f"output dtype {actual.dtype}, expected {expected.dtype}"
        )

    tolerances = CORRECTNESS_TOLERANCES[dtype_name]
    try:
        torch.testing.assert_close(actual, expected, **tolerances)
    except AssertionError as exc:
        raise AssertionError(
            f"Attention correctness failed for seq_len {seq_len} "
            f"(rtol={tolerances['rtol']}, atol={tolerances['atol']}):\n{exc}"
        ) from exc


def _attention_reference(inputs: SharedInputs, *, causal: bool) -> torch.Tensor:
    q = inputs.q_bshd.transpose(1, 2)
    k = inputs.k_bshd.transpose(1, 2)
    v = inputs.v_bshd.transpose(1, 2)
    if q.shape[1] != k.shape[1]:
        if q.shape[1] % k.shape[1] != 0:
            raise ValueError(
                f"query heads ({q.shape[1]}) must be divisible by "
                f"KV heads ({k.shape[1]})"
            )
        groups = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    return out.transpose(1, 2).contiguous()
