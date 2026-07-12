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
        "kernelbench",
        "kernelfalcon",
        "ksearch",
        "lumen",
        "triton",
    )
}

LUMEN_ATTENTION_MODULE = "attn_06_inst_scheduling.py"


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
) -> None:
    if backend == "lumen":
        _run_lumen_backend(
            backend=backend,
            lumen_root=attn_root / BACKENDS[backend].directory,
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
        return

    path = attn_root / BACKENDS[backend].directory / "model.py"
    fn = build_model_fn(load_module(path), device=device, dtype=dtype)
    for s in seq_lens:
        x = shared[s]
        timing = benchmark_with_cudagraph(
            lambda x=x: fn(x.q_bshd, x.k_bshd, x.v_bshd),
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
            }
        )


def _run_lumen_backend(
    *,
    backend: str,
    lumen_root: Path,
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
    if dtype is not torch.bfloat16:
        raise ValueError("LUMEN attention benchmark currently supports only bf16")
    if not causal:
        raise ValueError("LUMEN attention benchmark is wired for causal prefill")

    mod = load_module(lumen_root / LUMEN_ATTENTION_MODULE)
    for s in seq_lens:
        x = shared[s]
        q = x.q_bshd.reshape(batch_size * s, num_q_heads, head_dim).contiguous()
        k = x.k_bshd.reshape(batch_size * s, num_kv_heads, head_dim).contiguous()
        v = x.v_bshd.reshape(batch_size * s, num_kv_heads, head_dim).contiguous()
        seq_ptr_cpu = torch.arange(
            0,
            (batch_size + 1) * s,
            s,
            dtype=torch.int32,
        )
        out = torch.empty_like(q)
        seq_ptr_cpu_list = mod._validate_packed_flash_attn_inputs(
            q,
            k,
            v,
            seq_ptr_cpu,
        )
        max_seq_len = max(
            (
                seq_ptr_cpu_list[idx + 1] - seq_ptr_cpu_list[idx]
                for idx in range(len(seq_ptr_cpu_list) - 1)
            ),
            default=0,
        )
        if q.shape[0] == 0 or max_seq_len == 0:
            raise ValueError("LUMEN attention benchmark requires non-empty sequences")
        seq_ptr_device = seq_ptr_cpu.to(device=device)
        row_tiles = (max_seq_len + mod.BLOCK_ROWS - 1) // mod.BLOCK_ROWS
        physical_tiles = (row_tiles + 1) // 2
        num_sequences = len(seq_ptr_cpu_list) - 1
        kernel = mod._flash_attn_packed_kernel
        timing = benchmark_with_cudagraph(
            lambda q=q,
            k=k,
            v=v,
            seq_ptr_device=seq_ptr_device,
            out=out,
            kernel=kernel,
            physical_tiles=physical_tiles,
            num_sequences=num_sequences: kernel[
                lambda: (
                    (physical_tiles, q.shape[1], num_sequences),
                    (mod.THREADS, 1, 1),
                )
            ](
                q,
                k,
                v,
                seq_ptr_device,
                out,
                q.shape[0],
                num_sequences,
                num_warps=mod.NUM_WARPS,
            ),
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
            }
        )
