#!/usr/bin/env python3
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable, List

import torch
import triton.language as tl


"""
AITER Triton MoE adapter.

Historical launch:
  python opt_kernel/moe-openai/05_aiter/tools/bench_moe_aiter_backends_cudagraph.py \
    --backends triton --ms-list 1024,2048,4096,8192,16384 \
    --dim 7168 --inter-dim 2048 --experts 32 --topk 4
"""


BLOCK_N = 128
BLOCK_K = 128
_RUNTIME_CACHE = None


def _import_first(names: tuple[str, ...], attr: str):
    last_error: Exception | None = None
    for name in names:
        try:
            module = __import__(name, fromlist=[attr])
            return getattr(module, attr)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"could not import {attr} from {names}") from last_error


def _load_runtime():
    global _RUNTIME_CACHE
    if _RUNTIME_CACHE is not None:
        return _RUNTIME_CACHE

    fused_moe_silu = _import_first(
        (
            "aiter.ops.triton.moe.moe_op_silu_fused",
            "aiter.ops.triton.moe_op_silu_fused",
        ),
        "fused_moe_silu",
    )
    fused_moe = _import_first(
        (
            "aiter.ops.triton.moe.moe_op",
            "aiter.ops.triton.moe_op",
        ),
        "fused_moe",
    )
    moe_align_block_size_triton = _import_first(
        (
            "aiter.ops.triton.moe.moe_align_block_size",
            "aiter.ops.triton.moe_align_block_size",
        ),
        "moe_align_block_size_triton",
    )
    get_optimal_moe_config_func = _import_first(
        ("aiter.ops.triton.utils.moe_config_utils",),
        "get_optimal_moe_config_func",
    )

    _RUNTIME_CACHE = SimpleNamespace(
        fused_moe_silu=fused_moe_silu,
        fused_moe=fused_moe,
        moe_align_block_size_triton=moe_align_block_size_triton,
        triton_cfg_fn=get_optimal_moe_config_func(
            torch.bfloat16,
            use_fp8_w8a8=True,
            use_int8_w8a16=False,
            use_int8_w8a8=False,
            use_int4_w4a16=False,
            use_mxfp4=False,
        ),
    )
    return _RUNTIME_CACHE


def _validate_shared_input(shared_input: Any, *, seq_len: int, dim: int, topk: int):
    for name in ("input_q", "topk_weights", "topk_ids", "input_scale"):
        if not hasattr(shared_input, name):
            raise ValueError(f"shared_input missing field: {name}")

    x = shared_input
    if tuple(x.input_q.shape) != (seq_len, dim):
        raise ValueError(
            f"input_q shape mismatch: expected {(seq_len, dim)}, "
            f"got {tuple(x.input_q.shape)}"
        )
    if tuple(x.topk_weights.shape) != (seq_len, topk):
        raise ValueError(
            f"topk_weights shape mismatch: expected {(seq_len, topk)}, "
            f"got {tuple(x.topk_weights.shape)}"
        )
    if tuple(x.topk_ids.shape) != (seq_len, topk):
        raise ValueError(
            f"topk_ids shape mismatch: expected {(seq_len, topk)}, "
            f"got {tuple(x.topk_ids.shape)}"
        )
    if tuple(x.input_scale.shape) != (seq_len, dim // BLOCK_K):
        raise ValueError(
            f"input_scale shape mismatch: expected {(seq_len, dim // BLOCK_K)}, "
            f"got {tuple(x.input_scale.shape)}"
        )
    return x


def _validate_shared_weights(
    shared_weights: Any,
    *,
    experts: int,
    dim: int,
    inter_dim: int,
):
    if not isinstance(shared_weights, dict):
        raise ValueError(
            f"shared_weights must be dict, got {type(shared_weights).__name__}"
        )
    for name in ("w1_q", "w2_q", "fc1_scale", "fc2_scale"):
        if name not in shared_weights:
            raise ValueError(f"shared_weights missing key: {name}")

    if tuple(shared_weights["w1_q"].shape) != (experts, inter_dim * 2, dim):
        raise ValueError(
            f"w1_q shape mismatch: expected {(experts, inter_dim * 2, dim)}, "
            f"got {tuple(shared_weights['w1_q'].shape)}"
        )
    if tuple(shared_weights["w2_q"].shape) != (experts, dim, inter_dim):
        raise ValueError(
            f"w2_q shape mismatch: expected {(experts, dim, inter_dim)}, "
            f"got {tuple(shared_weights['w2_q'].shape)}"
        )
    return shared_weights


def _quantize_fp8_block_per_token(
    x: torch.Tensor,
    *,
    block_k: int = BLOCK_K,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = x.shape
    if cols % block_k != 0:
        raise ValueError(f"cols must be divisible by {block_k}, got {cols}")
    dtype = getattr(torch, "float8_e4m3fnuz", None) or getattr(
        torch,
        "float8_e4m3fn",
        None,
    )
    if dtype is None:
        raise RuntimeError("torch float8 dtype is unavailable")
    xb = x.float().view(rows, cols // block_k, block_k)
    amax = xb.abs().amax(dim=-1, keepdim=True).clamp_min_(1e-6)
    q_scale = 240.0 / amax
    dq_scale = (1.0 / q_scale).squeeze(-1).contiguous()
    q = (xb * q_scale).to(dtype).view(rows, cols).contiguous()
    return q, dq_scale


class Model:
    def __init__(self, *, variant: str):
        if variant != "triton":
            raise ValueError(f"unsupported Triton variant: {variant}")
        self.variant = variant

    def build_cases(
        self,
        *,
        seq_len: int,
        shared_input: Any,
        shared_weights: dict[str, torch.Tensor],
        dim: int,
        inter_dim: int,
        experts: int,
        topk: int,
        input_dtype: str,
    ) -> List[Callable[[], torch.Tensor]]:
        if input_dtype != "fp8":
            raise ValueError("AITER Triton MoE benchmark currently supports only fp8 input")

        rt = _load_runtime()
        x = _validate_shared_input(
            shared_input,
            seq_len=seq_len,
            dim=dim,
            topk=topk,
        )
        w = _validate_shared_weights(
            shared_weights,
            experts=experts,
            dim=dim,
            inter_dim=inter_dim,
        )

        cfg = rt.triton_cfg_fn(seq_len)
        block_m = int(cfg["BLOCK_SIZE_M"])
        sorted_token_ids = torch.empty(
            (x.topk_ids.numel() + experts * (block_m - 1),),
            dtype=torch.int32,
            device=x.topk_ids.device,
        )
        expert_ids = torch.empty(
            (x.topk_ids.numel() + experts,),
            dtype=torch.int32,
            device=x.topk_ids.device,
        )
        num_tokens_post_pad = torch.empty(
            (1,),
            dtype=torch.int32,
            device=x.topk_ids.device,
        )
        sorted_token_ids.fill_(x.topk_ids.numel())
        rt.moe_align_block_size_triton(
            topk_ids=x.topk_ids.to(torch.int32).contiguous(),
            num_experts=experts,
            block_size=block_m,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_pad=num_tokens_post_pad,
        )

        fc1_scale_3d = w["fc1_scale"].view(
            experts,
            (inter_dim * 2) // BLOCK_N,
            dim // BLOCK_K,
        ).contiguous()
        fc2_scale_3d = w["fc2_scale"].view(
            experts,
            dim // BLOCK_N,
            inter_dim // BLOCK_K,
        ).contiguous()

        stage1 = torch.zeros(
            (seq_len * topk, inter_dim),
            dtype=torch.bfloat16,
            device=x.input_q.device,
        )
        stage2_in_q = torch.empty(
            (seq_len * topk, inter_dim),
            dtype=x.input_q.dtype,
            device=x.input_q.device,
        )
        stage2_in_scale = torch.empty(
            (seq_len * topk, inter_dim // BLOCK_K),
            dtype=torch.float32,
            device=x.input_q.device,
        )
        stage2 = torch.zeros(
            (seq_len, topk, dim),
            dtype=torch.bfloat16,
            device=x.input_q.device,
        )
        out = torch.zeros((seq_len, dim), dtype=torch.bfloat16, device=x.input_q.device)
        stage2_topk_ids = torch.zeros(
            (seq_len * topk, 1),
            dtype=torch.int32,
            device=x.input_q.device,
        )
        stage2_topk_weights = x.topk_weights.reshape(-1, 1).contiguous()

        def run_triton() -> torch.Tensor:
            stage1.zero_()
            stage2.zero_()
            out.zero_()
            rt.fused_moe_silu(
                x.input_q,
                w["w1_q"],
                stage1,
                x.input_scale.contiguous(),
                fc1_scale_3d,
                None,
                x.topk_weights,
                x.topk_ids,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_pad,
                False,
                topk,
                tl.bfloat16,
                use_fp8_w8a8=True,
                use_int8_w8a16=False,
                use_int4_w4a16=False,
                block_shape=[BLOCK_N, BLOCK_K],
                config=cfg,
            )

            q, scale = _quantize_fp8_block_per_token(stage1, block_k=BLOCK_K)
            stage2_in_q.copy_(q)
            stage2_in_scale.copy_(scale)

            rt.fused_moe(
                stage2_in_q,
                w["w2_q"],
                stage2,
                stage2_in_scale,
                fc2_scale_3d,
                None,
                stage2_topk_weights,
                stage2_topk_ids,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_pad,
                True,
                1,
                tl.bfloat16,
                use_fp8_w8a8=True,
                use_int8_w8a16=False,
                use_int4_w4a16=False,
                block_shape=[BLOCK_N, BLOCK_K],
                config=cfg,
            )
            torch.sum(stage2, dim=1, out=out)
            return out

        return [run_triton]
