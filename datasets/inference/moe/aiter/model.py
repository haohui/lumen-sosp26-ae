#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from types import SimpleNamespace
from typing import Any, Callable, Dict, List

try:
    import torch
except Exception:
    torch = None


FP8_MAX = 240.0
ROUTE_GROUP_SIZE = 32
_AITER_RUNTIME_CACHE = None


def resolve_backends(args: Any) -> List[str]:
    if bool(getattr(args, "run_aiter", False)):
        return ["asm"]
    out: List[str] = []
    if bool(getattr(args, "run_aiter_asm", False)):
        out.append("asm")
    if bool(getattr(args, "run_aiter_triton", False)):
        out.append("triton")
    return out


def _load_aiter_runtime(*, include_triton: bool):
    global _AITER_RUNTIME_CACHE
    if _AITER_RUNTIME_CACHE is not None and (
        not include_triton or "triton_moe" in _AITER_RUNTIME_CACHE
    ):
        return _AITER_RUNTIME_CACHE

    import aiter
    from aiter.ops.shuffle import shuffle_weight

    _AITER_RUNTIME_CACHE = {
        "aiter": aiter,
        "shuffle_weight": shuffle_weight,
    }
    if include_triton:
        from aiter.ops.triton.moe_align_block_size import moe_align_block_size_triton
        from aiter.ops.triton.moe_op import fused_moe as triton_moe
        from aiter.ops.triton.moe_op_silu_fused import fused_moe_silu as triton_moe_silu
        from aiter.ops.triton.utils.moe_config_utils import get_optimal_moe_config_func
        from aiter.ops.triton.utils.types import torch_to_triton_dtype

        triton_cfg_fn = get_optimal_moe_config_func(
            torch.bfloat16,
            use_fp8_w8a8=True,
            use_int8_w8a16=False,
            use_int8_w8a8=False,
            use_int4_w4a16=False,
            use_mxfp4=False,
        )
        _AITER_RUNTIME_CACHE.update(
            {
                "moe_align_block_size_triton": moe_align_block_size_triton,
                "triton_moe": triton_moe,
                "triton_moe_silu": triton_moe_silu,
                "torch_to_triton_dtype": torch_to_triton_dtype,
                "triton_cfg_fn": triton_cfg_fn,
            }
        )
    return _AITER_RUNTIME_CACHE


def _validate_shared_input(shared_input: Any, *, seq_len: int, dim: int, topk: int):
    # Contract: all MoE baselines consume one unified input pack built in
    # python/harness/bench/moe_runtime.py. No random input construction here.
    for name in ("input_q", "topk_weights", "topk_ids", "input_scale"):
        if not hasattr(shared_input, name):
            raise ValueError(f"shared_input missing field: {name}")

    x = shared_input
    if x.input_q.shape != (seq_len, dim):
        raise ValueError(f"input_q shape mismatch: expected {(seq_len, dim)}, got {tuple(x.input_q.shape)}")
    if x.topk_weights.shape != (seq_len, topk):
        raise ValueError(
            f"topk_weights shape mismatch: expected {(seq_len, topk)}, got {tuple(x.topk_weights.shape)}"
        )
    if x.topk_ids.shape != (seq_len, topk):
        raise ValueError(f"topk_ids shape mismatch: expected {(seq_len, topk)}, got {tuple(x.topk_ids.shape)}")
    if x.input_scale.shape[0] != seq_len:
        raise ValueError(f"input_scale shape mismatch on dim0: expected {seq_len}, got {x.input_scale.shape[0]}")
    return x


def _validate_shared_weights(shared_weights: Any, *, experts: int, dim: int, inter_dim: int):
    # Same rule as shared_input: weights come from unified outer construction only.
    if not isinstance(shared_weights, dict):
        raise ValueError(f"shared_weights must be dict, got {type(shared_weights).__name__}")
    for name in ("w1_q", "w2_q", "fc1_scale", "fc2_scale"):
        if name not in shared_weights:
            raise ValueError(f"shared_weights missing key: {name}")

    w = shared_weights
    if tuple(w["w1_q"].shape) != (experts, inter_dim * 2, dim):
        raise ValueError(
            f"w1_q shape mismatch: expected {(experts, inter_dim * 2, dim)}, got {tuple(w['w1_q'].shape)}"
        )
    if tuple(w["w2_q"].shape) != (experts, dim, inter_dim):
        raise ValueError(
            f"w2_q shape mismatch: expected {(experts, dim, inter_dim)}, got {tuple(w['w2_q'].shape)}"
        )
    return w


def _quantize_fp8_block_per_token(x: "torch.Tensor", block_k: int = 128) -> tuple["torch.Tensor", "torch.Tensor"]:
    rows, cols = x.shape
    if cols % block_k != 0:
        raise ValueError(f"cols must be divisible by {block_k}, got {cols}")
    xb = x.float().view(rows, cols // block_k, block_k)
    amax = xb.abs().amax(dim=-1, keepdim=True).clamp_min_(1e-6)
    q_scale = FP8_MAX / amax
    dq_scale = (1.0 / q_scale).squeeze(-1).contiguous()
    q = (xb * q_scale).to(torch.float8_e4m3fnuz).view(rows, cols).contiguous()
    return q, dq_scale


def _make_triton_sorted(
    topk_ids: "torch.Tensor",
    experts: int,
    block_m: int,
    moe_align_block_size_triton,
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    sorted_token_ids = torch.empty(
        (topk_ids.numel() + experts * (block_m - 1),),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    expert_ids = torch.empty(
        (topk_ids.numel() + experts,),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device=topk_ids.device)
    sorted_token_ids.fill_(topk_ids.numel())
    moe_align_block_size_triton(
        topk_ids=topk_ids,
        num_experts=experts,
        block_size=block_m,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_pad=num_tokens_post_pad,
    )
    return sorted_token_ids, expert_ids, num_tokens_post_pad


def _build_asm_sorted_routes(
    *,
    topk_ids: "torch.Tensor",
    topk_weights: "torch.Tensor",
    experts: int,
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    tokens, topk = topk_ids.shape
    max_num_tokens_padded = tokens * topk + experts * ROUTE_GROUP_SIZE - topk
    max_num_m_blocks = (max_num_tokens_padded + ROUTE_GROUP_SIZE - 1) // ROUTE_GROUP_SIZE

    init_val = (topk << 24) | tokens
    sorted_token_ids = torch.full(
        (max_num_tokens_padded,),
        init_val,
        dtype=torch.int64,
        device=topk_ids.device,
    )
    sorted_weights = torch.zeros((max_num_tokens_padded,), dtype=torch.float32, device=topk_weights.device)
    sorted_expert_ids = torch.full(
        (max_num_m_blocks,),
        -1,
        dtype=torch.int64,
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

        route_ids = (slot_ids.to(torch.int64) << 24) | token_ids.to(torch.int64)
        sorted_token_ids[sorted_ids_begin : sorted_ids_begin + tokens_num] = route_ids
        sorted_weights[sorted_ids_begin : sorted_ids_begin + tokens_num] = topk_weights[token_ids, slot_ids]

        sorted_expert_ids_num = (tokens_num + ROUTE_GROUP_SIZE - 1) // ROUTE_GROUP_SIZE
        tokens_num_pad = sorted_expert_ids_num * ROUTE_GROUP_SIZE
        sorted_expert_ids[sorted_expert_ids_begin : sorted_expert_ids_begin + sorted_expert_ids_num] = expert

        sorted_ids_begin += tokens_num_pad
        sorted_expert_ids_begin += sorted_expert_ids_num

    num_valid_ids = torch.empty((2,), dtype=torch.int64, device=topk_ids.device)
    num_valid_ids[0] = sorted_ids_begin
    num_valid_ids[1] = tokens
    return sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids


def build_cases_for_seq(
    *,
    args: Any,
    seq_len: int,
    shared_input,
    shared_weights,
    backends: List[str] | None = None,
) -> List[Callable[[], None]]:
    if torch is None:
        raise RuntimeError("torch is required")

    resolved = list(backends) if backends is not None else resolve_backends(args)
    if not resolved:
        return []
    valid = {"asm", "triton"}
    unknown = [b for b in resolved if b not in valid]
    if unknown:
        raise ValueError(f"unsupported AITER backends: {unknown}")

    needs_asm = "asm" in resolved
    needs_triton = "triton" in resolved

    rt = _load_aiter_runtime(include_triton=needs_triton)
    aiter = rt["aiter"]
    shuffle_weight = rt["shuffle_weight"]

    experts = int(args.experts)
    topk = int(args.topk)
    dim = int(args.dim)
    inter_dim = int(args.inter_dim)
    x = _validate_shared_input(shared_input, seq_len=seq_len, dim=dim, topk=topk)
    w = _validate_shared_weights(shared_weights, experts=experts, dim=dim, inter_dim=inter_dim)

    if needs_asm:
        w1_shuf = shuffle_weight(w["w1_q"], (16, 16))
        w2_shuf = shuffle_weight(w["w2_q"], (16, 16))
        input_scale_t = x.input_scale.t().contiguous()

        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids = _build_asm_sorted_routes(
            topk_ids=x.topk_ids.to(torch.int64).contiguous(),
            topk_weights=x.topk_weights.to(torch.float32).contiguous(),
            experts=experts,
        )
        sorted_ids_i32 = sorted_ids.to(torch.int32).contiguous()
        sorted_weights_f32 = sorted_weights.to(torch.float32).contiguous()
        sorted_expert_ids_i32 = sorted_expert_ids.to(torch.int32).contiguous()
        num_valid_ids_i32 = num_valid_ids.to(torch.int32).contiguous()
        out_asm = torch.zeros((seq_len, dim), dtype=torch.bfloat16, device=x.input_q.device)

    if needs_triton:
        moe_align_block_size_triton = rt["moe_align_block_size_triton"]
        triton_moe = rt["triton_moe"]
        triton_moe_silu = rt["triton_moe_silu"]
        torch_to_triton_dtype = rt["torch_to_triton_dtype"]
        triton_cfg = rt["triton_cfg_fn"](seq_len)
        triton_block_m = int(triton_cfg["BLOCK_SIZE_M"])
        triton_sorted_ids, triton_expert_ids, triton_num_post = _make_triton_sorted(
            x.topk_ids.to(torch.int32).contiguous(),
            experts,
            triton_block_m,
            moe_align_block_size_triton,
        )

        fc1_scale_3d = w["fc1_scale"].view(experts, (inter_dim * 2) // 128, dim // 128).contiguous()
        fc2_scale_3d = w["fc2_scale"].view(experts, dim // 128, inter_dim // 128).contiguous()

        stage1_triton = torch.zeros((seq_len * topk, inter_dim), dtype=torch.bfloat16, device=x.input_q.device)
        stage2_in_q = torch.zeros((seq_len * topk, inter_dim), dtype=torch.float8_e4m3fnuz, device=x.input_q.device)
        stage2_in_scale = torch.zeros((seq_len * topk, inter_dim // 128), dtype=torch.float32, device=x.input_q.device)
        stage2_triton = torch.zeros((seq_len, topk, dim), dtype=torch.bfloat16, device=x.input_q.device)
        out_triton = torch.zeros((seq_len, dim), dtype=torch.bfloat16, device=x.input_q.device)
        triton_stage2_topk_ids = torch.zeros((seq_len * topk, 1), dtype=torch.int32, device=x.input_q.device)
        triton_stage2_topk_weights = x.topk_weights.reshape(-1, 1).contiguous()

    def run_asm() -> None:
        out_asm.zero_()
        aiter.fmoe_fp8_blockscale_g1u1(
            out_asm,
            x.input_q,
            w1_shuf,
            w2_shuf,
            sorted_ids_i32,
            sorted_weights_f32,
            sorted_expert_ids_i32,
            num_valid_ids_i32,
            topk,
            input_scale_t,
            w["fc1_scale"],
            w["fc2_scale"],
            "",
            128,
            128,
            None,
        )

    def run_triton() -> None:
        stage1_triton.zero_()
        stage2_triton.zero_()
        out_triton.zero_()
        triton_moe_silu(
            x.input_q,
            w["w1_q"],
            stage1_triton,
            x.input_scale.contiguous(),
            fc1_scale_3d,
            None,
            x.topk_weights,
            x.topk_ids,
            triton_sorted_ids,
            triton_expert_ids,
            triton_num_post,
            False,
            topk,
            torch_to_triton_dtype[torch.bfloat16],
            use_fp8_w8a8=True,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            block_shape=[128, 128],
            config=triton_cfg,
        )
        q, s = _quantize_fp8_block_per_token(stage1_triton, block_k=128)
        stage2_in_q.copy_(q)
        stage2_in_scale.copy_(s)
        triton_moe(
            stage2_in_q,
            w["w2_q"],
            stage2_triton,
            stage2_in_scale,
            fc2_scale_3d,
            None,
            triton_stage2_topk_weights,
            triton_stage2_topk_ids,
            triton_sorted_ids,
            triton_expert_ids,
            triton_num_post,
            True,
            1,
            torch_to_triton_dtype[torch.bfloat16],
            use_fp8_w8a8=True,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            block_shape=[128, 128],
            config=triton_cfg,
        )
        torch.sum(stage2_triton, dim=1, out=out_triton)

    out: List[Callable[[], None]] = []
    if "asm" in resolved:
        out.append(run_asm)
    if "triton" in resolved:
        out.append(run_triton)
    return out


class Model:
    def __init__(self, *, variant: str):
        if variant not in {"aiter", "aiter_asm", "aiter_triton"}:
            raise ValueError(f"unsupported AITER variant: {variant}")
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
    ) -> List[Callable[[], None]]:
        args = SimpleNamespace(
            run_aiter=self.variant == "aiter",
            run_aiter_asm=self.variant == "aiter_asm",
            run_aiter_triton=self.variant == "aiter_triton",
            dim=dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            input_dtype=input_dtype,
        )
        return build_cases_for_seq(
            args=args,
            seq_len=seq_len,
            shared_input=shared_input,
            shared_weights=shared_weights,
        )


def _parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MoE AITER entry (called by benchmark_moe_unified_graph.py)")
    p.add_argument("--run-aiter", action="store_true")
    p.add_argument("--run-aiter-asm", action="store_true")
    p.add_argument("--run-aiter-triton", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse(argv)
    print(",".join(resolve_backends(args)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
