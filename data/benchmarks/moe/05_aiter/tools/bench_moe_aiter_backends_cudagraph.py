#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

try:
    from cudagraph_timer import CUDAGraphTimingResult, benchmark_with_cudagraph
except Exception:
    # Primary fallback for current layout: opt_kernel/perf_script/cudagraph_timer.py
    perf_script_dir = THIS_DIR.parents[2] / "perf_script"
    if str(perf_script_dir) not in sys.path:
        sys.path.insert(0, str(perf_script_dir))
    try:
        from cudagraph_timer import CUDAGraphTimingResult, benchmark_with_cudagraph
    except Exception:
        # Legacy fallback where timer lives in repo/script.
        repo_script_dir = THIS_DIR.parents[3] / "script"
        if str(repo_script_dir) not in sys.path:
            sys.path.insert(0, str(repo_script_dir))
        from cudagraph_timer import CUDAGraphTimingResult, benchmark_with_cudagraph

import moe_quant_ref

# AITER backends
import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe_2stages, moe_sorting
from aiter.ops.quant import get_hip_quant
from aiter.ops.shuffle import shuffle_weight
from aiter.ops.triton.moe_op import fused_moe as triton_moe
from aiter.ops.triton.moe_op_silu_fused import fused_moe_silu as triton_moe_silu
from aiter.ops.triton.moe_align_block_size import moe_align_block_size_triton
from aiter.ops.triton.utils.moe_config_utils import get_optimal_moe_config_func
from aiter.ops.triton.utils.types import torch_to_triton_dtype


FP8_MAX = 240.0


@dataclass
class CaseConfig:
    tokens: int
    dim: int
    inter_dim: int
    experts: int
    topk: int


@dataclass
class DiffStats:
    max_abs: float
    mean_abs: float
    p99_abs: float
    max_rel: float
    allclose: bool
    nan_count: int
    inf_count: int


@dataclass
class EventTimingResult:
    median_ms: float
    mean_ms: float
    stdev_ms: float
    p10_ms: float
    p90_ms: float
    cv: float
    graph_iters: int
    num_replays: int
    total_calls_per_sample: int
    suspicious: bool
    suspicious_reason: str
    samples_ms: list[float]



def _stats(a: torch.Tensor, b: torch.Tensor, atol: float, rtol: float) -> DiffStats:
    aa = a.float()
    bb = b.float()
    diff = (aa - bb).abs()
    rel = diff / aa.abs().clamp_min(1e-8)
    flat = diff.reshape(-1)
    # Avoid very large-tensor quantile failure on long sequences.
    if flat.numel() > 8_000_000:
        step = max(1, flat.numel() // 8_000_000)
        p99_src = flat[::step]
    else:
        p99_src = flat
    nan_count = int(torch.isnan(bb).sum().item())
    inf_count = int(torch.isinf(bb).sum().item())
    return DiffStats(
        max_abs=float(diff.max().item()),
        mean_abs=float(diff.mean().item()),
        p99_abs=float(torch.quantile(p99_src, 0.99).item()),
        max_rel=float(rel.max().item()),
        allclose=bool(torch.allclose(aa, bb, atol=atol, rtol=rtol)),
        nan_count=nan_count,
        inf_count=inf_count,
    )



def _quantize_fp8_block_per_token(x: torch.Tensor, block_k: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = x.shape
    if cols % block_k != 0:
        raise ValueError(f"cols must be divisible by {block_k}, got {cols}")
    xb = x.float().view(rows, cols // block_k, block_k)
    amax = xb.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
    q_scale = FP8_MAX / amax
    dq_scale = (1.0 / q_scale).squeeze(-1).contiguous()
    q = (xb * q_scale).to(torch.float8_e4m3fnuz).view(rows, cols).contiguous()
    return q, dq_scale



def _make_triton_sorted(topk_ids: torch.Tensor, experts: int, block_m: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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



def _run_timer(
    fn: Callable[[], None],
    *,
    device: torch.device,
    warmup: int,
    warmup_ms: float,
    graph_iters: int,
    trials: int,
    repeat_ms: float,
    pre_capture_iters: int,
    min_replays: int,
    max_replays: int,
    disable_cudagraph: bool,
) -> CUDAGraphTimingResult | EventTimingResult:
    if disable_cudagraph:
        return _run_timer_event(
            fn,
            device=device,
            warmup=warmup,
            warmup_ms=warmup_ms,
            trials=trials,
            repeat_ms=repeat_ms,
            min_replays=min_replays,
            max_replays=max_replays,
        )

    return benchmark_with_cudagraph(
        fn=fn,
        device=device,
        warmup=warmup,
        warmup_ms=warmup_ms,
        graph_iters=graph_iters,
        pre_capture_iters=pre_capture_iters,
        trial_count=trials,
        min_measure_ms=repeat_ms,
        min_replays=min_replays,
        max_replays=max_replays,
        use_default_stream=True,
        allow_suspicious=True,
    )


def _run_timer_event(
    fn: Callable[[], None],
    *,
    device: torch.device,
    warmup: int,
    warmup_ms: float,
    trials: int,
    repeat_ms: float,
    min_replays: int,
    max_replays: int,
) -> EventTimingResult:
    warmup_calls = max(1, int(warmup))
    for _ in range(warmup_calls):
        fn()
    torch.cuda.synchronize(device=device)

    target_warmup_ms = max(0.0, float(warmup_ms))
    if target_warmup_ms > 0.0:
        elapsed_ms = 0.0
        ev_warmup_start = torch.cuda.Event(enable_timing=True)
        ev_warmup_end = torch.cuda.Event(enable_timing=True)
        while elapsed_ms < target_warmup_ms:
            ev_warmup_start.record()
            fn()
            ev_warmup_end.record()
            torch.cuda.synchronize(device=device)
            elapsed_ms += float(ev_warmup_start.elapsed_time(ev_warmup_end))
            warmup_calls += 1

    ev_start = torch.cuda.Event(enable_timing=True)
    ev_end = torch.cuda.Event(enable_timing=True)
    ev_start.record()
    fn()
    ev_end.record()
    torch.cuda.synchronize(device=device)
    probe_ms = float(ev_start.elapsed_time(ev_end))

    base = max(1e-3, probe_ms)
    replays = int(round(float(repeat_ms) / base))
    replays = max(int(min_replays), replays)
    replays = min(int(max_replays), replays)
    replays = max(1, replays)

    samples: list[float] = []
    trial_count = max(1, int(trials))
    for _ in range(trial_count):
        ev_start.record()
        for _ in range(replays):
            fn()
        ev_end.record()
        torch.cuda.synchronize(device=device)
        total_ms = float(ev_start.elapsed_time(ev_end))
        samples.append(total_ms / float(replays))

    samples_sorted = sorted(samples)
    n = len(samples_sorted)
    mean_ms = float(statistics.fmean(samples_sorted))
    stdev_ms = float(statistics.pstdev(samples_sorted)) if n > 1 else 0.0
    median_ms = float(statistics.median(samples_sorted))
    p10_ms = float(samples_sorted[max(0, int(0.1 * (n - 1)))])
    p90_ms = float(samples_sorted[min(n - 1, int(0.9 * (n - 1)))])
    cv = float(stdev_ms / mean_ms) if mean_ms > 0 else 0.0

    return EventTimingResult(
        median_ms=median_ms,
        mean_ms=mean_ms,
        stdev_ms=stdev_ms,
        p10_ms=p10_ms,
        p90_ms=p90_ms,
        cv=cv,
        graph_iters=0,
        num_replays=replays,
        total_calls_per_sample=replays,
        suspicious=False,
        suspicious_reason="event_timer",
        samples_ms=[float(x) for x in samples],
    )



def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Benchmark AITER ASM/CK/Triton with project CUDA graph timer")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=20260329)
    p.add_argument("--dim", type=int, default=7168)
    p.add_argument("--inter-dim", type=int, default=2048)
    p.add_argument("--experts", type=int, default=32)
    p.add_argument("--topk", type=int, default=4)
    p.add_argument("--ms-list", type=str, default="1024,2048,4096,8192,16384")
    p.add_argument("--backends", type=str, default="asm,ck,triton")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--warmup-ms", type=float, default=200.0)
    p.add_argument("--repeat-ms", type=float, default=1000.0)
    p.add_argument("--trials", type=int, default=9)
    p.add_argument("--graph-iters", type=int, default=10)
    p.add_argument("--pre-capture-iters", type=int, default=3)
    p.add_argument("--min-replays", type=int, default=100)
    p.add_argument("--max-replays", type=int, default=200000)
    p.add_argument("--disable-cudagraph", action="store_true")
    p.add_argument("--atol", type=float, default=5e-2)
    p.add_argument("--rtol", type=float, default=5e-2)
    p.add_argument("--out-dir", type=str, default="")
    p.add_argument("--out-prefix", type=str, default="bench_aiter_backends_cudagraph")
    return p



def main() -> int:
    args = _build_parser().parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device not available")

    device = torch.device(args.device)
    torch.cuda.set_device(device)

    ms_list = [int(x.strip()) for x in args.ms_list.split(",") if x.strip()]
    backends = [x.strip().lower() for x in args.backends.split(",") if x.strip()]

    valid_backends = {"asm", "ck", "triton"}
    unknown = [b for b in backends if b not in valid_backends]
    if unknown:
        raise ValueError(f"unsupported backends: {unknown}")

    cfg = moe_quant_ref.MoeConfig(
        dim=args.dim,
        inter_dim=args.inter_dim,
        experts=args.experts,
        topk=args.topk,
    )
    ref_impl = moe_quant_ref.MoeSingleOpTorchRef(out_dtype=torch.bfloat16)
    ck_quant_func = get_hip_quant(QuantType.per_1x128)

    triton_cfg_fn = get_optimal_moe_config_func(
        torch.bfloat16,
        use_fp8_w8a8=True,
        use_int8_w8a16=False,
        use_int8_w8a8=False,
        use_int4_w4a16=False,
        use_mxfp4=False,
    )

    rows: list[dict[str, str]] = []

    for tokens in ms_list:
        print(f"\n=== M={tokens} ===", flush=True)

        data = moe_quant_ref.make_inputs(tokens=tokens, cfg=cfg, device=device, seed=args.seed)

        with torch.no_grad():
            out_ref = ref_impl.forward(
                data["input_q"],
                data["w1_q"],
                data["w2_q"],
                data["topk_weights"],
                data["topk_ids"],
                data["input_scale"],
                data["fc1_scale"],
                data["fc2_scale"],
            ).to(torch.bfloat16)

        # Common route buffers for ASM/CK (same route format as reference)
        sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids = ref_impl._build_sorted_routes(
            data["topk_ids"].to(torch.int64),
            data["topk_weights"].to(torch.float32),
            cfg.experts,
        )
        sorted_token_ids_i32 = sorted_token_ids.to(torch.int32).contiguous()
        sorted_weights_f32 = sorted_weights.to(torch.float32).contiguous()
        sorted_expert_ids_i32 = sorted_expert_ids.to(torch.int32).contiguous()
        num_valid_ids_i32 = num_valid_ids.to(torch.int32).contiguous()

        # CK route buffers use CK-supported block_m.
        ck_block_m = 64
        ck_sorted_ids, ck_sorted_weights, ck_sorted_expert_ids, ck_num_valid_ids, _ = moe_sorting(
            data["topk_ids"].to(torch.int32).contiguous(),
            data["topk_weights"].to(torch.float32).contiguous(),
            cfg.experts,
            cfg.dim,
            torch.bfloat16,
            block_size=ck_block_m,
        )

        # Prepare weights/layout for ASM
        w1_shuf = shuffle_weight(data["w1_q"], (16, 16))
        w2_shuf = shuffle_weight(data["w2_q"], (16, 16))

        # Prepare Triton route/layout
        triton_cfg = triton_cfg_fn(tokens)
        triton_block_m = int(triton_cfg["BLOCK_SIZE_M"])
        triton_sorted_ids, triton_expert_ids, triton_num_post = _make_triton_sorted(
            data["topk_ids"].to(torch.int32).contiguous(),
            cfg.experts,
            triton_block_m,
        )

        fc1_scale_3d = data["fc1_scale"].view(cfg.experts, (cfg.inter_dim * 2) // 128, cfg.dim // 128).contiguous()
        fc2_scale_3d = data["fc2_scale"].view(cfg.experts, cfg.dim // 128, cfg.inter_dim // 128).contiguous()

        # Buffers
        out_asm = torch.zeros((tokens, cfg.dim), dtype=torch.bfloat16, device=device)
        out_ck = torch.zeros((tokens, cfg.dim), dtype=torch.bfloat16, device=device)
        ck_stage1 = torch.zeros((tokens, cfg.topk, cfg.inter_dim), dtype=torch.bfloat16, device=device)
        ck_stage2_in_q = torch.zeros((tokens, cfg.topk, cfg.inter_dim), dtype=torch.float8_e4m3fnuz, device=device)
        ck_stage2_in_scale = torch.zeros((tokens, cfg.topk, cfg.inter_dim // 128), dtype=torch.float32, device=device)
        stage1_triton = torch.zeros((tokens * cfg.topk, cfg.inter_dim), dtype=torch.bfloat16, device=device)
        stage2_in_q = torch.zeros((tokens * cfg.topk, cfg.inter_dim), dtype=torch.float8_e4m3fnuz, device=device)
        stage2_in_scale = torch.zeros((tokens * cfg.topk, cfg.inter_dim // 128), dtype=torch.float32, device=device)
        stage2_triton = torch.zeros((tokens, cfg.topk, cfg.dim), dtype=torch.bfloat16, device=device)
        out_triton = torch.zeros((tokens, cfg.dim), dtype=torch.bfloat16, device=device)
        # Triton stage2 consumes route-wise A (M*topk rows). Use top_k=1 and route-wise weights/ids.
        triton_stage2_topk_ids = torch.zeros((tokens * cfg.topk, 1), dtype=torch.int32, device=device)
        triton_stage2_topk_weights = data["topk_weights"].reshape(-1, 1).contiguous()

        def run_asm() -> None:
            out_asm.zero_()
            aiter.fmoe_fp8_blockscale_g1u1(
                out_asm,
                data["input_q"],
                w1_shuf,
                w2_shuf,
                sorted_token_ids_i32,
                sorted_weights_f32,
                sorted_expert_ids_i32,
                num_valid_ids_i32,
                cfg.topk,
                data["input_scale"].t().contiguous(),
                data["fc1_scale"].contiguous(),
                data["fc2_scale"].contiguous(),
                "",
                128,
                128,
                None,
            )

        def run_ck() -> None:
            out_ck.zero_()
            ck_stage1.zero_()
            aiter.ck_moe_stage1_fwd(
                data["input_q"],
                w1_shuf,
                w2_shuf,
                ck_sorted_ids,
                ck_sorted_expert_ids,
                ck_num_valid_ids,
                ck_stage1,
                cfg.topk,
                kernelName="",
                activation=ActivationType.Silu,
                block_m=ck_block_m,
                quant_type=QuantType.per_1x128,
                sorted_weights=None,
                w1_scale=data["fc1_scale"].contiguous(),
                a1_scale=data["input_scale"].contiguous(),
            )

            q_ck, s_ck = ck_quant_func(
                ck_stage1,
                scale=None,
                quant_dtype=torch.float8_e4m3fnuz,
                num_rows=None,
                num_rows_factor=cfg.topk,
            )
            ck_stage2_in_q.copy_(q_ck.view(tokens, cfg.topk, cfg.inter_dim))
            ck_stage2_in_scale.copy_(s_ck)

            aiter.ck_moe_stage2_fwd(
                ck_stage2_in_q,
                w1_shuf,
                w2_shuf,
                ck_sorted_ids,
                ck_sorted_expert_ids,
                ck_num_valid_ids,
                out_ck,
                cfg.topk,
                kernelName="",
                activation=ActivationType.Silu,
                block_m=ck_block_m,
                quant_type=QuantType.per_1x128,
                sorted_weights=ck_sorted_weights,
                w2_scale=data["fc2_scale"].contiguous(),
                a2_scale=ck_stage2_in_scale,
            )

        def run_triton() -> None:
            stage1_triton.zero_()
            stage2_triton.zero_()
            out_triton.zero_()
            triton_moe_silu(
                data["input_q"],
                data["w1_q"],
                stage1_triton,
                data["input_scale"].contiguous(),
                fc1_scale_3d,
                None,
                data["topk_weights"],
                data["topk_ids"],
                triton_sorted_ids,
                triton_expert_ids,
                triton_num_post,
                False,
                cfg.topk,
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
                data["w2_q"],
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

        backend_map: dict[str, tuple[Callable[[], None], torch.Tensor]] = {
            "asm": (run_asm, out_asm),
            "ck": (run_ck, out_ck),
            "triton": (run_triton, out_triton),
        }

        for backend in backends:
            fn, out_buf = backend_map[backend]
            try:
                # One eager run for initial compile and output materialization
                with torch.no_grad():
                    fn()
                    torch.cuda.synchronize(device=device)

                timing = _run_timer(
                    fn,
                    device=device,
                    warmup=args.warmup,
                    warmup_ms=args.warmup_ms,
                    graph_iters=args.graph_iters,
                    trials=args.trials,
                    repeat_ms=args.repeat_ms,
                    pre_capture_iters=args.pre_capture_iters,
                    min_replays=args.min_replays,
                    max_replays=args.max_replays,
                    disable_cudagraph=args.disable_cudagraph,
                )

                # Refresh output once after timing (avoid stale buffer assumptions)
                with torch.no_grad():
                    fn()
                    torch.cuda.synchronize(device=device)
                st = _stats(out_ref, out_buf, atol=args.atol, rtol=args.rtol)

                row = {
                    "backend": backend,
                    "tokens": str(tokens),
                    "median_ms": f"{timing.median_ms:.6f}",
                    "mean_ms": f"{timing.mean_ms:.6f}",
                    "stdev_ms": f"{timing.stdev_ms:.6f}",
                    "p10_ms": f"{timing.p10_ms:.6f}",
                    "p90_ms": f"{timing.p90_ms:.6f}",
                    "cv": f"{timing.cv:.6f}",
                    "graph_iters": str(timing.graph_iters),
                    "num_replays": str(timing.num_replays),
                    "total_calls_per_sample": str(timing.total_calls_per_sample),
                    "suspicious": str(bool(timing.suspicious)),
                    "suspicious_reason": timing.suspicious_reason or "",
                    "allclose": str(st.allclose),
                    "max_abs": f"{st.max_abs:.6f}",
                    "mean_abs": f"{st.mean_abs:.6f}",
                    "p99_abs": f"{st.p99_abs:.6f}",
                    "max_rel": f"{st.max_rel:.6f}",
                    "nan_count": str(st.nan_count),
                    "inf_count": str(st.inf_count),
                }
                rows.append(row)

                print(
                    f"[{backend:6s}] M={tokens:5d} median={timing.median_ms:8.4f} ms "
                    f"allclose={st.allclose} max_abs={st.max_abs:.6f} nan={st.nan_count} inf={st.inf_count}",
                    flush=True,
                )
            except Exception as e:
                print(
                    f"[warn] backend={backend} tokens={tokens} failed: {type(e).__name__}: {e}",
                    flush=True,
                )
                continue

        del data, out_ref
        torch.cuda.empty_cache()

    today = dt.date.today().isoformat()
    out_dir = Path(args.out_dir) if args.out_dir else (THIS_DIR / "runs" / f"moe_baselines_{today}")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"{args.out_prefix}_{ts}.csv"
    json_path = out_dir / f"{args.out_prefix}_{ts}.json"

    fieldnames = list(rows[0].keys()) if rows else []
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

        with json_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "meta": {
                        "time": dt.datetime.now(dt.timezone.utc).isoformat(),
                        "device": args.device,
                        "seed": args.seed,
                        "dim": args.dim,
                        "inter_dim": args.inter_dim,
                        "experts": args.experts,
                        "topk": args.topk,
                        "ms_list": ms_list,
                        "backends": backends,
                        "warmup": args.warmup,
                        "warmup_ms": args.warmup_ms,
                        "repeat_ms": args.repeat_ms,
                        "trials": args.trials,
                        "graph_iters": args.graph_iters,
                        "pre_capture_iters": args.pre_capture_iters,
                        "min_replays": args.min_replays,
                        "max_replays": args.max_replays,
                        "disable_cudagraph": bool(args.disable_cudagraph),
                        "atol": args.atol,
                        "rtol": args.rtol,
                    },
                    "rows": rows,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    print(csv_path)
    print(json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
