#!/usr/bin/env python3
"""Unified attention benchmark with publication-grade CUDA Graph timing."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, List

from cudagraph_timer import CUDAGraphTimingResult, benchmark_with_cudagraph

try:
    import torch
except Exception:
    torch = None


@dataclass
class SharedInputs:
    q_bhsd: "torch.Tensor"
    k_bhsd: "torch.Tensor"
    v_bhsd: "torch.Tensor"
    q_bshd: "torch.Tensor"
    k_bshd: "torch.Tensor"
    v_bshd: "torch.Tensor"
    q_thd: "torch.Tensor"
    k_thd: "torch.Tensor"
    v_thd: "torch.Tensor"
    indptr: "torch.Tensor"
    total_tokens: int


def _attention_tflops(
    *,
    batch_size: int,
    seq_len: int,
    num_q_heads: int,
    head_dim: int,
    causal: bool,
    ms: float,
) -> float:
    if ms <= 0.0:
        return float("nan")
    if causal:
        # lq=lk=S => per head FLOPs ~= 2 * S^2 * D
        flops = 2.0 * batch_size * num_q_heads * seq_len * seq_len * head_dim
    else:
        # lq=lk=S => per head FLOPs ~= 4 * S^2 * D
        flops = 4.0 * batch_size * num_q_heads * seq_len * seq_len * head_dim
    return flops / (ms * 1.0e-3) / 1.0e12


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_model_fn(module: ModuleType, device: "torch.device", dtype: "torch.dtype"):
    if hasattr(module, "ModelNew"):
        model = module.ModelNew().to(device=device, dtype=dtype)
        return lambda q, k, v: model(q, k, v)
    if hasattr(module, "Model"):
        model = module.Model().to(device=device, dtype=dtype)
        return lambda q, k, v: model(q, k, v)
    if hasattr(module, "kernel_function"):
        return lambda q, k, v: module.kernel_function(q, k, v)
    raise RuntimeError("expected one of: ModelNew, Model, kernel_function")


def _parse_cpu_cores(spec: str) -> List[int]:
    cores: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            lo_s, hi_s = token.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if hi < lo:
                raise ValueError(f"Invalid cpu core range token: {token}")
            cores.update(range(lo, hi + 1))
        else:
            cores.add(int(token))
    if not cores:
        raise ValueError("No CPU cores parsed from --cpu-cores")
    return sorted(cores)


def _apply_cpu_affinity(spec: str) -> List[int]:
    if not spec:
        return []
    if not hasattr(os, "sched_setaffinity"):
        raise RuntimeError("CPU affinity is not supported on this platform")
    allowed = sorted(os.sched_getaffinity(0))
    cores = _parse_cpu_cores(spec)
    invalid = [c for c in cores if c not in allowed]
    if invalid:
        raise ValueError(f"Requested CPU cores not allowed in this runtime: {invalid}")
    os.sched_setaffinity(0, set(cores))
    return sorted(os.sched_getaffinity(0))


def _apply_visible_devices(hip_visible_devices: str) -> str:
    if not hip_visible_devices:
        return ""
    normalized = ",".join(x.strip() for x in hip_visible_devices.split(",") if x.strip())
    if not normalized:
        raise ValueError("Invalid --hip-visible-devices value")
    os.environ["HIP_VISIBLE_DEVICES"] = normalized
    if "ROCR_VISIBLE_DEVICES" in os.environ:
        os.environ.pop("ROCR_VISIBLE_DEVICES")
    return normalized


def _build_shared_inputs(
    *,
    seq_lens: List[int],
    batch_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    device: "torch.device",
    dtype: "torch.dtype",
    seed: int,
) -> Dict[int, SharedInputs]:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    out: Dict[int, SharedInputs] = {}
    for s in seq_lens:
        q_bhsd = torch.randn((batch_size, num_q_heads, s, head_dim), device=device, dtype=dtype, generator=g)
        k_bhsd = torch.randn((batch_size, num_kv_heads, s, head_dim), device=device, dtype=dtype, generator=g)
        v_bhsd = torch.randn((batch_size, num_kv_heads, s, head_dim), device=device, dtype=dtype, generator=g)

        # FlashAttention path uses [B, S, H, D].
        q_bshd = q_bhsd.permute(0, 2, 1, 3).contiguous()
        k_bshd = k_bhsd.permute(0, 2, 1, 3).contiguous()
        v_bshd = v_bhsd.permute(0, 2, 1, 3).contiguous()

        # FlashInfer ragged prefill path uses [total_tokens, H, D] + indptr.
        q_thd = q_bshd.view(batch_size * s, num_q_heads, head_dim)
        k_thd = k_bshd.view(batch_size * s, num_kv_heads, head_dim)
        v_thd = v_bshd.view(batch_size * s, num_kv_heads, head_dim)
        indptr = (torch.arange(0, batch_size + 1, device=device, dtype=torch.int32) * s).contiguous()

        out[s] = SharedInputs(
            q_bhsd=q_bhsd,
            k_bhsd=k_bhsd,
            v_bhsd=v_bhsd,
            q_bshd=q_bshd,
            k_bshd=k_bshd,
            v_bshd=v_bshd,
            q_thd=q_thd,
            k_thd=k_thd,
            v_thd=v_thd,
            indptr=indptr,
            total_tokens=batch_size * s,
        )
    return out


def _timer_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    fixed_repeat_calls: int | None = None
    if args.repeat > 0:
        fixed_repeat_calls = args.repeat
    return {
        "warmup": args.warmup,
        "warmup_ms": args.warmup_ms,
        "graph_iters": args.graph_iters,
        "pre_capture_iters": args.pre_capture_iters,
        "trial_count": args.timer_trials,
        "min_measure_ms": args.repeat_ms if args.measure_ms is None else args.measure_ms,
        "min_replays": args.min_replays,
        "max_replays": args.max_replays,
        "fixed_repeat_calls": fixed_repeat_calls,
        "suspicious_ratio_threshold": args.suspicious_ratio,
        "allow_suspicious": args.allow_suspicious_graph,
    }


def _time_one(
    call: Callable[[], None],
    *,
    device: "torch.device",
    args: argparse.Namespace,
) -> CUDAGraphTimingResult:
    return benchmark_with_cudagraph(
        fn=call,
        device=device,
        **_timer_kwargs(args),
    )


def _make_row(
    *,
    baseline: str,
    kernel_path: str,
    seq_len: int,
    num_q_heads: int,
    head_dim: int,
    batch_size: int,
    causal: bool,
    timing: CUDAGraphTimingResult | None,
    error: str | None = None,
) -> Dict[str, Any]:
    if timing is None:
        return {
            "baseline": baseline,
            "kernel_path": kernel_path,
            "seq_len": seq_len,
            "status": "error",
            "error": error,
        }
    median_ms = timing.median_ms
    return {
        "baseline": baseline,
        "kernel_path": kernel_path,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "num_q_heads": num_q_heads,
        "head_dim": head_dim,
        "causal": causal,
        "status": "ok",
        "median_ms": median_ms,
        "mean_ms": timing.mean_ms,
        "stdev_ms": timing.stdev_ms,
        "min_ms": timing.min_ms,
        "max_ms": timing.max_ms,
        "p10_ms": timing.p10_ms,
        "p90_ms": timing.p90_ms,
        "cv": timing.cv,
        "tflops_median": _attention_tflops(
            batch_size=batch_size,
            seq_len=seq_len,
            num_q_heads=num_q_heads,
            head_dim=head_dim,
            causal=causal,
            ms=median_ms,
        ),
        "eager_probe_ms": timing.eager_probe_ms,
        "suspicious": timing.suspicious,
        "suspicious_reason": timing.suspicious_reason,
        "num_replays": timing.num_replays,
        "graph_iters": timing.graph_iters,
        "warmup_calls": timing.warmup_calls,
        "total_calls_per_sample": timing.total_calls_per_sample,
        "trial_count": len(timing.samples_ms),
        "samples_ms": timing.samples_ms,
    }


def _print_row(row: Dict[str, Any]) -> None:
    s = int(row["seq_len"])
    if row.get("status") != "ok":
        print(f"  S={s:>6} | error | {row.get('error', 'unknown')}")
        return
    flag = " [SUSPICIOUS]" if row.get("suspicious") else ""
    print(
        f"  S={s:>6} | med={row['median_ms']:>8.4f} ms | "
        f"p10/p90={row['p10_ms']:.4f}/{row['p90_ms']:.4f} | "
        f"cv={100.0 * row['cv']:.2f}% | {row['tflops_median']:>8.2f} TFLOPS | "
        f"trials={row['trial_count']} replays={row['num_replays']}{flag}"
    )


def run_flashattention_baseline(
    *,
    seq_lens: List[int],
    shared_inputs: Dict[int, SharedInputs],
    device: "torch.device",
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    from flash_attn import flash_attn_func

    rows: List[Dict[str, Any]] = []
    name = "flashattention"
    print(f"\n[{name}]")
    for s in seq_lens:
        x = shared_inputs[s]

        def call() -> None:
            flash_attn_func(
                x.q_bshd,
                x.k_bshd,
                x.v_bshd,
                dropout_p=0.0,
                causal=args.causal,
            )

        try:
            timing = _time_one(call, device=device, args=args)
            row = _make_row(
                baseline=name,
                kernel_path="flash_attn.flash_attn_func",
                seq_len=s,
                num_q_heads=args.num_q_heads,
                head_dim=args.head_dim,
                batch_size=args.batch_size,
                causal=args.causal,
                timing=timing,
            )
        except Exception as e:
            row = _make_row(
                baseline=name,
                kernel_path="flash_attn.flash_attn_func",
                seq_len=s,
                num_q_heads=args.num_q_heads,
                head_dim=args.head_dim,
                batch_size=args.batch_size,
                causal=args.causal,
                timing=None,
                error=f"{type(e).__name__}: {e}",
            )
        _print_row(row)
        rows.append(row)
    return rows


def run_flashinfer_baseline(
    *,
    seq_lens: List[int],
    shared_inputs: Dict[int, SharedInputs],
    device: "torch.device",
    dtype: "torch.dtype",
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    import flashinfer

    rows: List[Dict[str, Any]] = []
    name = "flashinfer"
    print(f"\n[{name}]")
    for s in seq_lens:
        x = shared_inputs[s]
        workspace = torch.empty(args.flashinfer_workspace_mb * 1024 * 1024, dtype=torch.int8, device=device)
        wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
            workspace,
            "NHD",
            use_cuda_graph=False,
            qo_indptr_buf=x.indptr,
            kv_indptr_buf=x.indptr,
            backend=args.flashinfer_backend,
        )
        wrapper.plan(
            x.indptr,
            x.indptr,
            args.num_q_heads,
            args.num_kv_heads,
            args.head_dim,
            head_dim_vo=args.head_dim,
            causal=args.causal,
            q_data_type=dtype,
            kv_data_type=dtype,
        )

        def call() -> None:
            wrapper.run(x.q_thd, x.k_thd, x.v_thd)

        try:
            timing = _time_one(call, device=device, args=args)
            row = _make_row(
                baseline=name,
                kernel_path=f"flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(backend={args.flashinfer_backend})",
                seq_len=s,
                num_q_heads=args.num_q_heads,
                head_dim=args.head_dim,
                batch_size=args.batch_size,
                causal=args.causal,
                timing=timing,
            )
        except Exception as e:
            row = _make_row(
                baseline=name,
                kernel_path=f"flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(backend={args.flashinfer_backend})",
                seq_len=s,
                num_q_heads=args.num_q_heads,
                head_dim=args.head_dim,
                batch_size=args.batch_size,
                causal=args.causal,
                timing=None,
                error=f"{type(e).__name__}: {e}",
            )
        _print_row(row)
        rows.append(row)
    return rows


def run_python_kernel_baseline(
    name: str,
    kernel_path: Path,
    *,
    seq_lens: List[int],
    shared_inputs: Dict[int, SharedInputs],
    device: "torch.device",
    dtype: "torch.dtype",
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    module = _load_module(kernel_path)
    fn_model = _build_model_fn(module, device=device, dtype=dtype)
    print(f"\n[{name}] {kernel_path}")
    for s in seq_lens:
        x = shared_inputs[s]

        def call() -> None:
            fn_model(x.q_bhsd, x.k_bhsd, x.v_bhsd)

        try:
            timing = _time_one(call, device=device, args=args)
            row = _make_row(
                baseline=name,
                kernel_path=str(kernel_path),
                seq_len=s,
                num_q_heads=args.num_q_heads,
                head_dim=args.head_dim,
                batch_size=args.batch_size,
                causal=args.causal,
                timing=timing,
            )
        except Exception as e:
            row = _make_row(
                baseline=name,
                kernel_path=str(kernel_path),
                seq_len=s,
                num_q_heads=args.num_q_heads,
                head_dim=args.head_dim,
                batch_size=args.batch_size,
                causal=args.causal,
                timing=None,
                error=f"{type(e).__name__}: {e}",
            )
        _print_row(row)
        rows.append(row)
    return rows


def run_petit_baseline(
    *,
    seq_lens: List[int],
    shared_inputs: Dict[int, SharedInputs],
    device: "torch.device",
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    petit_root = Path(args.petit_root).resolve()
    if not petit_root.exists():
        raise RuntimeError(f"petit root does not exist: {petit_root}")
    if str(petit_root) not in sys.path:
        sys.path.insert(0, str(petit_root))

    import petit_kernel

    if not args.causal:
        raise RuntimeError("petit FlashMHA binding currently supports causal path only")

    rows: List[Dict[str, Any]] = []
    name = "petit"
    print(f"\n[{name}] root={petit_root}")
    for s in seq_lens:
        x = shared_inputs[s]

        def call() -> None:
            petit_kernel.flash_mha(
                x.q_bshd,
                x.k_bshd,
                x.v_bshd,
                args.petit_solution_id,
                seq_start=x.indptr,
            )

        try:
            timing = _time_one(call, device=device, args=args)
            row = _make_row(
                baseline=name,
                kernel_path=f"petit_kernel.flash_mha(solution_id={args.petit_solution_id})",
                seq_len=s,
                num_q_heads=args.num_q_heads,
                head_dim=args.head_dim,
                batch_size=args.batch_size,
                causal=args.causal,
                timing=timing,
            )
        except Exception as e:
            row = _make_row(
                baseline=name,
                kernel_path=f"petit_kernel.flash_mha(solution_id={args.petit_solution_id})",
                seq_len=s,
                num_q_heads=args.num_q_heads,
                head_dim=args.head_dim,
                batch_size=args.batch_size,
                causal=args.causal,
                timing=None,
                error=f"{type(e).__name__}: {e}",
            )
        _print_row(row)
        rows.append(row)
    return rows


def _collect_env_metadata(device: "torch.device") -> Dict[str, Any]:
    props = torch.cuda.get_device_properties(device)
    return {
        "hostname": socket.gethostname(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_hip_version": getattr(torch.version, "hip", None),
        "device_index": int(device.index if device.index is not None else 0),
        "device_name": props.name,
        "total_memory_gb": float(props.total_memory) / (1024.0**3),
        "multi_processor_count": int(getattr(props, "multi_processor_count", 0)),
    }


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    attn_root = repo_root / "data" / "benchmarks" / "attn"
    p = argparse.ArgumentParser(
        description="Unified attention timer with publication-grade CUDA Graph statistics."
    )
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument(
        "--hip-visible-devices",
        type=str,
        default="",
        help="Set HIP_VISIBLE_DEVICES for this process (e.g. '1' or '1,3').",
    )
    p.add_argument(
        "--cpu-cores",
        type=str,
        default="",
        help="Pin benchmark process to CPU cores, e.g. '0-15' or '0-7,16-23'.",
    )
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--seq-lens", type=str, default="1024,2048,4096,8192")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-q-heads", type=int, default=8)
    p.add_argument("--num-kv-heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--causal", dest="causal", action="store_true", default=True)
    p.add_argument("--non-causal", dest="causal", action="store_false")
    p.add_argument("--seed", type=int, default=20260314)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--warmup-ms", type=float, default=200.0)
    p.add_argument("--graph-iters", type=int, default=10)
    p.add_argument(
        "--repeat",
        type=int,
        default=0,
        help="Fixed total calls per sample (legacy mode). If 0, use --measure-ms auto sizing.",
    )
    p.add_argument("--timer-trials", type=int, default=9)
    p.add_argument("--repeat-ms", type=float, default=1000.0)
    p.add_argument("--measure-ms", type=float, default=None)
    p.add_argument("--min-replays", type=int, default=5)
    p.add_argument("--max-replays", type=int, default=200000)
    p.add_argument("--pre-capture-iters", type=int, default=3)
    p.add_argument("--suspicious-ratio", type=float, default=0.25)
    p.add_argument("--allow-suspicious-graph", action="store_true")

    p.add_argument("--run-flashinfer", dest="run_flashinfer", action="store_true", default=True)
    p.add_argument("--no-flashinfer", dest="run_flashinfer", action="store_false")
    p.add_argument("--run-flashattention", dest="run_flashattention", action="store_true", default=True)
    p.add_argument("--no-flashattention", dest="run_flashattention", action="store_false")
    p.add_argument("--run-petit", dest="run_petit", action="store_true", default=False)
    p.add_argument("--no-petit", dest="run_petit", action="store_false")
    p.add_argument(
        "--flashinfer-backend",
        type=str,
        default="auto",
        choices=["auto", "fa2", "aiter"],
    )
    p.add_argument("--flashinfer-workspace-mb", type=int, default=512)
    p.add_argument(
        "--petit-root",
        type=Path,
        default=repo_root / "petit-kernel",
        help="Path to petit-kernel python package root.",
    )
    p.add_argument(
        "--petit-solution-id",
        type=int,
        default=-1,
        help="MHA solution id for petit flash_mha; -1 means auto select.",
    )

    default_baselines = [
        attn_root / "06_aiter" / "best_kernel.py",
        attn_root / "05_HipKittens" / "best_kernel.py",
        attn_root / "03_kernelfalcon" / "best_kernel.py",
        attn_root / "04_ksearch" / "best_kernel.py",
        attn_root / "01_kernelbench" / "best_kernel.py",
        attn_root / "02_cudaforge" / "best_kernel.py",
    ]
    p.add_argument(
        "--baseline-kernel",
        action="append",
        type=Path,
        default=None,
        help="Path to baseline attention kernel python file; can be passed multiple times.",
    )
    p.add_argument("--json-out", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true")

    args = p.parse_args()
    if args.baseline_kernel is None:
        args.baseline_kernel = default_baselines
    return args


def main() -> None:
    args = parse_args()
    visible_devices = _apply_visible_devices(args.hip_visible_devices)
    cpu_affinity = _apply_cpu_affinity(args.cpu_cores)
    seq_lens = [int(x.strip()) for x in args.seq_lens.split(",") if x.strip()]
    baseline_paths = [Path(p) for p in args.baseline_kernel]

    if visible_devices:
        visible_count = len([x for x in visible_devices.split(",") if x])
        if args.device.startswith("cuda:"):
            local_idx = int(args.device.split(":", 1)[1])
            if local_idx < 0 or local_idx >= visible_count:
                raise ValueError(
                    f"--device {args.device} is out of range for visible devices [{visible_devices}] "
                    f"(use local index cuda:0..cuda:{visible_count - 1})"
                )

    effective_repeat_ms = args.repeat_ms if args.measure_ms is None else args.measure_ms
    mode = f"fixed_calls={args.repeat}" if args.repeat > 0 else f"auto_repeat_ms={effective_repeat_ms}"
    print(
        f"[config] device={args.device} dtype={args.dtype} seq_lens={seq_lens} "
        f"batch={args.batch_size} hq={args.num_q_heads} hkv={args.num_kv_heads} d={args.head_dim} "
        f"causal={args.causal} seed={args.seed} warmup={args.warmup} graph_iters={args.graph_iters} "
        f"timer_trials={args.timer_trials} mode={mode}"
    )
    print(
        f"[config] run_flashinfer={args.run_flashinfer} run_flashattention={args.run_flashattention} "
        f"flashinfer_backend={args.flashinfer_backend} flashinfer_workspace_mb={args.flashinfer_workspace_mb}"
    )
    print(
        f"[config] run_petit={args.run_petit} petit_root={args.petit_root} "
        f"petit_solution_id={args.petit_solution_id}"
    )
    print(
        f"[config] suspicious_ratio={args.suspicious_ratio} "
        f"allow_suspicious_graph={args.allow_suspicious_graph}"
    )
    if visible_devices:
        print(f"[config] HIP_VISIBLE_DEVICES={visible_devices}")
    if cpu_affinity:
        affinity_label = ",".join(str(c) for c in cpu_affinity)
        print(f"[config] CPU_AFFINITY={affinity_label}")
    print(f"[discover] baseline_kernels={len(baseline_paths)}")

    if args.dry_run:
        for p in baseline_paths:
            print(f"[baseline] {p}")
        if args.run_flashinfer:
            print("[dry-run] flashinfer enabled")
        if args.run_flashattention:
            print("[dry-run] flashattention enabled")
        if args.run_petit:
            print("[dry-run] petit enabled")
        return

    if torch is None:
        raise RuntimeError("torch is required. Activate ROCm/PyTorch environment first.")

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    shared_inputs = _build_shared_inputs(
        seq_lens=seq_lens,
        batch_size=args.batch_size,
        num_q_heads=args.num_q_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        device=device,
        dtype=dtype,
        seed=args.seed,
    )

    rows: List[Dict[str, Any]] = []
    if args.run_flashinfer:
        try:
            rows.extend(
                run_flashinfer_baseline(
                    seq_lens=seq_lens,
                    shared_inputs=shared_inputs,
                    device=device,
                    dtype=dtype,
                    args=args,
                )
            )
        except Exception as e:
            print(f"[skip] flashinfer failed: {type(e).__name__}: {e}")

    if args.run_flashattention:
        try:
            rows.extend(
                run_flashattention_baseline(
                    seq_lens=seq_lens,
                    shared_inputs=shared_inputs,
                    device=device,
                    args=args,
                )
            )
        except Exception as e:
            print(f"[skip] flashattention failed: {type(e).__name__}: {e}")

    if args.run_petit:
        try:
            rows.extend(
                run_petit_baseline(
                    seq_lens=seq_lens,
                    shared_inputs=shared_inputs,
                    device=device,
                    args=args,
                )
            )
        except Exception as e:
            print(f"[skip] petit failed: {type(e).__name__}: {e}")

    for idx, p in enumerate(baseline_paths, start=1):
        name = f"baseline{idx}::{p.parent.name}/{p.name}"
        if not p.exists():
            print(f"\n[{name}] skip | missing file: {p}")
            for s in seq_lens:
                rows.append(
                    _make_row(
                        baseline=name,
                        kernel_path=str(p),
                        seq_len=s,
                        num_q_heads=args.num_q_heads,
                        head_dim=args.head_dim,
                        batch_size=args.batch_size,
                        causal=args.causal,
                        timing=None,
                        error=f"missing file: {p}",
                    )
                )
            continue
        try:
            rows.extend(
                run_python_kernel_baseline(
                    name=name,
                    kernel_path=p,
                    seq_lens=seq_lens,
                    shared_inputs=shared_inputs,
                    device=device,
                    dtype=dtype,
                    args=args,
                )
            )
        except Exception as e:
            print(f"\n[{name}] skip | {type(e).__name__}: {e}")

    if args.json_out is not None:
        payload = {
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "config": {
                "device": args.device,
                "dtype": args.dtype,
                "hip_visible_devices": visible_devices,
                "cpu_cores": args.cpu_cores,
                "cpu_affinity_applied": cpu_affinity,
                "seq_lens": seq_lens,
                "batch_size": args.batch_size,
                "num_q_heads": args.num_q_heads,
                "num_kv_heads": args.num_kv_heads,
                "head_dim": args.head_dim,
                "causal": args.causal,
                "seed": args.seed,
                "warmup": args.warmup,
                "warmup_ms": args.warmup_ms,
                "graph_iters": args.graph_iters,
                "repeat": args.repeat,
                "timer_trials": args.timer_trials,
                "repeat_ms": args.repeat_ms,
                "measure_ms": args.measure_ms,
                "min_replays": args.min_replays,
                "max_replays": args.max_replays,
                "pre_capture_iters": args.pre_capture_iters,
                "suspicious_ratio": args.suspicious_ratio,
                "allow_suspicious_graph": args.allow_suspicious_graph,
                "run_flashinfer": args.run_flashinfer,
                "run_flashattention": args.run_flashattention,
                "run_petit": args.run_petit,
                "flashinfer_backend": args.flashinfer_backend,
                "flashinfer_workspace_mb": args.flashinfer_workspace_mb,
                "petit_root": str(args.petit_root),
                "petit_solution_id": args.petit_solution_id,
                "baseline_kernels": [str(p) for p in baseline_paths],
            },
            "environment": _collect_env_metadata(device),
            "rows": rows,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\n[json] wrote {args.json_out}")


if __name__ == "__main__":
    main()
