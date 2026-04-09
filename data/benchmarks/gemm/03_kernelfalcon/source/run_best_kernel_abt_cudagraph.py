#!/usr/bin/env python3
import argparse
import csv
import importlib.util
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple


@dataclass
class GemmCase:
    m: int
    n: int
    k: int


ROW_FIELDS = [
    "gpu_count",
    "backend",
    "dtype",
    "M",
    "N",
    "K",
    "warmup",
    "iters",
    "graph_iters",
    "l2_flush_mb",
    "timer",
    "timing_ms",
    "tflops",
    "status",
    "note",
]


def parse_sizes(sizes_arg: str) -> List[GemmCase]:
    out: List[GemmCase] = []
    for token in sizes_arg.split(","):
        token = token.strip()
        if not token:
            continue
        n = int(token)
        out.append(GemmCase(n, n, n))
    if not out:
        raise ValueError("No valid sizes parsed from --sizes")
    return out


def parse_cpu_cores(spec: str) -> List[int]:
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


def _read_cpu_stat() -> Dict[int, Tuple[int, int]]:
    stats: Dict[int, Tuple[int, int]] = {}
    with open("/proc/stat", "r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith("cpu"):
                continue
            fields = line.strip().split()
            name = fields[0]
            if name == "cpu" or not name[3:].isdigit():
                continue
            cpu_idx = int(name[3:])
            nums = [int(x) for x in fields[1:]]
            if len(nums) < 8:
                continue
            user, nice, system, idle, iowait, irq, softirq, steal = nums[:8]
            busy = user + nice + system + irq + softirq + steal
            total = busy + idle + iowait
            stats[cpu_idx] = (busy, total)
    return stats


def _sample_cpu_busy_ratio(interval_s: float = 0.35) -> Dict[int, float]:
    start = _read_cpu_stat()
    time.sleep(max(0.05, interval_s))
    end = _read_cpu_stat()
    ratios: Dict[int, float] = {}
    for cpu_idx, (busy0, total0) in start.items():
        if cpu_idx not in end:
            continue
        busy1, total1 = end[cpu_idx]
        dbusy = max(0, busy1 - busy0)
        dtotal = max(1, total1 - total0)
        ratios[cpu_idx] = float(dbusy) / float(dtotal)
    return ratios


def _parse_auto_cpu_core_count(spec: str, allowed_count: int) -> int:
    token = spec.strip().lower()
    if token == "auto":
        return min(16, allowed_count)
    if token.startswith("auto:"):
        count = int(token.split(":", 1)[1].strip())
        if count <= 0:
            raise ValueError("auto:N requires N > 0")
        return min(count, allowed_count)
    raise ValueError(f"Invalid auto cpu core spec: {spec}")


def _select_idle_cores(allowed_cores: List[int], count: int) -> List[int]:
    busy_ratio = _sample_cpu_busy_ratio()
    ranked = sorted(allowed_cores, key=lambda c: (busy_ratio.get(c, 1.0), c))
    return ranked[:count]


def apply_cpu_affinity(spec: str) -> List[int]:
    if not spec:
        return []
    if not hasattr(os, "sched_setaffinity"):
        raise RuntimeError("CPU affinity is not supported on this platform")
    allowed = sorted(os.sched_getaffinity(0))
    token = spec.strip().lower()
    if token.startswith("auto"):
        count = _parse_auto_cpu_core_count(token, len(allowed))
        cores = _select_idle_cores(allowed, count)
    else:
        cores = parse_cpu_cores(spec)
        invalid = [c for c in cores if c not in allowed]
        if invalid:
            raise ValueError(f"Requested CPU cores not allowed in this runtime: {invalid}")
    os.sched_setaffinity(0, set(cores))
    return sorted(os.sched_getaffinity(0))


def apply_visible_devices(hip_visible_devices: str) -> str:
    if not hip_visible_devices:
        return ""
    normalized = ",".join(x.strip() for x in hip_visible_devices.split(",") if x.strip())
    if not normalized:
        raise ValueError("Invalid --hip-visible-devices value")
    os.environ["HIP_VISIBLE_DEVICES"] = normalized
    if "ROCR_VISIBLE_DEVICES" in os.environ:
        os.environ.pop("ROCR_VISIBLE_DEVICES")
    return normalized


def alloc_l2_flush_tensor(l2_flush_mb: int, device: str):
    import torch

    if l2_flush_mb <= 0:
        return None
    num_bytes = l2_flush_mb * 1024 * 1024
    return torch.empty(num_bytes, dtype=torch.uint8, device=device)


def flush_l2(buf):
    if buf is None:
        return
    buf.random_(0, 255)


def benchmark_with_events(
    fn: Callable[[], None],
    device: str,
    warmup: int,
    repeat: int,
    pre_iter: Callable[[], None] | None = None,
) -> Tuple[float, int]:
    import torch

    for _ in range(max(1, warmup)):
        if pre_iter is not None:
            pre_iter()
        fn()
    torch.cuda.synchronize(device=device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    if pre_iter is not None:
        pre_iter()
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / float(max(1, repeat)), repeat


def _single_event_probe_ms(
    fn: Callable[[], None],
    device: str,
    pre_iter: Callable[[], None] | None = None,
) -> float:
    import torch

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    if pre_iter is not None:
        pre_iter()
    torch.cuda.synchronize(device=device)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def benchmark_with_cuda_graph(
    fn: Callable[[], None],
    device: str,
    warmup: int,
    repeat: int,
    graph_iters: int,
    pre_iter: Callable[[], None] | None = None,
) -> Tuple[float, int, bool]:
    import torch

    for _ in range(max(1, warmup)):
        if pre_iter is not None:
            pre_iter()
        fn()
    torch.cuda.synchronize(device=device)

    try:
        graph = torch.cuda.CUDAGraph()
        capture_stream = torch.cuda.Stream(device=device)
        current_stream = torch.cuda.current_stream(device=device)
        capture_stream.wait_stream(current_stream)
        with torch.cuda.stream(capture_stream):
            for _ in range(3):
                fn()
            with torch.cuda.graph(graph):
                for _ in range(max(1, graph_iters)):
                    fn()
        current_stream.wait_stream(capture_stream)
        torch.cuda.synchronize(device=device)

        num_replays = int(math.ceil(float(repeat) / float(max(1, graph_iters))))
        total_iters = max(1, num_replays * max(1, graph_iters))
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        if pre_iter is not None:
            pre_iter()
        start.record()
        for _ in range(num_replays):
            graph.replay()
        end.record()
        end.synchronize()
        med_ms = float(start.elapsed_time(end)) / float(total_iters)
        probe_ms = _single_event_probe_ms(fn=fn, device=device, pre_iter=None)
        if probe_ms > 0.0 and med_ms < 0.25 * probe_ms:
            ms, total_iters = benchmark_with_events(
                fn=fn,
                device=device,
                warmup=warmup,
                repeat=repeat,
                pre_iter=pre_iter,
            )
            return ms, total_iters, False
        return med_ms, total_iters, True
    except RuntimeError:
        ms, total_iters = benchmark_with_events(
            fn=fn,
            device=device,
            warmup=warmup,
            repeat=repeat,
            pre_iter=pre_iter,
        )
        return ms, total_iters, False


def benchmark_python_callable(fn: Callable[[], None], args: argparse.Namespace) -> Tuple[float, int, bool, int, int, float]:
    import torch

    flush_buf = alloc_l2_flush_tensor(args.l2_flush_mb, args.device)

    def pre_iter():
        flush_l2(flush_buf)

    warmup_iters = max(1, int(args.warmup))
    repeat_iters = max(1, int(args.iters))
    probe_ms = 0.0
    if args.warmup_ms > 0.0 or args.repeat_ms > 0.0:
        fn()
        torch.cuda.synchronize(device=args.device)
        probe_ms = _single_event_probe_ms(fn=fn, device=args.device, pre_iter=None)
        probe_ms = max(probe_ms, 1e-3)
        if args.warmup_ms > 0.0:
            warmup_iters = max(1, int(math.ceil(float(args.warmup_ms) / probe_ms)))
        if args.repeat_ms > 0.0:
            repeat_iters = max(1, int(math.ceil(float(args.repeat_ms) / probe_ms)))

    ms, total_iters = benchmark_with_events(
        fn=fn,
        device=args.device,
        warmup=warmup_iters,
        repeat=repeat_iters,
        pre_iter=pre_iter,
    )
    return ms, total_iters, False, warmup_iters, repeat_iters, probe_ms


def tflops_from_ms(m: int, n: int, k: int, ms: float) -> float:
    return (2.0 * m * n * k) / ((ms / 1e3) * 1e12)


def max_abs_err(actual, expected) -> float:
    import torch

    with torch.no_grad():
        return float((actual.float() - expected.float()).abs().max().item())


def max_rel_err(actual, expected, eps: float = 1e-12) -> float:
    import torch

    with torch.no_grad():
        actual_f = actual.float()
        expected_f = expected.float()
        denom = expected_f.abs().clamp_min(eps)
        rel = (actual_f - expected_f).abs() / denom
        return float(rel.max().item())


def evaluate_correctness(actual, expected, atol: float, rtol: float) -> Tuple[float, float, str]:
    import torch

    max_abs_error = max_abs_err(actual, expected)
    max_rel_error = max_rel_err(actual, expected)
    status = "ok" if torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol) else "error"
    return max_abs_error, max_rel_error, status


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(f"kernelfalcon_best_{int(time.time())}", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import kernel module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_fn(mod, a, b, out):
    import triton

    if hasattr(mod, "_matmul_kernel"):
        M, K = a.shape
        N = b.shape[0]
        if a.dtype == a.new_empty((), dtype=a.dtype).half().dtype:
            dtype_id = 0
        elif str(a.dtype).endswith("bfloat16"):
            dtype_id = 1
        else:
            dtype_id = 2

        def grid(meta):
            return (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)

        def fn():
            mod._matmul_kernel[grid](
                a,
                b,
                out,
                M,
                N,
                K,
                dtype_id,
                a.stride(0),
                a.stride(1),
                b.stride(1),
                b.stride(0),
                out.stride(0),
                out.stride(1),
            )

        return fn

    if not hasattr(mod, "kernel_function"):
        raise RuntimeError("module has neither _matmul_kernel nor kernel_function")

    def fn():
        y = mod.kernel_function(a, b)
        out.copy_(y)

    return fn


def rows_to_markdown(rows: List[Dict[str, str]]) -> str:
    if not rows:
        return ""
    header = "| " + " | ".join(ROW_FIELDS) + " |"
    sep = "|" + "|".join(["---"] * len(ROW_FIELDS)) + "|"
    lines = [header, sep]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(k, "")) for k in ROW_FIELDS) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark KernelFalcon best kernel in A@B^T mode with CUDA event timing")
    p.add_argument("--kernel", type=str, default="", help="Path to best_kernel.py")
    p.add_argument("--sizes", type=str, default="1024,2048,4096,8192,16384")
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16"])
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--warmup-ms", type=float, default=200.0)
    p.add_argument("--repeat-ms", type=float, default=1000.0)
    p.add_argument("--graph-iters", type=int, default=1)
    p.add_argument("--l2-flush-mb", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--hip-visible-devices", type=str, default="7")
    p.add_argument("--cpu-cores", type=str, default="auto:16")
    p.add_argument("--input-seed", type=int, default=20260402)
    p.add_argument("--correctness-tol", type=float, default=1.0)
    p.add_argument("--correctness-rtol", type=float, default=0.02)
    p.add_argument("--out-prefix", type=str, default="gemm_kernelfalcon_best_abt_event_gpu7_emptycard_emptycore_warm200ms_repeat1s")
    args = p.parse_args()

    script_path = Path(__file__).resolve()
    repo_dir = script_path.parents[1]
    results_dir = repo_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    kernel_path = Path(args.kernel).resolve() if args.kernel else (repo_dir / "best_kernel.py")
    if not kernel_path.exists():
        raise FileNotFoundError(f"kernel not found: {kernel_path}")

    visible_devices = apply_visible_devices(args.hip_visible_devices)
    cpu_affinity = apply_cpu_affinity(args.cpu_cores)
    if visible_devices:
        print(f"[config] HIP_VISIBLE_DEVICES={visible_devices}")
    if cpu_affinity:
        print(f"[config] CPU_AFFINITY={','.join(str(c) for c in cpu_affinity)}")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is not available")

    cases = parse_sizes(args.sizes)
    mod = load_module(kernel_path)
    rows: List[Dict[str, str]] = []

    for idx, case in enumerate(cases):
        m, n, k = case.m, case.n, case.k
        seed = args.input_seed + idx
        g = torch.Generator(device=args.device)
        g.manual_seed(seed)
        a = torch.randn((m, k), device=args.device, dtype=torch.bfloat16, generator=g)
        b = torch.randn((n, k), device=args.device, dtype=torch.bfloat16, generator=g)
        out = torch.empty((m, n), device=args.device, dtype=torch.bfloat16)

        fn = make_fn(mod, a, b, out)

        try:
            ms, total_iters, used_graph, warmup_iters, repeat_iters, probe_ms = benchmark_python_callable(fn=fn, args=args)
            fn()
            torch.cuda.synchronize(device=args.device)
            ref = a @ b.t()
            max_abs_error, max_rel_error, status = evaluate_correctness(out, ref, args.correctness_tol, args.correctness_rtol)
            note = (
                f"kernelfalcon_best A@B^T; max_abs_err={max_abs_error:.6f}; max_rel_err={max_rel_error:.6f}; "
                f"atol={args.correctness_tol}; rtol={args.correctness_rtol}; input_seed={seed}; "
                f"target_warmup_ms={args.warmup_ms}; target_repeat_ms={args.repeat_ms}; "
                f"resolved_warmup_iters={warmup_iters}; resolved_repeat_iters={repeat_iters}; probe_ms={probe_ms:.6f}"
            )
            tflops = tflops_from_ms(m, n, k, ms) if ms > 0.0 else None
            rows.append(
                {
                    "gpu_count": "1",
                    "backend": "KernelFalcon-best(A@B^T)",
                    "dtype": "BF16",
                    "M": str(m),
                    "N": str(n),
                    "K": str(k),
                    "warmup": str(warmup_iters),
                    "iters": str(total_iters),
                    "graph_iters": str(args.graph_iters),
                    "l2_flush_mb": str(args.l2_flush_mb),
                    "timer": "cuda_event",
                    "timing_ms": f"{ms:.6f}",
                    "tflops": "" if tflops is None else f"{tflops:.6f}",
                    "status": status,
                    "note": note,
                }
            )
            print(f"[ok] m=n=k={m}: {ms:.6f} ms, {tflops:.6f} TFLOPS, status={status}")
        except Exception as e:
            rows.append(
                {
                    "gpu_count": "1",
                    "backend": "KernelFalcon-best(A@B^T)",
                    "dtype": "BF16",
                    "M": str(m),
                    "N": str(n),
                    "K": str(k),
                    "warmup": str(args.warmup),
                    "iters": str(args.iters),
                    "graph_iters": str(args.graph_iters),
                    "l2_flush_mb": str(args.l2_flush_mb),
                    "timer": "cuda_event",
                    "timing_ms": "",
                    "tflops": "",
                    "status": "error",
                    "note": f"{type(e).__name__}: {e}",
                }
            )
            print(f"[error] m=n=k={m}: {type(e).__name__}: {e}")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_path = results_dir / f"{args.out_prefix}_{ts}.csv"
    md_path = results_dir / f"{args.out_prefix}_{ts}.md"

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ROW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    md_path.write_text(rows_to_markdown(rows), encoding="utf-8")

    latest_csv = repo_dir / f"{args.out_prefix}.csv"
    latest_md = repo_dir / f"{args.out_prefix}.md"
    latest_csv.write_text(csv_path.read_text(encoding="utf-8"), encoding="utf-8")
    latest_md.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"[saved] csv={csv_path}")
    print(f"[saved] md={md_path}")
    print(f"[saved] latest_csv={latest_csv}")
    print(f"[saved] latest_md={latest_md}")


if __name__ == "__main__":
    main()
