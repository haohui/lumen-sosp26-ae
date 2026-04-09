#!/usr/bin/env python3
import argparse
import csv
import importlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import sysconfig
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple


@dataclass
class GemmCase:
    m: int
    n: int
    k: int


@dataclass
class SharedInputs:
    a_mk: "torch.Tensor"
    b_kn: "torch.Tensor"
    seed: int


CASES_BF16: List[GemmCase] = [
    GemmCase(1024, 1024, 1024),
    GemmCase(2048, 2048, 2048),
    GemmCase(4096, 4096, 4096),
    GemmCase(8192, 8192, 8192),
    GemmCase(9216, 9216, 9216),
    GemmCase(14592, 14592, 14592),
    GemmCase(16384, 16384, 16384),
]
CASE_MAP_BF16: Dict[int, GemmCase] = {c.m: c for c in CASES_BF16}

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

BACKEND_LABELS = {
    "aiter": "AITER-gemm_a16w16_asm",
    "aiter_triton": "AITER-triton-gemm_a16w16",
    "triton_official": "Triton-official-matmul",
    "hipkittens_triton_v01": "HipKittens-triton_gemm_v01",
    "hipkittens_triton_v01_remap_xcd": "HipKittens-triton_gemm_v01+remap_xcd",
    "hipkittens": "HipKittens-cdna3",
    "hipblaslt": "hipBLASLt",
    "kernelbench_codex": "KernelBench-codex-ModelNew",
}

HIPBLASLT_INTERNAL_SRC = Path(__file__).resolve().with_name("hipblaslt_internal_ext.cpp")


def parse_sizes(sizes_arg: str) -> List[GemmCase]:
    if not sizes_arg:
        return list(CASES_BF16)
    out: List[GemmCase] = []
    for token in sizes_arg.split(","):
        token = token.strip()
        if not token:
            continue
        n = int(token)
        if n not in CASE_MAP_BF16:
            allowed = ",".join(str(k) for k in sorted(CASE_MAP_BF16))
            raise ValueError(f"Unsupported size {n}. Allowed sizes: {allowed}")
        out.append(CASE_MAP_BF16[n])
    if not out:
        raise ValueError("No valid sizes parsed from --sizes.")
    return out


def parse_cpu_cores(spec: str) -> List[int]:
    cores: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            parts = token.split("-", 1)
            if len(parts) != 2 or not parts[0] or not parts[1]:
                raise ValueError(f"Invalid cpu core range token: {token}")
            lo, hi = int(parts[0]), int(parts[1])
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
        count_str = token.split(":", 1)[1].strip()
        if not count_str:
            raise ValueError("Invalid --cpu-cores auto format. Use auto or auto:N")
        count = int(count_str)
        if count <= 0:
            raise ValueError("auto:N requires N > 0")
        return min(count, allowed_count)
    raise ValueError(f"Invalid auto cpu core spec: {spec}")


def _select_idle_cores(allowed_cores: List[int], count: int) -> List[int]:
    busy_ratio = _sample_cpu_busy_ratio()
    ranked = sorted(
        allowed_cores,
        key=lambda c: (busy_ratio.get(c, 1.0), c),
    )
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
    # Avoid PyTorch HIP parser conflicts when ROCR_VISIBLE_DEVICES is preset.
    if "ROCR_VISIBLE_DEVICES" in os.environ:
        os.environ.pop("ROCR_VISIBLE_DEVICES")
    return normalized


def tflops_from_ms(m: int, n: int, k: int, ms: float) -> float:
    return (2.0 * m * n * k) / ((ms / 1e3) * 1e12)


def dtype_label(dtype_arg: str) -> str:
    if dtype_arg == "bf16":
        return "BF16"
    if dtype_arg == "fp16":
        return "FP16"
    raise ValueError(f"Unsupported dtype: {dtype_arg}")


def torch_dtype_from_arg(dtype_arg: str):
    import torch

    if dtype_arg == "bf16":
        return torch.bfloat16
    if dtype_arg == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {dtype_arg}")


def build_shared_inputs(case: GemmCase, device: str, seed: int, dtype_arg: str) -> SharedInputs:
    import torch

    g = torch.Generator(device=device)
    g.manual_seed(seed)
    td = torch_dtype_from_arg(dtype_arg)
    a_mk = torch.randn((case.m, case.k), device=device, dtype=td, generator=g)
    b_kn = torch.randn((case.k, case.n), device=device, dtype=td, generator=g)
    return SharedInputs(a_mk=a_mk, b_kn=b_kn, seed=seed)


def append_result_row(
    rows: List[Dict[str, str]],
    *,
    backend: str,
    case: GemmCase,
    args: argparse.Namespace,
    timer: str,
    status: str,
    note: str,
    timing_ms: Optional[float] = None,
    tflops: Optional[float] = None,
    warmup: Optional[int] = None,
    iters: Optional[int] = None,
    graph_iters: Optional[str] = None,
) -> None:
    rows.append(
        {
            "gpu_count": "1",
            "backend": backend,
            "dtype": dtype_label(args.dtype),
            "M": str(case.m),
            "N": str(case.n),
            "K": str(case.k),
            "warmup": str(args.warmup if warmup is None else warmup),
            "iters": str(args.iters if iters is None else iters),
            "graph_iters": str(args.graph_iters if graph_iters is None else graph_iters),
            "l2_flush_mb": str(args.l2_flush_mb),
            "timer": timer,
            "timing_ms": "" if timing_ms is None else f"{timing_ms:.6f}",
            "tflops": "" if tflops is None else f"{tflops:.6f}",
            "status": status,
            "note": note,
        }
    )


def append_exception_row(
    rows: List[Dict[str, str]],
    *,
    backend: str,
    case: GemmCase,
    args: argparse.Namespace,
    exc: Exception,
) -> None:
    append_result_row(
        rows,
        backend=backend,
        case=case,
        args=args,
        timer="cuda_graph_event",
        status="error",
        note=f"{type(exc).__name__}: {exc}",
    )


def benchmark_python_callable(
    fn: Callable[[], None],
    args: argparse.Namespace,
) -> Tuple[float, int, bool, int, int, float]:
    flush_buf = alloc_l2_flush_tensor(args.l2_flush_mb, args.device)

    def pre_iter():
        flush_l2(flush_buf)

    warmup_iters = max(1, int(args.warmup))
    repeat_iters = max(1, int(args.iters))
    probe_ms = 0.0
    if args.warmup_ms > 0.0 or args.repeat_ms > 0.0:
        # Ensure JIT/first-launch overhead is paid before probe.
        import torch

        fn()
        torch.cuda.synchronize(device=args.device)
        # Use kernel-only probe (no L2 flush), because warmup/repeat conversion
        # should track steady-state per-iteration kernel runtime.
        probe_ms = _single_event_probe_ms(fn=fn, device=args.device, pre_iter=None)
        probe_ms = max(probe_ms, 1e-3)
        if args.warmup_ms > 0.0:
            warmup_iters = max(1, int(math.ceil(float(args.warmup_ms) / probe_ms)))
        if args.repeat_ms > 0.0:
            repeat_iters = max(1, int(math.ceil(float(args.repeat_ms) / probe_ms)))

    return benchmark_with_cuda_graph(
        fn=fn,
        device=args.device,
        warmup=warmup_iters,
        repeat=repeat_iters,
        graph_iters=args.graph_iters,
        pre_iter=pre_iter,
    ) + (warmup_iters, repeat_iters, probe_ms)


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
    max_abs_error = max_abs_err(actual, expected)
    max_rel_error = max_rel_err(actual, expected)
    import torch

    status = "ok" if torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol) else "error"
    return max_abs_error, max_rel_error, status


def tflops_if_valid(status: str, m: int, n: int, k: int, ms: float) -> Optional[float]:
    # Always report throughput from measured latency, even when correctness status is "error".
    # This keeps performance visibility while status still indicates numerical pass/fail.
    del status
    if ms <= 0.0:
        return None
    return tflops_from_ms(m, n, k, ms)


def require_shared_inputs(shared: SharedInputs | None, backend: str) -> SharedInputs:
    if shared is None:
        raise RuntimeError(f"shared inputs unavailable for backend={backend}")
    return shared


def alloc_l2_flush_tensor(l2_flush_mb: int, device: str):
    if l2_flush_mb <= 0:
        return None
    import torch

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


def _compile_hipkittens_kernel(
    case_n: int,
    hipcc: str,
    kernels_dir: Path,
    tk_root: Path,
) -> Tuple[str, Path]:
    src = kernels_dir / f"kernel_{case_n}.cpp"
    if not src.exists():
        raise FileNotFoundError(f"HipKittens source not found: {src}")

    module_name = f"tk_kernel_{case_n}"
    patched_src = kernels_dir / f"{module_name}__autogen.cpp"
    src_text = src.read_text(encoding="utf-8")
    marker = "PYBIND11_MODULE(tk_kernel, m)"
    if marker not in src_text:
        raise RuntimeError(f"Failed to patch HipKittens source: marker '{marker}' not found in {src}")
    patched_src.write_text(
        src_text.replace(marker, f"PYBIND11_MODULE({module_name}, m)", 1),
        encoding="utf-8",
    )

    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not ext_suffix:
        ext_suffix = ".so"
    out_name = f"{module_name}{ext_suffix}"
    out_path = kernels_dir / out_name

    pybind_includes = (
        subprocess.check_output([sys.executable, "-m", "pybind11", "--includes"], text=True)
        .strip()
        .split()
    )
    libdir = sysconfig.get_config_var("LIBDIR")
    cmd = [
        hipcc,
        str(patched_src),
        "-DKITTENS_CDNA3",
        "--offload-arch=gfx942",
        "-std=c++20",
        "-w",
        f"-I{tk_root / 'include'}",
        f"-I{tk_root / 'prototype'}",
        *pybind_includes,
        "-shared",
        "-fPIC",
        "-Rpass-analysis=kernel-resource-usage",
        "-I/opt/rocm/include/hip",
    ]
    if libdir:
        cmd.append(f"-L{libdir}")
    cmd.extend(["-lpthread", "-ldl", "-lutil", "-lm", "-o", str(out_path)])
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=str(kernels_dir))
    if proc.returncode != 0:
        detail = (proc.stdout + "\n" + proc.stderr).strip()
        raise RuntimeError(f"HipKittens compile failed:\n{detail}")
    return module_name, out_path


def _load_hipkittens_module(module_name: str, so_path: Path):
    if module_name in sys.modules:
        del sys.modules[module_name]
    importlib.invalidate_caches()
    spec = importlib.util.spec_from_file_location(module_name, str(so_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import HipKittens extension from {so_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[module_name] = mod
    return mod


def _normalize_hip_codegen(src: str) -> str:
    src = src.replace("__hip_bfloat16", "hip_bfloat16")
    src = src.replace("__float2bfloat16(", "hip_bfloat16(")
    src = src.replace("__bfloat162float(", "static_cast<float>(")
    src = src.replace(
        "#include <ATen/cuda/CUDAContext.h>",
        "#include <ATen/hip/HIPContext.h>",
    )
    src = src.replace(
        "#include <ATen/cuda/CUDAContextLight.h>",
        "#include <ATen/hip/HIPContext.h>",
    )
    src = src.replace(
        "at::cuda::getCurrentCUDAStreamMasqueradingAsHIP()",
        "at::hip::getCurrentHIPStreamMasqueradingAsCUDA()",
    )
    return src


def _load_kernelbench_modelnew(kernel_src_path: Path):
    src = kernel_src_path.read_text(encoding="utf-8")
    src = _normalize_hip_codegen(src)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(src)
        tmp_path = Path(f.name)
    module_name = f"kb_codex_modelnew_{os.getpid()}_{abs(hash(str(tmp_path)))}"
    spec = importlib.util.spec_from_file_location(module_name, str(tmp_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import ModelNew from {kernel_src_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "ModelNew"):
        raise RuntimeError(f"ModelNew not found in {kernel_src_path}")
    return mod.ModelNew


def _prepare_hipblaslt_env() -> Dict[str, str]:
    env = dict(os.environ)
    candidates = [
        "/opt/rocm/lib/hipblaslt/library",
        "/opt/rocm/lib",
        "/opt/rocm-7.2.0/lib/hipblaslt/library",
        "/opt/rocm-7.2.0/lib/llvm/lib",
        "/opt/rocm-7.2.0/lib",
        "/opt/rocm-7.1.1/lib/hipblaslt/library",
        "/opt/rocm-7.1.1/lib",
        "/opt/rocm-7.1.0/lib/hipblaslt/library",
        "/opt/rocm-7.1.0/lib/llvm/lib",
        "/opt/rocm-7.1.0/lib",
        "/usr/lib/x86_64-linux-gnu",
        "/lib/x86_64-linux-gnu",
    ]
    existing = [p for p in candidates if os.path.isdir(p)]
    prev = env.get("LD_LIBRARY_PATH", "")
    extra = ":".join(existing)
    if extra and prev:
        env["LD_LIBRARY_PATH"] = f"{extra}:{prev}"
    elif extra:
        env["LD_LIBRARY_PATH"] = extra
    return env


def _parse_hipblaslt_output(stdout: str) -> Tuple[float, float]:
    lines = [ln.rstrip("\n") for ln in stdout.splitlines()]
    header = None
    values = None
    for i, line in enumerate(lines):
        if line.startswith("[0]:"):
            header = [x.strip() for x in line[4:].split(",")]
            for j in range(i + 1, len(lines)):
                cand = lines[j].strip()
                if not cand or cand.startswith("["):
                    continue
                if "," in cand:
                    values = next(csv.reader([cand]))
                    break
            break
    if not header or not values or len(header) != len(values):
        raise RuntimeError("Failed to parse hipBLASLt benchmark output row.")
    row = {k: v.strip() for k, v in zip(header, values)}
    gflops = None
    for key in ("hipblaslt-Gflops", "gflops", "Gflops"):
        if key in row and row[key]:
            gflops = float(row[key])
            break
    if gflops is None:
        raise RuntimeError("Cannot find Gflops field in hipBLASLt output.")
    if "us" not in row or not row["us"]:
        raise RuntimeError("Cannot find us field in hipBLASLt output.")
    us = float(row["us"])
    return us / 1000.0, gflops / 1000.0


_HIPBLASLT_INTERNAL_MODULE = None


def _load_hipblaslt_internal_module():
    global _HIPBLASLT_INTERNAL_MODULE
    if _HIPBLASLT_INTERNAL_MODULE is not None:
        return _HIPBLASLT_INTERNAL_MODULE

    from torch.utils.cpp_extension import load_inline

    if not HIPBLASLT_INTERNAL_SRC.exists():
        raise FileNotFoundError(f"hipBLASLt extension source not found: {HIPBLASLT_INTERNAL_SRC}")

    os.environ.setdefault("CXX", "hipcc")
    os.environ.setdefault("MAX_JOBS", "4")
    cpp_src = HIPBLASLT_INTERNAL_SRC.read_text(encoding="utf-8")

    mod = load_inline(
        name="kb_hipblaslt_internal_ext",
        cpp_sources=cpp_src,
        functions=["hipblaslt_bf16_mm_out"],
        extra_cflags=["-O3"],
        extra_ldflags=["-L/opt/rocm/lib", "-lhipblaslt", "-Wl,-rpath,/opt/rocm/lib"],
        with_cuda=False,
        verbose=False,
    )
    _HIPBLASLT_INTERNAL_MODULE = mod
    return _HIPBLASLT_INTERNAL_MODULE


def run_aiter_case(
    case: GemmCase,
    args: argparse.Namespace,
    rows: List[Dict[str, str]],
    shared: SharedInputs,
) -> None:
    import torch
    import aiter

    m, n, k = case.m, case.n, case.k
    a = shared.a_mk
    b = shared.b_kn.t().contiguous()
    y = torch.empty((m, n), device=args.device, dtype=torch.float32)

    def fn():
        aiter.gemm_a16w16_asm(a, b, y)

    ms, total_iters, used_graph, warmup_iters, repeat_iters, probe_ms = benchmark_python_callable(fn=fn, args=args)
    ref = a @ b.t()
    max_abs_error, max_rel_error, status = evaluate_correctness(
        y, ref, args.correctness_tol, args.correctness_rtol
    )
    append_result_row(
        rows,
        backend=BACKEND_LABELS["aiter"],
        case=case,
        args=args,
        timer="cuda_graph_event" if used_graph else "cuda_event_fallback",
        status=status,
        note=(
            f"python callable benchmark; max_abs_err={max_abs_error:.6f}; max_rel_err={max_rel_error:.6f}; "
            f"atol={args.correctness_tol}; rtol={args.correctness_rtol}; shared_input_seed={shared.seed}; "
            f"target_warmup_ms={args.warmup_ms}; target_repeat_ms={args.repeat_ms}; "
            f"resolved_warmup_iters={warmup_iters}; resolved_repeat_iters={repeat_iters}; "
            f"probe_ms={probe_ms:.6f}"
        ),
        timing_ms=ms,
        tflops=tflops_if_valid(status, m, n, k, ms),
        warmup=warmup_iters,
        iters=total_iters,
    )


def run_aiter_triton_case(
    case: GemmCase,
    args: argparse.Namespace,
    rows: List[Dict[str, str]],
    shared: SharedInputs,
) -> None:
    import torch
    from aiter.ops.triton.gemm_a16w16 import gemm_a16w16 as triton_gemm_a16w16

    m, n, k = case.m, case.n, case.k
    a = shared.a_mk
    b = shared.b_kn.t().contiguous()
    y = torch.empty((m, n), device=args.device, dtype=a.dtype)

    def fn():
        triton_gemm_a16w16(a, b, dtype=a.dtype, y=y)

    ms, total_iters, used_graph, warmup_iters, repeat_iters, probe_ms = benchmark_python_callable(fn=fn, args=args)
    ref = a @ b.t()
    max_abs_error, max_rel_error, status = evaluate_correctness(
        y, ref, args.correctness_tol, args.correctness_rtol
    )
    append_result_row(
        rows,
        backend=BACKEND_LABELS["aiter_triton"],
        case=case,
        args=args,
        timer="cuda_graph_event" if used_graph else "cuda_event_fallback",
        status=status,
        note=(
            f"python callable benchmark; max_abs_err={max_abs_error:.6f}; max_rel_err={max_rel_error:.6f}; "
            f"atol={args.correctness_tol}; rtol={args.correctness_rtol}; shared_input_seed={shared.seed}; "
            f"target_warmup_ms={args.warmup_ms}; target_repeat_ms={args.repeat_ms}; "
            f"resolved_warmup_iters={warmup_iters}; resolved_repeat_iters={repeat_iters}; "
            f"probe_ms={probe_ms:.6f}"
        ),
        timing_ms=ms,
        tflops=tflops_if_valid(status, m, n, k, ms),
        warmup=warmup_iters,
        iters=total_iters,
    )


def run_triton_official_case(
    case: GemmCase,
    args: argparse.Namespace,
    rows: List[Dict[str, str]],
    shared: SharedInputs,
) -> None:
    import torch
    from triton_official_matmul import matmul_bf16 as triton_official_matmul_bf16

    if args.dtype != "bf16":
        raise RuntimeError("triton_official backend currently supports BF16 only")

    m, n, k = case.m, case.n, case.k
    a = shared.a_mk
    b = shared.b_kn
    y = torch.empty((m, n), device=args.device, dtype=a.dtype)

    def fn():
        triton_official_matmul_bf16(a, b, out=y)

    ms, total_iters, used_graph, warmup_iters, repeat_iters, probe_ms = benchmark_python_callable(fn=fn, args=args)
    ref = a @ b
    max_abs_error, max_rel_error, status = evaluate_correctness(
        y, ref, args.correctness_tol, args.correctness_rtol
    )
    append_result_row(
        rows,
        backend=BACKEND_LABELS["triton_official"],
        case=case,
        args=args,
        timer="cuda_graph_event" if used_graph else "cuda_event_fallback",
        status=status,
        note=(
            f"python callable benchmark; max_abs_err={max_abs_error:.6f}; max_rel_err={max_rel_error:.6f}; "
            f"atol={args.correctness_tol}; rtol={args.correctness_rtol}; shared_input_seed={shared.seed}; "
            f"target_warmup_ms={args.warmup_ms}; target_repeat_ms={args.repeat_ms}; "
            f"resolved_warmup_iters={warmup_iters}; resolved_repeat_iters={repeat_iters}; "
            f"probe_ms={probe_ms:.6f}"
        ),
        timing_ms=ms,
        tflops=tflops_if_valid(status, m, n, k, ms),
        warmup=warmup_iters,
        iters=total_iters,
    )


def run_hipkittens_case(
    case: GemmCase,
    args: argparse.Namespace,
    rows: List[Dict[str, str]],
    shared: SharedInputs,
) -> None:
    module_name, so_path = _compile_hipkittens_kernel(
        case_n=case.n,
        hipcc=args.hipcc,
        kernels_dir=Path(args.hipkittens_kernels_dir),
        tk_root=Path(args.hipkittens_root),
    )
    tk_kernel = _load_hipkittens_module(module_name, so_path)

    m, n, k = case.m, case.n, case.k
    a = shared.a_mk
    b = shared.b_kn.t().contiguous()
    c = shared.a_mk.new_zeros((m, n))

    def fn():
        tk_kernel.dispatch_micro(a, b, c)

    ms, total_iters, used_graph, warmup_iters, repeat_iters, probe_ms = benchmark_python_callable(fn=fn, args=args)
    ref = a @ b.t()
    max_abs_error, max_rel_error, status = evaluate_correctness(
        c, ref, args.correctness_tol, args.correctness_rtol
    )
    append_result_row(
        rows,
        backend=BACKEND_LABELS["hipkittens"],
        case=case,
        args=args,
        timer="cuda_graph_event" if used_graph else "cuda_event_fallback",
        status=status,
        note=(
            f"python callable benchmark; max_abs_err={max_abs_error:.6f}; max_rel_err={max_rel_error:.6f}; "
            f"atol={args.correctness_tol}; rtol={args.correctness_rtol}; shared_input_seed={shared.seed}; "
            f"target_warmup_ms={args.warmup_ms}; target_repeat_ms={args.repeat_ms}; "
            f"resolved_warmup_iters={warmup_iters}; resolved_repeat_iters={repeat_iters}; "
            f"probe_ms={probe_ms:.6f}"
        ),
        timing_ms=ms,
        tflops=tflops_if_valid(status, m, n, k, ms),
        warmup=warmup_iters,
        iters=total_iters,
    )


def run_hipkittens_triton_v01_case(
    case: GemmCase,
    args: argparse.Namespace,
    rows: List[Dict[str, str]],
    shared: SharedInputs,
) -> None:
    import torch
    from hipkittens_triton_gemm_v01_matmul import get_last_launch_meta
    from hipkittens_triton_gemm_v01_matmul import matmul_bf16 as hipkittens_triton_v01_matmul_bf16

    if args.dtype != "bf16":
        raise RuntimeError("hipkittens_triton_v01 backend currently supports BF16 only")

    m, n, k = case.m, case.n, case.k
    a = shared.a_mk
    b = shared.b_kn
    y = torch.empty((m, n), device=args.device, dtype=a.dtype)

    def fn():
        hipkittens_triton_v01_matmul_bf16(a, b, out=y)

    ms, total_iters, used_graph, warmup_iters, repeat_iters, probe_ms = benchmark_python_callable(fn=fn, args=args)
    ref = a @ b
    max_abs_error, max_rel_error, status = evaluate_correctness(
        y, ref, args.correctness_tol, args.correctness_rtol
    )
    best_meta = get_last_launch_meta()
    best_meta_note = ""
    if best_meta:
        best_meta_note = f"; best_config={json.dumps(best_meta, sort_keys=True)}"
    append_result_row(
        rows,
        backend=BACKEND_LABELS["hipkittens_triton_v01"],
        case=case,
        args=args,
        timer="cuda_graph_event" if used_graph else "cuda_event_fallback",
        status=status,
        note=(
            f"python callable benchmark; max_abs_err={max_abs_error:.6f}; max_rel_err={max_rel_error:.6f}; "
            f"atol={args.correctness_tol}; rtol={args.correctness_rtol}; shared_input_seed={shared.seed}; "
            f"target_warmup_ms={args.warmup_ms}; target_repeat_ms={args.repeat_ms}; "
            f"resolved_warmup_iters={warmup_iters}; resolved_repeat_iters={repeat_iters}; "
            f"probe_ms={probe_ms:.6f}{best_meta_note}"
        ),
        timing_ms=ms,
        tflops=tflops_if_valid(status, m, n, k, ms),
        warmup=warmup_iters,
        iters=total_iters,
    )


def run_hipkittens_triton_v01_remap_xcd_case(
    case: GemmCase,
    args: argparse.Namespace,
    rows: List[Dict[str, str]],
    shared: SharedInputs,
) -> None:
    import torch
    from hipkittens_triton_gemm_v01_remap_xcd_matmul import get_last_launch_meta
    from hipkittens_triton_gemm_v01_remap_xcd_matmul import matmul_bf16 as hipkittens_triton_v01_remap_xcd_matmul_bf16

    if args.dtype != "bf16":
        raise RuntimeError("hipkittens_triton_v01_remap_xcd backend currently supports BF16 only")

    m, n, k = case.m, case.n, case.k
    a = shared.a_mk
    b = shared.b_kn
    y = torch.empty((m, n), device=args.device, dtype=a.dtype)

    def fn():
        hipkittens_triton_v01_remap_xcd_matmul_bf16(a, b, out=y)

    ms, total_iters, used_graph, warmup_iters, repeat_iters, probe_ms = benchmark_python_callable(fn=fn, args=args)
    ref = a @ b
    max_abs_error, max_rel_error, status = evaluate_correctness(
        y, ref, args.correctness_tol, args.correctness_rtol
    )
    best_meta = get_last_launch_meta()
    best_meta_note = ""
    if best_meta:
        best_meta_note = f"; best_config={json.dumps(best_meta, sort_keys=True)}"
    append_result_row(
        rows,
        backend=BACKEND_LABELS["hipkittens_triton_v01_remap_xcd"],
        case=case,
        args=args,
        timer="cuda_graph_event" if used_graph else "cuda_event_fallback",
        status=status,
        note=(
            f"python callable benchmark; max_abs_err={max_abs_error:.6f}; max_rel_err={max_rel_error:.6f}; "
            f"atol={args.correctness_tol}; rtol={args.correctness_rtol}; shared_input_seed={shared.seed}; "
            f"target_warmup_ms={args.warmup_ms}; target_repeat_ms={args.repeat_ms}; "
            f"resolved_warmup_iters={warmup_iters}; resolved_repeat_iters={repeat_iters}; "
            f"probe_ms={probe_ms:.6f}{best_meta_note}"
        ),
        timing_ms=ms,
        tflops=tflops_if_valid(status, m, n, k, ms),
        warmup=warmup_iters,
        iters=total_iters,
    )


def run_kernelbench_codex_case(
    case: GemmCase,
    args: argparse.Namespace,
    rows: List[Dict[str, str]],
    shared: SharedInputs,
) -> None:
    model_cls = _load_kernelbench_modelnew(Path(args.kernelbench_kernel_src))
    model = model_cls().to(device=args.device, dtype=shared.a_mk.dtype)

    m, n, k = case.m, case.n, case.k
    a = shared.a_mk
    b = shared.b_kn

    def fn():
        _ = model(a, b)

    ms, total_iters, used_graph, warmup_iters, repeat_iters, probe_ms = benchmark_python_callable(fn=fn, args=args)

    out = model(a, b)
    ref = a @ b
    max_abs_error, max_rel_error, status = evaluate_correctness(
        out, ref, args.correctness_tol, args.correctness_rtol
    )
    append_result_row(
        rows,
        backend=BACKEND_LABELS["kernelbench_codex"],
        case=case,
        args=args,
        timer="cuda_graph_event" if used_graph else "cuda_event_fallback",
        status=status,
        note=(
            f"python callable benchmark; max_abs_err={max_abs_error:.6f}; max_rel_err={max_rel_error:.6f}; "
            f"atol={args.correctness_tol}; rtol={args.correctness_rtol}; shared_input_seed={shared.seed}; "
            f"target_warmup_ms={args.warmup_ms}; target_repeat_ms={args.repeat_ms}; "
            f"resolved_warmup_iters={warmup_iters}; resolved_repeat_iters={repeat_iters}; "
            f"probe_ms={probe_ms:.6f}"
        ),
        timing_ms=ms,
        tflops=tflops_if_valid(status, m, n, k, ms),
        warmup=warmup_iters,
        iters=total_iters,
    )


def run_hipblaslt_case_external(
    case: GemmCase,
    args: argparse.Namespace,
    rows: List[Dict[str, str]],
) -> None:
    precision = "bf16_r" if args.dtype == "bf16" else "f16_r"
    cmd = [
        args.hipblaslt_bench_bin,
        "--function",
        "matmul",
        "--precision",
        precision,
        "--compute_type",
        "f32_r",
        "--transA",
        "N",
        "--transB",
        "N",
        "-m",
        str(case.m),
        "-n",
        str(case.n),
        "-k",
        str(case.k),
        "--iters",
        str(args.iters),
        "--cold_iters",
        str(args.warmup),
        "--rotating",
        str(args.l2_flush_mb),
        "--use_gpu_timer",
        "--device",
        str(args.hipblaslt_device),
    ]
    proc = subprocess.run(
        cmd,
        env=_prepare_hipblaslt_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        detail = (proc.stdout + "\n" + proc.stderr).strip()
        append_result_row(
            rows,
            backend=BACKEND_LABELS["hipblaslt"],
            case=case,
            args=args,
            timer="hipblaslt_gpu_event",
            status="error",
            note=detail.splitlines()[-1] if detail else f"rc={proc.returncode}",
            graph_iters="N/A",
        )
        return

    ms, tflops = _parse_hipblaslt_output(proc.stdout)
    append_result_row(
        rows,
        backend=BACKEND_LABELS["hipblaslt"],
        case=case,
        args=args,
        timer="hipblaslt_gpu_event",
        status="ok",
        note="external hipblaslt-bench path (not capturable by python cuda graph)",
        timing_ms=ms,
        tflops=tflops,
        graph_iters="N/A",
    )


def run_hipblaslt_case_internal(
    case: GemmCase,
    args: argparse.Namespace,
    rows: List[Dict[str, str]],
    shared: SharedInputs,
) -> None:
    import torch

    if args.dtype != "bf16":
        raise RuntimeError("internal hipBLASLt extension only supports BF16; use --hipblaslt-mode external for FP16")

    hipblaslt_mod = _load_hipblaslt_internal_module()
    m, n, k = case.m, case.n, case.k
    a = shared.a_mk
    b = shared.b_kn
    d = torch.empty((m, n), dtype=torch.bfloat16, device=args.device)

    def fn():
        hipblaslt_mod.hipblaslt_bf16_mm_out(a, b, d)

    ms, total_iters, used_graph, warmup_iters, repeat_iters, probe_ms = benchmark_python_callable(fn=fn, args=args)
    ref = a @ b
    max_abs_error, max_rel_error, status = evaluate_correctness(
        d, ref, args.correctness_tol, args.correctness_rtol
    )
    append_result_row(
        rows,
        backend=BACKEND_LABELS["hipblaslt"],
        case=case,
        args=args,
        timer="cuda_graph_event" if used_graph else "cuda_event_fallback",
        status=status,
        note=(
            f"internal hipBLASLt extension (A@B via col-major reinterpretation); max_abs_err={max_abs_error:.6f}; "
            f"max_rel_err={max_rel_error:.6f}; atol={args.correctness_tol}; rtol={args.correctness_rtol}; "
            f"shared_input_seed={shared.seed}; target_warmup_ms={args.warmup_ms}; "
            f"target_repeat_ms={args.repeat_ms}; resolved_warmup_iters={warmup_iters}; "
            f"resolved_repeat_iters={repeat_iters}; probe_ms={probe_ms:.6f}"
        ),
        timing_ms=ms,
        tflops=tflops_if_valid(status, m, n, k, ms),
        warmup=warmup_iters,
        iters=total_iters,
    )


def run_hipblaslt_case(
    case: GemmCase,
    args: argparse.Namespace,
    rows: List[Dict[str, str]],
    shared: SharedInputs | None,
) -> None:
    if args.hipblaslt_mode == "external":
        run_hipblaslt_case_external(case, args, rows)
        return

    try:
        if shared is None:
            raise RuntimeError("internal hipBLASLt mode requires shared inputs but none were prepared")
        run_hipblaslt_case_internal(case, args, rows, shared)
    except Exception as e:
        if args.hipblaslt_mode == "internal_fallback_external":
            append_result_row(
                rows,
                backend=BACKEND_LABELS["hipblaslt"],
                case=case,
                args=args,
                timer="cuda_graph_event",
                status="error",
                note=f"internal path failed: {type(e).__name__}: {e}; fallback to external",
            )
            run_hipblaslt_case_external(case, args, rows)
            return
        raise


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_codex_kernel = (
        repo_root
        / "kb_eval_pipeline"
        / "runs"
        / "prompt_methods_bf16_20260307"
        / "generated"
        / "method_b_kernel_shape_agnostic_bf16.py"
    )

    p = argparse.ArgumentParser(
        description="Run unified GEMM baseline table with CUDA Graph timing where possible."
    )
    p.add_argument("--sizes", type=str, default="")
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument(
        "--warmup-ms",
        type=float,
        default=0.0,
        help="If >0, override --warmup by converting warmup duration (ms) to per-case iterations via single-event probe.",
    )
    p.add_argument(
        "--repeat-ms",
        type=float,
        default=0.0,
        help="If >0, override --iters by converting repeat duration (ms) to per-case iterations via single-event probe.",
    )
    p.add_argument("--graph-iters", type=int, default=1)
    p.add_argument("--l2-flush-mb", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument(
        "--hip-visible-devices",
        type=str,
        default="",
        help="Set HIP_VISIBLE_DEVICES/ROCR_VISIBLE_DEVICES for this benchmark process (e.g. '1' or '1,3').",
    )
    p.add_argument(
        "--cpu-cores",
        type=str,
        default="",
        help="Pin benchmark process to CPU cores: explicit '0-15' / '0-7,16-23' or auto-select idle cores via 'auto' / 'auto:N'.",
    )
    p.add_argument(
        "--input-seed",
        type=int,
        default=20260307,
        help="Base RNG seed for per-case shared GEMM inputs used across backends.",
    )
    p.add_argument(
        "--backends",
        type=str,
        default="aiter,hipkittens,hipblaslt,kernelbench_codex",
        help="Comma-separated from: aiter,aiter_triton,triton_official,hipkittens_triton_v01,hipkittens_triton_v01_remap_xcd,hipkittens,hipblaslt,kernelbench_codex",
    )
    p.add_argument(
        "--kernelbench-kernel-src",
        type=str,
        default=str(default_codex_kernel),
    )
    p.add_argument("--correctness-tol", type=float, default=1.0)
    p.add_argument("--correctness-rtol", type=float, default=0.02)
    p.add_argument("--out-date", type=str, default="2026-03-07")
    p.add_argument("--out-prefix", type=str, default="gemm_user_table_cudagraph_unified")
    p.add_argument(
        "--hipkittens-kernels-dir",
        type=str,
        default=str(repo_root / "third_party" / "HipKittens" / "analysis" / "bf16_gemm" / "mi325x"),
    )
    p.add_argument(
        "--hipkittens-root",
        type=str,
        default=str(repo_root / "third_party" / "HipKittens"),
    )
    p.add_argument("--hipcc", type=str, default="/opt/rocm/bin/hipcc")
    p.add_argument(
        "--hipblaslt-bench-bin",
        type=str,
        default=str(repo_root / "third_party" / "hipBLASLt" / "build_kb" / "clients" / "hipblaslt-bench"),
    )
    p.add_argument("--hipblaslt-device", type=int, default=0)
    p.add_argument(
        "--hipblaslt-mode",
        type=str,
        default="internal",
        choices=["internal", "external", "internal_fallback_external"],
        help="hipBLASLt path: internal extension call, external bench binary, or internal with external fallback.",
    )
    args = p.parse_args()

    visible_devices = apply_visible_devices(args.hip_visible_devices)
    cpu_affinity = apply_cpu_affinity(args.cpu_cores)
    if visible_devices:
        visible_count = len([x for x in visible_devices.split(",") if x])
        if args.device.startswith("cuda:"):
            local_idx = int(args.device.split(":", 1)[1])
            if local_idx < 0 or local_idx >= visible_count:
                raise ValueError(
                    f"--device {args.device} is out of range for visible devices [{visible_devices}] "
                    f"(use local index cuda:0..cuda:{visible_count - 1})"
                )
        if args.hipblaslt_mode == "external" and (args.hipblaslt_device < 0 or args.hipblaslt_device >= visible_count):
            raise ValueError(
                f"--hipblaslt-device {args.hipblaslt_device} is out of range for visible devices [{visible_devices}]"
            )

    if visible_devices:
        print(f"[config] HIP_VISIBLE_DEVICES={visible_devices}")
    if cpu_affinity:
        affinity_label = ",".join(str(c) for c in cpu_affinity)
        print(f"[config] CPU_AFFINITY={affinity_label}")

    cases = parse_sizes(args.sizes)
    selected = {x.strip() for x in args.backends.split(",") if x.strip()}
    if args.dtype == "fp16":
        for unsupported in ("hipkittens", "hipkittens_triton_v01", "hipkittens_triton_v01_remap_xcd", "kernelbench_codex"):
            if unsupported in selected:
                selected.remove(unsupported)
                print(f"[warn] backend={unsupported} is BF16-only in current setup; skipped for FP16 run")
        if "hipblaslt" in selected and args.hipblaslt_mode != "external":
            print("[warn] forcing hipBLASLt mode to external for FP16 run")
            args.hipblaslt_mode = "external"

    out_dir = repo_root / "logs" / "gemm_baselines" / args.out_date
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{args.out_prefix}.csv"
    md_path = out_dir / f"{args.out_prefix}.md"

    rows: List[Dict[str, str]] = []
    backend_order = [
        "aiter",
        "aiter_triton",
        "triton_official",
        "hipkittens_triton_v01",
        "hipkittens_triton_v01_remap_xcd",
        "hipkittens",
        "hipblaslt",
        "kernelbench_codex",
    ]
    needs_shared_inputs = any(
        (key in selected) and (key != "hipblaslt" or args.hipblaslt_mode != "external")
        for key in backend_order
    )

    def run_backend_for_case(backend_key: str, case: GemmCase, shared: SharedInputs | None) -> None:
        if backend_key == "aiter":
            run_aiter_case(case, args, rows, require_shared_inputs(shared, backend_key))
            return
        if backend_key == "aiter_triton":
            run_aiter_triton_case(case, args, rows, require_shared_inputs(shared, backend_key))
            return
        if backend_key == "triton_official":
            run_triton_official_case(case, args, rows, require_shared_inputs(shared, backend_key))
            return
        if backend_key == "hipkittens_triton_v01":
            run_hipkittens_triton_v01_case(case, args, rows, require_shared_inputs(shared, backend_key))
            return
        if backend_key == "hipkittens_triton_v01_remap_xcd":
            run_hipkittens_triton_v01_remap_xcd_case(case, args, rows, require_shared_inputs(shared, backend_key))
            return
        if backend_key == "hipkittens":
            run_hipkittens_case(case, args, rows, require_shared_inputs(shared, backend_key))
            return
        if backend_key == "hipblaslt":
            run_hipblaslt_case(case, args, rows, shared)
            return
        if backend_key == "kernelbench_codex":
            run_kernelbench_codex_case(case, args, rows, require_shared_inputs(shared, backend_key))
            return
        raise ValueError(f"Unknown backend: {backend_key}")

    for case_idx, case in enumerate(cases):
        shared_inputs = None
        if needs_shared_inputs:
            shared_inputs = build_shared_inputs(
                case=case,
                device=args.device,
                seed=args.input_seed + case_idx,
                dtype_arg=args.dtype,
            )

        for backend_key in backend_order:
            if backend_key not in selected:
                continue
            try:
                run_backend_for_case(backend_key, case, shared_inputs)
            except Exception as exc:
                append_exception_row(
                    rows,
                    backend=BACKEND_LABELS[backend_key],
                    case=case,
                    args=args,
                    exc=exc,
                )

        print(f"[done] M=N=K={case.n}")

    fields = ROW_FIELDS
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("## Unified GEMM Baselines (CUDA Graph Timing)\n\n")
        if args.warmup_ms > 0.0 or args.repeat_ms > 0.0:
            f.write(
                f"Settings: warmup_ms={args.warmup_ms}, repeat_ms={args.repeat_ms}, "
                f"graph_iters={args.graph_iters}, l2_flush_mb={args.l2_flush_mb}.\n\n"
            )
        else:
            f.write(
                f"Settings: warmup={args.warmup}, iters={args.iters}, graph_iters={args.graph_iters}, "
                f"l2_flush_mb={args.l2_flush_mb}.\n\n"
            )
        if visible_devices:
            f.write(f"Visible devices: `{visible_devices}`\n\n")
        if cpu_affinity:
            f.write(f"CPU affinity: `{','.join(str(c) for c in cpu_affinity)}`\n\n")
        f.write("| " + " | ".join(fields) + " |\n")
        f.write("|" + "|".join(["---"] * len(fields)) + "|\n")
        for row in rows:
            f.write("| " + " | ".join(row.get(k, "") for k in fields) + " |\n")

    print(csv_path)
    print(md_path)
    print(json.dumps({"rows": len(rows)}, ensure_ascii=True))


if __name__ == "__main__":
    main()
