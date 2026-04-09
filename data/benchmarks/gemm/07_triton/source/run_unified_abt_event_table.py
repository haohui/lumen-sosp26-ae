#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import sysconfig
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


@dataclass
class Inputs:
    a_mk: "torch.Tensor"
    b_nk: "torch.Tensor"
    seed: int


SIZE_LIST = [1024, 2048, 4096, 8192, 16384]
BACKEND_ORDER = [
    "hipblaslt",
    "aiter",
    "hipkittens",
    "triton",
    "kernelfalcon",
    "ksearch",
    "kernelbench",
    "cudaforge",
]
BACKEND_LABEL = {
    "hipblaslt": "HipBlasLt",
    "aiter": "AITER",
    "hipkittens": "HipKittens",
    "triton": "Triton",
    "kernelfalcon": "KernelFalcon",
    "ksearch": "KSearch",
    "kernelbench": "KernelBench",
    "cudaforge": "CUDAForge",
}


def parse_sizes(sizes_arg: str) -> List[GemmCase]:
    if not sizes_arg:
        vals = SIZE_LIST
    else:
        vals = [int(x.strip()) for x in sizes_arg.split(",") if x.strip()]
    out = [GemmCase(v, v, v) for v in vals]
    if not out:
        raise ValueError("No sizes")
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
                raise ValueError(f"Invalid core range: {token}")
            cores.update(range(lo, hi + 1))
        else:
            cores.add(int(token))
    return sorted(cores)


def _read_cpu_stat() -> Dict[int, Tuple[int, int]]:
    stats: Dict[int, Tuple[int, int]] = {}
    with open("/proc/stat", "r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith("cpu"):
                continue
            cols = line.strip().split()
            name = cols[0]
            if name == "cpu" or not name[3:].isdigit():
                continue
            idx = int(name[3:])
            nums = [int(x) for x in cols[1:]]
            if len(nums) < 8:
                continue
            user, nice, system, idle, iowait, irq, softirq, steal = nums[:8]
            busy = user + nice + system + irq + softirq + steal
            total = busy + idle + iowait
            stats[idx] = (busy, total)
    return stats


def _sample_cpu_busy_ratio(interval_s: float = 0.35) -> Dict[int, float]:
    s0 = _read_cpu_stat()
    time.sleep(max(0.05, interval_s))
    s1 = _read_cpu_stat()
    out: Dict[int, float] = {}
    for c, (b0, t0) in s0.items():
        if c not in s1:
            continue
        b1, t1 = s1[c]
        db = max(0, b1 - b0)
        dt = max(1, t1 - t0)
        out[c] = float(db) / float(dt)
    return out


def _pick_idle_cores(allowed: List[int], count: int) -> List[int]:
    busy = _sample_cpu_busy_ratio()
    ranked = sorted(allowed, key=lambda x: (busy.get(x, 1.0), x))
    return ranked[:count]


def apply_cpu_affinity(spec: str) -> List[int]:
    if not spec:
        return []
    if not hasattr(os, "sched_setaffinity"):
        raise RuntimeError("CPU affinity unsupported on this platform")
    allowed = sorted(os.sched_getaffinity(0))
    token = spec.strip().lower()
    if token.startswith("auto"):
        if token == "auto":
            count = min(16, len(allowed))
        else:
            count = int(token.split(":", 1)[1])
            count = min(count, len(allowed))
        cores = _pick_idle_cores(allowed, count)
    else:
        cores = parse_cpu_cores(spec)
        invalid = [c for c in cores if c not in allowed]
        if invalid:
            raise ValueError(f"CPU cores not allowed: {invalid}")
    os.sched_setaffinity(0, set(cores))
    return sorted(os.sched_getaffinity(0))


def apply_visible_devices(hip_visible_devices: str) -> str:
    if not hip_visible_devices:
        return ""
    normalized = ",".join(x.strip() for x in hip_visible_devices.split(",") if x.strip())
    if not normalized:
        raise ValueError("Invalid --hip-visible-devices")
    os.environ["HIP_VISIBLE_DEVICES"] = normalized
    os.environ.pop("ROCR_VISIBLE_DEVICES", None)
    return normalized


def alloc_l2_flush_tensor(l2_flush_mb: int, device: str):
    import torch

    if l2_flush_mb <= 0:
        return None
    return torch.empty(l2_flush_mb * 1024 * 1024, dtype=torch.uint8, device=device)


def flush_l2(buf):
    if buf is None:
        return
    buf.random_(0, 255)


def single_event_ms(fn: Callable[[], None], device: str) -> float:
    import torch

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(device=device)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def benchmark_events(
    fn: Callable[[], None],
    device: str,
    warmup_iters: int,
    repeat_iters: int,
    pre_iter: Optional[Callable[[], None]] = None,
) -> float:
    import torch

    for _ in range(max(1, warmup_iters)):
        if pre_iter is not None:
            pre_iter()
        fn()
    torch.cuda.synchronize(device=device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    if pre_iter is not None:
        pre_iter()
    start.record()
    for _ in range(max(1, repeat_iters)):
        fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / float(max(1, repeat_iters))


def resolve_iters(fn: Callable[[], None], device: str, warmup_ms: float, repeat_ms: float) -> Tuple[int, int, float]:
    probe = max(single_event_ms(fn, device), 1e-3)
    warmup_iters = max(1, int(math.ceil(warmup_ms / probe)))
    repeat_iters = max(1, int(math.ceil(repeat_ms / probe)))
    return warmup_iters, repeat_iters, probe


def tflops(ms: float, m: int, n: int, k: int) -> float:
    return (2.0 * m * n * k) / ((ms / 1e3) * 1e12)


def max_abs_rel(actual, ref) -> Tuple[float, float]:
    import torch

    with torch.no_grad():
        a = actual.float()
        r = ref.float()
        abs_err = float((a - r).abs().max().item())
        rel_err = float(((a - r).abs() / r.abs().clamp_min(1e-12)).max().item())
    return abs_err, rel_err


def load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_aiter_module(home_root: Path):
    aiter_repo_root = home_root / "aiter"
    if not aiter_repo_root.is_dir():
        raise FileNotFoundError(f"AITER repo not found: {aiter_repo_root}")
    if str(aiter_repo_root) not in sys.path:
        sys.path.insert(0, str(aiter_repo_root))
    stale = sys.modules.get("aiter")
    if stale is not None and getattr(stale, "__file__", None) is None:
        del sys.modules["aiter"]
    # AITER jit helper uses PyTorch cpp_extension compiler ABI check, which invokes
    # `hipcc -v` and may fail with non-zero in this environment. Skip ABI check.
    os.environ.setdefault("TORCH_DONT_CHECK_COMPILER_ABI", "1")
    importlib.invalidate_caches()
    aiter = importlib.import_module("aiter")
    if not hasattr(aiter, "gemm_a16w16_asm"):
        raise AttributeError("aiter.gemm_a16w16_asm is not available in loaded module")
    return aiter


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


def load_hipblaslt_internal_module(ext_src: Path):
    from torch.utils.cpp_extension import load_inline

    if not ext_src.exists():
        raise FileNotFoundError(f"hipBLASLt internal ext source not found: {ext_src}")

    os.environ.setdefault("CXX", "hipcc")
    os.environ.setdefault("MAX_JOBS", "4")
    cpp_src = ext_src.read_text(encoding="utf-8")
    mod = load_inline(
        name="kb_hipblaslt_internal_ext_abt_event",
        cpp_sources=cpp_src,
        functions=["hipblaslt_bf16_mm_out"],
        extra_cflags=["-O3"],
        extra_ldflags=["-L/opt/rocm/lib", "-lhipblaslt", "-Wl,-rpath,/opt/rocm/lib"],
        with_cuda=False,
        verbose=False,
    )
    return mod


def compile_hipkittens_kernel(case_n: int, hipcc: str, kernels_dir: Path, tk_root: Path, build_dir: Path) -> Tuple[str, Path]:
    src = kernels_dir / f"kernel_{case_n}.cpp"
    if not src.exists():
        raise FileNotFoundError(f"HipKittens source not found: {src}")

    module_name = f"tk_kernel_unified_{case_n}"
    build_dir.mkdir(parents=True, exist_ok=True)
    patched_src = build_dir / f"{module_name}__autogen.cpp"
    text = src.read_text(encoding="utf-8")
    marker = "PYBIND11_MODULE(tk_kernel, m)"
    if marker not in text:
        raise RuntimeError(f"Cannot patch HipKittens source marker in {src}")
    patched_src.write_text(text.replace(marker, f"PYBIND11_MODULE({module_name}, m)", 1), encoding="utf-8")

    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    out_name = f"{module_name}{ext_suffix}"
    out_path = build_dir / out_name

    pybind_includes = subprocess.check_output([sys.executable, "-m", "pybind11", "--includes"], text=True).strip().split()
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
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=str(build_dir))
    if proc.returncode != 0:
        raise RuntimeError(f"HipKittens compile failed:\n{proc.stdout}\n{proc.stderr}")
    return module_name, out_path


def load_so_module(module_name: str, so_path: Path):
    if module_name in sys.modules:
        del sys.modules[module_name]
    importlib.invalidate_caches()
    spec = importlib.util.spec_from_file_location(module_name, str(so_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import so: {so_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[module_name] = mod
    return mod


def build_inputs(case: GemmCase, device: str, seed: int):
    import torch

    g = torch.Generator(device=device)
    g.manual_seed(seed)
    a = torch.randn((case.m, case.k), dtype=torch.bfloat16, device=device, generator=g)
    b = torch.randn((case.n, case.k), dtype=torch.bfloat16, device=device, generator=g)
    return Inputs(a_mk=a, b_nk=b, seed=seed)


def benchmark_backend_case(
    backend_key: str,
    case: GemmCase,
    inp: Inputs,
    modules: dict,
    args: argparse.Namespace,
    flush_buf,
) -> dict:
    import torch

    m, n, k = case.m, case.n, case.k
    a = inp.a_mk
    b_nk = inp.b_nk
    b_kn = b_nk.t().contiguous()

    backend_name = BACKEND_LABEL[backend_key]
    row = {
        "backend": backend_name,
        "M": m,
        "N": n,
        "K": k,
        "timer": "cuda_event",
        "status": "ok",
        "timing_ms": "",
        "tflops": "",
        "warmup": "",
        "iters": "",
        "note": "",
    }

    def pre_iter():
        flush_l2(flush_buf)

    try:
        if backend_key == "hipblaslt":
            hipblaslt_mod = modules["hipblaslt"]
            out = torch.empty((m, n), dtype=torch.bfloat16, device=args.device)

            def fn():
                hipblaslt_mod.hipblaslt_bf16_mm_out(a, b_kn, out)

            warmup_iters, repeat_iters, probe = resolve_iters(fn, args.device, args.warmup_ms, args.repeat_ms)
            ms = benchmark_events(fn, args.device, warmup_iters, repeat_iters, pre_iter=pre_iter)
            ref = a @ b_nk.t()
            abs_err, rel_err = max_abs_rel(out, ref)
            row["note"] = f"max_abs_err={abs_err:.6f}; max_rel_err={rel_err:.6f}; seed={inp.seed}; probe_ms={probe:.6f}"

        elif backend_key == "aiter":
            aiter = modules["aiter"]

            out = torch.empty((m, n), dtype=torch.float32, device=args.device)

            def fn():
                aiter.gemm_a16w16_asm(a, b_nk, out)

            warmup_iters, repeat_iters, probe = resolve_iters(fn, args.device, args.warmup_ms, args.repeat_ms)
            ms = benchmark_events(fn, args.device, warmup_iters, repeat_iters, pre_iter=pre_iter)
            ref = a @ b_nk.t()
            abs_err, rel_err = max_abs_rel(out, ref)
            row["note"] = f"max_abs_err={abs_err:.6f}; max_rel_err={rel_err:.6f}; seed={inp.seed}; probe_ms={probe:.6f}"

        elif backend_key == "hipkittens":
            hk_kt_cache = modules.setdefault("hipkittens_cpp_cache", {})
            if n not in hk_kt_cache:
                module_name, so_path = compile_hipkittens_kernel(
                    case_n=n,
                    hipcc=args.hipcc,
                    kernels_dir=modules["hipkittens_kernels_dir"],
                    tk_root=modules["hipkittens_root"],
                    build_dir=modules["hipkittens_build_dir"],
                )
                hk_kt_cache[n] = load_so_module(module_name, so_path)
            tk_kernel = hk_kt_cache[n]
            out = torch.empty((m, n), dtype=torch.bfloat16, device=args.device)

            def fn():
                tk_kernel.dispatch_micro(a, b_nk, out)

            warmup_iters, repeat_iters, probe = resolve_iters(fn, args.device, args.warmup_ms, args.repeat_ms)
            ms = benchmark_events(fn, args.device, warmup_iters, repeat_iters, pre_iter=pre_iter)
            ref = a @ b_nk.t()
            abs_err, rel_err = max_abs_rel(out, ref)
            row["note"] = f"max_abs_err={abs_err:.6f}; max_rel_err={rel_err:.6f}; seed={inp.seed}; probe_ms={probe:.6f}"

        elif backend_key == "triton":
            triton_official = modules["triton_official"]
            out = torch.empty((m, n), dtype=torch.bfloat16, device=args.device)

            def fn():
                triton_official.matmul_bf16(a, b_kn, out=out)

            warmup_iters, repeat_iters, probe = resolve_iters(fn, args.device, args.warmup_ms, args.repeat_ms)
            ms = benchmark_events(fn, args.device, warmup_iters, repeat_iters, pre_iter=pre_iter)
            ref = a @ b_nk.t()
            abs_err, rel_err = max_abs_rel(out, ref)
            row["note"] = f"max_abs_err={abs_err:.6f}; max_rel_err={rel_err:.6f}; seed={inp.seed}; probe_ms={probe:.6f}"

        elif backend_key == "kernelfalcon":
            kf = modules["kernelfalcon"]
            out = torch.empty((m, n), dtype=torch.bfloat16, device=args.device)

            def fn():
                y = kf.kernel_function(a, b_nk, transpose_b=True)
                out.copy_(y)

            warmup_iters, repeat_iters, probe = resolve_iters(fn, args.device, args.warmup_ms, args.repeat_ms)
            ms = benchmark_events(fn, args.device, warmup_iters, repeat_iters, pre_iter=pre_iter)
            ref = a @ b_nk.t()
            abs_err, rel_err = max_abs_rel(out, ref)
            row["note"] = f"max_abs_err={abs_err:.6f}; max_rel_err={rel_err:.6f}; seed={inp.seed}; probe_ms={probe:.6f}"

        elif backend_key == "ksearch":
            ks_model = modules["ksearch_model"]
            out = torch.empty((m, n), dtype=torch.bfloat16, device=args.device)

            def fn():
                y = ks_model(a, b_kn)
                out.copy_(y)

            warmup_iters, repeat_iters, probe = resolve_iters(fn, args.device, args.warmup_ms, args.repeat_ms)
            ms = benchmark_events(fn, args.device, warmup_iters, repeat_iters, pre_iter=pre_iter)
            ref = a @ b_nk.t()
            abs_err, rel_err = max_abs_rel(out, ref)
            row["note"] = f"max_abs_err={abs_err:.6f}; max_rel_err={rel_err:.6f}; seed={inp.seed}; probe_ms={probe:.6f}"

        elif backend_key == "kernelbench":
            kb_model = modules["kernelbench_model"]
            out = torch.empty((m, n), dtype=torch.bfloat16, device=args.device)

            def fn():
                y = kb_model(a, b_kn)
                out.copy_(y)

            warmup_iters, repeat_iters, probe = resolve_iters(fn, args.device, args.warmup_ms, args.repeat_ms)
            ms = benchmark_events(fn, args.device, warmup_iters, repeat_iters, pre_iter=pre_iter)
            ref = a @ b_nk.t()
            abs_err, rel_err = max_abs_rel(out, ref)
            row["note"] = f"max_abs_err={abs_err:.6f}; max_rel_err={rel_err:.6f}; seed={inp.seed}; probe_ms={probe:.6f}"

        elif backend_key == "cudaforge":
            cf_model = modules["cudaforge_model"]
            out = torch.empty((m, n), dtype=torch.bfloat16, device=args.device)

            def fn():
                y = cf_model(a, b_kn)
                out.copy_(y)

            warmup_iters, repeat_iters, probe = resolve_iters(fn, args.device, args.warmup_ms, args.repeat_ms)
            ms = benchmark_events(fn, args.device, warmup_iters, repeat_iters, pre_iter=pre_iter)
            ref = a @ b_nk.t()
            abs_err, rel_err = max_abs_rel(out, ref)
            row["note"] = f"max_abs_err={abs_err:.6f}; max_rel_err={rel_err:.6f}; seed={inp.seed}; probe_ms={probe:.6f}"

        else:
            raise ValueError(f"Unknown backend: {backend_key}")

        row["warmup"] = warmup_iters
        row["iters"] = repeat_iters
        row["timing_ms"] = f"{ms:.6f}"
        row["tflops"] = f"{tflops(ms, m, n, k):.6f}"
        return row
    except Exception as e:
        row["status"] = "error"
        row["note"] = f"{type(e).__name__}: {e}"
        return row


def write_long_csv(path: Path, rows: List[dict]) -> None:
    fields = ["backend", "M", "N", "K", "warmup", "iters", "timer", "timing_ms", "tflops", "status", "note"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def build_pivot(rows: List[dict], sizes: List[int]) -> List[Tuple[str, List[str]]]:
    by_backend: Dict[str, Dict[int, str]] = {}
    for r in rows:
        b = r["backend"]
        m = int(r["M"])
        val = r["timing_ms"] if r["status"] == "ok" and r["timing_ms"] else "ERR"
        by_backend.setdefault(b, {})[m] = val
    out = []
    for k in BACKEND_ORDER:
        b = BACKEND_LABEL[k]
        vals = [by_backend.get(b, {}).get(s, "NA") for s in sizes]
        out.append((b, vals))
    return out


def write_pivot_md(path: Path, pivot_rows: List[Tuple[str, List[str]]], sizes: List[int], args: argparse.Namespace) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("## Unified GEMM Clean Table\n\n")
        f.write("All rows use: `A@B^T`, `cuda_event`, `warmup=200ms`, `repeat=1s`.\n\n")
        f.write(f"Device: `{args.device}`, HIP_VISIBLE_DEVICES=`{args.hip_visible_devices}`\n\n")
        header = "| iT | " + " | ".join(str(s) for s in sizes) + " |\n"
        sep = "|---|" + "|".join(["---"] * len(sizes)) + "|\n"
        f.write(header)
        f.write(sep)
        for name, vals in pivot_rows:
            f.write("| " + name + " | " + " | ".join(vals) + " |\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Unified ABt + cuda_event benchmark table")
    p.add_argument("--sizes", type=str, default="1024,2048,4096,8192,16384")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--hip-visible-devices", type=str, default="7")
    p.add_argument("--cpu-cores", type=str, default="auto:16")
    p.add_argument("--warmup-ms", type=float, default=200.0)
    p.add_argument("--repeat-ms", type=float, default=1000.0)
    p.add_argument("--l2-flush-mb", type=int, default=256)
    p.add_argument("--input-seed", type=int, default=20260402)
    p.add_argument("--out-prefix", type=str, default="gemm_clean_abt_event_unified_gpu7_emptycard_emptycore_warm200ms_repeat1s")
    p.add_argument("--backends", type=str, default="hipblaslt,aiter,hipkittens,triton,kernelfalcon,ksearch,kernelbench,cudaforge")
    p.add_argument("--hipcc", type=str, default="/opt/rocm/bin/hipcc")
    args = p.parse_args()

    sizes = [c.m for c in parse_sizes(args.sizes)]
    cases = [GemmCase(s, s, s) for s in sizes]

    # Important: avoid compiling unsupported gfx targets when importing ksearch extension.
    os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx942")
    os.environ.setdefault("MAX_JOBS", "8")

    visible = apply_visible_devices(args.hip_visible_devices)
    affinity = apply_cpu_affinity(args.cpu_cores)
    if visible:
        print(f"[config] HIP_VISIBLE_DEVICES={visible}")
    if affinity:
        print(f"[config] CPU_AFFINITY={','.join(str(c) for c in affinity)}")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is not available")

    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]  # .../opt_kernel/gemm-openai
    kb_root = repo_root.parents[1]      # .../kernel_benchmark
    home_root = kb_root.parent          # .../home/<user>
    if str(home_root) not in sys.path:
        sys.path.insert(0, str(home_root))

    modules: dict = {}

    # Load modules once (compile-heavy paths happen here).
    print("[load] modules start")
    modules["triton_official"] = load_module_from_path("triton_official_matmul_abt_event", repo_root / "07_triton" / "source" / "triton_official_matmul.py")
    modules["kernelfalcon"] = load_module_from_path("kernelfalcon_best_abt_event", repo_root / "03_kernelfalcon" / "best_kernel.py")

    kb01 = load_module_from_path("kernelbench01_best_abt_event", repo_root / "01_kernelbench" / "best_kernel.py")
    modules["kernelbench_model"] = kb01.ModelNew().to(device=args.device, dtype=torch.bfloat16).eval()

    cf02 = load_module_from_path("cudaforge02_best_abt_event", repo_root / "02_cudaforge" / "best_kernel.py")
    modules["cudaforge_model"] = cf02.ModelNew().to(device=args.device, dtype=torch.bfloat16).eval()

    ks04 = load_module_from_path("ksearch04_best_abt_event", repo_root / "04_ksearch" / "best_kernel.py")
    modules["ksearch_model"] = ks04.ModelNew().to(device=args.device, dtype=torch.bfloat16).eval()
    modules["aiter"] = load_aiter_module(home_root)

    modules["hipkittens_kernels_dir"] = kb_root / "third_party" / "HipKittens" / "analysis" / "bf16_gemm" / "mi325x"
    modules["hipkittens_root"] = kb_root / "third_party" / "HipKittens"
    modules["hipkittens_build_dir"] = repo_root / "07_triton" / "build_hipkittens_unified"

    hipblaslt_src = repo_root / "07_triton" / "source" / "hipblaslt_internal_ext.cpp"
    if not hipblaslt_src.exists():
        hipblaslt_src = kb_root / "scripts" / "hipblaslt_internal_ext.cpp"
    modules["hipblaslt"] = load_hipblaslt_internal_module(hipblaslt_src)
    print("[load] modules done")

    selected = [x.strip() for x in args.backends.split(",") if x.strip()]
    for k in selected:
        if k not in BACKEND_LABEL:
            raise ValueError(f"Unknown backend key: {k}")

    rows: List[dict] = []
    flush_buf = alloc_l2_flush_tensor(args.l2_flush_mb, args.device)

    for i, case in enumerate(cases):
        inp = build_inputs(case, args.device, args.input_seed + i)
        for bk in BACKEND_ORDER:
            if bk not in selected:
                continue
            print(f"[run] backend={BACKEND_LABEL[bk]} m={case.m}")
            row = benchmark_backend_case(bk, case, inp, modules, args, flush_buf)
            rows.append(row)
            print(f"[done] backend={BACKEND_LABEL[bk]} m={case.m} status={row['status']} ms={row['timing_ms']}")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = repo_root / "results_clean"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{args.out_prefix}_{ts}.csv"
    md_path = out_dir / f"{args.out_prefix}_{ts}.md"
    latest_csv = repo_root / f"{args.out_prefix}.csv"
    latest_md = repo_root / f"{args.out_prefix}.md"

    write_long_csv(csv_path, rows)
    pivot = build_pivot(rows, sizes)
    write_pivot_md(md_path, pivot, sizes, args)
    latest_csv.write_text(csv_path.read_text(encoding="utf-8"), encoding="utf-8")
    latest_md.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"[saved] {csv_path}")
    print(f"[saved] {md_path}")
    print(f"[saved] {latest_csv}")
    print(f"[saved] {latest_md}")


if __name__ == "__main__":
    main()
