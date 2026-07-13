#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Benchmark local Lumen flash-attention ablations and write a CSV."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from types import ModuleType

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPO_ROOT / "scripts" / "benchmark"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from backends import load_module  # noqa: E402


def _enable_attn_opt() -> None:
    try:
        from avelang import knobs as avelang_knobs

        avelang_knobs.amdgpu.enable_attn_opt = True
    except Exception:
        os.environ["ENABLE_ATTN_OPT"] = "1"


ABLATIONS = [
    ("Naive", "attn_01_naive.py", False),
    ("Transpose V", "attn_02_trans_v.py", False),
    ("+Async memcpy", "attn_03_async_lds.py", True),
    ("+Bank conflict", "attn_04_shm_bank_conflicts.py", True),
    ("+Pipeline+WS", "attn_05_wg_specialization.py", True),
    ("+Inst. schedule (All)", "attn_06_inst_scheduling.py", True),
]
SEQ_LENS = [1024, 2048, 4096, 8192, 16384]
BATCH_SIZE = 16
Q_HEADS = 8
KV_HEADS = 1
HEAD_DIM = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark local Lumen flash-attention ablations and write a CSV."
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="CSV file to create.",
    )
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=SEQ_LENS,
        help=f"Sequence lengths to benchmark. Default: {' '.join(map(str, SEQ_LENS))}",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260314)
    return parser.parse_args()


def build_inputs(seq_len: int, seed: int) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed + seq_len)
    shape = (BATCH_SIZE, seq_len)
    q = torch.randn(
        (*shape, Q_HEADS, HEAD_DIM),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ).view(-1, Q_HEADS, HEAD_DIM)
    k = torch.randn(
        (*shape, KV_HEADS, HEAD_DIM),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ).view(-1, KV_HEADS, HEAD_DIM)
    v = torch.randn(
        (*shape, KV_HEADS, HEAD_DIM),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ).view(-1, KV_HEADS, HEAD_DIM)
    seq_ptr = torch.arange(BATCH_SIZE + 1, device="cuda", dtype=torch.int32).mul_(
        seq_len
    )
    return q, k, v, seq_ptr


def attention_tflops(seq_len: int, mean_ms: float) -> float:
    if mean_ms <= 0.0:
        return float("nan")
    flops = 2.0 * BATCH_SIZE * Q_HEADS * seq_len * seq_len * HEAD_DIM * 2.0
    return flops / (mean_ms * 1.0e-3) / 1.0e12


def launch_avelang_kernel(
    module: ModuleType,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_ptr: torch.Tensor,
    out: torch.Tensor,
    seq_len: int,
    mirror_q_tiles: bool,
) -> None:
    row_tiles = (seq_len + module.BLOCK_ROWS - 1) // module.BLOCK_ROWS
    physical_tiles = (row_tiles + 1) // 2 if mirror_q_tiles else row_tiles
    module._flash_attn_packed_kernel[
        lambda: ((physical_tiles, Q_HEADS, BATCH_SIZE), (module.THREADS, 1, 1))
    ](
        q,
        k,
        v,
        seq_ptr,
        out,
        q.shape[0],
        BATCH_SIZE,
        num_warps=module.NUM_WARPS,
    )


def benchmark(
    module: ModuleType,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_ptr: torch.Tensor,
    out: torch.Tensor,
    seq_len: int,
    mirror_q_tiles: bool,
    *,
    warmup: int,
    repeat: int,
) -> float:
    """Return mean kernel time using CUDA events around graph replay."""
    for _ in range(max(0, warmup)):
        launch_avelang_kernel(module, q, k, v, seq_ptr, out, seq_len, mirror_q_tiles)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    with torch.cuda.graph(graph, stream=capture_stream):
        launch_avelang_kernel(module, q, k, v, seq_ptr, out, seq_len, mirror_q_tiles)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(max(1, repeat)):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / max(1, repeat)


def main() -> None:
    _enable_attn_opt()
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the attention ablation benchmark.")
    if any(seq_len <= 0 for seq_len in args.seq_lens):
        raise ValueError("sequence lengths must be positive")

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lumen_dir = REPO_ROOT / "datasets" / "inference" / "attention" / "lumen"
    inputs = {
        seq_len: build_inputs(seq_len, args.seed) for seq_len in args.seq_lens
    }

    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=["name", "seq_len", "mean_ms", "tflops"],
        )
        writer.writeheader()

        for name, filename, mirror_q_tiles in ABLATIONS:
            print(f"\n=== {name}: {filename} ===", flush=True)
            module = load_module(lumen_dir / filename)

            for seq_len in args.seq_lens:
                q, k, v, seq_ptr = inputs[seq_len]
                out = torch.empty_like(q)
                mean_ms = benchmark(
                    module,
                    q,
                    k,
                    v,
                    seq_ptr,
                    out,
                    seq_len,
                    mirror_q_tiles,
                    warmup=args.warmup,
                    repeat=args.repeat,
                )
                tflops = attention_tflops(seq_len, mean_ms)
                writer.writerow(
                    {
                        "name": name,
                        "seq_len": seq_len,
                        "mean_ms": f"{mean_ms:.6f}",
                        "tflops": f"{tflops:.6f}",
                    }
                )
                output_file.flush()
                print(
                    f"seq_len={seq_len} mean_ms={mean_ms:.6f} "
                    f"tflops={tflops:.6f}",
                    flush=True,
                )

    print(f"Results CSV: {output_path}")


if __name__ == "__main__":
    main()
