# Benchmark Harness README

This directory contains the benchmark entry points used to time OpenAI-baseline kernels for:

- GEMM
- Attention
- MoE

All timing scripts use CUDA Graph replay timing with configurable warmup/repeat windows.

## Files

- `setup_env.sh`: Environment setup and pinned-version checks. Also rebuilds missing HipKittens `.so` modules from pinned source.
- `run_all.py`: Unified entry point that runs GEMM/Attention/MoE and prints a markdown table.
- `benchmark_gemm_unified_graph.py`: GEMM-only timing script.
- `benchmark_attention_unified_graph.py`: Attention-only timing script.
- `benchmark_moe_unified_graph.py`: MoE-only timing script.
- `cudagraph_timer.py`: Shared timer utility used by benchmark scripts.

## Quick Start

From repository root:

```bash
bash python/harness/bench/setup_env.sh
python3 python/harness/bench/run_all.py --mode run --domain all
```

## `setup_env.sh` Usage

```bash
bash python/harness/bench/setup_env.sh [options]
```

Common options:

- `--python <bin>`: Python executable (default: `python3`).
- `--reinstall-aiter`: Force reinstall pinned `amd-aiter`.
- `--skip-smoke`: Skip post-setup import/version checks.
- `--dry-run`: Print commands without executing.

## `run_all.py` Usage

`run_all.py` has two modes:

- `--mode locked`: Print locked table only (no benchmark run).
- `--mode run`: Run benchmarks and parse fresh JSON outputs.

Example (run all domains with paper-style timing):

```bash
python3 python/harness/bench/run_all.py \
  --mode run \
  --domain all \
  --warmup-ms 1000 \
  --repeat-ms 5000 \
  --graph-iters 10 \
  --timer-trials 9 \
  --min-replays 2 \
  --max-replays 3
```

Smoke test (small workloads only):

```bash
python3 python/harness/bench/run_all.py \
  --mode run \
  --domain all \
  --workloads 1024 \
  --warmup-ms 200 \
  --repeat-ms 1000
```

Pin GPU and CPU cores:

```bash
python3 python/harness/bench/run_all.py \
  --mode run \
  --domain gemm \
  --hip-visible-devices 7 \
  --cpu-cores 0-15
```

Write markdown table to file:

```bash
python3 python/harness/bench/run_all.py \
  --mode run \
  --domain all \
  --write-md data/benchmarks/retime.md
```

## Per-Domain Scripts

Each script can be run directly for debugging or isolated reruns.

GEMM:

```bash
python3 python/harness/bench/benchmark_gemm_unified_graph.py \
  --sizes 1024,2048 \
  --warmup-ms 1000 \
  --repeat-ms 5000 \
  --graph-iters 10 \
  --timer-trials 9 \
  --min-replays 2 \
  --max-replays 3 \
  --run-aiter \
  --run-hipblaslt \
  --run-hipkittens \
  --json-out logs/gemm_debug.json
```

Attention:

```bash
python3 python/harness/bench/benchmark_attention_unified_graph.py \
  --seq-lens 1024,2048 \
  --warmup-ms 1000 \
  --repeat-ms 5000 \
  --graph-iters 10 \
  --timer-trials 9 \
  --min-replays 2 \
  --max-replays 3 \
  --no-flashinfer \
  --no-flashattention \
  --baseline-kernel data/benchmarks/attn/01_kernelbench/best_kernel.py \
  --baseline-kernel data/benchmarks/attn/02_cudaforge/best_kernel.py \
  --baseline-kernel data/benchmarks/attn/03_kernelfalcon/best_kernel.py \
  --baseline-kernel data/benchmarks/attn/04_ksearch/best_kernel.py \
  --baseline-kernel data/benchmarks/attn/05_HipKittens/best_kernel.py \
  --baseline-kernel data/benchmarks/attn/06_aiter/best_kernel.py \
  --json-out logs/attention_debug.json
```

MoE:

```bash
python3 python/harness/bench/benchmark_moe_unified_graph.py \
  --seq-lens 1024,2048 \
  --warmup-ms 1000 \
  --repeat-ms 5000 \
  --graph-iters 10 \
  --timer-trials 9 \
  --min-replays 2 \
  --max-replays 3 \
  --baseline-kernel data/benchmarks/moe/01_kernelbench/best_kernel.py \
  --baseline-kernel data/benchmarks/moe/02_cudaforge/best_kernel.py \
  --baseline-kernel data/benchmarks/moe/03_kernelfalcon/best_kernel.py \
  --baseline-kernel data/benchmarks/moe/04_ksearch/best_kernel.py \
  --json-out logs/moe_debug.json
```

## Output

- `run_all.py --mode run` prints a markdown table to stdout.
- If `--out-dir` is set, domain JSON files are written there (`gemm.json`, `attention.json`, `moe_best.json`).
- If `--write-md` is set, markdown output is also saved to that path.
