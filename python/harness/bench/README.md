# Benchmark Harness

This directory keeps the benchmark runner minimal and modular.

## Scripts

- `run_all.py`: one-command runner for GEMM/Attention/MoE.
- `benchmark_gemm_unified_graph.py`: GEMM domain benchmark.
- `benchmark_attention_unified_graph.py`: Attention domain benchmark.
- `benchmark_moe_unified_graph.py`: MoE domain benchmark.
- `cudagraph_timer.py`: shared CUDA-graph timing implementation.
- `common.py`: shared CLI/runtime helpers.
- `gemm_runtime.py`: GEMM-specific runtime loaders (HipBlasLt/HipKittens).
- `attn_runtime.py`: Attention shared input generation + FLOPs helper.
- `moe_runtime.py`: MoE shared input generation + AITER helper bridge.
- `report_table.py`: JSON-to-markdown table parser/renderer.

## Setup

```bash
bash python/harness/bench/setup_env.sh
```

## Run All Domains

```bash
python3 python/harness/bench/run_all.py \
  --domain all \
  --warmup-ms 1000 \
  --repeat-ms 5000 \
  --graph-iters 10 \
  --timer-trials 9 \
  --min-replays 5
```

## Smoke Test

```bash
python3 python/harness/bench/run_all.py \
  --domain all \
  --workloads 1024 \
  --warmup-ms 200 \
  --repeat-ms 1000
```

## Pin GPU + CPU Cores

```bash
python3 python/harness/bench/run_all.py \
  --domain gemm \
  --hip-visible-devices 7 \
  --cpu-cores 0-15
```

## Domain-Only Runs

GEMM:

```bash
python3 python/harness/bench/benchmark_gemm_unified_graph.py \
  --run-aiter --run-hipblaslt --run-hipkittens \
  --sizes 1024,2048 \
  --warmup-ms 1000 --repeat-ms 5000 --graph-iters 10 --timer-trials 9 --min-replays 5
```

Attention:

```bash
python3 python/harness/bench/benchmark_attention_unified_graph.py \
  --seq-lens 1024,2048 \
  --warmup-ms 1000 --repeat-ms 5000 --graph-iters 10 --timer-trials 9 --min-replays 5
```

MoE:

```bash
python3 python/harness/bench/benchmark_moe_unified_graph.py \
  --seq-lens 1024,2048 \
  --warmup-ms 1000 --repeat-ms 5000 --graph-iters 10 --timer-trials 9 --min-replays 5
```

## Output

- `run_all.py` prints a markdown table in terminal.
- Intermediate JSON files are written only to a temporary directory and removed automatically.
