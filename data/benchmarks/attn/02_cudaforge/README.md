# 02_cudaforge (self-contained submit folder)

## Submit Entry

- submit this file: `kernel.py`
- backup copy: `best_kernel.py` (same content)

## Workload

- `bf16, KV=1, head_dim=128, num_q_heads=8, batch_size=16, causal=True`
- sequence lengths: `1024, 2048, 4096, 8192, 16384`
- correctness tolerance: `1e-2`

## Latest Graph Benchmark

- file: `output/benchmark_attention_unified_graph_bestkernel_only_latest.json`
- config: `warmup_ms=200, repeat_ms=1000, timer_trials=5, graph_iters=1, min_replays=1, cpu_cores=0-15`
- median latency (ms):
  - `1024`: `7.2068`
  - `2048`: `28.6897`
  - `4096`: `110.0055`
  - `8192`: `434.3635`
  - `16384`: `1769.9286`
- all 5 workloads: `suspicious=False`

## Latest Correctness (No-Graph)

- dir: `output/correctness_nograph_latest/`
- all 5 workloads: `max_abs_err=0.015625`

## Included

- `kernel.py`, `best_kernel.py`
- `prompt/`
- `trace_input_output/`
- `traffic/`
- `output/`
