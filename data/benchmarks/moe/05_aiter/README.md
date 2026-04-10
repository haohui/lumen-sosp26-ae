# 05_aiter (MoE)

This baseline is executed through the pinned `amd-aiter==0.1.10.post3` runtime.

## Files Kept

- `run_aiter.py`: wrapper that runs only the MoE AITER baseline via the unified harness.
- `tools/bench_moe_aiter_backends_cudagraph.py`: helper script used by the harness to benchmark AITER backends.
- `tools/moe_quant_ref.py`: input/reference utilities used by the helper script.

## Kernel Source Location

The runtime kernels are loaded from the installed `amd-aiter` package (set up by `python/harness/bench/setup_env.sh`), not from local generated artifact dumps.

## Run AITER-only MoE

```bash
python3 data/benchmarks/moe/05_aiter/run_aiter.py \
  --device cuda:0 \
  --seq-lens 1024,2048 \
  --warmup-ms 1000 \
  --repeat-ms 5000 \
  --graph-iters 10 \
  --timer-trials 9 \
  --min-replays 5
```

