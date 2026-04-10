# HipBlasLt GEMM Baseline

- Extension source: `src/hipblaslt_internal_ext.cpp`
- Runtime dependency: system ROCm libs under `/opt/rocm*/lib`
- Single-baseline wrapper: `run_hipblaslt.py`

Run only HipBlasLt GEMM:

```bash
python3 data/benchmarks/gemm/06_hipblaslt/run_hipblaslt.py \
  --device cuda:0 \
  --sizes 1024,2048 \
  --warmup-ms 1000 \
  --repeat-ms 5000 \
  --graph-iters 10 \
  --timer-trials 9 \
  --min-replays 5
```
