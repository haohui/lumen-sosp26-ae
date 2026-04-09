# 08_hipketten

2026-03-30 + 2026-03-31`HipKittens-triton_gemm_v01` / `+remap_xcd` / `+remap_xcd+stagger_k`  artifacts 

## 

- `../../../logs/gemm/08_hipketten/csv/root/gemm_hipkittens_triton_v01_cudagraph_gpu7_emptycard_emptycore_warm200ms_repeat1s.csv`
- `gemm_hipkittens_triton_v01_cudagraph_gpu7_emptycard_emptycore_warm200ms_repeat1s.md`
- `../../../logs/gemm/08_hipketten/csv/root/gemm_hipkittens_v01_vs_remap_xcd_cudagraph_gpu7_emptycard_emptycore_warm200ms_repeat1s.csv`
- `gemm_hipkittens_v01_vs_remap_xcd_cudagraph_gpu7_emptycard_emptycore_warm200ms_repeat1s.md`
- `../../../logs/gemm/08_hipketten/csv/root/gemm_hipkittens_triton_v01_remap_xcd_autotune_cudagraph_gpu7_emptycard_emptycore_warm200ms_repeat1s_round2.csv`
- `gemm_hipkittens_triton_v01_remap_xcd_autotune_cudagraph_gpu7_emptycard_emptycore_warm200ms_repeat1s_round2.md`
- `../../../logs/gemm/08_hipketten/csv/root/gemm_hipkittens_triton_v01_remap_xcd_autotune_with_xcd_cudagraph_gpu7_emptycard_emptycore_warm200ms_repeat1s.csv`
- `gemm_hipkittens_triton_v01_remap_xcd_autotune_with_xcd_cudagraph_gpu7_emptycard_emptycore_warm200ms_repeat1s.md`
- `../../../logs/gemm/08_hipketten/csv/root/gemm_hipkittens_triton_v01_remap_xcd_staggerk_autotune_cudagraph_gpu7_emptycard_emptycore_warm200ms_repeat1s.csv`
- `gemm_hipkittens_triton_v01_remap_xcd_staggerk_autotune_cudagraph_gpu7_emptycard_emptycore_warm200ms_repeat1s.md`
- `../../../logs/gemm/08_hipketten/csv/results/gemm_hipkittens_triton_v01_cudagraph_gpu7_emptycard_emptycore_warm200ms_repeat1s.csv`
- `results/gemm_hipkittens_triton_v01_cudagraph_gpu7_emptycard_emptycore_warm200ms_repeat1s.md`
- `../../../logs/gemm/08_hipketten/csv/results/gemm_hipkittens_v01_vs_remap_xcd_cudagraph_gpu7_emptycard_emptycore_warm200ms_repeat1s.csv`
- `results/gemm_hipkittens_v01_vs_remap_xcd_cudagraph_gpu7_emptycard_emptycore_warm200ms_repeat1s.md`
- `results/remap_xcd_ab_report_20260330.md`
- `results/autotune_table_remap_xcd_round2_20260330.md`
- `../../../logs/gemm/08_hipketten/csv/results/gemm_hipkittens_triton_v01_remap_xcd_staggerk_autotune_cudagraph_gpu7_emptycard_emptycore_warm200ms_repeat1s.csv`
- `results/gemm_hipkittens_triton_v01_remap_xcd_staggerk_autotune_cudagraph_gpu7_emptycard_emptycore_warm200ms_repeat1s.md`
- `../../../logs/gemm/08_hipketten/csv/results/gemm_hipkittens_triton_v01_remap_xcd_autotune_with_xcd_cudagraph_gpu7_emptycard_emptycore_warm200ms_repeat1s.csv`
- `results/gemm_hipkittens_triton_v01_remap_xcd_autotune_with_xcd_cudagraph_gpu7_emptycard_emptycore_warm200ms_repeat1s.md`
- `results/autotune_table_with_stagger_k_20260331.md`
- `results/stagger_k_ab_report_20260331.md`
- `auto_tune_with_xcd/` workload  source/ttir/ttgir/llir/amdgcn/hsaco/json  stagger_k autotune

## BF16, GPU7, warmup=200ms, repeat=1s

| M | N | K | warmup | iters | timer | timing_ms | tflops | status |
|---:|---:|---:|---:|---:|---|---:|---:|---|
| 1024 | 1024 | 1024 | 2064 | 10320 | cuda_graph_event | 0.031046 | 69.170262 | ok |
| 2048 | 2048 | 2048 | 1507 | 7531 | cuda_graph_event | 0.060869 | 282.245502 | ok |
| 4096 | 4096 | 4096 | 444 | 2218 | cuda_graph_event | 0.373220 | 368.251853 | ok |
| 8192 | 8192 | 8192 | 70 | 349 | cuda_graph_event | 2.764233 | 397.763667 | ok |
| 16384 | 16384 | 16384 | 9 | 44 | cuda_graph_event | 22.682182 | 387.797487 | ok |

## BF16, GPU7, remap_xcd + stagger_k autotune, warmup=200ms, repeat=1s

| M | N | K | REMAP_XCD | STAGGER_K | warmup | iters | timer | timing_ms | tflops | status |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|
| 1024 | 1024 | 1024 | 1 | 1 | 1855 | 9273 | cuda_event_fallback | 0.032256 | 66.576361 | ok |
| 2048 | 2048 | 2048 | 0 | 1 | 1356 | 6780 | cuda_event_fallback | 0.061102 | 281.168097 | ok |
| 4096 | 4096 | 4096 | 1 | 8 | 470 | 2348 | cuda_graph_event | 0.339053 | 405.361830 | ok |
| 8192 | 8192 | 8192 | 0 | 8 | 75 | 372 | cuda_graph_event | 2.666227 | 412.384809 | ok |
| 16384 | 16384 | 16384 | 0 | 4 | 10 | 48 | cuda_graph_event | 20.803305 | 422.821897 | ok |

## BF16, GPU7, remap_xcd autotune with XCD, warmup=200ms, repeat=1s2026-03-31 

| M | N | K | REMAP_XCD | STAGGER_K | warmup | iters | timer | timing_ms | tflops | status |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|
| 1024 | 1024 | 1024 | 1 | 8 | 1913 | 9561 | cuda_graph_event | 0.032354 | 66.374736 | ok |
| 2048 | 2048 | 2048 | 1 | 1 | 1309 | 6544 | cuda_graph_event | 0.063078 | 272.358480 | ok |
| 4096 | 4096 | 4096 | 0 | 8 | 409 | 2044 | cuda_graph_event | 0.409573 | 335.566088 | ok |
| 8192 | 8192 | 8192 | 0 | 2 | 64 | 318 | cuda_graph_event | 3.221051 | 341.351832 | ok |
| 16384 | 16384 | 16384 | 0 | 1 | 8 | 38 | cuda_graph_event | 26.521247 | 331.662125 | ok |

## .amdgcn

- [mnk_1024x1024x1024](kernels/mnk_1024x1024x1024/mnk_1024x1024x1024_kernel.amdgcn)
- [mnk_2048x2048x2048](kernels/mnk_2048x2048x2048/mnk_2048x2048x2048_kernel.amdgcn)
- [mnk_4096x4096x4096](kernels/mnk_4096x4096x4096/mnk_4096x4096x4096_kernel.amdgcn)
- [mnk_8192x8192x8192](kernels/mnk_8192x8192x8192/mnk_8192x8192x8192_kernel.amdgcn)
- [mnk_16384x16384x16384](kernels/mnk_16384x16384x16384/mnk_16384x16384x16384_kernel.amdgcn)

##  artifacts

- `kernels_manifest.json`
- `triton_cache_manifest.json`
- `source/`
- `source/hipkittens_triton_gemm_v01_remap_xcd_matmul.py`
- `workloads/`
- `triton_cache_v01/`
- `../../../logs/gemm/08_hipketten/triton_cache_smoke/`
