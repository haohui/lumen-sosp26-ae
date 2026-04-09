# auto tune with X CD

Run date: 2026-03-31
Backend: hipkittens_triton_v01_remap_xcd (with stagger_k autotune)
GPU: HIP_VISIBLE_DEVICES=7 (empty card)
CPU affinity: 0-15 (empty core set)
Timing: warmup=200ms, repeat=1000ms, graph_iters=1

## Results
- ../../../../logs/gemm/08_hipketten/csv/results/gemm_hipkittens_triton_v01_remap_xcd_staggerk_autotune_cudagraph_gpu7_emptycard_emptycore_warm200ms_repeat1s.csv
- results/gemm_hipkittens_triton_v01_remap_xcd_staggerk_autotune_cudagraph_gpu7_emptycard_emptycore_warm200ms_repeat1s.md
- results/autotune_table_with_stagger_k_20260331.md
- results/stagger_k_ab_report_20260331.md

## Kernel dumps per workload
- kernels/mnk_<M>x<N>x<K>/*_kernel.{source,ttir,ttgir,llir,amdgcn,hsaco,json}
- kernels/mnk_<M>x<N>x<K>/kernel_meta.json
- kernels/mnk_<M>x<N>x<K>/*_kernel_group.json
- kernels_manifest.json
- triton_cache_manifest.json
- kernels/_shared/*.so
