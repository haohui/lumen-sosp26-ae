# 05_aiter / triton

 2026-03-30  docker (`kernel-benchmark-rocm-traffic`)  AITER Triton GEMMBF16

## 

```bash
cd /workspace/kernel_benchmark
TRITON_CACHE_DIR=/workspace/kernel_benchmark/opt_kernel/gemm-openai/05_aiter/triton/triton_cache \
python3 scripts/run_gemm_baselines_cudagraph.py \
  --dtype bf16 \
  --device cuda:0 \
  --hip-visible-devices 7 \
  --cpu-cores auto:16 \
  --backends aiter_triton \
  --sizes 1024,2048,4096,8192,16384 \
  --correctness-tol 1.0 \
  --correctness-rtol 0.02 \
  --out-date 2026-03-30 \
  --out-prefix gemm_aiter_triton_cudagraph_gpu7_auto16_5sizes_bf16_repo05
```

## 

- `results/gemm_aiter_triton_cudagraph_gpu7_auto16_5sizes_bf16_repo05.csv`
- `results/gemm_aiter_triton_cudagraph_gpu7_auto16_5sizes_bf16_repo05.md`
- `results/gemm_aiter_triton_vs_triton_official_cudagraph_gpu7_auto16_5sizes_bf16.csv`
- `results/gemm_aiter_triton_vs_triton_official_cudagraph_gpu7_auto16_5sizes_bf16.md`

## 

- `source/gemm_a16w16.py`
- `source/gemm_a16w16_kernel.py`
- `source/MI300X-GEMM-A16W16.json`

## Triton 

- `kernels/mnk_<M>x<N>x<K>/`
  -  workload  `mnk_<M>x<N>x<K>_kernel.{source,ttir,ttgir,llir,amdgcn,hsaco,json}``mnk_<M>x<N>x<K>_kernel_group.json`  `kernel_meta.json`
- `kernels/_shared/``__triton_launcher*.so`, `hip_utils*.so`
- workload `kernels_manifest.json`
- `triton_cache/`raw

## workload 

-  workload  `workloads/mnk_<M>x<N>x<K>/row.{csv,json}`
