# 07_triton

 2026-03-30  docker (`kernel-benchmark-rocm-traffic`)  Triton  GEMMBF16

## 

```bash
cd /workspace/kernel_benchmark
TRITON_CACHE_DIR=/workspace/kernel_benchmark/opt_kernel/gemm-openai/07_triton/triton_cache_official \
python3 scripts/run_gemm_baselines_cudagraph.py \
  --dtype bf16 \
  --device cuda:0 \
  --hip-visible-devices 7 \
  --cpu-cores auto:16 \
  --backends triton_official \
  --sizes 1024,2048,4096,8192,16384 \
  --correctness-tol 1.0 \
  --correctness-rtol 0.02 \
  --out-date 2026-03-30 \
  --out-prefix gemm_triton_official_cudagraph_gpu7_auto16_5sizes_bf16_repo09_v2
```

## 

- `results/gemm_triton_official_cudagraph_gpu7_auto16_5sizes_bf16_repo09_v2.csv`
- `results/gemm_triton_official_cudagraph_gpu7_auto16_5sizes_bf16_repo09_v2.md`
- `results/gemm_triton_official_cudagraph_gpu7_auto16_5sizes_bf16_repo09.csv`
- `results/gemm_triton_official_cudagraph_gpu7_auto16_5sizes_bf16_repo09.md`
- `results/gemm_aiter_triton_vs_triton_official_cudagraph_gpu7_auto16_5sizes_bf16.csv`
- `results/gemm_aiter_triton_vs_triton_official_cudagraph_gpu7_auto16_5sizes_bf16.md`

## 

- `source/triton_official_matmul.py`
- `source/run_gemm_baselines_cudagraph.py`

## Triton 

-  Triton `3.5.1+rocm7.1.1.gita272dfa8`
  - `triton_cache_manifest.json`  `triton_version`
-  `2026-03-30``triton-lang/triton` GitHub Releases  `v3.6.0` `2026-01-21`
  - https://github.com/triton-lang/triton/releases
  - https://github.com/triton-lang/triton/releases/tag/v3.6.0

##  .so /  / IR

- `kernels/mnk_<M>x<N>x<K>/`
  -  workload  `mnk_<M>x<N>x<K>_kernel.{source,ttir,ttgir,llir,amdgcn,hsaco,json}``mnk_<M>x<N>x<K>_kernel_group.json`  `kernel_meta.json`
- `kernels/_shared/``__triton_launcher*.so`, `hip_utils*.so`
- workload `kernels_manifest.json`
- `triton_cache_official/`raw

 5  workload  5  `kernels_manifest.json`  5  `hash/cache_dir` `mnk_<M>x<N>x<K>_kernel.*`

## workload 

-  workload  `workloads/mnk_<M>x<N>x<K>/row.{csv,json}`

## HipKittens `triton_gemm_v01.py`docker


`https://github.com/HazyResearch/HipKittens/blob/4d15d8e92dfc65b6b33c36ad8b6a7e883c5f7245/analysis/baselines/gemm/triton_gemm_v01.py`



```bash
cd /data01/home/daifeng/kernel_benchmark/opt_kernel/gemm-openai/07_triton/source
./run_hipkittens_triton_gemm_v01_in_docker.sh substrate_rocm711_wheatopt 7
```



-  clone HipKittens  checkout  `4d15d8e...` `third_party/HipKittens` 
-  `/dev/kfd`  `torch.cuda.is_available()`
- `results/hipkittens_triton_gemm_v01_<commit>_gpu<id>_<date>.log`



- `results/hipkittens_triton_gemm_v01_4d15d8e_gpu7_20260330.md`
- `results/hipkittens_triton_gemm_v01_4d15d8e_gpu7_20260330.log`
