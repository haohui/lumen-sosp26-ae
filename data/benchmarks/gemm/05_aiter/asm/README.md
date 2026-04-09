# 05_aiter / asm

 2026-03-30  docker (`kernel-benchmark-rocm-traffic`)  AITER ASM GEMMBF16

## 

```bash
cd /workspace/kernel_benchmark
python3 scripts/run_gemm_baselines_cudagraph.py \
  --dtype bf16 \
  --device cuda:0 \
  --hip-visible-devices 7 \
  --cpu-cores auto:16 \
  --backends aiter \
  --sizes 1024,2048,4096,8192,16384 \
  --correctness-tol 1.0 \
  --correctness-rtol 0.02 \
  --out-date 2026-03-30 \
  --out-prefix gemm_aiter_asm_cudagraph_gpu7_auto16_5sizes_bf16_repo05
```

## 

- `results/gemm_aiter_asm_cudagraph_gpu7_auto16_5sizes_bf16_repo05.csv`
- `results/gemm_aiter_asm_cudagraph_gpu7_auto16_5sizes_bf16_repo05.md`

## 

- `bf16gemm_outf32_tn_64x64_pf3.co`1024
- `bf16gemm_outf32_tn_96x64_pf3.co`2048/4096/8192/16384

## 

- `artifacts/module_gemm_a16w16_asm.so`
- `artifacts/module_aiter_enum.so`
- `artifacts/bf16gemm_outf32_tn_64x64_pf3.co`
- `artifacts/bf16gemm_outf32_tn_96x64_pf3.co`
- `artifacts/bf16gemm_outf32.csv`
- `artifacts/codegen.py`
- `artifacts/20260313_aiter_kernel_select.log`

## workload 

-  workload  `workloads/mnk_<M>x<N>x<K>/row.{csv,json}`
