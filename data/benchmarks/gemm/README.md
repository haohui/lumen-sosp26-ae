# GEMM OpenAI Index

 `data/benchmarks/gemm` 

## Directory Map

- [01_kernelbench](./01_kernelbench)
- [02_cudaforge](./02_cudaforge)
- [03_kernelfalcon](./03_kernelfalcon)
- [04_ksearch](./04_ksearch)
- [05_aiter](./05_aiter)
- [06_hipblaslt](./06_hipblaslt)
- [07_triton](./07_triton)
- [08_hipketten](./08_hipketten)

## Baseline -> Kernel Mapping

| Baseline | Kernel / Entry |
|---|---|
| KernelBench | [01_kernelbench/best_kernel.py](./01_kernelbench/best_kernel.py) |
| CUDAForge | [02_cudaforge/best_kernel.py](./02_cudaforge/best_kernel.py) |
| KernelFalcon | [03_kernelfalcon/best_kernel.py](./03_kernelfalcon/best_kernel.py) |
| KSearch | [04_ksearch/best_kernel.cu](./04_ksearch/best_kernel.cu) |
| AITER | [05_aiter/triton/source/gemm_a16w16_kernel.py](./05_aiter/triton/source/gemm_a16w16_kernel.py) |
| HipBlasLt | [06_hipblaslt/README.md](./06_hipblaslt/README.md), [source](./06_hipblaslt/src/hipblaslt_internal_ext.cpp) |
| Triton | [07_triton/best_kernel.py](./07_triton/best_kernel.py) |
| HipKittens | [08_hipketten/best_kernel.py](./08_hipketten/best_kernel.py) |

## KSearch Per-Workload Results

- workload manifest: [04_ksearch/specific_workload/manifest_specific_workload.json](./04_ksearch/specific_workload/manifest_specific_workload.json)

| Workload | Kernel | Result Record |
|---|---|---|
| wl01 (1024) | [best_kernel.cu](./04_ksearch/specific_workload/wl01_m1024_n1024_k1024/best_kernel.cu) | [best_solution.json](./04_ksearch/specific_workload/wl01_m1024_n1024_k1024/best_solution.json), [eval reports](./04_ksearch/specific_workload/wl01_m1024_n1024_k1024/gemm_bf16_var_mnk/eval/gemm_bf16_var_mnk) |
| wl02 (2048) | [best_kernel.cu](./04_ksearch/specific_workload/wl02_m2048_n2048_k2048/best_kernel.cu) | [best_solution.json](./04_ksearch/specific_workload/wl02_m2048_n2048_k2048/best_solution.json), [eval reports](./04_ksearch/specific_workload/wl02_m2048_n2048_k2048/gemm_bf16_var_mnk/eval/gemm_bf16_var_mnk) |
| wl03 (4096) | [best_kernel.cu](./04_ksearch/specific_workload/wl03_m4096_n4096_k4096/best_kernel.cu) | [eval reports](./04_ksearch/specific_workload/wl03_m4096_n4096_k4096/gemm_bf16_var_mnk/eval/gemm_bf16_var_mnk) |
| wl04 (8192) | [best_kernel.cu](./04_ksearch/specific_workload/wl04_m8192_n8192_k8192/best_kernel.cu) | [best_solution.json](./04_ksearch/specific_workload/wl04_m8192_n8192_k8192/best_solution.json), [eval reports](./04_ksearch/specific_workload/wl04_m8192_n8192_k8192/gemm_bf16_var_mnk/eval/gemm_bf16_var_mnk) |
| wl05 (16384) | [best_kernel.cu](./04_ksearch/specific_workload/wl05_m16384_n16384_k16384/best_kernel.cu) | [best_solution.json](./04_ksearch/specific_workload/wl05_m16384_n16384_k16384/best_solution.json), [eval reports](./04_ksearch/specific_workload/wl05_m16384_n16384_k16384/gemm_bf16_var_mnk/eval/gemm_bf16_var_mnk) |

## Result Table

- Unified retime table: `data/benchmarks/retime.md`
