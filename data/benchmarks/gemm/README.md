# GEMM Index

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

## Baseline -> Entry

| Baseline | Entry |
|---|---|
| KernelBench | [01_kernelbench/best_kernel.py](./01_kernelbench/best_kernel.py) |
| CUDAForge | [02_cudaforge/best_kernel.py](./02_cudaforge/best_kernel.py) |
| KernelFalcon | [03_kernelfalcon/best_kernel.py](./03_kernelfalcon/best_kernel.py) |
| KSearch | [04_ksearch/best_kernel.py](./04_ksearch/best_kernel.py) |
| AITER | [05_aiter/best_kernel.py](./05_aiter/best_kernel.py) |
| HipBLASLt | [06_hipblaslt/run_hipblaslt.py](./06_hipblaslt/run_hipblaslt.py) |
| Triton | [07_triton/best_kernel.py](./07_triton/best_kernel.py) |
| HipKittens | [08_hipketten/best_kernel.py](./08_hipketten/best_kernel.py) |

## Result Table

- Unified retime table: `data/benchmarks/retime.md`
