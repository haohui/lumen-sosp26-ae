# Attention d128 OpenAI Index

 `data/benchmarks/attn` 

## Directory Map

- [01_kernelbench](./01_kernelbench)
- [02_cudaforge](./02_cudaforge)
- [03_kernelfalcon](./03_kernelfalcon)
- [04_ksearch](./04_ksearch)
- [05_HipKittens](./05_HipKittens)
- [06_aiter](./06_aiter)

## Baseline -> Kernel Mapping

| Baseline | Kernel / Entry |
|---|---|
| KernelBench | [01_kernelbench/best_kernel.py](./01_kernelbench/best_kernel.py) |
| CUDAForge | [02_cudaforge/best_kernel.py](./02_cudaforge/best_kernel.py) |
| KernelFalcon | [03_kernelfalcon/best_kernel.py](./03_kernelfalcon/best_kernel.py) |
| KSearch (global) | [04_ksearch/best_kernel.cu](./04_ksearch/best_kernel.cu) |
| KSearch wl01 | [04_ksearch/specific_workload/wl01_seq1024_bs16_kv1/best_kernel.cu](./04_ksearch/specific_workload/wl01_seq1024_bs16_kv1/best_kernel.cu) |
| KSearch wl02 | [04_ksearch/specific_workload/wl02_seq2048_bs16_kv1/best_kernel.cu](./04_ksearch/specific_workload/wl02_seq2048_bs16_kv1/best_kernel.cu) |
| KSearch wl03 | [04_ksearch/specific_workload/wl03_seq4096_bs16_kv1/best_kernel.cu](./04_ksearch/specific_workload/wl03_seq4096_bs16_kv1/best_kernel.cu) |
| KSearch wl04 | [04_ksearch/specific_workload/wl04_seq8192_bs16_kv1/best_kernel.cu](./04_ksearch/specific_workload/wl04_seq8192_bs16_kv1/best_kernel.cu) |
| KSearch wl05 | [04_ksearch/specific_workload/wl05_seq16384_bs16_kv1/best_kernel.cu](./04_ksearch/specific_workload/wl05_seq16384_bs16_kv1/best_kernel.cu) |
| AITER (0.1.10.post3) | [06_aiter/best_kernel.py](./06_aiter/best_kernel.py) |
| HipKittens | [05_HipKittens/best_kernel.py](./05_HipKittens/best_kernel.py) |

## Result Table

- Unified retime table: `data/benchmarks/retime.md`
