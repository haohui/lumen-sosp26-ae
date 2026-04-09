# MoE OpenAI Index

 `data/benchmarks/moe` 

## Directory Map

- [01_kernelbench](./01_kernelbench)
- [02_cudaforge](./02_cudaforge)
- [03_kernelfalcon](./03_kernelfalcon)
- [04_ksearch](./04_ksearch)
- [05_aiter](./05_aiter)
- [reports](./reports)

## Agentic MoE (fused runtime, authoritative)

| Baseline | Kernel / Entry |
|---|---|
| 01_kernelbench | [01_kernelbench/best_kernel.py](./01_kernelbench/best_kernel.py) |
| 02_cudaforge | [02_cudaforge/best_kernel.py](./02_cudaforge/best_kernel.py) |
| 03_kernelfalcon | [03_kernelfalcon/best_kernel.py](./03_kernelfalcon/best_kernel.py) |
| 04_ksearch | [04_ksearch/best_kernel.py](./04_ksearch/best_kernel.py) |

## AITER Backends

| Backend | Entry |
|---|---|
| ASM | [05_aiter/ASM/src/moe_op.py](./05_aiter/ASM/src/moe_op.py) |
| CK | [05_aiter/CK/src/moe_op.py](./05_aiter/CK/src/moe_op.py) |
| Triton | [05_aiter/Triton/src/moe_op.py](./05_aiter/Triton/src/moe_op.py) |

## CUDAForge Disasm

- [02_cudaforge/disasm/rocprof_20260408_024717](./02_cudaforge/disasm/rocprof_20260408_024717)

## 01 KernelBench Trace Layout

- Run I/O artifacts (prompt, generated kernel, generation config):
  - [01_kernelbench/trace_input_output/run_io_prompt_kernel_config](./01_kernelbench/trace_input_output/run_io_prompt_kernel_config)
- MITM traffic capture artifacts:
  - [01_kernelbench/trace_input_output/traffic_capture_mitm](./01_kernelbench/trace_input_output/traffic_capture_mitm)
- Prepared reference snapshot (`_prepared_refs`) is archived outside this package.

## Result Table

- Unified retime table: `data/benchmarks/retime.md`
