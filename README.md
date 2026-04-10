# Lumen Artifact Evaluation Repository

This repository contains the data, generated kernels, and benchmarking harness
for the Lumen paper.

## Installation

The Python environment for this repository is expected to be managed with
`uv`. From the repository root:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[eval,dev]"
```

You will need to install substrate to evaluate the performance of the DSL kernels.

For formatting and linting:

```bash
ruff format .
ruff check .
```

## Scope

The artifact is intended to support the evaluation of Lumen on:

- BF16 GEMM
- Flash attention
- Fused Mixture-of-Experts (MoE)
- KernelBench tasks

It also includes outputs from baseline agentic kernel-generation frameworks,
including KernelBench, KSearch, CUDAForge, and KernelFalcon, together with the
corresponding Lumen-generated implementations.

## Evaluation Environment

The experiments reported in the paper were run on a server with:

- 2x AMD EPYC 9554 CPUs
- 2 TB DDR5 memory
- 8x AMD Instinct MI300X GPUs
- Ubuntu 22.04.5 LTS
- Linux 5.15.0
- ROCm 7.1.1

## Repository Layout

The repository is organized with benchmark code under `python/` and benchmark
artifacts under `data/`:

- `python/harness/bench/`: benchmark entry points and shared timing/runtime
  helpers (`run_all.py`, domain benchmarks, CUDA Graph timer, setup script).
- `data/benchmarks/gemm/`: GEMM baseline kernels and wrappers.
- `data/benchmarks/attn/`: attention baseline kernels.
- `data/benchmarks/moe/`: MoE baseline kernels and AITER wrapper/helper files.
- `data/benchmarks/retime.md`: unified timing reference table.
