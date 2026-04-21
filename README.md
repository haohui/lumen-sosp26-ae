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

The repository is organized so Python code lives under `python/`, while
benchmark inputs and generated outputs live under `data/`:

- `python/lumen_artifact/`: shared Python implementation used by benchmarking,
  generation, and evaluation scripts. Common helpers, configuration loaders,
  result parsing, and reusable benchmark utilities should live here.
- `python/harness/bench/`: benchmarking entry points that import shared logic
  from `python/lumen_artifact/`.
- `python/harness/generation/`: generation drivers and thin entry-point scripts
  that import shared logic from `python/lumen_artifact/`.
- `data/benchmarks/gemm/`, `data/benchmarks/attn/`, `data/benchmarks/moe/`: kernel
  implementations produced by Lumen and by baseline agentic systems for GEMM,
  flash attention, and fused MoE.
- `data/benchmarks/kernelbench/oracle/`: expert-optimized KernelBench reference
  implementations.
- `data/benchmarks/kernelbench/lumen/`: Lumen-generated KernelBench solutions,
  including multiple optimization rounds when applicable.
- `data/invariant/`: artifact-evaluation assets for the MFMA invariant ablation,
  including the vendored `kb_eval_pipeline/` workspace, prompt templates, and
  compact experiment `runs/`.
