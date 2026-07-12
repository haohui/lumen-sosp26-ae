# Lumen Artifact Evaluation Repository

This repository contains the data, generated kernels, and benchmarking harness
for the Lumen SOSP'26 paper.

## Installation

The Python environment for this repository is expected to be managed with
`uv`. From the repository root:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
```

You will need to install substrate to evaluate the performance of the DSL kernels.

For formatting and linting:

```bash
ruff format .
ruff check .
```

## Benchmark Source Setup

The benchmark harness expects the AITER source checkout pinned in
`third_party/aiter.source`. Prepare it from the repository root:

```bash
scripts/benchmark/prepare_sources.sh --install-aiter
```

The HipKittens GEMM backend uses an external HipKittens checkout. With the
checkout pinned to `7d58fa1026b4582a75ebdaf7ab5e45e3747a2b7b`, install its
separate binding package as follows:

```bash
HIPKITTENS_ROOT=/path/to/HipKittens uv pip install -e ./packages/hipkittens
```

`HIPKITTENS_ROOT` is required at build time and is not assumed to live under
this repository.

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

The repository is organized around executable reproduction scripts and their
inputs:

- `docker/`: reproducible ROCm evaluation image and its build instructions.
- `scripts/figure1/`, `scripts/figure2/`, and `scripts/table2/`: entry points
  for reproducing the paper figures and tables.
- `scripts/benchmark/`: shared CUDA-graph benchmark harnesses for GEMM,
  attention, and MoE; they emit JSONL timing records.
- `scripts/generation/`: generation and result-staging drivers.
- `datasets/inference/`: Lumen and baseline kernel implementations consumed by
  the benchmark scripts.
- `python/lumen/`: shared Python implementation for generation, evaluation,
  configuration, and reporting.
- `packages/hipkittens/`: Python bindings for the pinned HipKittens kernels.
- `third_party/`: pinned source manifests and local patches for external
  baselines and dependencies.
- `prompts/` and `skills/`: prompt resources and kernel-development context
  used by the generation workflows.

## Reproducing the evaluation

The repository provides a number of scripts under the `scripts/` directory to reproduce the evaluation results in the paper:

- `figure1/validate_invariant.py` validates the data flow invariants described in Figure 1.
- `table2/benchmark.py` reproduces the benchmark results of Table 2.
- `table2/agent_generate.py` regenerates the GPU kernels with KernelFalcon, KSearch, KernelBench, CUDAForge. 
- `figure2/bench_attn_ablation.py` regenerates the ablation of optimizations on the flash attention kernel.

Note that for generation tasks, you will need to set the environment various `LUMEN_GENERATION_API_URL` and `LUMEN_GENERATION_API_KEY` to point to a valid API endpoint of the  chat completion API. 

## Intermediate data and trace

We also provide the traces of agent harness and interactions of LLM. For cost reasons we regenerate the trace with DeepSeek V4. The traces are available at `data/traces`. 

For the generations of GEMM, Flash Attention and MoE, we use [mitmproxy](https://pypi.org/project/mitmproxy/) to collect the interactions with LLM with PII information redacted.

We collect the session traces from Codex for the generations and optimizations on KernelBench problems.
