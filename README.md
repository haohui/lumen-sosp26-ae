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

## Flash Attention Ablation

The flash-attention optimization ablation is driven by
`dev-support/ablation_flash_attention.py`. The script checks out the six
recorded Avelang flash-attention commits, rebuilds Avelang with Ninja at each
commit, benchmarks sequence lengths `1024, 2048, 4096, 8192, 16384`, writes a
CSV result file, and emits a grouped bar chart.

From this repository:

```bash
python dev-support/ablation_flash_attention.py /workspace/ae/avelang /workspace
```

The first argument is the Avelang checkout to benchmark and defaults to
`/workspace/ae/avelang`. The second argument is the output directory and
defaults to `/workspace`. The script expects the Avelang checkout to have a
clean worktree because it uses `git checkout` to move between commits. It
restores the original branch or detached commit before exiting.

The benchmark command run at each commit is equivalent to:

```bash
PYTHONPATH=/workspace/ae/avelang/python /opt/venv/bin/python \
  benchmark/attention/bench_flash_attn_amdgpu.py \
  --batch-size 16 --seq-len <SEQ_LEN> --q-heads 8 --kv-heads 1 --head-dim 128
```

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
