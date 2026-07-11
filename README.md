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

- `python/lumen/`: shared Python implementation used by benchmarking,
  generation, and evaluation scripts. Common helpers, configuration loaders,
  result parsing, and reusable benchmark utilities should live here.
- `scripts/benchmark/`: standalone artifact benchmarking tools that emit JSONL
  timing records.
- `data/benchmarks/gemm/`, `data/benchmarks/attn/`, `data/benchmarks/moe/`: kernel
  implementations produced by Lumen and by baseline agentic systems for GEMM,
  flash attention, and fused MoE.
- `data/benchmarks/kernelbench/lumen/`: Lumen-generated KernelBench solutions,
  including multiple optimization rounds when applicable.

## Reproducing the evaluation

The repository provides a number of scripts under the `scripts/` directory to reproduce the evaluation results in the paper:

- `table2/benchmark.py` reproduces the benchmark results of Table 2.
- `table2/agent_generate.py` regenerates the GPU kernels with KernelFalcon, KSearch, KernelBench, CUDAForge. 

Note that for generation tasks, you will need to set the environment various `LUMEN_GENERATION_API_URL` and `LUMEN_GENERATION_API_KEY` to point to a valid API endpoint of the  chat completion API. 

## Intermediate data and trace

We also provide the traces of agent harness and interactions of LLM. For cost reasons we regenerate the trace with DeepSeek V4. The traces are available at `data/traces`. 

For the generations of GEMM, Flash Attention and MoE, we use [mitmproxy](https://pypi.org/project/mitmproxy/) to collect the interactions with LLM with PII information redacted.

We collect the session traces from Codex for the generations and optimizations on KernelBench problems.
