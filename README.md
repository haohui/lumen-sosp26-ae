# Lumen Artifact Evaluation Repository

This repository contains the data, generated kernels, and benchmarking harness
for the Lumen SOSP'26 paper.

## Paper Reference

**Guiding Agentic GPU Kernel Optimization with Data Flow Invariants**

- Haohui Mai (The Hong Kong University of Science and Technology),
  <haohui@ust.hk>
- Xiaoyan Guo (University of Chinese Academy of Sciences, China),
  <guoxiaoyan24s@ict.ac.cn>
- Xiangyun Ding (University of California, Riverside), <xding047@ucr.edu>
- Daifeng Li (HKUST), <fengli1702@gmail.com>
- Qiuchu Yu (University of Chinese Academy of Sciences, China),
  <yuqiuchu19@mails.ucas.ac.cn>
- Chenzhun Guo (Xi'an Jiaotong University, China), <wsgcz@stu.xjtu.edu.cn>
- Cong Wang (Tsinghua University), <wcon006@gmail.com>
- Jiacheng Zhao (Institute of Computing Technology, Chinese Academy of
  Sciences), <zhaojiacheng@ict.ac.cn>
- Christos Kozyrakis (Stanford University), <kozyraki@stanford.edu>
- Binhang Yuan (HKUST), <biyuan@ust.hk>

## Scope

The artifact supports the evaluation of Lumen on:

- BF16 GEMM;
- flash attention;
- fused Mixture-of-Experts (MoE); and
- KernelBench tasks.

It also includes outputs from KernelBench, KSearch, CUDAForge, and
KernelFalcon, together with the corresponding Lumen-generated implementations.

## Evaluation Environment

The experiments reported in the artifact-evaluation (AE) version of the paper
used **GPT-5.3-Codex with high reasoning effort** as the agent's LLM backend.
Due to API cost and regional constraints on our proxy server, the evaluator
entry point [`run_experiments.py`](run_experiments.py) currently provides free
access only to the **DeepSeek-V4** API. Its default model is
`deepseek-v4-flash`. Consequently, newly generated kernels and trajectories may
differ from the paper's GPT-5.3-Codex results. The archived traces identify the
backend used for each recorded run.

The paper measurements were collected on:

- 2x AMD EPYC 9554 CPUs;
- 2 TB DDR5 memory;
- 8x AMD Instinct MI300X GPUs (`gfx942`);
- Ubuntu 22.04.5 LTS and Linux 5.15.0; and
- ROCm 7.1.1.

The supplied Docker environment uses a newer Ubuntu 24.04/ROCm 7.2.2 base for
ease of distribution. Performance numbers can vary across hardware, ROCm,
compiler, and model-backend versions. Table 3 generation and optimization are
configured for eight GPUs; the other experiments can run on one MI300X.

## Dependencies

The provided Docker image is the reference installation. It contains the
following explicit dependencies (see [`docker/Dockerfile`](docker/Dockerfile)
for the executable specification):

- **Platform:** Linux on an AMD ROCm-capable GPU; the image is based on
  `rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0` and reuses
  its Python 3.12, PyTorch, and Triton installations.
- **System build tools and libraries:** `build-essential`, `cmake`, `git`,
  `ninja-build`, `patch`, `pkg-config`, `python3-dev`, `nodejs`, `npm`, Z3,
  `libz3-dev`, `libedit-dev`, `libncurses-dev`, `libxml2-dev`, `libzstd-dev`,
  OpenSSH client, and CA certificates.
- **Compiler and DSL:** the Lumen branches of LLVM/Clang/MLIR (installed under
  `/opt/llvm`) and AveLang. Building these components from source requires SSH
  access to their repositories; evaluators using the supplied image do not need
  that access.
- **Agent runtime:** the OpenAI Codex CLI (`@openai/codex`) plus `openai`,
  `anthropic`, `litellm`, `requests`, `python-dotenv`, `pydra-config`, and
  `omegaconf`.
- **Python scientific and harness packages:** Jinja2, NumPy, pandas,
  matplotlib, SymPy, NetworkX, einops, psutil, datasets, transformers, Modal,
  tabulate, tomli, mitmproxy, pytest, pybind11, scikit-build-core,
  setuptools-scm, setuptools, wheel, ninja, and pip.
- **GPU kernel packages:** FlashInfer `v0.3.1+amd.1`,
  `flashinfer-bench-ksearch`, AITER `v0.1.10.post3`, and the local HipKittens
  Python package built for `gfx942`.
- **Pinned external systems:** KernelBench, K-Search, KernelFalcon/KernelAgent,
  CUDAForge, and HipKittens. Their upstream URLs, commits, and local patches are
  recorded in `third_party/*/source.toml` and prepared by
  `dev-support/prepare-dependency.py`.

To use the pre-installed environment, activate its virtual environment from the
repository root:

```bash
source .venv/bin/activate
```

All commands below assume this environment and a working ROCm device. Generation
experiments additionally require network access to the configured LLM endpoint;
KernelBench runs also download the `ScalingIntelligence/KernelBench` dataset
from Hugging Face when it is not cached.

Validate an environment independently with:

```bash
python scripts/validate_environment.py --require-codex --require-api --require-hf
```

The validator checks GPU availability, ROCm, Python/build dependencies, Codex,
the configured generation API, Hugging Face access, and free disk space, then
prints a configuration summary. `run_experiments.py` calls it automatically
before running experiments and makes checks fatal only when the selected
experiments require that resource. Use `--help` to adjust the expected GPU
count, output path, or minimum free space.

## Repository-to-Paper Map

| Paper result or topic | Experiment entry point | Main repository components |
| --- | --- | --- |
| Figure 1: data-flow invariant validation | `scripts/figure1/validate_invariant.py` | `datasets/inference/attention/lumen/attn_07_invariants.py` and AveLang's invariant validator |
| Table 2: end-to-end GEMM, attention, and MoE comparison | `scripts/table2/benchmark.py` | `scripts/benchmark/` and the system-specific implementations under `datasets/inference/{gemm,attention,moe}/` |
| Table 2: agentic baseline generation | `scripts/table2/agent_generate.py` | `prompts/{kernelbench,cudaforge,kernelfalcon,ksearch}/`, the prepared third-party systems, and the corresponding `datasets/inference/` backend directories |
| Table 2: Lumen optimization sequences | `scripts/table2/optimizer.py` | `python/lumen/harness/datasets/lumen/`, its per-workload prompts, and the staged Lumen kernels under `datasets/inference/` |
| Figure 2: flash-attention optimization ablation | `scripts/figure2/bench_attn_ablation.py` | `datasets/inference/attention/lumen/attn_01_naive.py` through `attn_06_inst_scheduling.py` |
| Table 3: KernelBench generation, context ablation, and invariant-guided optimization | `scripts/table3/run_kernelbench_table3.py` | `scripts/table3/config/`, `python/lumen/harness/datasets/kernelbench/`, and `data/seeds/` |
| Table 3: reported summary rows | `scripts/table3/kernelbench_table.py` | `scripts/table3/kb_table/` and the archived generation/optimization traces in `data/traces/` |

Shared Python code lives under `python/lumen/`. The standalone tools in
`scripts/benchmark/` emit JSONL timing records and are invoked by the Table 2
driver rather than being separate paper experiments.

## Reproducing the Evaluation

Run commands from the repository root. List all experiment selectors with:

```bash
python run_experiments.py --list
```

### One-command full reproduction

The following runs every generation, optimization, benchmark, validation, and
summary stage in dependency order:

```bash
python run_experiments.py
```

By default, unified-runner logs and collected outputs are written to
`runs/artifact_experiments_<UTC timestamp>/`. Use `--output-dir PATH` to choose
a stable location, `--gpu-id N` for the single-GPU experiments, or
`--keep-going` to continue independent experiments after a failure. The Table 3
configs use GPU IDs 0-7 regardless of `--gpu-id`. Check the evaluator API before
starting generation with:

```bash
python run_experiments.py api-check
```

Every experiment entry point prints a final PASS/FAIL block with its reason,
result paths, log paths, and the paper figure or table to compare. The unified
runner also prints an aggregate block and stores each experiment's complete
stdout/stderr in `<output-dir>/logs/`; direct invocations report stdout/stderr as
the log unless that experiment creates its own trace log.

The runner supplies the evaluator DeepSeek-V4 endpoint by default. To use a
different OpenAI-compatible Responses endpoint, set
`LUMEN_GENERATION_API_URL`, `LUMEN_GENERATION_API_KEY`, and
`LUMEN_GENERATION_MODEL`, or pass `--api-url` and `--model`.

### Figure 1: invariant validation

**Purpose:** compile and execute the final Lumen flash-attention kernel with
AveLang data-flow invariant validation enabled.

**Prerequisites:** one ROCm GPU, AveLang, the Lumen LLVM build, and PyTorch.

**Command:**

```bash
python run_experiments.py figure1 --output-dir runs/figure1
```

**Outputs:** a pass/fail message and `runs/figure1/logs/figure1.log`; this
validation does not produce a numeric data file.

### Table 2: agentic baseline generation

**Purpose:** regenerate GEMM, attention, and MoE kernels with KernelBench,
CUDAForge, KernelFalcon, and K-Search.

**Prerequisites:** one ROCm GPU, the prepared third-party checkouts, Codex CLI,
and access to the generation API. This stage incurs LLM calls. Use
`--table2-baseline` and `--table2-task` to select one system or workload.

**Command:**

```bash
python run_experiments.py table2-generation \
  --output-dir runs/table2-generation \
  --generation-rounds 10
```

The runner prints each direct API attempt immediately. By default, Table 2
KernelBench generation makes at most two attempts per task, with a five-minute
timeout. The full runner uses a 65,536 output-token limit only with the artifact's
default DeepSeek-V4 API; direct runs and other APIs retain the script's 32,768
default. Adjust these bounds with
`--kernelbench-attempts`, `--generation-api-timeout-seconds`, and
`--generation-max-output-tokens`. Add `--table2-no-stage` to retain diagnostic
outputs only in the run workspace without replacing included dataset kernels.

**Outputs:** native run artifacts under
`runs/table2-generation/table2/generation/<task>/<baseline>/`, a runner log,
and staged generated kernels in
`datasets/inference/<task>/<baseline>/`. The staging step updates working-tree
files; use `scripts/table2/agent_generate.py --no-stage` for an exploratory run
that must not stage generated kernels.

### Table 2: Lumen optimization sequences

**Purpose:** replay the guided, multi-round Lumen optimization sequences for
GEMM, flash attention, and fused MoE.

**Prerequisites:** one ROCm GPU, AveLang, the Lumen compiler, Codex CLI, and
access to the generation API. This stage incurs LLM calls.

**Command:**

```bash
python run_experiments.py table2-optimization \
  --output-dir runs/table2-optimization \
  --gpu-id 0
```

**Outputs:** one directory per workload under
`runs/table2-optimization/table2/optimization/`. Each round contains its
prompt, input and output model, Codex result/trajectory, evaluation result, and
status; logs are under `runs/table2-optimization/logs/`.

To run or resume one workload directly (and to set the paper backend
explicitly), use:

```bash
PYTHONPATH=python python scripts/table2/optimizer.py gemm \
  --gpu-id 0 --model gpt-5.3-codex --reasoning-effort high
```

Replace `gemm` with `attn` or `moe`. See `--help` for `--resume-run` and custom
prompt options.

### Table 2: performance benchmark

**Purpose:** benchmark all available Lumen and baseline implementations for the
three workloads and render the throughput cells used by Table 2.

**Prerequisites:** one ROCm GPU and all backend packages/kernels to be measured;
no LLM API is needed. A backend that fails is reported as a missing (`-`) cell.

**Command:**

```bash
python run_experiments.py table2-benchmark \
  --output-dir runs/table2-benchmark \
  --warmup 10 --benchmark-repeat 100
```

**Outputs:** `gemm.jsonl`, `attention.jsonl`, `moe.jsonl`, and `table2.csv`
under `runs/table2-benchmark/table2/benchmark/`, plus the runner log.

### Figure 2: flash-attention ablation

**Purpose:** benchmark each successive Lumen flash-attention optimization at
sequence lengths 1K, 2K, 4K, 8K, and 16K.

**Prerequisites:** one ROCm GPU, AveLang, the Lumen compiler, and enough GPU
memory for batch size 16 at the selected sequence lengths; no LLM API is needed.

**Command:**

```bash
python run_experiments.py figure2 \
  --output-dir runs/figure2 \
  --warmup 10 --benchmark-repeat 100
```

**Outputs:** `runs/figure2/figure2/attention_ablation.csv`, with optimization
name, sequence length, mean latency, and TFLOP/s, plus the runner log.

**Expected variation:** We tested Figure 2 five times on one isolated AMD
Instinct MI300X (`gfx942`) with ROCm 7.2.2 and PyTorch 2.10.0+ROCm, using the
fixed seed, batch size 16, 10 warmups, and 100 timed graph replays shown above.
All five runs completed successfully. Across the 30 figure points, the
five-run medians had 0.65% median absolute error from the paper: 25/30 were
within 3% and 29/30 were within 7%. The median run-to-run spread was 1.55%;
27/30 points had at most 7% spread, while isolated slow measurements raised
the largest spread to 17.56%. The cold, short `Naive` 1K case was the only
systematic exception, measuring 22.65% below the paper.

For validation, use the median of at least five complete runs. An expected
reproduction has 29/30 median points within 7% of the reported values, permits
up to 25% lower throughput for the clock-sensitive `Naive` 1K point, and
preserves the large throughput gain from the first three variants to the bank
conflict, pipeline/workgroup-specialization, and final scheduler variants.
Rerun an individual trial when a point is more than 7% from the other trials;
do not use a single slow trial as the paper comparison. The five validation
CSVs and logs are under `runs/figure2-variation/`.

### Table 3: KernelBench generation

**Purpose:** generate KernelBench Level 1 and Level 2 solutions both with the
full AveLang in-context material and with language-specification-only context.

**Prerequisites:** eight MI300X GPUs as configured, the Hugging Face
KernelBench dataset, Codex CLI, and access to the generation API. This is a
large LLM/GPU experiment.

**Command:**

```bash
python run_experiments.py table3-generation \
  --output-dir runs/table3-generation
```

**Outputs:** per-problem workspaces, generated models, trajectories, and
evaluation results in the `run_dir` values declared by
`scripts/table3/config/kernelbench_generation_*.toml` (under `data/traces/`),
Table 3 logs in `data/traces/logs/`, and a unified-runner log under the selected
output directory. A single configuration can be run with, for example,
`PYTHONPATH=python python scripts/table3/run_kernelbench_table3.py generation
level1 full`.

### Table 3: invariant-guided optimization

**Purpose:** optimize the provided naive AveLang KernelBench candidates with
and without invariant guidance for Levels 1 and 2.

**Prerequisites:** the same eight-GPU/API setup as Table 3 generation and
`data/seeds/kernelbench_optimization_naive_avelang_seeds.tar.xz` (included).

**Command:**

```bash
python run_experiments.py table3-optimization \
  --output-dir runs/table3-optimization
```

**Outputs:** per-problem optimization rounds, trajectories, and evaluation
results under the optimization `run_dir` values in `scripts/table3/config/`,
plus logs in `data/traces/logs/` and the unified output directory. A single
profile can be run with, for example,
`PYTHONPATH=python python scripts/table3/run_kernelbench_table3.py optimization
level1 invariants`.

### Table 3: summary

**Purpose:** compute the Table 3 correctness, speedup, context-ablation, token,
and invariant-ablation statistics from trace archives. The default command
summarizes the included DeepSeek-V4 archives, so it does not require generation.

**Prerequisites:** Python and the two included KernelBench trace archives; no
GPU or LLM API is needed.

**Command:**

```bash
python run_experiments.py table3-summary \
  --output-dir runs/table3-summary
```

**Outputs:** `runs/table3-summary/table3/table3.csv` and a runner log. Run
`python scripts/table3/kernelbench_table.py --help` to select different
archives, run-directory names, JSON output, or an observed-run denominator.

## Traces

`data/traces/` contains archives of agent trajectories. Each round typically
includes the agent trace, prompt, input and generated model files, evaluation
configuration and result, and per-round metadata. The archives are:

- `kernelbench_generation_dsv4-07-13-2026.tar.xz`: DeepSeek-V4 KernelBench
  generation trajectories for Levels 1 and 2, each run with and without DSL
  examples.
- `kernelbench_optimization_dsv4-07-13-2026.tar.xz`: DeepSeek-V4 KernelBench
  optimization trajectories for Levels 1 and 2, each run with and without
  invariant guidance.
- `lumen_optimization_codex-07-13-2026.tar.xz`: GPT-5.3-Codex optimization
  trajectories for the Lumen GEMM, flash-attention, and fused-MoE kernels.
- `table2_agentic_generation_gpt53codex-07-13-2026.tar.xz`: redacted
  GPT-5.3-Codex HTTP traffic from the agentic-generation runs for CUDAForge,
  KernelBench, KernelFalcon, and K-Search across GEMM, flash attention, and
  fused MoE.

The trace bundles contain recorded interaction data only; they are not needed
to benchmark the included kernels or reproduce the archived Table 3 summary.
Any PII in the captured LLM interactions has been redacted.
