# Kernel Benchmark Invariant Study

This repository contains the evaluation pipeline used to study how much
explicit invariants help an agent write optimized GPU kernels. The core
comparison is:

- `optimization_rounds/`: prompt includes invariant information.
- `optimization_rounds_no_invariants/`: prompt removes the invariant block.

## Required Workspace Layout

The prompts and saved run artifacts use absolute paths. Put the repositories in
these locations before running experiments:

```text
/workspace/kb_eval_pipeline
/workspace/substrate
```

`/workspace/kb_eval_pipeline` is this repository.

`/workspace/substrate` is the Substrate DSL repository used by the generated
kernels and tests.

## Substrate Cleanup Before Running

Before running the no-invariant experiments, remove or edit any Substrate files
that reveal the MFMA invariant mapping. Otherwise the agent may simply read the
answer from examples instead of deriving it.

In particular, check files such as:

```text
/workspace/substrate/test/examples/gemm/amdgpu/test_gemm_mfma.py
```

and remove the parts that expose:

- MFMA operand lane/register packing.
- accumulator index to output row/column mapping.
- complete example kernels that already implement the target mapping.
- other previous kernels or tests that encode the same invariant.

The no-invariant condition is only meaningful if `/workspace/substrate` does
not contain a readable implementation of the hidden invariant.

## Main Components

`harness/`
: KernelBench-based compile, correctness, profiling, timing, and event logging
tools.

`optimization_loop/`
: Multi-round Codex optimization runner and prompt templates for `gemm` and
`conv`.

`runs/`
: Saved experiment outputs: prompts, generated kernels, Codex traces, metadata,
and per-round tables.

`compute_api_request_duration.py`
: Post-processing script for Codex `agent_trace.jsonl` files. It estimates LLM
request durations and trace wall-clock durations.

`compute_harness_event_duration.py`
: Post-processing script for `harness_events.jsonl` files. It sums evaluator
calls, evaluator wall-clock time, compilation time, and execution/profile time.

## Running Optimization

Example with invariants:

```bash
cd /workspace/kb_eval_pipeline
python optimization_loop/run_optimization_loop.py \
  --run-id kb1 \
  --problems p01,p02,p03 \
  --parallel-devices 0,1,2 \
  --agent codex \
  --template gemm \
  --max-rounds 3
```

Example without invariants:

```bash
cd /workspace/kb_eval_pipeline
python optimization_loop/run_optimization_loop.py \
  --run-id kb1 \
  --problems p01,p02,p03 \
  --parallel-devices 0,1,2 \
  --agent codex \
  --template gemm \
  --no-invariant \
  --max-rounds 3
```

Use `--template conv` for the conv prompt set.

## Reporting Table Metrics

The main reporting table is:

| Setting | Pass@1 % | Cost($) | # of LLM msgs | Wall Clock(s) | LLM time(s) | Compilation(s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L1 |  |  |  |  |  |  |
| L1+Inv |  |  |  |  |  |  |
| L2 |  |  |  |  |  |  |
| L2+Inv |  |  |  |  |  |  |

Rows are defined as:

- `L1`: `runs/kb1`, without invariants, usually `optimization_rounds_no_invariants`.
- `L1+Inv`: `runs/kb1`, with invariants, usually `optimization_rounds`.
- `L2`: `runs/kb2`, without invariants, usually `optimization_rounds_no_invariants`.
- `L2+Inv`: `runs/kb2`, with invariants, usually `optimization_rounds`.

### Pass@1 %

`Pass@1 %` is the percentage of problems whose final optimized kernel is correct
after all optimization rounds.

Correctness alone is not enough. Each passing kernel should also be manually
inspected and marked failed if it uses reward hacking, puts the required MFMA
path off the main execution path, bypasses the intended optimization, or falls
back to unrelated framework operators instead of solving the requested kernel.

### Cost($)

The Codex trace records token usage. Sum the relevant input/output/cache token
counts for the selected runs, then multiply by the model price used in the
experiment.

### # of LLM msgs, Wall Clock(s), and LLM time(s)

Use `compute_api_request_duration.py` on saved `agent_trace.jsonl` files.

The script has two useful modes:

- `--mode jsonl`: one output row per inferred LLM API request, with
  `api_request.duration_ms`, `response_items`, `function_calls`, and `token
  usage`.
- `--mode trace-jsonl`: one output row per agent trace, with
  `wall_clock.duration_ms`.

In the table:

- `# of LLM msgs` = sum of `response_items` from `--mode jsonl`.
- `Wall Clock(s)` = sum of `wall_clock.duration_ms` from `--mode trace-jsonl` / 1000.
- `LLM time(s)` = sum of `api_request.duration_ms` from `--mode jsonl` / 1000.

Example for one setting; change `kb1` and `optimization_rounds` as needed:

```bash
cd /workspace/kb_eval_pipeline

find runs/kb1 -path '*/optimization_rounds/round*/agent_trace.jsonl' \
  -print0 \
  | xargs -0 python3 compute_api_request_duration.py --mode jsonl \
  | jq -s '{llm_msgs:(map(.response_items)|add), llm_time_s:((map(."api_request.duration_ms")|add)/1000)}'

find runs/kb1 -path '*/optimization_rounds/round*/agent_trace.jsonl' \
  -print0 \
  | xargs -0 python3 compute_api_request_duration.py --mode trace-jsonl \
  | jq -s '{wall_clock_s:((map(."wall_clock.duration_ms")|add)/1000)}'
```

### Compilation(s)

Use `compute_harness_event_duration.py` on saved `harness_events.jsonl` files.

The script outputs one row per `harness_events.jsonl` file:

- `harness_runs`: number of evaluator/harness calls.
- `evaluation.duration_s`: sum of `harness_eval_end.duration_s`.
- `compilation.duration_s`: sum of `compilation_end.duration_s`.
- `execution.duration_s`: sum of `execution_end.duration_s`.

Example for running this script:

```bash
cd /workspace/kb_eval_pipeline

find runs/kb1 -path '*/optimization_rounds/round*/harness_events.jsonl' \
  -print0 \
  | xargs -0 python3 compute_harness_event_duration.py \
  | jq -s '{runs:(map(.harness_runs)|add), compilation_s:(map(."compilation.duration_s")|add), execution_s:(map(."execution.duration_s")|add)}'
```
