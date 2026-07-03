# Optimization loop

Multi-round orchestration for **KernelBench-style problems**: copy an existing
eval round as seed, then run **Codex** for several rounds. Each round improves
the previous candidate, runs
`harness/tools/run_kernelbench_case.py`, and
records results under `optimization_rounds/`. Conv templates are different:
they rely on their own pytest validation and do not use the KernelBench case
harness.

All paths below are relative to the `kb_eval_pipeline/` repository root.

## Layout (this directory)

| Path | Role |
| --- | --- |
| `run_optimization_loop.py` | CLI entrypoint: seed, prompts, agent subprocess, harness eval, `TABLE.md` / `meta.json` updates. |
| `gemm/` | Gemm prompt template (`HINTS.md`, `TABLE.md`, optional no-invariants prompt). |
| `conv/` | Conv prompt template. Conv rounds use pytest validation instead of `run_kernelbench_case.py`. |

Add another operator family by creating `optimization_loop/<name>/` with the same template files and passing `--template <name>`.
If `--no-invariant` is enabled, prompt 1 is replaced by
`optimization_loop/<name>/prompt1_no_invariants.md`.

## Problem directory input

You can target either a single problem with `--problem-dir`, or a serial batch with `--run-id` plus `--problems`.

- `--problem-dir` must point to one KernelBench problem folder (for example
  `runs/kb1/p01`) that already contains one or
  more `roundN/` subdirectories.
- `--run-id kb1 --problems p09,p12` runs the same optimization command serially
  for `runs/kb1/p09` and then
  `runs/kb1/p12`.
- `--parallel-devices 0,1,2,3,4,5,6,7` lets a batch run spread problems across those GPUs in parallel, one problem per GPU worker.

The orchestrator:

1. Picks the **highest-numbered** `roundN` as the **source round**.
2. Creates `<problem-dir>/optimization_rounds/round0` by copying that source.
3. Copies `TABLE.md` from the chosen template if it is missing.

Optimization rounds are `optimization_rounds/round1`, `round2`, … (see below).

## Prompt schedule

`HINTS.md` is split into numbered sections (headers matching `## N : Title` / similar—see `HINT_HEADER_RE` in `run_optimization_loop.py`). These sections now act as the round prompt schedule.

Round *k* (`k >= 1`) uses section *k* from `HINTS.md`. If a later round has
no matching section, the last section is reused.

This makes it straightforward to define exactly three prompts and run them in order on the same kernel.

The number of optimization rounds actually run is still controlled by `--max-rounds`.

## Per-round artifacts

For each `roundN` (`N >= 1`):

- **Inputs copied from `round{N-1}`:** `input_model.py`, `eval_config.json`; previous `output_model_new.py` → this round’s `candidate_input.py` and initial `output_model_new.py`.
- **Written by orchestrator:** `prompt.txt`, `meta.json`; gemm rounds also
  record harness timing events.
- **Written by agent:** updated `output_model_new.py` (required contract).
- **On failure:** `error.txt` may be present.

## Generated summaries

- **`optimization_rounds/TABLE.md`:** human-written prefix is preserved; after `<!-- AUTO-GENERATED HISTORY BELOW -->` the script appends a history table and notes from each round’s `meta.json`.
## Resuming

If `roundK/meta.json` already has a completed `optimization_loop.status`, that round is **skipped**. Re-run the same command to continue later rounds.

## Requirements

- From the **repository root**: `codex` on `PATH`.
- Python env able to run `run_kernelbench_case.py` (same as normal KernelBench eval).

## Usage

From the `kb_eval_pipeline/` repo root:

```bash
python optimization_loop/run_optimization_loop.py \
  --problem-dir runs/<run_id>/p01 \
  --agent codex \
  --template gemm \
  --max-rounds 3
```

Serial batch example:

```bash
python optimization_loop/run_optimization_loop.py \
  --run-id kb1 \
  --problems p09,p12 \
  --parallel-devices 0,1,2,3,4,5,6,7 \
  --agent codex \
  --template gemm \
  --max-rounds 3
```

See **`--help`** for the full flag list (model, effort, agent timeout, GPU
device, correctness trials, `--timing-method`, `--measure-performance` /
`--no-measure-performance`, repeated `--agent-arg`, Codex sandbox bypass
toggles, etc.). Evaluator flags are forwarded to `run_kernelbench_case.py`;
behavior of list vs directory inputs is documented in
`harness/README.md`.

## Related docs

- Harness: `harness/README.md`
- Run artifact layout: `runs/README.md`
