# Optimization loop

Multi-round orchestration for **KernelBench-style problems**: copy an existing
eval round as seed, then run **Codex** or **Claude** for several rounds. Each
round improves the previous candidate, runs
`data/invariant/kb_eval_pipeline/harness/tools/run_kernelbench_case.py`, and
records results under `optimization_rounds/`.

All paths below are relative to the repository root `lumen-sosp26-ae/`.

The checked-in `data/invariant/kb_eval_pipeline/runs/` directory is a compact
archive for inspection. Its `optimization_rounds*` directories keep only final
kernels and summaries, not the full prompts, per-round metadata, and agent
logs produced by this loop. Use a fresh output suffix or an untrimmed workspace
when resuming or regenerating full optimization traces.

## Layout (this directory)

| Path | Role |
| --- | --- |
| `run_optimization_loop.py` | CLI entrypoint: seed, prompts, agent subprocess, harness eval, `TABLE.md` / `meta.json` updates. |
| `gemm/` | Default **template** (`TASK.md`, `HINTS.md`, `TABLE.md`) with `{{PLACEHOLDER}}` fields filled on first run. |
| `AGENTS.md` | Optional human-written guidance for agents (profiling, skills paths); not consumed by the script. |

Add another operator family by creating `optimization_loop/<name>/` with the same three template files and passing `--template <name>`.

## Problem directory input

You can target either a single problem with `--problem-dir`, or a serial batch with `--run-id` plus `--problems`.

- `--problem-dir` must point to one KernelBench problem folder (for example
  `data/invariant/kb_eval_pipeline/runs/kb1/p01`) that already contains one or
  more `roundN/` subdirectories.
- `--run-id kb1 --problems p09,p12` runs the same optimization command serially
  for `data/invariant/kb_eval_pipeline/runs/kb1/p09` and then
  `data/invariant/kb_eval_pipeline/runs/kb1/p12`.
- `--parallel-devices 0,1,2,3,4,5,6,7` lets a batch run spread problems across those GPUs in parallel, one problem per GPU worker.
- When `--agent claude` is used without `--output-suffix`, results default to `optimization_rounds_claude/` (or `optimization_rounds_no_invariants_claude/`) so they do not collide with Codex runs.

The orchestrator:

1. Picks the **highest-numbered** `roundN` as the **source round**.
2. Creates `<problem-dir>/<output-dir>/round0` by copying that source (seed metadata is written into `round0/meta.json`).
3. If `<output-dir>/{TASK,HINTS,TABLE}.md` are missing, copies them from the chosen template and substitutes context (problem id, backend, precision, paths, etc.).

Optimization rounds are `<output-dir>/round1`, `round2`, ... (see below).

## Prompt schedule

`HINTS.md` is split into numbered sections (headers matching `## N : Title` / similar—see `HINT_HEADER_RE` in `run_optimization_loop.py`). These sections now act as the round prompt schedule.

With the default `--prompt-mode sequential`, **round *k*** (`k >= 1`) uses only section *k* from `HINTS.md`. With `--prompt-mode cumulative`, round *k* includes all sections `1..k`.

This makes it straightforward to define exactly three prompts and run them in order on the same kernel.

The number of optimization rounds actually run is still controlled by `--max-rounds`.

## Per-round artifacts

For each `roundN` (`N >= 1`):

- **Inputs copied from `round{N-1}`:** `input_model.py`, `eval_config.json`; previous `output_model_new.py` → this round’s `candidate_input.py` and initial `output_model_new.py`.
- **Written by orchestrator:** `prompt.txt`, `meta.json`, `token_usage.json`.
- **Written by agent:** updated `output_model_new.py` (required contract).
- **On failure:** `error.txt` may be present.

## Generated summaries

- **`<output-dir>/TABLE.md`:** human-written prefix is preserved; after `<!-- AUTO-GENERATED HISTORY BELOW -->` the script appends a history table and notes from each round's `meta.json`.
- **`<output-dir>/meta.json`:** run-level summary (all rounds, best speedup among correctness-passing rows, token summary, etc.).
- **`<output-dir>/token_usage_summary.json`:** aggregated token usage across rounds when available from the agent session logs.

## Resuming

If `roundK/meta.json` already has a completed `optimization_loop.status`, that round is **skipped**. Re-run the same command to continue later rounds.

## Requirements

- From the **repository root**: `codex` and/or `claude-yolo` on `PATH` when
  using `--agent claude` (the script invokes the `claude-yolo` CLI).
- Python env able to run `run_kernelbench_case.py` (same as normal KernelBench eval).

## Claude provider selection

The script now supports two Claude provider env sets:

- `--claude-provider default`
  Uses the current shell's `ANTHROPIC_BASE_URL` and `ANTHROPIC_AUTH_TOKEN`.
- `--claude-provider zhipu`
  Replaces those at launch time with:
  - `ANTHROPIC_BASE_URL_ZHIPU`
  - `ANTHROPIC_AUTH_TOKEN_ZHIPU` when `--claude-zhipu-key-slot 1`
  - `ANTHROPIC_AUTH_TOKEN_ZHIPU_2` when `--claude-zhipu-key-slot 2`

Example shell setup:

```bash
export ANTHROPIC_BASE_URL="..."
export ANTHROPIC_AUTH_TOKEN="..."

export ANTHROPIC_BASE_URL_ZHIPU="..."
export ANTHROPIC_AUTH_TOKEN_ZHIPU="..."
export ANTHROPIC_AUTH_TOKEN_ZHIPU_2="..."
```

Then choose at runtime:

```bash
--agent claude --claude-provider default
```

or:

```bash
--agent claude --claude-provider zhipu
```

Use the second key:

```bash
--agent claude --claude-provider zhipu --claude-zhipu-key-slot 2
```

## Usage

From the `lumen-sosp26-ae/` repo root:

```bash
python3 data/invariant/kb_eval_pipeline/optimization_loop/run_optimization_loop.py \
  --problem-dir data/invariant/kb_eval_pipeline/runs/<run_id>/p01 \
  --agent claude \
  --claude-provider default \
  --template gemm \
  --max-rounds 3
```

Serial batch example:

```bash
python3 data/invariant/kb_eval_pipeline/optimization_loop/run_optimization_loop.py \
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
`data/invariant/kb_eval_pipeline/harness/README.md`.

## Related docs

- Harness: `data/invariant/kb_eval_pipeline/harness/README.md`
- Run artifact layout: `data/invariant/kb_eval_pipeline/runs/README.md`
