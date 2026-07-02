# Run Artifacts

This directory stores optimization-loop outputs. Paths in this document are
relative to the `kb_eval_pipeline/` repository root.

## Directory Layout

```text
runs/
├── kb1/
│   └── p01/
│       ├── round0/
│       ├── optimization_rounds/
│       │   ├── TABLE.md
│       │   ├── round0/
│       │   ├── round1/
│       │   └── ...
│       └── optimization_rounds_no_invariants/
│           ├── TABLE.md
│           ├── round0/
│           ├── round1/
│           └── ...
└── kb2/
    └── p09/
        └── ...
```

`kb1/` and `kb2/` are KernelBench level folders.

`pXX/` is one problem.

`round0/` directly under `pXX/` is the naive implementation to optimize.

`optimization_rounds/` contains runs with invariants.

`optimization_rounds_no_invariants/` contains runs without the invariant in prompt.

## Optimization Root Files

Each `optimization_rounds*` directory contains:

`TABLE.md`
: A human-readable per-round summary table. It is regenerated from each
round's `meta.json` and is only for quick inspection.

`round0/`
: The seed baseline to optimize. It is not an agent
optimization round.

`round1/`, `round2/`, ...
: Agent optimization rounds. Each round starts from the previous round's
`output_model_new.py` and apply the prompt using agent.

## Per-Round Files

`input_model.py`
: The reference model for this problem using pytorch. The evaluator imports this to build
inputs and compare correctness.

`candidate_input.py`
: The starting kernel given to the agent for this round. For `round1`, this is
copied from the seed output. For later rounds, it is copied from the previous
round's `output_model_new.py`.

`output_model_new.py`
: The final kernel written by the agent for this round. This is the file to
inspect for reward hacking, missing MFMA main path, or skipped optimization.

`prompt.txt`
: The exact prompt sent to Codex agent for this round after template rendering and
path substitution.

`agent_trace.jsonl`
: Raw Codex session trace copied from `~/.codex/sessions`. This is the source
for LLM message counts, wall-clock/API timing post-processing, tool calls, and
what files the agent read or modified.

`meta.json`
: Compact round metadata written by the optimization loop: problem id, agent
exit status, correctness/runtime fields when an evaluator ran, trace path,
round number, timestamps, and error state.

`eval_config.json`
: Evaluator configuration for the problem. It records the benchmark/evaluation
settings needed to reproduce the run.

`harness_events.jsonl`
: Gemm-only evaluator timeline emitted during agent debug eval. It records
harness phases such as compilation and execution/profile timing. Conv rounds
do not use the KernelBench case harness in this pipeline, so this file is not
expected there.

`case.txt`
: Gemm-only round-local case list. It contains exactly one line: the absolute
path of the current round directory. Conv rounds do not need or generate this
file.

`error.txt`
: Optional. Present only when the agent or evaluator failed. It contains the
short failure reason.
