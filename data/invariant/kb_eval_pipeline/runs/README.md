# Run artifact layout

This directory contains a compact archive of invariant-ablation outputs for
AE inspection. All paths below are relative to the repository root
`lumen-sosp26-ae/`.

## Top-level runs

`kb1/` and `kb2/` are the two KernelBench run groups included in the artifact.
Each contains problem directories named `p<id>`, for example `p01`, `p80`, or
`p100`.

```text
runs/
├── kb1/
│   └── p01/
└── kb2/
    └── p09/
```

## Problem directory

Each problem directory may contain original seed rounds plus one or more
optimization-variant directories:

| Directory | Meaning |
| --- | --- |
| `round0/`, `round1/`, ... | Original imported seed/evaluation attempts. |
| `optimization_rounds/` | Codex run with invariant prompts. |
| `optimization_rounds_no_invariants/` | Codex run without the invariant prompt. |
| `optimization_rounds_claude/` | GLM-5 run through the Claude-compatible CLI with invariant prompts. |
| `optimization_rounds_no_invariants_claude/` | GLM-5 run through the Claude-compatible CLI without the invariant prompt. |

A typical compact problem directory looks like this:

```text
p80/
├── round0/
│   ├── input_model.py
│   ├── output_model_new.py
│   ├── eval_config.json
│   └── meta.json
├── optimization_rounds/
│   ├── final_summary.json
│   ├── token_usage_summary.json
│   └── round3/
│       └── output_model_new.py
└── optimization_rounds_no_invariants/
    ├── final_summary.json
    ├── token_usage_summary.json
    └── round3/
        └── output_model_new.py
```

Some problems or variants may be missing if that run was not launched or did
not produce an artifact.

## Optimization round contents

The `optimization_rounds*` directories are intentionally trimmed. For each
variant, the archive keeps only:

| File | Meaning |
| --- | --- |
| `roundN/output_model_new.py` | Final available optimized implementation, where `roundN` is the highest-numbered round that produced `output_model_new.py`. |
| `final_summary.json` | Final selected result summary when present. |
| `token_usage_summary.json` | Aggregated token usage across rounds when present. |

Intermediate prompts, per-round metadata, agent stdout/stderr, and tool traces
are omitted from the checked-in archive to keep the AE payload small.

## Run-level summaries

The checked-in results are intended for inspecting final kernels and summary
metrics. Intermediate logs and traces are not included in this compact archive,
so these directories are not a full resume point for the optimization loop.

To launch new runs, use the optimization loop documented in
`data/invariant/kb_eval_pipeline/optimization_loop/README.md`.
