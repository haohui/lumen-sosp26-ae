# Invariant AE data

This directory packages the artifact-evaluation assets for the MFMA invariant
ablation. All paths below are relative to the repository root
`lumen-sosp26-ae/`.

The AE material is intentionally split into two reviewable parts:

1. Templates, harness glue, and documentation.
2. Compact run artifacts.

This split keeps the code needed to reproduce the data separate from the data
produced by the experiments.

## Directory map

| Path | Purpose |
| --- | --- |
| `data/invariant/kb_eval_pipeline/optimization_loop/` | Multi-round optimization driver and prompt templates. |
| `data/invariant/kb_eval_pipeline/harness/` | Vendored KernelBench-style evaluator used by the optimization loop. |
| `data/invariant/kb_eval_pipeline/runs/` | Compact experiment outputs for `kb1` and `kb2`. |

## Variants

The compact run archive uses these optimization-variant directories:

| Variant | Agent | Prompt setting | Output directory |
| --- | --- | --- | --- |
| with invariants | Codex | MFMA invariant prompts enabled | `optimization_rounds/` |
| without invariants | Codex | MFMA invariant prompt disabled | `optimization_rounds_no_invariants/` |
| with invariants, GLM-5 | GLM-5 through the Claude-compatible CLI | MFMA invariant prompts enabled | `optimization_rounds_claude/` |
| without invariants, GLM-5 | GLM-5 through the Claude-compatible CLI | MFMA invariant prompt disabled | `optimization_rounds_no_invariants_claude/` |

## Reproducibility notes

- The wrappers default to the vendored workspace under
  `data/invariant/kb_eval_pipeline/`; no sibling checkout is required.
- Run the commands from the repository root.
- The GLM-5 variants expect Zhipu-compatible Claude environment variables to
  be configured before launch. See
  `data/invariant/kb_eval_pipeline/optimization_loop/README.md` for the exact
  variables.
- The archived `runs/` tree was cleaned for artifact packaging. Each
  `optimization_rounds*` directory keeps only the final available
  `output_model_new.py`, `final_summary.json`, and `token_usage_summary.json`;
  intermediate logs, traces, prompts, crash dumps such as `gpucore.*`, and
  Python `__pycache__/` directories were removed.

## Related documentation

- Optimization loop:
  `data/invariant/kb_eval_pipeline/optimization_loop/README.md`
- Run artifact layout:
  `data/invariant/kb_eval_pipeline/runs/README.md`
- Harness:
  `data/invariant/kb_eval_pipeline/harness/README.md`
