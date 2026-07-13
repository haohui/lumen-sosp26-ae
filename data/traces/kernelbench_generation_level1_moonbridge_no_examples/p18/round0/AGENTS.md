# AveLang DSL - KernelBench Agent Context

This working directory is a single KernelBench problem.
Your task is to write an optimized AMD GPU kernel in AveLang DSL that matches
the behavior of `input_model.py`.
Output must go to `output_model_new.py`.

The reference model is in `input_model.py`. Preserve its public input and output
contract exactly. Do not edit `input_model.py`, `eval_config.json`, or
`prompt.txt`.

## Workspace Isolation

Treat this directory as the only experiment workspace. You may read files in the
current directory, the reference notes listed below.

Do not inspect, copy from, or search any `runs/` directory outside this current
round. In particular, never read another problem's or another run's
`output_model_new.py`, `input_model.py`, `eval_result.json`, `codex_result.json`,
`trace.jsonl`, or `meta.json`.

## Reference Notes

- AveLang syntax/API: `/root/lumen-clean/lumen-sosp26-ae-origin/skills/languages/avelang-language-spec.md`

Use only this language specification as a local reference.
Do not inspect local AveLang examples, generic kernel knowledge, or optimization technique notes.

Do not invent Triton-, TVM-, CUDA-, or PyTorch-fallback APIs when an AveLang
implementation is required.

## Critical Code Constraints

- All `@avelang.jit` kernel functions must be defined at module top level.
  Never nest them inside other functions or classes.
- AveLang compiles kernels at import time; runtime kernel definition is not
  supported.
- Keep `ModelNew` compatible with `get_inputs()` and `get_init_inputs()` from
  `input_model.py`.

## Self-Verification Loop

After writing `output_model_new.py`, evaluate correctness and performance with
the KernelBench harness. Use the same Python environment that launched the
generation runner; do not use bare `python`, `python3`, `uv`, or a different
environment for self-checks.

```bash
source /opt/venv/bin/activate
export PYTHONPATH=/root/lumen-clean/lumen-sosp26-ae-origin/python:$PYTHONPATH
/opt/venv/bin/python /root/lumen-clean/lumen-sosp26-ae-origin/python/lumen/tools/cli/kernelbench_graph_eval.py \
  --mode generated \
  --original input_model.py \
  --generated output_model_new.py \
  --eval-config eval_config.json \
  --json-output eval_result.json
```

The harness writes one JSON object to `eval_result.json`; check `compiled`,
`correctness`, `runtime`, and `ref_runtime`. If `compiled` or `correctness` is
false, fix `output_model_new.py` and re-run.
