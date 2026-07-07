# Substrate DSL - Project Context

This working directory is a single KernelBench problem.
Your task: write an optimized AMD GPU kernel in **Substrate DSL** that matches
the behavior of `input_model.py`.
Output must go to `output_model_new.py`.

The reference model is in `input_model.py`. Preserve its public input and output
contract exactly.

## Substrate Reference Skills

All skills live in `{skills_root}/`.
Before writing any code, read the required skills in this order:

1. **`substrate-language-spec`** - complete DSL syntax, typing, launch semantics,
   memory/layout conventions. Always required.
   Path: `{lang_spec_skill}`

{examples_section}

Do not Glob or search other directories for skills.
Only read skills from `{skills_root}/`.
Do not invent Triton-, TVM-, or CUDA-style APIs absent from the skill files.

## Critical Code Constraints

- All `@substrate.jit` kernel functions must be defined at module top level.
  Never nest them inside other functions or classes.
- Substrate compiles kernels at import time; runtime kernel definition is not
  supported.

## Self-Verification Loop

After writing `output_model_new.py`, evaluate correctness and performance with
the KernelBench harness. Use the same Python environment that launched the
generation runner; do not use bare `python`, `python3`, `uv`, or a different
environment for self-checks.

```bash
{venv_activation}
export PYTHONPATH={python_root}:$PYTHONPATH
{python_executable} {bench_script} \
  --mode generated \
  --original input_model.py \
  --generated output_model_new.py \
  --eval-config eval_config.json \
  --json-output eval_result.json
```

Use the same environment block above for any ad-hoc import or forward-pass
checks.

The harness writes one JSON object to `eval_result.json`; check
`compiled`, `correctness`, `runtime`, and `ref_runtime`. If `compiled` or
`correctness` is false, fix `output_model_new.py` and re-run.

Do not edit `input_model.py`, `eval_config.json`, or `prompt.txt`.
