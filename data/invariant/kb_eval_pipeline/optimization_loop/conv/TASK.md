# Conv2D Optimization Task

This file is the human-maintained task specification for the Conv2D optimization loop.
The orchestrator reads it and injects it into every round prompt.

## Problem Context

- Problem directory: `{{PROBLEM_DIR}}`
- Problem id: `{{PROBLEM_ID}}`
- Problem name: `{{PROBLEM_NAME}}`
- Seed source round: `{{SOURCE_ROUND_NAME}}` at `{{SOURCE_ROUND_PATH}}`
- Optimization workspace: `{{OPTIMIZATION_ROOT}}`

## Evaluation Contract

- Backend: `{{BACKEND}}`
- Precision: `{{PRECISION}}`
- GPU arch: `{{GPU_ARCH}}`
- The evaluator is `kb_eval_pipeline/harness/tools/run_kernelbench_case.py`.
- The reference semantics always come from `roundN/input_model.py`.
- The optimization starting point for each new round comes from the previous round candidate.

## Required Behavior

1. Preserve correctness against the reference model.
2. Optimize the previous candidate incrementally instead of rewriting from scratch unless the previous design is clearly blocking correctness.
3. Prefer robust Conv2D improvements that match the existing Substrate and AMD GPU examples in `/workspace/substrate`.
4. Keep the code easy to validate with the existing tests and KernelBench harness.
5. Respect the ordered hints. Only the hints injected into the current prompt are allowed for that round.

## Output Constraints

- The agent must read the current round's `candidate_input.py` as the kernel starting point.
- The agent must leave the final candidate in the current round's `output_model_new.py`.
- The agent must not edit `TASK.md`, `HINTS.md`, `TABLE.md`, or previous round directories.
- The agent should provide a short textual summary of the concrete kernel changes at the end of the round.

## Operator-Specific Goal

Push the Conv2D kernel toward a correct MFMA implementation on AMD GPUs while preserving the existing launch structure and public behavior.
