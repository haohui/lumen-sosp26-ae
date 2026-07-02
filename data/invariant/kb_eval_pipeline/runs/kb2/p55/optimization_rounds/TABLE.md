# GEMM Optimization History

This file is shared state between optimization rounds.
The orchestrator rewrites the history section after each round while preserving this intro.

Interpretation rules for the agent:
- Treat the recorded outcomes as the authoritative history of what has already been tried.
- Avoid repeating a failed optimization unless the new round has a clearly different reason to retry it.
- Prefer incremental decisions that build on the previous round instead of restarting from scratch.

<!-- AUTO-GENERATED HISTORY BELOW -->

| round | prompts | status | compiled | correctness | speedup | ref_us | new_us |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | seed baseline | seeded | None | None | - | - | - |
| 1 | 1 | eval_failed | True | False | - | 0.786 | - |
| 2 | 2 | eval_failed | True | False | - | 0.785 | - |
| 3 | 3 | completed | True | True | 0.9900 | 0.793 | 0.801 |
