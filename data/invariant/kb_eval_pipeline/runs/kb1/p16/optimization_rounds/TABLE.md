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
| 1 | 1 | completed | True | True | 0.0143 | 0.249 | 17.400 |
| 2 | 2 | completed | True | True | 0.0326 | 0.270 | 8.270 |
| 3 | 3 | completed | True | True | 0.0311 | 0.253 | 8.140 |
