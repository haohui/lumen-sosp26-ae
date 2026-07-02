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
| 1 | 1 | completed | True | True | 36.4230 | 0.279 | 0.008 |
| 2 | 2 | completed | True | True | 65.9574 | 0.279 | 0.004 |
| 3 | 3 | completed | True | True | 60.1713 | 0.281 | 0.005 |
