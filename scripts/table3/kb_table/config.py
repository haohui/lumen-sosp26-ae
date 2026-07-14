"""Static paths and default run directory names for Table 3."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_RUN_DIRS = {
    "generation_l1": "kernelbench_generation_level1_dsv4",
    "generation_l2": "kernelbench_generation_level2_dsv4",
    "generation_no_examples_l1": "kernelbench_generation_level1_dsv4_no_examples",
    "generation_no_examples_l2": "kernelbench_generation_level2_dsv4_no_examples",
    "optimization_invariants_l1": "kernelbench_optimization_level1_invariants",
    "optimization_invariants_l2": "kernelbench_optimization_level2_invariants",
    "optimization_no_invariants_l1": "kernelbench_optimization_level1_no_invariants",
    "optimization_no_invariants_l2": "kernelbench_optimization_level2_no_invariants",
}
