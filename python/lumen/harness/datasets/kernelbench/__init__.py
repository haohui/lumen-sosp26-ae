"""KernelBench evaluation helpers."""

from typing import Any

__all__ = ["evaluate_generated_model", "evaluate_reference_file"]


def __getattr__(name: str) -> Any:
    if name == "evaluate_reference_file":
        from lumen.harness.datasets.kernelbench.evaluator import evaluate_reference_file

        return evaluate_reference_file
    if name == "evaluate_generated_model":
        from lumen.harness.datasets.kernelbench.evaluator import (
            evaluate_generated_model,
        )

        return evaluate_generated_model
    raise AttributeError(name)
