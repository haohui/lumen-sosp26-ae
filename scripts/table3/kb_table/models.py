"""Small data containers used by the KernelBench Table 3 summarizer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RoundRecord:
    problem_id: int
    round_index: int
    round_dir: Path
    correct: bool
    speedup: float | None
    reward_hacking: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return (
            self.correct
            and self.speedup is not None
            and self.speedup > 0
            and not self.reward_hacking
        )


@dataclass(frozen=True)
class GenerationStats:
    denominator: int
    valid_count: int
    geom: float | None
    min_speedup: float | None
    max_speedup: float | None
    gt1_count: int
    pass1: int
    pass3: int
    avg_files_read: float | None


@dataclass(frozen=True)
class OptimizationStats:
    denominator: int
    pass_final: int
    avg_token_usage: float | None
