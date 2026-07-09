"""Invariant prompt experiment for KernelBench generation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal

from lumen.harness.datasets.kernelbench.generation.artifacts import (
    write_problem_meta,
)
from lumen.harness.datasets.kernelbench.generation.orchestrator import (
    run_generation_experiment,
)
from lumen.harness.datasets.kernelbench.generation.prompt import (
    WRITE_FILE_DIRECTIVE,
)
from lumen.harness.datasets.kernelbench.generation.round_runner import (
    run_generation_round,
)
from lumen.harness.datasets.kernelbench.generation.types import (
    GenerationConfig,
    WorkArgs,
)
from lumen.harness.datasets.kernelbench.generation.workspace import (
    prepare_invariant_round_workspace,
)

InvariantPromptVariant = Literal["invariants", "no-invariants"]

_HINT_HEADER_RE = re.compile(r"^##\s*(?:Hint\s*)?(\d+)\s*[:.\-]\s*(.+?)\s*$")


@dataclass(frozen=True)
class _HintSection:
    number: int
    markdown: str


def run_invariant_generation(
    config: GenerationConfig,
    *,
    prompt_variant: InvariantPromptVariant = "invariants",
) -> None:
    """Run the invariant experiment on the standard generation framework."""
    if prompt_variant not in ("invariants", "no-invariants"):
        raise ValueError(f"unknown invariant prompt variant: {prompt_variant}")

    rounds_subdir = invariant_rounds_subdir(prompt_variant)
    run_generation_experiment(
        config,
        lambda work, generation_config, dataset, run_dir: generate_invariant_problem(
            work,
            generation_config,
            dataset,
            run_dir,
            prompt_variant=prompt_variant,
        ),
        action="Generating invariant variants for",
        rounds_subdir=rounds_subdir,
        problem_meta_subdir=rounds_subdir,
    )


def generate_invariant_problem(
    work: WorkArgs,
    config: GenerationConfig,
    dataset: Any,
    run_dir: Path,
    *,
    prompt_variant: InvariantPromptVariant,
) -> bool:
    """Generate a chained sequence of invariant optimization rounds."""
    problem = dataset.get_problem_by_id(work.problem_id)
    problem_dir = run_dir / f"p{work.problem_id:02d}"
    rounds_dir = problem_dir / invariant_rounds_subdir(prompt_variant)
    rounds_dir.mkdir(parents=True, exist_ok=True)

    round_metas: list[dict[str, Any]] = []
    any_correct = False
    candidate_src: str | None = None
    for round_idx in range(max(1, int(config.codex.max_retries))):
        round_dir = rounds_dir / f"round{round_idx}"
        workspace = prepare_invariant_round_workspace(
            round_dir,
            ref_arch_src=problem.code,
            candidate_src=candidate_src,
            evaluation=config.evaluation,
            prompt_transform=lambda base_prompt, index=round_idx: (
                build_invariant_prompt(
                    base_prompt,
                    round_index=index,
                    prompt_variant=prompt_variant,
                )
            ),
        )
        round_meta, round_correct, _ = run_generation_round(
            tag=f"p{work.problem_id:02d}/round{round_idx}",
            round_dir=round_dir,
            work=work,
            config=config,
            problem_name=problem.name,
            ref_arch_src=problem.code,
            prepared_workspace=workspace,
        )
        round_metas.append(round_meta)
        any_correct = any_correct or round_correct
        output_path = round_dir / "output_model_new.py"
        codex_result_path = round_dir / "codex_result.json"
        codex_result = (
            json.loads(codex_result_path.read_text(encoding="utf-8"))
            if codex_result_path.is_file()
            else {}
        )
        if not codex_result.get("ok") or not output_path.is_file():
            break
        candidate_src = output_path.read_text(encoding="utf-8")
        if not candidate_src.strip():
            break

    write_problem_meta(
        rounds_dir,
        problem_id=work.problem_id,
        problem_name=problem.name,
        round_metas=round_metas,
    )
    return any_correct or bool(round_metas)


def invariant_rounds_subdir(prompt_variant: InvariantPromptVariant) -> str:
    """Return the per-problem directory for an invariant prompt variant."""
    if prompt_variant == "invariants":
        return "optimization_rounds"
    if prompt_variant == "no-invariants":
        return "optimization_rounds_no_invariant"
    raise ValueError(f"unknown invariant prompt variant: {prompt_variant}")


def build_invariant_prompt(
    base_prompt: str,
    *,
    round_index: int,
    prompt_variant: InvariantPromptVariant = "invariants",
) -> str:
    """Add only the guidance assigned to this invariant round."""
    if round_index < 0:
        raise ValueError("round_index must be non-negative")

    hints = _load_gemm_hints(prompt_variant)
    selected = [hint for hint in hints if hint.number == round_index + 1]
    if not selected:
        raise ValueError(
            f"the GEMM invariant template has no hint for round {round_index}"
        )

    experiment_prompt = "\n\n".join(hint.markdown for hint in selected)
    block = (
        "## Optimization guidance\n\n"
        "Write `output_model_new.py` so that it applies the optimization "
        "guidance below.\n\n"
        f"{experiment_prompt.strip()}"
    )
    return _insert_before_write_directive(base_prompt, block)


def _load_gemm_hints(
    prompt_variant: InvariantPromptVariant,
) -> list[_HintSection]:
    full_template = _template_resource("HINTS.md").read_text(encoding="utf-8")
    hints = _parse_hints(full_template)
    if prompt_variant == "invariants":
        return hints

    no_invariants = _parse_hints(
        _template_resource("prompt1_no_invariants.md").read_text(encoding="utf-8")
    )
    if not no_invariants:
        raise ValueError("the no-invariants template contains no hints")
    return [no_invariants[0], *(hint for hint in hints if hint.number != 1)]


def _parse_hints(text: str) -> list[_HintSection]:
    sections: list[_HintSection] = []
    current_number: int | None = None
    current_lines: list[str] = []

    for line in text.splitlines():
        match = _HINT_HEADER_RE.match(line.strip())
        if match:
            if current_number is not None:
                sections.append(
                    _HintSection(current_number, "\n".join(current_lines).strip())
                )
            current_number = int(match.group(1))
            current_lines = [line]
        elif current_number is not None:
            current_lines.append(line)

    if current_number is not None:
        sections.append(
            _HintSection(current_number, "\n".join(current_lines).strip())
        )
    return sections


def _insert_before_write_directive(base_prompt: str, block: str) -> str:
    prompt = base_prompt.rstrip()
    if prompt.endswith(WRITE_FILE_DIRECTIVE):
        prompt = prompt[: -len(WRITE_FILE_DIRECTIVE)].rstrip()
        return f"{prompt}\n\n{block.strip()}{WRITE_FILE_DIRECTIVE}"
    return f"{prompt}\n\n{block.strip()}\n"


def _template_resource(name: str):
    return files("lumen.harness.datasets.kernelbench").joinpath(
        "optimization_templates",
        "gemm",
        name,
    )
