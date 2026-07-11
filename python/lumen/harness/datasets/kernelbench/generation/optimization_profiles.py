"""Declarative prompt profiles for iterative KernelBench optimization."""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib.resources import files
from typing import Literal

TemplateFamily = Literal["gemm", "conv"]


@dataclass(frozen=True)
class OptimizationProfile:
    name: str
    prompt_config_name: str
    prompt_name: str
    round_resources: dict[TemplateFamily, tuple[str, ...]]

    def template_family(self, ref_arch_src: str) -> TemplateFamily:
        if re.search(
            r"\b(?:nn\.)?Conv[123]d\b|\bF\.conv[123]d\b|\btorch\.conv[123]d\b|"
            r"\bconv[123]d\b|\bconvolution\b",
            ref_arch_src,
            flags=re.IGNORECASE,
        ):
            return "conv"
        return "gemm"

    def guidance(self, ref_arch_src: str, round_index: int) -> tuple[str, str]:
        if round_index < 0:
            raise ValueError("round_index must be non-negative")
        family = self.template_family(ref_arch_src)
        resources = self.round_resources[family]
        if round_index >= len(resources):
            raise ValueError(
                f"profile {self.name!r} has no guidance for round {round_index}"
            )
        resource = resources[round_index]
        text = _resource(family, resource).read_text(encoding="utf-8")
        return family, _hint_for_round(text, round_index + 1)

    def round_count(self, ref_arch_src: str) -> int:
        return len(self.round_resources[self.template_family(ref_arch_src)])


_FULL = ("HINTS.md", "HINTS.md", "HINTS.md")
PROFILES = {
    "invariants": OptimizationProfile(
        name="invariants",
        prompt_config_name="optimization_prompt_config.toml",
        prompt_name="avelang_amd_invariant",
        round_resources={"gemm": _FULL, "conv": _FULL},
    ),
    "no-invariants": OptimizationProfile(
        name="no-invariants",
        prompt_config_name="optimization_prompt_config.toml",
        prompt_name="avelang_amd_invariant",
        round_resources={
            "gemm": ("prompt1_no_invariants.md", "HINTS.md", "HINTS.md"),
            "conv": ("prompt1_no_invariants.md", "HINTS.md"),
        },
    ),
}


def get_profile(name: str) -> OptimizationProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(PROFILES))
        raise ValueError(
            f"unknown optimization profile {name!r}; choose one of {choices}"
        ) from exc


def _resource(family: TemplateFamily, name: str):
    return files("lumen.harness.datasets.kernelbench").joinpath(
        "optimization_templates", family, name
    )


def _hint_for_round(text: str, number: int) -> str:
    header = re.compile(r"^##\s*(?:Hint\s*)?(\d+)\s*[:.\-]\s*.+?\s*$")
    lines: list[str] = []
    active = False
    for line in text.splitlines():
        match = header.match(line.strip())
        if match:
            if active:
                break
            active = int(match.group(1)) == number
        if active:
            lines.append(line)
    if not lines:
        raise ValueError(f"guidance resource has no Hint {number}")
    return "\n".join(lines).strip()
