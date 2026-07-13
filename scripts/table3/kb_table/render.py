"""Markdown and LaTeX table rendering."""

from __future__ import annotations

from typing import Any

from kb_table.models import GenerationStats, OptimizationStats


def print_markdown(rows: dict[int, dict[str, Any]]) -> None:
    print(
        "| Level | Valid% | GeoMean | Min | Max | >1x | Pass@1% | Pass@3% | "
        "Avg. files read | Pass@1% | Avg. token usage |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for level in (1, 2):
        print("| " + " | ".join(markdown_cells(level, rows[level])) + " |")


def print_latex(rows: dict[int, dict[str, Any]]) -> None:
    print(r"\begin{tabular}{lrrrrrrrrrr}")
    print(r"\toprule")
    print(
        r"& \multicolumn{5}{c}{Performance} "
        r"& \multicolumn{3}{c}{Adding in-context examples} "
        r"& \multicolumn{2}{c}{Adding Invariants} \\"
    )
    print(
        r"& Valid\% & GeoMean & Min & Max & $>1\times$ "
        r"& Pass@1\% & Pass@3\% & Avg. files read "
        r"& Pass@1\% & Avg. token usage \\"
    )
    print(r"\midrule")
    for level in (1, 2):
        print(" & ".join(latex_cells(level, rows[level])) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


def markdown_cells(level: int, row: dict[str, Any]) -> list[str]:
    gen, gen_no, opt_no, opt_inv = unpack_row(row)
    return [
        f"Level {level}",
        format_percent(gen.valid_count, gen.denominator),
        format_float(gen.geom),
        format_float(gen.min_speedup),
        format_float(gen.max_speedup),
        str(gen.gt1_count),
        format_arrow_count(gen_no.pass1, gen.pass1),
        format_arrow_count(gen_no.pass3, gen.pass3),
        format_arrow_float(gen_no.avg_files_read, gen.avg_files_read),
        format_arrow_count(opt_no.pass_final, opt_inv.pass_final),
        format_arrow_tokens(opt_no.avg_token_usage, opt_inv.avg_token_usage),
    ]


def latex_cells(level: int, row: dict[str, Any]) -> list[str]:
    gen, gen_no, opt_no, opt_inv = unpack_row(row)
    return [
        f"Level {level}",
        format_percent(gen.valid_count, gen.denominator).replace("%", r"\%"),
        format_float(gen.geom),
        format_float(gen.min_speedup),
        format_float(gen.max_speedup),
        str(gen.gt1_count),
        format_arrow_count(gen_no.pass1, gen.pass1, latex=True),
        format_arrow_count(gen_no.pass3, gen.pass3, latex=True),
        format_arrow_float(gen_no.avg_files_read, gen.avg_files_read, latex=True),
        format_arrow_count(opt_no.pass_final, opt_inv.pass_final, latex=True),
        format_arrow_tokens(
            opt_no.avg_token_usage,
            opt_inv.avg_token_usage,
            latex=True,
        ),
    ]


def unpack_row(
    row: dict[str, Any],
) -> tuple[GenerationStats, GenerationStats, OptimizationStats, OptimizationStats]:
    return (
        row["generation"],
        row["generation_no_examples"],
        row["optimization_no_invariants"],
        row["optimization_invariants"],
    )


def format_percent(count: int, denominator: int) -> str:
    if denominator <= 0:
        return "--"
    return f"{100.0 * count / denominator:.0f}%"


def format_float(value: float | None) -> str:
    return "--" if value is None else f"{value:.2f}"


def format_arrow_count(left: int, right: int, *, latex: bool = False) -> str:
    arrow = r"$\rightarrow$" if latex else "->"
    return f"{left}{arrow}{right}"


def format_arrow_float(
    left: float | None,
    right: float | None,
    *,
    latex: bool = False,
) -> str:
    return format_arrow(format_float(left), format_float(right), latex=latex)


def format_arrow_tokens(
    left: float | None,
    right: float | None,
    *,
    latex: bool = False,
) -> str:
    return format_arrow(format_tokens(left), format_tokens(right), latex=latex)


def format_arrow(left: str, right: str, *, latex: bool) -> str:
    arrow = r"$\rightarrow$" if latex else "->"
    if left == "--" or right == "--":
        return f"{left} {arrow} {right}"
    return f"{left}{arrow}{right}"


def format_tokens(value: float | None) -> str:
    if value is None:
        return "--"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:.0f}"
