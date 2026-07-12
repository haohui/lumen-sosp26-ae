"""MoE-specific configuration for the shared Lumen optimization runtime."""

from __future__ import annotations

from pathlib import Path

from lumen.harness.datasets.lumen.optimization_runtime import OptimizationSpec


MOE_WORKLOADS = (1024, 2048, 4096, 8192, 16384)


def moe_optimization_spec(repo_root: Path) -> OptimizationSpec:
    prompt_root = (
        repo_root
        / "python"
        / "lumen"
        / "harness"
        / "datasets"
        / "lumen"
        / "moe"
        / "prompts"
    )
    return OptimizationSpec(
        domain="moe",
        run_slug="moe",
        default_kernel=(
            repo_root
            / "datasets"
            / "inference"
            / "moe"
            / "lumen"
            / "moe_01_baseline.py"
        ),
        default_prompts=tuple(
            prompt_root / f"optimization-{index:02d}.md" for index in range(2, 8)
        ),
        workloads=MOE_WORKLOADS,
        workload_key="tokens",
        workload_flag="--tokens",
        benchmark_script="bench_moe.py",
        adapter_source=_benchmark_adapter_source(repo_root),
    )


def _benchmark_adapter_source(repo_root: Path) -> str:
    template_path = repo_root / "datasets" / "inference" / "moe" / "lumen" / "model.py"
    source = template_path.read_text(encoding="utf-8")
    module_location = (
        '_THIS_DIR = Path(__file__).resolve().parent\n'
        '_MOE_MODULE = "fused_moe.py"'
    )
    candidate_location = (
        '_THIS_DIR = Path(__file__).resolve().parents[4]\n'
        '_MOE_MODULE = "output_model_new.py"'
    )
    if module_location not in source:
        raise ValueError(f"unexpected MoE benchmark adapter layout: {template_path}")
    return source.replace(module_location, candidate_location, 1)
