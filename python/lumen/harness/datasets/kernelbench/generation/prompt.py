"""Prompt construction for Lumen KernelBench generation."""

from __future__ import annotations

import json
import os
import shlex
import sys
from collections.abc import Callable
from importlib.resources import as_file, files
from importlib.util import find_spec
from pathlib import Path

PROMPT_OPTION = "zero_shot"
AVELANG_PROMPT_NAME = "avelang_amd_full"
INVARIANT_PROMPT_NAME = "avelang_amd_invariant"

WRITE_FILE_DIRECTIVE = (
    "\n\nWrite your complete final implementation directly to the file "
    "`output_model_new.py` in the current working directory using your file "
    "writing tools. Do not print the code in your response text; write it to "
    "the file only. Do not inspect or copy from any other `runs/` directory, "
    "previous generated kernel, optimization round, or candidate output."
)


def build_avelang_prompt(
    ref_arch_src: str,
    *,
    precision: str = "bf16",
) -> str:
    """Render the Lumen AveLang prompt with KernelBench's prompt renderer.

    KernelBench is deliberately used as an external dependency here. This module
    owns only Lumen's prompt configuration and imports the renderer lazily so
    non-KernelBench workflows do not need the dependency at import time.
    """
    prompt = _render_avelang_prompt(
        ref_arch_src,
        prompt_name=AVELANG_PROMPT_NAME,
        prompt_config_name="prompt_config.toml",
        precision=precision,
    )

    return prompt.rstrip() + WRITE_FILE_DIRECTIVE


def build_invariant_avelang_prompt(
    ref_arch_src: str,
    *,
    has_candidate: bool,
    precision: str = "bf16",
) -> str:
    """Render an invariant-round prompt from the authoritative Torch model."""
    prompt = _render_avelang_prompt(
        ref_arch_src,
        prompt_name=INVARIANT_PROMPT_NAME,
        prompt_config_name="invariant_prompt_config.toml",
        precision=precision,
    ).rstrip() + WRITE_FILE_DIRECTIVE
    if not has_candidate:
        return prompt

    candidate_guidance = (
        "## Starting candidate\n\n"
        "The original Torch reference remains in `input_model.py`. Start this "
        "optimization round from the existing AveLang implementation in "
        "`candidate_input.py`; do not regenerate a fresh implementation from "
        "the Torch reference. Preserve the reference semantics while applying "
        "this round's optimization guidance. `candidate_input.py` is read-only; "
        "edit only `output_model_new.py`."
    )
    return _insert_before_write_directive(prompt, candidate_guidance)


def _render_avelang_prompt(
    ref_arch_src: str,
    *,
    prompt_name: str,
    prompt_config_name: str,
    precision: str,
) -> str:
    if not ref_arch_src.strip():
        raise ValueError("ref_arch_src must not be empty")

    from kernelbench.prompt_constructor_toml import get_custom_prompt

    with as_file(_prompt_resource(prompt_config_name)) as prompt_config:
        return get_custom_prompt(
            prompt_name,
            ref_arch_src=ref_arch_src,
            backend="avelang",
            option=PROMPT_OPTION,
            precision=precision,
            prompts_toml=str(prompt_config),
        )


def render_agents_md(
    *,
    python_root: str | Path | None = None,
    skills_root: str | Path | None = None,
    python_executable: str | Path = sys.executable,
) -> str:
    """Render the `AGENTS.md` guidance used by Codex workspaces."""
    return _render_agents_md(
        python_root=python_root,
        skills_root=skills_root,
        python_executable=python_executable,
        invariant=False,
    )


def render_invariant_agents_md(
    *,
    python_root: str | Path | None = None,
    skills_root: str | Path | None = None,
    python_executable: str | Path = sys.executable,
) -> str:
    """Render invariant guidance exposing only the AveLang language spec."""
    return _render_agents_md(
        python_root=python_root,
        skills_root=skills_root,
        python_executable=python_executable,
        invariant=True,
    )


def _render_agents_md(
    *,
    python_root: str | Path | None,
    skills_root: str | Path | None,
    python_executable: str | Path,
    invariant: bool,
) -> str:
    python_path = (
        Path(python_root) if python_root is not None else _discover_python_root()
    )
    skills_path = (
        Path(skills_root)
        if skills_root is not None
        else _discover_skills_root(python_root=python_path)
    )
    executable_path = Path(python_executable)
    activate_path = executable_path.parent / "activate"
    if activate_path.is_file():
        venv_activation = f"source {shlex.quote(str(activate_path))}"
    else:
        venv_activation = (
            f"# No activate script found for {shlex.quote(str(executable_path))}; "
            "use the executable below directly."
        )

    skills_section = (
        _render_invariant_skills_section(skills_path)
        if invariant
        else _render_skills_section(skills_path)
    )
    template = _prompt_resource("codex_context.md").read_text(encoding="utf-8")
    return template.format(
        python_root=shlex.quote(str(python_path)),
        python_executable=shlex.quote(str(executable_path)),
        bench_script=shlex.quote(str(_discover_kernelbench_cli_path())),
        venv_activation=venv_activation,
        skills_section=skills_section,
    )


def write_generation_workspace(
    work_dir: str | Path,
    *,
    ref_arch_src: str,
    precision: str = "bf16",
    gpu_arch: str = "gfx942",
    eval_num_correct_trials: int = 5,
    eval_num_perf_trials: int = 10,
    python_root: str | Path | None = None,
    skills_root: str | Path | None = None,
    python_executable: str | Path = sys.executable,
) -> dict[str, Path]:
    """Write prompt-side files for one Codex KernelBench workspace.

    This writes only the inputs needed before agent execution. It does not run
    Codex, evaluate the generated kernel, or import KernelBench datasets.
    """
    prompt = build_avelang_prompt(ref_arch_src, precision=precision)
    agents = render_agents_md(
        python_root=python_root,
        skills_root=skills_root,
        python_executable=python_executable,
    )
    return _write_workspace_files(
        work_dir,
        ref_arch_src=ref_arch_src,
        prompt=prompt,
        agents=agents,
        precision=precision,
        gpu_arch=gpu_arch,
        eval_num_correct_trials=eval_num_correct_trials,
        eval_num_perf_trials=eval_num_perf_trials,
    )


def write_invariant_workspace(
    work_dir: str | Path,
    *,
    ref_arch_src: str,
    candidate_src: str | None,
    precision: str = "bf16",
    gpu_arch: str = "gfx942",
    eval_num_correct_trials: int = 5,
    eval_num_perf_trials: int = 10,
    python_root: str | Path | None = None,
    skills_root: str | Path | None = None,
    python_executable: str | Path = sys.executable,
    prompt_transform: Callable[[str], str] | None = None,
) -> dict[str, Path]:
    """Write one invariant round without exposing optimization references."""
    prompt = build_invariant_avelang_prompt(
        ref_arch_src,
        has_candidate=candidate_src is not None,
        precision=precision,
    )
    if prompt_transform is not None:
        prompt = prompt_transform(prompt)
    agents = render_invariant_agents_md(
        python_root=python_root,
        skills_root=skills_root,
        python_executable=python_executable,
    )
    files_written = _write_workspace_files(
        work_dir,
        ref_arch_src=ref_arch_src,
        prompt=prompt,
        agents=agents,
        precision=precision,
        gpu_arch=gpu_arch,
        eval_num_correct_trials=eval_num_correct_trials,
        eval_num_perf_trials=eval_num_perf_trials,
    )
    if candidate_src is not None:
        path = Path(work_dir).expanduser()
        candidate_path = path / "candidate_input.py"
        output_path = path / "output_model_new.py"
        candidate_path.write_text(candidate_src, encoding="utf-8")
        output_path.write_text(candidate_src, encoding="utf-8")
        files_written["candidate_input"] = candidate_path
        files_written["output_model_new"] = output_path
    return files_written


def _write_workspace_files(
    work_dir: str | Path,
    *,
    ref_arch_src: str,
    prompt: str,
    agents: str,
    precision: str,
    gpu_arch: str,
    eval_num_correct_trials: int,
    eval_num_perf_trials: int,
) -> dict[str, Path]:
    path = Path(work_dir).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    files_written = {
        "input_model": path / "input_model.py",
        "prompt": path / "prompt.txt",
        "agents": path / "AGENTS.md",
        "eval_config": path / "eval_config.json",
    }
    files_written["input_model"].write_text(ref_arch_src, encoding="utf-8")
    files_written["prompt"].write_text(prompt, encoding="utf-8")
    files_written["agents"].write_text(agents, encoding="utf-8")
    files_written["eval_config"].write_text(
        json.dumps(
            {
                "backend": "avelang",
                "gpu_arch": gpu_arch,
                "precision": precision,
                "num_correct_trials": int(eval_num_correct_trials),
                "num_trials": int(eval_num_perf_trials),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return files_written


def _render_skills_section(skills_root: Path) -> str:
    if not skills_root.is_dir():
        return (
            f"No local skills directory was found at `{skills_root}`. Use the "
            "AveLang constraints in `prompt.txt` as the source of truth."
        )

    spec = skills_root / "languages" / "avelang-language-spec.md"
    examples = skills_root / "languages" / "avelang" / "index.md"
    kernels = skills_root / "knowledges" / "kernels"
    techniques = skills_root / "knowledges" / "techniques"

    lines: list[str] = []
    if spec.is_file():
        lines.append(f"- AveLang syntax/API: `{spec}`")
    if examples.is_file():
        lines.append(f"- AveLang examples: `{examples}`")
    if kernels.is_dir():
        lines.append(f"- Generic kernel knowledge: `{kernels}`")
    if techniques.is_dir():
        lines.append(f"- Optimization techniques: `{techniques}`")
    if lines:
        return "\n".join(lines)

    return (
        f"No AveLang prompt files were found under `{skills_root}`. Use "
        "the AveLang constraints in `prompt.txt` as the source of truth."
    )


def _render_invariant_skills_section(skills_root: Path) -> str:
    spec = skills_root / "languages" / "avelang-language-spec.md"
    if spec.is_file():
        return f"- AveLang syntax/API: `{spec}`"
    return (
        f"No AveLang language spec was found at `{spec}`. Use the "
        "AveLang constraints in `prompt.txt` as the source of truth."
    )


def _insert_before_write_directive(prompt: str, block: str) -> str:
    stripped = prompt.rstrip()
    if stripped.endswith(WRITE_FILE_DIRECTIVE):
        stripped = stripped[: -len(WRITE_FILE_DIRECTIVE)].rstrip()
        return f"{stripped}\n\n{block.strip()}{WRITE_FILE_DIRECTIVE}"
    return f"{stripped}\n\n{block.strip()}\n"


def _prompt_resource(name: str):
    return files("lumen.harness.datasets.kernelbench").joinpath("prompts", name)


def _discover_python_root(package_name: str = "lumen") -> Path:
    spec = find_spec(package_name)
    if spec is not None and spec.submodule_search_locations:
        return Path(next(iter(spec.submodule_search_locations))).resolve().parent
    return Path.cwd()


def _discover_module_file(module_name: str) -> Path | None:
    spec = find_spec(module_name)
    if spec is None or not spec.origin:
        return None
    return Path(spec.origin).resolve()


def _discover_kernelbench_cli_path() -> Path:
    module_path = _discover_module_file("lumen.tools.cli.kernelbench_graph_eval")
    if module_path is not None:
        return module_path
    return (
        _discover_python_root()
        / "lumen"
        / "tools"
        / "cli"
        / "kernelbench_graph_eval.py"
    )


def _discover_skills_root(
    *,
    python_root: str | Path | None = None,
    env_var: str = "LUMEN_SKILLS_ROOT",
) -> Path:
    env_path = os.environ.get(env_var)
    if env_path:
        return Path(env_path).expanduser()

    root = Path(python_root) if python_root is not None else _discover_python_root()
    for start in (root, Path.cwd()):
        start = start.resolve()
        for parent in (start, *start.parents):
            candidate = parent / "skills"
            if candidate.is_dir():
                return candidate

    return Path.cwd() / "skills"
