from __future__ import annotations

import shlex
import sys
from pathlib import Path


def render_claude_context(
    *,
    python_root: str | Path,
    skills_root: str | Path,
    use_example_skills: bool,
    eval_num_correct_trials: int,
    eval_num_perf_trials: int,
    eval_timing_method: str,
    python_executable: str | Path = sys.executable,
) -> str:
    python_path = Path(python_root)
    executable_path = Path(python_executable)
    skills_path = Path(skills_root)
    bench_script = (
        python_path / "lumen_artifact" / "bench" / "tools" / "run_kernelbench_case.py"
    )
    template = Path(__file__).with_name("claude_context.md").read_text(encoding="utf-8")
    examples_skill = skills_path / "substrate-examples" / "SKILL.md"
    if use_example_skills:
        examples_section = (
            "2. **`substrate-examples`** - router skill that lists which category "
            "sub-skills are available. Read it, then read the ONE sub-skill whose\n"
            "   category best matches the target operator.\n"
            f"   Path: `{examples_skill}`"
        )
    else:
        examples_section = (
            "Example skills are disabled for this run. Use only "
            "`substrate-language-spec`."
        )
    activate_path = executable_path.parent / "activate"
    if activate_path.is_file():
        venv_activation = f"source {shlex.quote(str(activate_path))}"
    else:
        venv_activation = (
            f"# No activate script found for {shlex.quote(str(executable_path))}; "
            "use the executable below directly."
        )
    return template.format(
        python_root=shlex.quote(str(python_path)),
        python_executable=shlex.quote(str(executable_path)),
        bench_script=shlex.quote(str(bench_script)),
        venv_activation=venv_activation,
        skills_root=skills_path,
        lang_spec_skill=skills_path / "substrate-language-spec" / "SKILL.md",
        examples_section=examples_section,
        eval_num_correct_trials=eval_num_correct_trials,
        eval_num_perf_trials=eval_num_perf_trials,
        eval_timing_method=eval_timing_method,
    )
