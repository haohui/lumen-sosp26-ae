from __future__ import annotations
"""Prompt builder for Mind‑Evolution CUDA‑kernel search (seed‑kernel version).

Generates a **single prompt** that contains:
1. Target GPU spec (from `prompts/hardware/gpu_specs.py`)
2. **Few‑shot pair** – original *and* optimised model code blocks
3. Source architecture (`class Model`) that needs to be optimised
4. Existing kernel summaries (optional, for diversity context)
5. A **diversity requirement** section ensuring the new kernel differs from all previous ones
6. Output requirements

CLI usage
---------
```bash
python -m prompts.build_prompt KernelBench/level1/19_ReLU.py \
       --gpu "MI300X" -o prompt.txt
```
"""

import argparse
import importlib.util
import sys
from pathlib import Path
from string import Template
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]  # project root
HW_FILE = ROOT / "prompts/hardware/gpu_specs.py"  # GPU spec table

# --------------------------------------------------
# Few‑shot pair  (before / after)
# --------------------------------------------------
FEWSHOT_BASE = ROOT / "prompts/few_shot/model_ex_add.py"   # original Model
FEWSHOT_NEW = ROOT / "prompts/few_shot/model_new_ex_add.py"  # optimised ModelNew
FEWSHOT_BASE_HIP = ROOT / "prompts/few_shot_hip/model_ex_add.py"
FEWSHOT_NEW_HIP = ROOT / "prompts/few_shot_hip/model_new_ex_add.py"

# ---------------------------------------------------------------------------
# Prompt template (with diversity requirement)
# ---------------------------------------------------------------------------
test = Template(
    dedent(
        """ 
You write custom $BACKEND_LABEL kernels to replace the pytorch operators in the given architecture 
to get speedups.You have complete freedom to choose the set of operators you want to replace. You may
make the decision to replace some operators with custom $BACKEND_LABEL kernels and leave others
unchanged. You may replace multiple operators with custom implementations, consider
operator fusion opportunities (combining multiple operators into a single kernel, for
example, combining matmul+relu), or algorithmic changes (such as online softmax). You are
only limited by your imagination.

Here\’s an example to show you the syntax of inline embedding custom $BACKEND_LABEL operators in torch: 
The example given architecture is:
‘‘‘
$few_base
‘‘‘
The example new arch with custom $BACKEND_LABEL kernels looks like this:
‘‘‘
$few_new
‘‘‘

You are given the following architecture:
$arch_src

And the kernel you need to implement is:
```python
$kernel_src
```

Target data type requirement:
- Primary compute dtype must be BF16 (`torch.bfloat16`).
- Generated $BACKEND_LABEL code should implement the BF16 path directly, not by silently switching main math to FP32.
- `ModelNew.forward` final returned tensor dtype must match reference dtype (for this task: BF16).
- Do NOT return FP32 attention output and rely on evaluator-side dtype cast.
- FP32 accumulation is allowed internally, but final output writeback/return must be BF16.

Hard constraint (NO-HACK):
- Do NOT call framework attention core operators, directly or indirectly.
- Forbidden APIs include (non-exhaustive):
  - `torch.nn.functional.scaled_dot_product_attention`
  - `torch.nn.functional.multi_head_attention_forward`
  - `torch.nn.MultiheadAttention`
  - `flash_attn*` / `flash_attention*`
- Do NOT implement attention via framework dense ops either (forbidden fallback path):
  - `torch.matmul` / `torch.bmm` / `torch.einsum` for QK^T or AV
  - `torch.softmax` / `torch.nn.functional.softmax` on attention scores
- Implement the attention math with your own custom kernels and wrappers.

Optimize the architecture named Model with custom $BACKEND_LABEL operators! Name your optimized
output architecture ModelNew. Output the new code in codeblocks. Please generate real
code, NOT pseudocode, make sure the code compiles and is fully functional. Just output
the new model code, no other text, and NO testing code!
"""
    )
)
TEMPLATE = Template(
    dedent(
        """
Task
----
Generate **hand‑written $BACKEND_LABEL kernels** that replace *all* PyTorch operator(s)
inside the original `class Model` (shown later).  You may fuse multiple
operators into a single kernel if that yields better performance.  Leave any
non‑replaced parts of the model unchanged.

OUTPUT RULES (STRICT) ────────────────────────────────────────────────
1. Inside the block, follow **exactly** this order:
   1. Imports – `torch`, `torch.nn`, `load_inline`.
   2. `source` – triple‑quoted $BACKEND_LABEL string(s) (kernel + host wrapper).
   3. `cpp_src` – prototypes for *all* kernels you expose.
   4. **One** `load_inline` call per kernel group.
   5. `class ModelNew(nn.Module)` – mirrors original inputs/outputs but calls
      your $BACKEND_LABEL kernels.
2. **Do NOT include** testing code, `if __name__ == "__main__"`, or extra prose.

Hard constraint (NO-HACK):
- Do NOT call framework attention core operators, directly or indirectly.
- Forbidden APIs include (non-exhaustive):
  - `torch.nn.functional.scaled_dot_product_attention`
  - `torch.nn.functional.multi_head_attention_forward`
  - `torch.nn.MultiheadAttention`
  - `flash_attn*` / `flash_attention*`
- Do NOT implement attention via framework dense ops either (forbidden fallback path):
  - `torch.matmul` / `torch.bmm` / `torch.einsum` for QK^T or AV
  - `torch.softmax` / `torch.nn.functional.softmax` on attention scores
- Implement attention math in your own custom $BACKEND_LABEL kernels.

DType parity constraint (STRICT):
- Keep output dtype aligned with the PyTorch reference output dtype.
- Avoid "cast-to-pass" patterns (e.g., returning FP32 then expecting external cast).


Few‑shot example (reference only – do **not** echo):
**Original**
```python
$few_base
```
**Optimised**
```python
$few_new
```

Target architecture (to optimise):
```python
$arch_src
```

Optimize the architecture named Model with custom $BACKEND_LABEL operators! Name your optimized
output architecture ModelNew. Output the new code in codeblocks. Please generate real
code, NOT pseudocode, make sure the code compiles and is fully functional. Just output
the new model code, no other text, and NO testing code!

Example:
```python
# <complete ModelNew code>
```
# ==========================================================
"""
    )
)
default_system_prompt = """\
You are a senior GPU-kernel optimisation specialist. Your job is to generate a high-quality,
compilable, and runnable Python script that builds and launches **hand-written GPU kernels**.

NO-HACK POLICY (STRICT):
- Never call framework attention core operators.
- Forbidden APIs include:
  - torch.nn.functional.scaled_dot_product_attention
  - torch.nn.functional.multi_head_attention_forward
  - torch.nn.MultiheadAttention
  - flash_attn / flash_attention families
- Never use framework dense-op fallback to realize attention:
  - torch.matmul / torch.bmm / torch.einsum for QK^T or AV
  - torch.softmax / torch.nn.functional.softmax on attention scores
- Produce true custom kernel implementations instead.

DTYPE PARITY POLICY (STRICT):
- `ModelNew.forward` final returned tensor dtype must match reference output dtype.
- Internal FP32 accumulation is allowed, but do not return FP32 and rely on external casting.

OUTPUT RULES (STRICT):
output the code within:
```python
# <complete ModelNew code>
```

"""
# ---------------------------------------------------------------------------
# GPU spec loader
# ---------------------------------------------------------------------------


def _load_gpu_spec() -> dict:  # noqa: D401
    """Import `gpu_specs.py` and return the GPU_SPEC_INFO dict (robust across Python versions)."""
    spec = importlib.util.spec_from_file_location("gpu_specs", HW_FILE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {HW_FILE}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["gpu_specs"] = module  # avoid re‑import
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    if not hasattr(module, "GPU_SPEC_INFO"):
        raise AttributeError("GPU_SPEC_INFO not defined in gpu_specs.py")
    return module.GPU_SPEC_INFO  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Prompt builder core
# ---------------------------------------------------------------------------

def build_seed_prompt(
    arch_path: Path,
    gpu_name: str | None = None,
    backend: str = "cuda",
) -> str:
    """Build LLM prompt for CUDA‑kernel optimisation (seed generation)."""
    gpu_info = _load_gpu_spec()

    # Auto‑detect GPU if not provided
    if gpu_name is None:
        try:
            import torch
            gpu_name = torch.cuda.get_device_name(0)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("CUDA device not found – pass --gpu <name>.") from exc

    if gpu_name in gpu_info:
        info = gpu_info[gpu_name]
    else:
        info = {
            "GPU Architecture": "Unknown",
            "GPU Name": gpu_name,
            "Note": "No predefined spec.",
        }
    gpu_arch = info.get("GPU Architecture", "Unknown")
    arch_src = "\n".join(
        f"• {k}: {v}" for k, v in info.items() if k != "GPU Architecture"
    ) if gpu_arch != "Unknown" else "Not Specified"

    backend_is_hip = (backend or "cuda").lower() == "hip"
    backend_label = "HIP/ROCm" if backend_is_hip else "CUDA"
    few_base_path = FEWSHOT_BASE_HIP if backend_is_hip and FEWSHOT_BASE_HIP.exists() else FEWSHOT_BASE
    few_new_path = FEWSHOT_NEW_HIP if backend_is_hip and FEWSHOT_NEW_HIP.exists() else FEWSHOT_NEW

    few_base = few_base_path.read_text().strip()
    few_new = few_new_path.read_text().strip()
    kernel_src = Path(arch_path).read_text().strip()

    return test.substitute(
        few_base=few_base,
        few_new=few_new,
        arch_src=arch_src,
        kernel_src=kernel_src,
        BACKEND_LABEL=backend_label,
    )


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

def _cli() -> None:  # noqa: D401
    parser = argparse.ArgumentParser(
        description="Build LLM prompt for CUDA‑kernel optimisation (seed generation)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("model_py", help="Path to .py containing class Model")
    parser.add_argument("--gpu", default=None, help="GPU name key in gpu_specs.py")
    parser.add_argument("-o", "--out", help="Save prompt to file")
    args = parser.parse_args()

    prompt = build_seed_prompt(Path(args.model_py), args.gpu)

    if args.out:
        Path(args.out).write_text(prompt)
        print(f"[✓] Prompt saved to {args.out}")
    else:
        print(prompt)


if __name__ == "__main__":  # pragma: no cover
    _cli()
