---
name: avelang-examples
description: >
  Router for verified AveLang DSL kernel examples on AMD MI300X (BF16). Read this
  skill first, pick a category, then open the matching sub-skill for copy-ready
  tiling, launch, and memory patterns (adapt only math and indexing).
tags: [avelang, amd, kernel, router, bf16]
---

# AveLang verified examples — router

**Purpose.** Sub-skills hold **verified** kernels (correct + benchmarked vs. PyTorch on MI300X, BF16). This file only routes you to the right one; patterns live in the category skills.

**Workflow**

1. Classify the operator (table below).
2. Open the sub-skill for your category (links below).
3. Reuse its **structure** (tiling, loads/stores, launch shapes, vector width). Change only **math, shapes, and indexing** for your problem.
4. For DSL syntax and API limits, use `avelang-language-spec`. Do not add APIs that are not in the chosen example or the language spec.

## Category → sub-skill

| Pick when the problem is… | Sub-skill folder | Skill name (for tooling) |
|---------------------------|------------------|---------------------------|
| Dense / batched GEMM, MFMA-oriented tiling | `avelang-examples-gemm` | `avelang-examples-gemm` |
| Sum / max / min over axes, norm, softmax, argmax, logsumexp | `avelang-examples-reduction` | `avelang-examples-reduction` |
| Nothing above fits cleanly | `avelang-examples-general` | `avelang-examples-general` |

**If unsure:** start with `avelang-examples-general`, or combine a **structurally similar** category (e.g. treat a weird reduce like reduction + custom combine) and still follow the spec.

## Sub-skills

- [avelang-examples-gemm]
- [avelang-examples-reduction]
- [avelang-examples-general]

[avelang-examples-gemm]: ../avelang-examples-gemm/SKILL.md
[avelang-examples-reduction]: ../avelang-examples-reduction/SKILL.md
[avelang-examples-general]: ../avelang-examples-general/SKILL.md
