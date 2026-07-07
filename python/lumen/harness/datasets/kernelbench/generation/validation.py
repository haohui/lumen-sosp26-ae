"""Static validation for generated AveLang kernels."""

from __future__ import annotations

import ast
import re


def validate_generated_avelang(code: str) -> tuple[bool, list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    stripped = _strip_comments(code)

    if re.search(r"\bsubstrate\b", stripped):
        errors.append("Uses retired substrate package name; use avelang")
    if "@avelang.jit" not in stripped:
        errors.append("Missing @avelang.jit decorator")
    if "avelang.language" not in stripped:
        errors.append("Missing avelang.language import")
    if not re.search(
        r"\b[A-Za-z_]\w*\.(thread_id|block_id|make_tensor|make_shared|range|convert|view|amdgpu|nvvm)\b",
        stripped,
    ):
        errors.append("No AveLang language operations found")
    if re.search(r"\btry\s*:", stripped) or re.search(r"\bexcept\b", stripped):
        errors.append("Contains try-except block")
    if _has_pass_statement(code):
        errors.append("Contains pass statement")

    fallback_ops = [
        "torch.mm",
        "torch.bmm",
        "torch.matmul",
        "torch.conv2d",
        "torch.nn.functional",
    ]
    for op in fallback_ops:
        if op in stripped:
            warnings.append(f"Uses possible PyTorch fallback op: {op}")

    return not errors, errors, warnings


def _strip_comments(code: str) -> str:
    lines = []
    for line in code.splitlines():
        line = line.split("#", 1)[0]
        line = line.split("//", 1)[0]
        lines.append(line)
    return "\n".join(lines)


def _has_pass_statement(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return any(isinstance(node, ast.Pass) for node in ast.walk(tree))
