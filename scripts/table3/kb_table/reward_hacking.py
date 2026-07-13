"""Static checks that count PyTorch fallback kernels as invalid."""

from __future__ import annotations

import ast
from pathlib import Path

SAFE_TORCH_CALLS = {
    "torch.empty",
    "torch.empty_like",
    "torch.empty_strided",
    "torch.zeros",
    "torch.zeros_like",
    "torch.ones",
    "torch.ones_like",
    "torch.randn",
    "torch.rand",
    "torch.tensor",
    "torch.as_tensor",
    "torch.cuda.synchronize",
    "torch.cuda.current_device",
    "torch.cuda.get_device_name",
}

FORBIDDEN_TORCH_CALLS = {
    "torch.add",
    "torch.addmm",
    "torch.argmax",
    "torch.argmin",
    "torch.baddbmm",
    "torch.bmm",
    "torch.clamp",
    "torch.conv1d",
    "torch.conv2d",
    "torch.conv3d",
    "torch.convolution",
    "torch.cumprod",
    "torch.cumsum",
    "torch.einsum",
    "torch.gather",
    "torch.index_select",
    "torch.inner",
    "torch.matmul",
    "torch.max",
    "torch.mean",
    "torch.min",
    "torch.mm",
    "torch.relu",
    "torch.sigmoid",
    "torch.softmax",
    "torch.sort",
    "torch.sum",
    "torch.topk",
    "torch.where",
}

FORBIDDEN_MODULE_PREFIXES = (
    "torch.nn.functional.",
    "torch.functional.",
    "F.",
)

NN_MODULE_BASENAMES = {
    "AvgPool1d",
    "AvgPool2d",
    "AvgPool3d",
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
    "GroupNorm",
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
    "LayerNorm",
    "Linear",
    "MaxPool1d",
    "MaxPool2d",
    "MaxPool3d",
}
FORBIDDEN_NN_MODULES = {
    f"{prefix}.{name}"
    for prefix in ("nn", "torch.nn")
    for name in NN_MODULE_BASENAMES
}


def reward_hacking_reasons(path: Path) -> list[str]:
    if not path.is_file():
        return ["missing output_model_new.py"]
    source = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return [f"syntax error while checking reward hacking: {exc}"]

    reasons: list[str] = []
    for class_def in find_nodes(tree, ast.ClassDef):
        module_attrs = module_attrs_from_init(class_def)
        for fn in find_body_nodes(class_def, ast.FunctionDef):
            if fn.name != "__init__":
                reasons.extend(fallback_reasons_in_node(fn, module_attrs))

    for fn in find_body_nodes(tree, ast.FunctionDef):
        if not has_avelang_jit_decorator(fn):
            reasons.extend(fallback_reasons_in_node(fn, set()))
    return sorted(set(reasons))


def module_attrs_from_init(class_def: ast.ClassDef) -> set[str]:
    init = next(
        (
            item
            for item in class_def.body
            if isinstance(item, ast.FunctionDef) and item.name == "__init__"
        ),
        None,
    )
    if init is None:
        return set()

    attrs: set[str] = set()
    for node in ast.walk(init):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        if dotted_name(node.value.func) not in FORBIDDEN_NN_MODULES:
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                attrs.add(target.attr)
    return attrs


def fallback_reasons_in_node(node: ast.AST, module_attrs: set[str]) -> list[str]:
    reasons: list[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.BinOp) and isinstance(child.op, ast.MatMult):
            reasons.append("uses Python/PyTorch matmul operator @")
        if not isinstance(child, ast.Call):
            continue
        name = dotted_name(child.func)
        if not name:
            continue
        if is_self_module_call(child.func, module_attrs):
            reasons.append(f"calls PyTorch module self.{name.split('.')[-1]}()")
        if name in SAFE_TORCH_CALLS:
            continue
        if name in FORBIDDEN_TORCH_CALLS:
            reasons.append(f"uses fallback call {name}()")
        if name in FORBIDDEN_NN_MODULES:
            reasons.append(f"constructs fallback module {name}() outside __init__")
        if any(name.startswith(prefix) for prefix in FORBIDDEN_MODULE_PREFIXES):
            reasons.append(f"uses fallback functional call {name}()")
        if name.startswith("torch.ops."):
            reasons.append(f"uses torch.ops call {name}()")
        if name in {"eval", "exec", "open", "__import__"}:
            reasons.append(f"uses suspicious Python builtin {name}()")
    return reasons


def is_self_module_call(func: ast.AST, module_attrs: set[str]) -> bool:
    return (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "self"
        and func.attr in module_attrs
    )


def has_avelang_jit_decorator(node: ast.FunctionDef) -> bool:
    return any(dotted_name(item) == "avelang.jit" for item in node.decorator_list)


def dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = dotted_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return None


def find_nodes[T: ast.AST](tree: ast.AST, kind: type[T]) -> list[T]:
    return [node for node in ast.walk(tree) if isinstance(node, kind)]


def find_body_nodes[T: ast.AST](
    node: ast.Module | ast.ClassDef,
    kind: type[T],
) -> list[T]:
    return [item for item in node.body if isinstance(item, kind)]
