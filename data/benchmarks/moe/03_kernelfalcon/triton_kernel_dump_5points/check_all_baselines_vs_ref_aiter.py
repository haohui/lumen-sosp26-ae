#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import socket
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import torch

BLOCK_N = 128
BLOCK_K = 128


@dataclass
class SharedInputs:
    input_q: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    input_scale: torch.Tensor


def _load_module(path: Path) -> ModuleType:
    module_name = f"kb_dyn_{path.stem}_{abs(hash(str(path.resolve()))):x}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _build_model_fn(module: ModuleType, device: torch.device):
    if hasattr(module, "ModelNew"):
        model = module.ModelNew()
        if hasattr(model, "to"):
            model = model.to(device=device)
        return lambda *xs: model(*xs)
    if hasattr(module, "Model"):
        model = module.Model()
        if hasattr(model, "to"):
            model = model.to(device=device)
        return lambda *xs: model(*xs)
    if hasattr(module, "kernel_function"):
        return lambda *xs: module.kernel_function(*xs)
    if hasattr(module, "run"):
        return lambda *xs: module.run(*xs)
    raise RuntimeError("expected one of: ModelNew, Model, kernel_function, run")


def _run_model_once(
    fn_model: Callable[..., Any],
    *,
    x: SharedInputs,
    shared_weights: dict[str, torch.Tensor],
) -> torch.Tensor:
    with torch.inference_mode():
        y = fn_model(
            x.input_q,
            shared_weights["w1_q"],
            shared_weights["w2_q"],
            x.topk_weights,
            x.topk_ids,
            x.input_scale,
            shared_weights["fc1_scale"],
            shared_weights["fc2_scale"],
        )
    if not isinstance(y, torch.Tensor):
        raise RuntimeError(f"kernel output must be torch.Tensor, got {type(y).__name__}")
    return y


def _max_rel_err(actual: torch.Tensor, expect: torch.Tensor) -> float:
    diff = (actual - expect).abs()
    denom = torch.maximum(expect.abs(), torch.tensor(1e-6, device=expect.device))
    return float((diff / denom).max().item())


def _build_shared_inputs(
    *,
    seq_lens: list[int],
    dim: int,
    experts: int,
    topk: int,
    input_dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> dict[int, SharedInputs]:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    out: dict[int, SharedInputs] = {}
    hidden_blocks = dim // BLOCK_K

    for s in seq_lens:
        input_q = torch.randn((s, dim), dtype=torch.float32, device=device, generator=g).mul_(1.0).add_(0.1)
        input_q = input_q.to(input_dtype).contiguous()

        input_scale = (
            torch.randn((s, hidden_blocks), dtype=torch.float32, device=device, generator=g).mul_(2e-2).add_(1e-1)
        ).clamp_min_(1e-8).contiguous()

        scores = torch.randn((s, experts), dtype=torch.float32, device=device, generator=g)
        topk_val, topk_idx = torch.topk(scores, k=topk, dim=-1, largest=True, sorted=True)
        topk_ids = topk_idx.to(torch.int32).contiguous()
        topk_weights = torch.softmax(topk_val, dim=-1).to(torch.float32).contiguous()

        out[s] = SharedInputs(
            input_q=input_q,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            input_scale=input_scale,
        )
    return out


def _build_shared_weights(
    *,
    dim: int,
    inter_dim: int,
    experts: int,
    input_dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> dict[str, torch.Tensor]:
    g = torch.Generator(device=device)
    g.manual_seed(seed + 17)

    w1_q = torch.randn(
        (experts, inter_dim * 2, dim),
        dtype=torch.float32,
        device=device,
        generator=g,
    ).mul_(8.0).to(input_dtype).contiguous()

    w2_q = torch.randn(
        (experts, dim, inter_dim),
        dtype=torch.float32,
        device=device,
        generator=g,
    ).mul_(8.0).to(input_dtype).contiguous()

    fc1_scale = (
        torch.randn(
            (experts, ((inter_dim * 2) // BLOCK_N) * (dim // BLOCK_K)),
            dtype=torch.float32,
            device=device,
            generator=g,
        ).mul_(2e-3).add_(1e-2)
    ).clamp_min_(1e-8).contiguous()

    fc2_scale = (
        torch.randn(
            (experts, (dim // BLOCK_N) * (inter_dim // BLOCK_K)),
            dtype=torch.float32,
            device=device,
            generator=g,
        ).mul_(2e-3).add_(1e-2)
    ).clamp_min_(1e-8).contiguous()

    return {
        "w1_q": w1_q,
        "w2_q": w2_q,
        "fc1_scale": fc1_scale,
        "fc2_scale": fc2_scale,
    }


def _compare(a: torch.Tensor, b: torch.Tensor, atol: float, rtol: float) -> dict[str, Any]:
    a32 = a.to(torch.float32)
    b32 = b.to(torch.float32)
    abs_err = float((a32 - b32).abs().max().item())
    rel_err = _max_rel_err(a32, b32)
    ok = bool(torch.allclose(a32, b32, atol=atol, rtol=rtol))
    return {
        "max_abs_err": abs_err,
        "max_rel_err": rel_err,
        "allclose": ok,
        "atol": float(atol),
        "rtol": float(rtol),
    }


def _run_aiter_fp8_blockscale(
    *,
    x: SharedInputs,
    shared_weights: dict[str, torch.Tensor],
) -> torch.Tensor:
    import aiter
    from aiter.fused_moe import moe_sorting

    model_dim = int(x.input_q.shape[-1])
    topk = int(x.topk_ids.shape[-1])
    experts = int(shared_weights["w1_q"].shape[0])
    out = torch.empty((x.input_q.shape[0], model_dim), dtype=torch.bfloat16, device=x.input_q.device)

    sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        x.topk_ids,
        x.topk_weights,
        experts,
        model_dim,
        torch.bfloat16,
    )

    aiter.fmoe_fp8_blockscale_g1u1(
        out,
        x.input_q,
        shared_weights["w1_q"],
        shared_weights["w2_q"],
        sorted_token_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        topk,
        x.input_scale,
        shared_weights["fc1_scale"],
        shared_weights["fc2_scale"],
        "",
        128,
        128,
        None,
    )
    return out


def _run_aiter_torch_blockscale(
    *,
    x: SharedInputs,
    shared_weights: dict[str, torch.Tensor],
) -> torch.Tensor:
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import torch_moe_stage1, torch_moe_stage2

    token_num = int(x.input_q.shape[0])
    topk = int(x.topk_ids.shape[-1])
    inter_dim = int(shared_weights["w2_q"].shape[-1])

    # Stage-1 dequantizes (input,w1) with per-1x128 block scales.
    a2 = torch_moe_stage1(
        x.input_q,
        shared_weights["w1_q"],
        shared_weights["w2_q"],
        x.topk_weights,
        x.topk_ids,
        dtype=torch.bfloat16,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x128,
        a1_scale=x.input_scale,
        w1_scale=shared_weights["fc1_scale"],
        doweight=False,
    )

    # Stage-2 expects activation scales for per-1x128 mode; stage-1 output here is already
    # dequantized bf16, so use identity scales.
    a2_scale = torch.ones((token_num * topk, inter_dim // 128), dtype=torch.float32, device=x.input_q.device)
    out = torch_moe_stage2(
        a2,
        shared_weights["w1_q"],
        shared_weights["w2_q"],
        x.topk_weights,
        x.topk_ids,
        dtype=torch.bfloat16,
        quant_type=QuantType.per_1x128,
        w2_scale=shared_weights["fc2_scale"],
        a2_scale=a2_scale,
        doweight=True,
    )
    return out


def _default_baselines(best_root: Path) -> list[Path]:
    baselines = sorted(best_root.glob("*/fused/best_kernel.py")) + sorted(best_root.glob("*/nofused/best_kernel.py"))
    return [p.resolve() for p in baselines if p.is_file()]


def _write_markdown(result: dict[str, Any], md_path: Path) -> None:
    seq_lens = result["config"]["seq_lens"]
    rows = result["rows"]
    lines: list[str] = []
    lines.append("## MoE Baseline Consistency Check (vs PyTorch ref + AITER)")
    lines.append("")
    lines.append(
        "Settings: "
        f"device={result['config']['device']}, "
        f"dim={result['config']['dim']}, inter_dim={result['config']['inter_dim']}, "
        f"experts={result['config']['experts']}, topk={result['config']['topk']}, "
        f"seq_lens={seq_lens}, seed={result['config']['seed']}, "
        f"atol={result['config']['atol']}, rtol={result['config']['rtol']}"
    )
    lines.append("")

    lines.append("### Reference vs AITER")
    lines.append("")
    lines.append("| seq | max_abs_err | max_rel_err | allclose |")
    lines.append("|---|---:|---:|---:|")
    for seq in seq_lens:
        c = result["ref_vs_aiter"][str(seq)]
        lines.append(
            f"| {seq} | {c['max_abs_err']:.6g} | {c['max_rel_err']:.6g} | {str(bool(c['allclose']))} |"
        )
    lines.append("")

    lines.append("### Baselines")
    lines.append("")
    lines.append("| baseline | seq | status | vs_ref_allclose | vs_aiter_allclose | max_abs_vs_ref | max_abs_vs_aiter |")
    lines.append("|---|---:|---|---:|---:|---:|---:|")
    for r in rows:
        if r.get("status") != "ok":
            lines.append(f"| {r['baseline']} | - | error | - | - | - | - |")
            continue
        for seq in seq_lens:
            item = r["per_seq"][str(seq)]
            lines.append(
                f"| {r['baseline']} | {seq} | ok | "
                f"{str(bool(item['vs_ref']['allclose']))} | {str(bool(item['vs_aiter']['allclose']))} | "
                f"{item['vs_ref']['max_abs_err']:.6g} | {item['vs_aiter']['max_abs_err']:.6g} |"
            )
    lines.append("")

    lines.append("### Verdict")
    lines.append("")
    lines.append(f"- all_ref_vs_aiter_allclose: {result['summary']['all_ref_vs_aiter_allclose']}")
    lines.append(f"- all_baselines_allclose_vs_ref_and_aiter: {result['summary']['all_baselines_allclose_vs_ref_and_aiter']}")
    lines.append("- baseline_overview:")
    for b, ok in result["summary"]["baseline_allclose_map"].items():
        lines.append(f"  - {b}: {ok}")
    lines.append("")

    md_path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check all moe-openai baselines against torch ref and aiter")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seq-lens", type=str, default="1024,2048,4096,8192,16384")
    p.add_argument("--dim", type=int, default=7168)
    p.add_argument("--inter-dim", type=int, default=2048)
    p.add_argument("--experts", type=int, default=32)
    p.add_argument("--topk", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260319)
    p.add_argument("--atol", type=float, default=0.5)
    p.add_argument("--rtol", type=float, default=0.05)
    p.add_argument(
        "--best-kernels-root",
        type=Path,
        default=Path("/data01/home/daifeng/kernel_benchmark/opt_kernel/moe-openai/best_kernels"),
    )
    p.add_argument(
        "--reference-kernel",
        type=Path,
        default=Path("/data01/home/daifeng/kernel_benchmark/opt_kernel/moe/moe_reference_pytorch.py"),
    )
    p.add_argument("--json-out", type=Path, required=True)
    p.add_argument("--md-out", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP runtime unavailable")

    seq_lens = [int(x.strip()) for x in args.seq_lens.split(",") if x.strip()]
    input_dtype = getattr(torch, "float8_e4m3fnuz", None)
    if input_dtype is None:
        input_dtype = getattr(torch, "float8_e4m3fn", None)
    if input_dtype is None:
        raise RuntimeError("fp8 dtype float8_e4m3fnuz/float8_e4m3fn unavailable")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)

    shared_inputs = _build_shared_inputs(
        seq_lens=seq_lens,
        dim=args.dim,
        experts=args.experts,
        topk=args.topk,
        input_dtype=input_dtype,
        device=device,
        seed=args.seed,
    )
    shared_weights = _build_shared_weights(
        dim=args.dim,
        inter_dim=args.inter_dim,
        experts=args.experts,
        input_dtype=input_dtype,
        device=device,
        seed=args.seed,
    )

    ref_mod = _load_module(args.reference_kernel)
    ref_fn = _build_model_fn(ref_mod, device=device)
    ref_outputs: dict[int, torch.Tensor] = {}
    aiter_outputs: dict[int, torch.Tensor] = {}
    aiter_backend = "fmoe_fp8_blockscale_g1u1"
    aiter_backend_error = ""

    for s in seq_lens:
        x = shared_inputs[s]
        ref_outputs[s] = _run_model_once(ref_fn, x=x, shared_weights=shared_weights).detach()

    try:
        for s in seq_lens:
            x = shared_inputs[s]
            aiter_outputs[s] = _run_aiter_fp8_blockscale(x=x, shared_weights=shared_weights).detach()
            torch.cuda.synchronize(device=device)
    except Exception as e:
        aiter_backend = "torch_moe_stage1_stage2_per_1x128_fallback"
        aiter_backend_error = f"{type(e).__name__}: {e}"
        print(
            f"[warn] aiter backend {aiter_backend} because primary failed: {aiter_backend_error}",
            flush=True,
        )
        aiter_outputs = {}
        for s in seq_lens:
            x = shared_inputs[s]
            aiter_outputs[s] = _run_aiter_torch_blockscale(x=x, shared_weights=shared_weights).detach()
            torch.cuda.synchronize(device=device)

    ref_vs_aiter: dict[str, dict[str, Any]] = {}
    for s in seq_lens:
        ref_vs_aiter[str(s)] = _compare(ref_outputs[s], aiter_outputs[s], args.atol, args.rtol)

    baselines = _default_baselines(args.best_kernels_root)
    rows: list[dict[str, Any]] = []
    baseline_allclose_map: dict[str, bool] = {}

    for idx, path in enumerate(baselines, start=1):
        base_name = str(path.relative_to(args.best_kernels_root))
        print(f"[baseline {idx}/{len(baselines)}] {base_name}", flush=True)
        row: dict[str, Any] = {
            "baseline": base_name,
            "path": str(path),
            "status": "ok",
            "per_seq": {},
        }
        try:
            mod = _load_module(path)
            fn = _build_model_fn(mod, device=device)
            ok_all = True
            for s in seq_lens:
                x = shared_inputs[s]
                y = _run_model_once(fn, x=x, shared_weights=shared_weights).detach()
                vs_ref = _compare(y, ref_outputs[s], args.atol, args.rtol)
                vs_aiter = _compare(y, aiter_outputs[s], args.atol, args.rtol)
                row["per_seq"][str(s)] = {
                    "vs_ref": vs_ref,
                    "vs_aiter": vs_aiter,
                }
                ok_all = ok_all and bool(vs_ref["allclose"]) and bool(vs_aiter["allclose"])
                torch.cuda.synchronize(device=device)
            baseline_allclose_map[base_name] = bool(ok_all)
        except Exception as e:
            row["status"] = "error"
            row["error"] = f"{type(e).__name__}: {e}"
            row["traceback_tail"] = "\n".join(traceback.format_exc().splitlines()[-30:])
            baseline_allclose_map[base_name] = False
        rows.append(row)
        torch.cuda.empty_cache()

    summary = {
        "all_ref_vs_aiter_allclose": all(bool(ref_vs_aiter[str(s)]["allclose"]) for s in seq_lens),
        "all_baselines_allclose_vs_ref_and_aiter": all(bool(v) for v in baseline_allclose_map.values()) if baseline_allclose_map else False,
        "baseline_allclose_map": baseline_allclose_map,
    }

    payload: dict[str, Any] = {
        "config": {
            "device": str(args.device),
            "seq_lens": seq_lens,
            "dim": int(args.dim),
            "inter_dim": int(args.inter_dim),
            "experts": int(args.experts),
            "topk": int(args.topk),
            "seed": int(args.seed),
            "atol": float(args.atol),
            "rtol": float(args.rtol),
            "best_kernels_root": str(args.best_kernels_root),
            "reference_kernel": str(args.reference_kernel),
            "baseline_count": len(baselines),
        },
        "environment": {
            "hostname": socket.gethostname(),
            "torch_version": torch.__version__,
            "torch_hip_version": getattr(torch.version, "hip", None),
            "device_name": torch.cuda.get_device_properties(device).name,
            "hip_visible_devices": str(__import__("os").environ.get("HIP_VISIBLE_DEVICES", "")),
            "aiter_backend": aiter_backend,
            "aiter_backend_error": aiter_backend_error,
        },
        "ref_vs_aiter": ref_vs_aiter,
        "rows": rows,
        "summary": summary,
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2))
    _write_markdown(payload, args.md_out)

    print(f"[saved] {args.json_out}")
    print(f"[saved] {args.md_out}")
    print(f"[summary] all_ref_vs_aiter_allclose={summary['all_ref_vs_aiter_allclose']}")
    print(f"[summary] all_baselines_allclose_vs_ref_and_aiter={summary['all_baselines_allclose_vs_ref_and_aiter']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
