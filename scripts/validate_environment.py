#!/usr/bin/env python3
"""Validate the environment used by the Lumen artifact experiments."""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_API_URL = "http://47.79.17.216:8080/v1"
DEFAULT_API_KEY = "lumen-ae"
DEFAULT_MODEL = "deepseek-v4-flash"
HF_DATASET_URL = "https://huggingface.co/api/datasets/ScalingIntelligence/KernelBench"
PYTHON_DEPENDENCIES = (
    "aiter",
    "avelang",
    "datasets",
    "flashinfer",
    "hipkittens",
    "jinja2",
    "kernelbench",
    "numpy",
    "openai",
    "pandas",
    "torch",
    "transformers",
    "triton",
)


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str
    required: bool = True
    skipped: bool = False

    @property
    def label(self) -> str:
        if self.skipped:
            return "SKIP"
        if self.ok:
            return "PASS"
        return "FAIL" if self.required else "WARN"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the Lumen GPU, software, services, and disk space."
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "runs")
    parser.add_argument("--min-gpus", type=non_negative_int, default=1)
    parser.add_argument("--min-disk-gb", type=positive_float, default=20.0)
    parser.add_argument("--require-codex", action="store_true")
    parser.add_argument("--require-api", action="store_true")
    parser.add_argument("--require-hf", action="store_true")
    parser.add_argument(
        "--skip-network",
        action="store_true",
        help="Skip API and Hugging Face requests (useful for command dry runs).",
    )
    parser.add_argument(
        "--api-url",
        default=os.environ.get("LUMEN_GENERATION_API_URL", DEFAULT_API_URL),
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("LUMEN_GENERATION_API_KEY", DEFAULT_API_KEY),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("LUMEN_GENERATION_MODEL", DEFAULT_MODEL),
    )
    return parser.parse_args(argv)


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def check_gpu(min_gpus: int) -> tuple[Check, str]:
    try:
        import torch

        count = torch.cuda.device_count()
        names = [torch.cuda.get_device_name(index) for index in range(count)]
        detail = f"{count} device(s)"
        if names:
            counts = Counter(names)
            detail += ": " + ", ".join(
                f"{number}x {name}" for name, number in counts.items()
            )
        required = min_gpus > 0
        available = count > 0 and count >= min_gpus
        return Check("GPU", available, detail, required=required), detail
    except Exception as exc:
        required = min_gpus > 0
        detail = f"unavailable ({short_error(exc)})"
        return Check("GPU", False, detail, required=required), detail


def check_rocm(required: bool) -> tuple[Check, str]:
    hipcc = shutil.which("hipcc")
    try:
        import torch

        torch_hip = torch.version.hip or "unknown"
    except Exception:
        torch_hip = "unknown"
    if not hipcc:
        detail = f"hipcc not found; PyTorch HIP {torch_hip}"
        return Check("ROCm", False, detail, required=required), detail
    version = command_first_line([hipcc, "--version"])
    detail = f"PyTorch HIP {torch_hip}; {version}"
    return Check("ROCm", True, detail, required=required), detail


def check_dependencies() -> Check:
    missing = [
        name for name in PYTHON_DEPENDENCIES if importlib.util.find_spec(name) is None
    ]
    commands = ("clang", "cmake", "ninja")
    missing_commands = [name for name in commands if shutil.which(name) is None]
    if not missing and not missing_commands:
        return Check(
            "Dependencies",
            True,
            (
                f"{len(PYTHON_DEPENDENCIES)} Python modules and "
                f"{len(commands)} build tools"
            ),
        )
    parts = []
    if missing:
        parts.append("missing Python modules: " + ", ".join(missing))
    if missing_commands:
        parts.append("missing commands: " + ", ".join(missing_commands))
    return Check("Dependencies", False, "; ".join(parts))


def check_codex(required: bool) -> tuple[Check, str]:
    codex = shutil.which("codex")
    if not codex:
        detail = "not found in PATH"
        return Check("Codex", False, detail, required=required), detail
    version = command_first_line([codex, "--version"])
    return Check("Codex", True, version, required=required), version


def check_api(args: argparse.Namespace) -> Check:
    required = bool(args.require_api)
    config = f"{args.model} at {args.api_url.rstrip('/')}"
    if args.skip_network:
        return Check(
            "Generation API",
            True,
            f"not probed; configured {config}",
            required,
            True,
        )
    if not args.api_url or not args.api_key:
        return Check(
            "Generation API",
            False,
            "URL or API key is not configured",
            required,
        )
    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=args.api_key,
            base_url=args.api_url.rstrip("/"),
            timeout=20.0,
        )
        response = client.responses.create(
            model=args.model,
            input="Reply with exactly LUMEN_OK.",
            max_output_tokens=32,
        )
        text = response.output_text.strip()
        if "LUMEN_OK" not in text:
            raise RuntimeError(f"unexpected response {text[:80]!r}")
        return Check("Generation API", True, config, required)
    except Exception as exc:
        return Check("Generation API", False, f"{config}: {short_error(exc)}", required)


def check_hugging_face(args: argparse.Namespace) -> Check:
    required = bool(args.require_hf)
    if args.skip_network:
        return Check("Hugging Face", True, "access not probed", required, True)
    try:
        request = urllib.request.Request(
            HF_DATASET_URL,
            headers={"User-Agent": "lumen-environment-validator/1.0"},
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            status = response.status
        return Check(
            "Hugging Face",
            status == 200,
            f"KernelBench dataset HTTP {status}",
            required,
        )
    except Exception as exc:
        return Check("Hugging Face", False, short_error(exc), required)


def check_disk(output_dir: Path, minimum_gb: float) -> tuple[Check, str]:
    probe = output_dir.expanduser().resolve()
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    free_gb = usage.free / 1024**3
    detail = f"{free_gb:.1f} GiB free at {probe} (minimum {minimum_gb:.1f} GiB)"
    return Check("Disk", free_gb >= minimum_gb, detail), detail


def command_first_line(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        text = result.stdout.strip() or result.stderr.strip()
        return text.splitlines()[0] if text else "version unknown"
    except Exception as exc:
        return f"version check failed ({short_error(exc)})"


def short_error(exc: Exception) -> str:
    return " ".join(str(exc).split())[:240] or type(exc).__name__


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gpu, gpu_summary = check_gpu(args.min_gpus)
    rocm, rocm_summary = check_rocm(required=args.min_gpus > 0)
    codex, codex_summary = check_codex(args.require_codex)
    disk, disk_summary = check_disk(args.output_dir, args.min_disk_gb)
    checks = [
        gpu,
        rocm,
        check_dependencies(),
        codex,
        check_api(args),
        check_hugging_face(args),
        disk,
    ]

    print("Lumen environment validation")
    for check in checks:
        print(f"[{check.label}] {check.name}: {check.detail}")

    failures = [check for check in checks if not check.ok and check.required]
    print("\nEnvironment/configuration summary")
    print(f"  Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"  GPU: {gpu_summary}")
    print(f"  ROCm: {rocm_summary}")
    print(f"  Codex: {codex_summary}")
    print(f"  Generation API: {args.model} at {args.api_url.rstrip('/')}")
    print(f"  Output: {args.output_dir.expanduser().resolve()}")
    print(f"  Disk: {disk_summary}")
    if failures:
        names = ", ".join(check.name for check in failures)
        print(f"  Result: NOT READY ({names})")
        return 1
    print("  Result: READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
