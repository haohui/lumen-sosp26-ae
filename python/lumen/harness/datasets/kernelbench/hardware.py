"""Small GPU hardware helpers used by the harness runtime."""

from __future__ import annotations

import os

import torch


NVIDIA_ARCHS = ["Maxwell", "Pascal", "Volta", "Turing", "Ampere", "Hopper", "Ada", "Blackwell"]
AMD_ARCHS = ["gfx942", "gfx950"]


def get_gpu_vendor(device: torch.device | int | None = None) -> str:
    """Return ``nvidia``, ``amd``, or ``unknown`` for the selected CUDA device."""
    if not torch.cuda.is_available():
        return "unknown"
    if device is None:
        device = torch.cuda.current_device()
    name = torch.cuda.get_device_name(device).upper()
    if "NVIDIA" in name:
        return "nvidia"
    if "AMD" in name or "MI3" in name:
        return "amd"
    return "unknown"


def set_gpu_arch(arch_list: list[str]) -> None:
    """Set the torch arch environment variable for NVIDIA or AMD builds."""
    nvidia_archs: list[str] = []
    amd_archs: list[str] = []

    for arch in arch_list:
        if arch in NVIDIA_ARCHS:
            nvidia_archs.append(arch)
        elif arch in AMD_ARCHS:
            amd_archs.append(arch)
        else:
            raise ValueError(
                f"Invalid architecture: {arch}. Must be one of NVIDIA: {NVIDIA_ARCHS} or AMD: {AMD_ARCHS}"
            )

    if nvidia_archs and amd_archs:
        raise ValueError(f"Cannot mix NVIDIA and AMD architectures: {nvidia_archs} vs {amd_archs}")

    if nvidia_archs:
        os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(nvidia_archs)
    elif amd_archs:
        os.environ["PYTORCH_ROCM_ARCH"] = ";".join(amd_archs)
