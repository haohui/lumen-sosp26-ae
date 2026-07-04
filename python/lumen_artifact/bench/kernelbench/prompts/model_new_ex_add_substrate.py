import torch
import torch.nn as nn
import torch.nn.functional as F
import substrate
import substrate.language as S


@substrate.jit
def add_kernel(
    a: S.Tensor((1, 128), S.f32),
    b: S.Tensor((1, 128), S.f32),
    out: S.Tensor((1, 128), S.f32),
):
    tid = S.thread_id(0)
    if tid < 128:
        out[0, tid] = a[0, tid] + b[0, tid]


def substrate_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    This function wraps the Substrate kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Launches the Substrate kernel.
    """
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA/HIP device."
    assert a.shape == (1, 128) and b.shape == (1, 128), "Example kernel expects shape (1, 128)."

    a = a.contiguous()
    b = b.contiguous()

    # Prepare output tensor
    out = torch.empty_like(a)

    # Launch the Substrate kernel
    add_kernel[lambda: ((1, 1, 1), (128, 1, 1))](a, b, out)
    return out


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        # Instead of "return a + b", call our Substrate-based addition
        return substrate_add(a, b)
