import torch
import torch.nn as nn
import substrate
import substrate.language as S

BLOCK_SIZE: S.constexpr = 256


@substrate.jit
def gelu_kernel(
    x_ptr: S.Pointer(S.bf16),
    out_ptr: S.Pointer(S.bf16),
    n: S.i32,
):
    idx = S.block_id(0) * S.block_dim(0) + S.thread_id(0)

    if idx < n:
        layout = S.make_layout((n,), (1,))
        x = S.make_tensor(x_ptr, S.bf16, layout)
        out = S.make_tensor(out_ptr, S.bf16, layout)

        val_bf16 = x[idx]

        # Convert to f32 for accurate computation of GELU
        val = S.convert(val_bf16, S.f32)

        # GELU: 0.5 * x * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        # sqrt(2/pi) ≈ 0.7978845608028654
        sqrt_2_over_pi = S.convert(0.7978845608028654, S.f32)
        coeff = S.convert(0.044715, S.f32)
        half = S.convert(0.5, S.f32)
        one = S.convert(1.0, S.f32)

        # x^3
        x_cubed = val * val * val

        # x + 0.044715 * x^3
        inner = val + coeff * x_cubed

        # sqrt(2/pi) * (x + 0.044715 * x^3)
        scaled = sqrt_2_over_pi * inner

        # tanh(sqrt(2/pi) * (x + 0.044715 * x^3))
        tanh_val = S.tanh(scaled)

        # 1.0 + tanh(...)
        one_plus_tanh = one + tanh_val

        # 0.5 * x * (1.0 + tanh(...))
        result = half * val * one_plus_tanh

        # Convert back to bf16
        out[idx] = S.convert(result, S.bf16)


def gelu_substrate(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Tensors must be on CUDA/HIP device."

    orig_dtype = x.dtype
    x_contig = x.contiguous().to(torch.bfloat16)

    n = x_contig.numel()

    out = torch.empty_like(x_contig)

    # Compute grid dimensions
    grid_size = (n + BLOCK_SIZE - 1) // BLOCK_SIZE

    gelu_kernel[lambda: ((grid_size, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_contig, out, n
    )

    return out.to(orig_dtype)


class ModelNew(nn.Module):
    """
    Implementation of the GELU activation function from Google BERT repo.
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies GELU activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with GELU applied, same shape as input.
        """
        return gelu_substrate(x)
