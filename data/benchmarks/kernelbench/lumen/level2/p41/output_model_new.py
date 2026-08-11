import torch
import torch.nn as nn
import substrate
import substrate.language as S


ACT_BLOCK_SIZE: S.constexpr = 256

batch_size = 16384
in_features = 4096
out_features = 4096


@substrate.jit
def gelu_relu_bf16_inplace_kernel(
    x_ptr: S.Pointer(S.bf16),
    n: S.u32,
):
    layout = S.make_layout((n,), (1,))
    x = S.make_tensor(x_ptr, S.bf16, layout)

    bid = S.convert(S.block_id(0), S.u32)
    bdim = S.convert(S.block_dim(0), S.u32)
    tid = S.convert(S.thread_id(0), S.u32)
    idx = bid * bdim + tid

    if idx < n:
        xv = S.convert(x[idx], S.f32)
        zero = S.convert(0.0, S.f32)

        if xv <= zero:
            x[idx] = S.convert(0.0, S.bf16)
        else:
            half = S.convert(0.5, S.f32)
            one = S.convert(1.0, S.f32)
            inv_sqrt2 = S.convert(0.7071067811865476, S.f32)
            gelu = half * xv * (one + S.erf(xv * inv_sqrt2))
            x[idx] = S.convert(gelu, S.bf16)


def substrate_gelu_relu_(x: torch.Tensor) -> torch.Tensor:
    if not x.is_cuda:
        raise ValueError("substrate_gelu_relu_ expects CUDA/HIP tensors.")
    if x.dtype != torch.bfloat16:
        raise TypeError(f"substrate_gelu_relu_ expects bf16 input, got {x.dtype}.")

    x = x.contiguous()
    n = x.numel()
    grid = ((n + ACT_BLOCK_SIZE - 1) // ACT_BLOCK_SIZE, 1, 1)
    gelu_relu_bf16_inplace_kernel[lambda: (grid, (ACT_BLOCK_SIZE, 1, 1))](x.view(-1), n)
    return x


class ModelNew(nn.Module):
    """
    Use native BF16 Linear + BatchNorm for exactness, then fuse GELU + ReLU with
    a single in-place Substrate kernel.
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.batch_norm = nn.BatchNorm1d(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)
        x = x.contiguous()

        x = self.gemm(x)
        x = self.batch_norm(x)
        x = substrate_gelu_relu_(x)

        if orig_dtype != torch.bfloat16:
            x = x.to(orig_dtype)
        return x


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features]
