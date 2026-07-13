import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE: al.constexpr = 256


@avelang.jit
def rmsnorm_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    total_positions: al.i32,
    F: al.i32,
    D1D2: al.i32,
    D2: al.i32,
    eps: al.f32,
):
    """
    RMS Normalization: for each (batch, d1, d2) position,
    compute sum of squares over F=num_features, then divide by sqrt(mean + eps).

    Launch: grid = (num_blocks, 1, 1), block = (BLOCK_SIZE, 1, 1)
    Each thread handles one independent position, looping serially over F.
    """
    tid = al.thread_id(0)
    bid = al.block_id(0)

    pos = bid * BLOCK_SIZE + tid
    if pos < total_positions:
        # Unflatten pos -> (batch, d1, d2)
        batch = pos // D1D2
        rem = pos - batch * D1D2
        d1 = rem // D2
        d2 = rem - d1 * D2

        # 4D layout: (batch, F, D1, D2) row-major
        stride_b = F * D1D2
        stride_f = D1D2
        stride_d1 = D2
        stride_d2 = al.convert(1, al.i32)

        layout_4d = al.make_layout(
            (total_positions // D1D2, F, D1D2 // D2, D2),
            (stride_b, stride_f, stride_d1, stride_d2),
        )
        x = al.make_tensor(x_ptr, al.bf16, layout_4d)
        out = al.make_tensor(out_ptr, al.bf16, layout_4d)

        # Pass 1: accumulate sum of squares over features
        sum_sq = al.convert(0.0, al.f32)
        for f in al.range(F):
            val = al.convert(x[batch, f, d1, d2], al.f32)
            sum_sq = sum_sq + val * val

        # Compute RMS
        F_f32 = al.convert(F, al.f32)
        rms = al.sqrt(sum_sq / F_f32 + eps)

        # Pass 2: normalize and write back
        for f in al.range(F):
            val = al.convert(x[batch, f, d1, d2], al.f32)
            result = val / rms
            out[batch, f, d1, d2] = al.convert(result, al.bf16)


def avelang_rmsnorm(x: torch.Tensor, eps: float) -> torch.Tensor:
    assert x.is_cuda, "Tensors must be on CUDA/HIP device."

    B = x.shape[0]
    F = x.shape[1]
    D1 = x.shape[2]
    D2 = x.shape[3]

    x_bf16 = x.contiguous().to(torch.bfloat16)
    total_positions = B * D1 * D2
    D1D2 = D1 * D2

    out = torch.empty_like(x_bf16)

    num_blocks = (total_positions + BLOCK_SIZE - 1) // BLOCK_SIZE

    rmsnorm_kernel[lambda: ((num_blocks, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_bf16, out, total_positions, F, D1D2, D2, eps
    )

    return out.to(x.dtype)


class ModelNew(nn.Module):
    """
    Optimized model that performs RMS Normalization using AveLang DSL.
    """
    def __init__(self, num_features: int, eps: float = 1e-5):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_rmsnorm(x, self.eps)
import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE: al.constexpr = 256


@avelang.jit
def rmsnorm_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    total_positions: al.i32,
    F: al.i32,
    D1D2: al.i32,
    D2: al.i32,
    eps: al.f32,
):
    """
    RMS Normalization: for each (batch, d1, d2) position,
    compute sum of squares over F=num_features, then divide by sqrt(mean + eps).

    Launch: grid = (num_blocks, 1, 1), block = (BLOCK_SIZE, 1, 1)
    Each thread handles one independent position, looping serially over F.
    """
    tid = al.thread_id(0)
    bid = al.block_id(0)

    pos = bid * BLOCK_SIZE + tid
    if pos < total_positions:
        # Unflatten pos -> (batch, d1, d2)
        batch = pos // D1D2
        rem = pos - batch * D1D2
        d1 = rem // D2
        d2 = rem - d1 * D2

        # 4D layout: (batch, F, D1, D2) row-major
        stride_b = F * D1D2
        stride_f = D1D2
        stride_d1 = D2
        stride_d2 = al.convert(1, al.i32)

        layout_4d = al.make_layout(
            (total_positions // D1D2, F, D1D2 // D2, D2),
            (stride_b, stride_f, stride_d1, stride_d2),
        )
        x = al.make_tensor(x_ptr, al.bf16, layout_4d)
        out = al.make_tensor(out_ptr, al.bf16, layout_4d)

        # Pass 1: accumulate sum of squares over features
        sum_sq = al.convert(0.0, al.f32)
        for f in al.range(F):
            val = al.convert(x[batch, f, d1, d2], al.f32)
            sum_sq = sum_sq + val * val

        # Compute RMS
        F_f32 = al.convert(F, al.f32)
        rms = al.sqrt(sum_sq / F_f32 + eps)

        # Pass 2: normalize and write back
        for f in al.range(F):
            val = al.convert(x[batch, f, d1, d2], al.f32)
            result = val / rms
            out[batch, f, d1, d2] = al.convert(result, al.bf16)


def avelang_rmsnorm(x: torch.Tensor, eps: float) -> torch.Tensor:
    assert x.is_cuda, "Tensors must be on CUDA/HIP device."

    B = x.shape[0]
    F = x.shape[1]
    D1 = x.shape[2]
    D2 = x.shape[3]

    x_bf16 = x.contiguous().to(torch.bfloat16)
    total_positions = B * D1 * D2
    D1D2 = D1 * D2

    out = torch.empty_like(x_bf16)

    num_blocks = (total_positions + BLOCK_SIZE - 1) // BLOCK_SIZE

    rmsnorm_kernel[lambda: ((num_blocks, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_bf16, out, total_positions, F, D1D2, D2, eps
    )

    return out.to(x.dtype)


class ModelNew(nn.Module):
    """
    Optimized model that performs RMS Normalization using AveLang DSL.
    """
    def __init__(self, num_features: int, eps: float = 1e-5):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_rmsnorm(x, self.eps)
import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE: al.constexpr = 256


@avelang.jit
def rmsnorm_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    total_positions: al.i32,
    F: al.i32,
    D1D2: al.i32,
    D2: al.i32,
    eps: al.constexpr,
):
    """
    RMS Normalization: for each (batch, d1, d2) position,
    compute sum of squares over F=num_features, then divide by sqrt(mean + eps).

    Launch: grid = (num_blocks, 1, 1), block = (BLOCK_SIZE, 1, 1)
    Each thread handles one independent position, looping serially over F.
    """
    tid = al.thread_id(0)
    bid = al.block_id(0)

    eps_f32 = al.convert(eps, al.f32)

    pos = bid * BLOCK_SIZE + tid
    if pos < total_positions:
        # Unflatten pos -> (batch, d1, d2)
        batch = pos // D1D2
        rem = pos - batch * D1D2
        d1 = rem // D2
        d2 = rem - d1 * D2

        # 4D layout: (batch, F, D1, D2) row-major
        stride_b = F * D1D2
        stride_f = D1D2
        stride_d1 = D2
        stride_d2 = al.convert(1, al.i32)

        layout_4d = al.make_layout(
            (total_positions // D1D2, F, D1D2 // D2, D2),
            (stride_b, stride_f, stride_d1, stride_d2),
        )
        x = al.make_tensor(x_ptr, al.bf16, layout_4d)
        out = al.make_tensor(out_ptr, al.bf16, layout_4d)

        # Pass 1: accumulate sum of squares over features
        sum_sq = al.convert(0.0, al.f32)
        for f in al.range(F):
            val = al.convert(x[batch, f, d1, d2], al.f32)
            sum_sq = sum_sq + val * val

        # Compute RMS
        F_f32 = al.convert(F, al.f32)
        rms = al.sqrt(sum_sq / F_f32 + eps_f32)

        # Pass 2: normalize and write back
        for f in al.range(F):
            val = al.convert(x[batch, f, d1, d2], al.f32)
            result = val / rms
            out[batch, f, d1, d2] = al.convert(result, al.bf16)


def avelang_rmsnorm(x: torch.Tensor, eps: float) -> torch.Tensor:
    assert x.is_cuda, "Tensors must be on CUDA/HIP device."

    B = x.shape[0]
    F = x.shape[1]
    D1 = x.shape[2]
    D2 = x.shape[3]

    x_bf16 = x.contiguous().to(torch.bfloat16)
    total_positions = B * D1 * D2
    D1D2 = D1 * D2

    out = torch.empty_like(x_bf16)

    num_blocks = (total_positions + BLOCK_SIZE - 1) // BLOCK_SIZE

    rmsnorm_kernel[lambda: ((num_blocks, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_bf16, out, total_positions, F, D1D2, D2, eps=eps
    )

    return out.to(x.dtype)


class ModelNew(nn.Module):
    """
    Optimized model that performs RMS Normalization using AveLang DSL.
    """
    def __init__(self, num_features: int, eps: float = 1e-5):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_rmsnorm(x, self.eps)
