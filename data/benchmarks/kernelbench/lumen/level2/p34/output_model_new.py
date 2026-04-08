import torch
import torch.nn as nn
import substrate
import substrate.language as S

WARP_SIZE = 64
LN_DIM = 64


@substrate.jit
def layernorm_gelu_scale_kernel(
    input: S.Pointer(S.bf16),
    output: S.Pointer(S.bf16),
    gamma: S.Pointer(S.bf16),
    beta: S.Pointer(S.bf16),
    eps: S.constexpr,
    scaling_factor: S.constexpr,
    num_groups: S.i32,
):
    """Fused LayerNorm + GELU + Scale kernel optimized for bf16."""
    tid = S.thread_id(0)
    group_id = S.block_id(0)

    if group_id >= num_groups:
        return

    # Create tensor views with dynamic shape
    layout = S.make_layout((num_groups, LN_DIM), (LN_DIM, 1))
    input_tensor = S.make_tensor(input, S.bf16, layout)
    output_tensor = S.make_tensor(output, S.bf16, layout)
    gamma_tensor = S.make_tensor(gamma, S.bf16, S.make_layout((LN_DIM,), (1,)))
    beta_tensor = S.make_tensor(beta, S.bf16, S.make_layout((LN_DIM,), (1,)))

    # Shared memory for parallel reduction
    shared = S.make_shared((WARP_SIZE,), S.f32)

    # Load input value and convert to f32 for precision
    val_bf16 = input_tensor[group_id, tid]
    val = S.convert(val_bf16, S.f32)

    # Store for reduction
    shared[tid] = val
    S.syncthreads()

    # Tree reduction for sum (mean calculation)
    if tid < 32:
        shared[tid] = shared[tid] + shared[tid + 32]
    S.syncthreads()
    if tid < 16:
        shared[tid] = shared[tid] + shared[tid + 16]
    S.syncthreads()
    if tid < 8:
        shared[tid] = shared[tid] + shared[tid + 8]
    S.syncthreads()
    if tid < 4:
        shared[tid] = shared[tid] + shared[tid + 4]
    S.syncthreads()
    if tid < 2:
        shared[tid] = shared[tid] + shared[tid + 2]
    S.syncthreads()
    if tid < 1:
        shared[tid] = shared[tid] + shared[tid + 1]
    S.syncthreads()

    mean = shared[0] / LN_DIM
    S.syncthreads()

    # Compute (x - mean)^2 for variance
    diff = val - mean
    shared[tid] = diff * diff
    S.syncthreads()

    # Tree reduction for variance
    if tid < 32:
        shared[tid] = shared[tid] + shared[tid + 32]
    S.syncthreads()
    if tid < 16:
        shared[tid] = shared[tid] + shared[tid + 16]
    S.syncthreads()
    if tid < 8:
        shared[tid] = shared[tid] + shared[tid + 8]
    S.syncthreads()
    if tid < 4:
        shared[tid] = shared[tid] + shared[tid + 4]
    S.syncthreads()
    if tid < 2:
        shared[tid] = shared[tid] + shared[tid + 2]
    S.syncthreads()
    if tid < 1:
        shared[tid] = shared[tid] + shared[tid + 1]
    S.syncthreads()

    var = shared[0] / LN_DIM
    rstd = 1.0 / S.sqrt(var + eps)

    # Load gamma and beta, convert to f32
    gamma_val = S.convert(gamma_tensor[tid], S.f32)
    beta_val = S.convert(beta_tensor[tid], S.f32)

    # Normalize
    normalized = diff * rstd
    norm_val = gamma_val * normalized + beta_val

    # GELU activation using tanh approximation
    # GELU(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_over_pi = 0.7978845608028654
    coeff = 0.044715
    x_val = norm_val
    x3 = x_val * x_val * x_val
    inner = sqrt_2_over_pi * (x_val + coeff * x3)
    tanh_inner = S.tanh(inner)
    gelu = x_val * 0.5 * (1.0 + tanh_inner)

    # Scale
    result = gelu * scaling_factor

    # Store as bf16
    output_tensor[group_id, tid] = S.convert(result, S.bf16)


def fused_layernorm_gelu_scale(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias_param: torch.Tensor,
    eps: float,
    scaling_factor: float,
) -> torch.Tensor:
    """Fused LayerNorm + GELU + Scale with bf16 optimization."""
    input_dtype = x.dtype
    x_bf16 = x.to(torch.bfloat16)
    weight_bf16 = weight.to(torch.bfloat16).contiguous()
    bias_bf16 = bias_param.to(torch.bfloat16).contiguous()

    N, C, D, H, W = x_bf16.shape
    num_groups = N * C * D * H
    output_bf16 = torch.empty_like(x_bf16)

    layernorm_gelu_scale_kernel[lambda: ((num_groups, 1, 1), (WARP_SIZE, 1, 1))](
        x_bf16,
        output_bf16,
        weight_bf16,
        bias_bf16,
        eps,
        scaling_factor,
        num_groups,
    )

    return output_bf16.to(input_dtype)


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias=True,
        eps=1e-5,
        scaling_factor=1.0,
    ):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias
        )
        self.layer_norm = nn.LayerNorm(out_channels, eps=eps)
        self.scaling_factor = scaling_factor
        self.eps = eps

    def forward(self, x):
        # ConvTranspose3d using PyTorch optimized implementation
        x = self.conv_transpose(x)

        # Fused LayerNorm + GELU + Scale using Substrate kernel (bf16 optimized)
        x = fused_layernorm_gelu_scale(
            x,
            self.layer_norm.weight,
            self.layer_norm.bias,
            self.eps,
            self.scaling_factor,
        )
        return x
