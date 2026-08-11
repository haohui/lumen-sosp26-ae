import torch
import torch.nn as nn
import substrate
import substrate.language as S


BLOCK_SIZE: S.constexpr = 256
GRID_SIZE: S.constexpr = 1024
ELEMENTS_PER_LAUNCH: S.constexpr = BLOCK_SIZE * GRID_SIZE  # 262144


@substrate.jit
def fused_activation_kernel(
    x: S.Tensor((ELEMENTS_PER_LAUNCH,), S.f32),
    bias: S.Tensor((32,), S.f32),
    out: S.Tensor((ELEMENTS_PER_LAUNCH,), S.f32),
    n: S.i32,
    offset: S.i32,
    out_channels: S.i32,
    spatial_size: S.i32,
):
    """Fused kernel: ReLU -> LeakyReLU -> GELU -> Sigmoid -> Bias"""
    tid = S.thread_id(0)
    block_id = S.block_id(0)

    idx = block_id * BLOCK_SIZE + tid
    global_idx = offset + idx

    if global_idx < n:
        x_val = x[idx]

        # Constants
        zero = S.convert(0.0, S.f32)
        one = S.convert(1.0, S.f32)
        half = S.convert(0.5, S.f32)
        leaky_slope = S.convert(0.01, S.f32)
        gelu_coef = S.convert(0.7978845608028654, S.f32)
        gelu_const = S.convert(0.044715, S.f32)
        log2_e = S.convert(1.4426950408889634, S.f32)

        # Step 1: ReLU
        relu_val = x_val if x_val >= zero else zero

        # Step 2: LeakyReLU
        negative_part = leaky_slope * relu_val
        leaky_val = relu_val if relu_val >= negative_part else negative_part

        # Step 3: GELU (approximation using tanh)
        x_cubed = leaky_val * leaky_val * leaky_val
        inner = gelu_coef * (leaky_val + gelu_const * x_cubed)
        exp_arg = inner * log2_e + inner * log2_e  # 2 * inner * log2_e
        exp_val = S.exp2(exp_arg)
        tanh_val = (exp_val - one) / (exp_val + one)
        gelu_val = half * leaky_val * (one + tanh_val)

        # Step 4: Sigmoid
        neg_x_scaled = zero - gelu_val * log2_e
        exp_negx = S.exp2(neg_x_scaled)
        sigmoid_val = one / (one + exp_negx)

        # Step 5: Add bias
        idx_in_batch = global_idx - (global_idx // (out_channels * spatial_size)) * (out_channels * spatial_size)
        c_idx = idx_in_batch // spatial_size
        bias_val = bias[c_idx]

        # Store result
        final_val = sigmoid_val + bias_val
        out[idx] = final_val


def fused_activation(
    x: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Apply fused activation: ReLU -> LeakyReLU -> GELU -> Sigmoid -> Bias"""
    assert x.is_cuda, "Input must be on CUDA/HIP device."

    batch_size = x.shape[0]
    out_channels = x.shape[1]
    depth_out = x.shape[2]
    height_out = x.shape[3]
    width_out = x.shape[4]
    spatial_size = depth_out * height_out * width_out
    n = batch_size * out_channels * spatial_size

    x_f32 = x.to(torch.float32).contiguous()
    x_flat = x_f32.view(-1)
    bias_f32 = bias.view(-1).to(torch.float32).contiguous()
    out_flat = torch.empty_like(x_flat)

    for chunk_offset in range(0, n, ELEMENTS_PER_LAUNCH):
        chunk_size = min(ELEMENTS_PER_LAUNCH, n - chunk_offset)

        x_chunk = x_flat[chunk_offset:chunk_offset + ELEMENTS_PER_LAUNCH]
        out_chunk = out_flat[chunk_offset:chunk_offset + ELEMENTS_PER_LAUNCH]

        if chunk_size < ELEMENTS_PER_LAUNCH:
            x_padded = torch.zeros(ELEMENTS_PER_LAUNCH, dtype=torch.float32, device=x.device)
            x_padded[:chunk_size] = x_chunk
            x_chunk = x_padded
            out_temp = torch.zeros(ELEMENTS_PER_LAUNCH, dtype=torch.float32, device=x.device)
        else:
            out_temp = out_chunk

        fused_activation_kernel[lambda: ((GRID_SIZE, 1, 1), (BLOCK_SIZE, 1, 1))](
            x_chunk, bias_f32, out_temp, n, chunk_offset, out_channels, spatial_size
        )

        if chunk_size < ELEMENTS_PER_LAUNCH:
            out_flat[chunk_offset:chunk_offset + chunk_size] = out_temp[:chunk_size]

    return out_flat.view(x.shape).to(torch.bfloat16)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv(x)
        x_bf16 = x.to(torch.bfloat16)
        bias_bf16 = self.bias.to(torch.bfloat16)
        x = fused_activation(x_bf16, bias_bf16)
        return x


def get_inputs():
    return [torch.rand(64, 8, 32, 64, 64)]


def get_init_inputs():
    return [8, 32, 3, (32, 1, 1, 1)]
