import torch
import torch.nn as nn
import substrate
import substrate.language as S
import math

# Constants for BF16 precision
LOG2_E = 1.4426950408889634  # log2(e) for exp(x) = exp2(x * log2(e))

# Thread block size
BLOCK_SIZE = 256


@substrate.jit
def mish_scalar(x: S.f32) -> S.f32:
    """
    Compute mish activation: x * tanh(softplus(x))
    where softplus(x) = log(1 + exp(x))
    """
    # Compute exp(x) using exp2
    log2_e = S.convert(LOG2_E, S.f32)
    exp_x = S.exp2(x * log2_e)

    # Compute softplus(x) = log(1 + exp(x))
    one_plus_exp = S.convert(1.0, S.f32) + exp_x
    softplus_x = S.log(one_plus_exp)

    # Compute mish = x * tanh(softplus_x)
    tanh_softplus = S.tanh(softplus_x)
    result = x * tanh_softplus

    return result


def create_fused_kernel(subtract_total: float):
    """
    Create a fused kernel with the specific subtract value compiled in.
    """
    @substrate.jit
    def fused_subtract_mish_kernel(
        input_ptr: S.Pointer(S.bf16),
        output_ptr: S.Pointer(S.bf16),
        n_elements: S.u32,
    ):
        """
        Fused kernel: subtract value and apply mish activation.
        """
        SUBTRACT_VAL = S.convert(subtract_total, S.f32)

        tid = S.thread_id(0)
        bid = S.block_id(0)

        # Create tensor views with BF16 type
        layout = S.make_layout((n_elements,), (1,))
        g_in = S.make_tensor(input_ptr, S.bf16, layout)
        g_out = S.make_tensor(output_ptr, S.bf16, layout)

        # Linear index
        linear_idx = bid * S.convert(BLOCK_SIZE, S.u32) + tid

        # Each thread processes one element
        if linear_idx < n_elements:
            # Load BF16 value
            val_bf16 = g_in[linear_idx]

            # Convert to f32 for computation
            val = S.convert(val_bf16, S.f32)

            # Subtract
            val = val - SUBTRACT_VAL

            # Apply mish activation
            val = mish_scalar(val)

            # Convert back to BF16 and store
            g_out[linear_idx] = S.convert(val, S.bf16)

    return fused_subtract_mish_kernel


class ModelNew(nn.Module):
    """
    Optimized model using fused Substrate kernels for post-conv operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super(ModelNew, self).__init__()
        # Use PyTorch's optimized Conv2d
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2
        self.total_subtract = subtract_value_1 + subtract_value_2

        # Create kernel with the specific subtract value
        self._kernel = create_fused_kernel(self.total_subtract)

    def forward(self, x):
        # Apply convolution using PyTorch's optimized implementation
        x = self.conv(x)

        # Convert to BF16 for kernel processing
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)

        # Flatten for kernel processing
        original_shape = x.shape
        x_flat = x.flatten()

        n_elements = x_flat.numel()
        output = torch.empty_like(x_flat)

        if n_elements > 0:
            # Calculate grid dimensions
            num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
            if num_blocks == 0:
                num_blocks = 1

            self._kernel[lambda: ((num_blocks, 1, 1), (BLOCK_SIZE, 1, 1))](
                x_flat, output, n_elements
            )

        # Reshape back to original shape
        x = output.view(original_shape)

        return x


batch_size = 128
in_channels = 8
out_channels = 64
height, width = 256, 256
kernel_size = 3
subtract_value_1 = 0.5
subtract_value_2 = 0.2


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2]
