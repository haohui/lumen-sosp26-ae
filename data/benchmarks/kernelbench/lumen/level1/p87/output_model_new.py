import torch
import torch.nn as nn
import math
from substrate_kernels.amdgpu_gemm import gemm_1stage_pipeline_transposed_b
from substrate_kernels.amdgpu_conv2d import (
    _transpose_input_nchw_to_nhwc_kernel,
    _transpose_input_launch_config,
)


class ModelNew(nn.Module):
    """
    Optimized pointwise 2D convolution using Substrate's MFMA-tiled GEMM kernel
    with optimized NCHW->NHWC transpose.
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(ModelNew, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias

        # Weight shape: (out_channels, in_channels, 1, 1) for Conv2d
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, 1, 1))

        if bias:
            self.bias_param = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias_param', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias_param is not None:
            fan_in = self.in_channels
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_param, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure BF16 and contiguous
        x = x.to(dtype=torch.bfloat16).contiguous()

        batch_size, in_channels, height, width = x.shape
        out_channels = self.out_channels

        # GEMM dimensions: M = batch * H * W, K = in_channels, N = out_channels
        M = batch_size * height * width
        K = in_channels
        N = out_channels

        # Allocate workspace for NHWC input
        x_nhwc = torch.empty(
            (batch_size, height, width, in_channels),
            dtype=torch.bfloat16,
            device=x.device
        )

        # Transpose input from NCHW to NHWC using optimized kernel
        transpose_grid, transpose_block = _transpose_input_launch_config(
            batch_size, in_channels, height, width
        )
        _transpose_input_nchw_to_nhwc_kernel[lambda: (transpose_grid, transpose_block)](
            x,
            x_nhwc,
            batch_size,
            in_channels,
            height,
            width
        )

        # Reshape to (M, K) for GEMM - this is just a view, no copy
        A = x_nhwc.view(M, K)

        # Get weight in (N, K) format for transposed_b GEMM
        weight = self.weight.data.squeeze(-1).squeeze(-1).contiguous()

        # Run GEMM: C = A @ B^T where B has shape (N, K)
        C = gemm_1stage_pipeline_transposed_b(A, weight)

        # Reshape output to (batch, H, W, N) and transpose to NCHW
        # The output reshape is just a view, and permute is lazy in PyTorch
        output = C.view(batch_size, height, width, out_channels).permute(0, 3, 1, 2)

        # Add bias if needed
        if self.bias_param is not None:
            output = output + self.bias_param.view(1, -1, 1, 1)

        return output
