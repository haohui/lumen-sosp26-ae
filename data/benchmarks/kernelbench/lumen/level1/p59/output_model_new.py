import torch
import torch.nn as nn
import math
import substrate
import substrate.language as S


# Block tile sizes
BLOCK_SIZE = 256


@substrate.jit
def conv3d_asymmetric_kernel(
    input_ptr: S.Pointer(S.bf16),
    weight_ptr: S.Pointer(S.bf16),
    output_ptr: S.Pointer(S.bf16),
    batch_size: S.i32,
    in_channels: S.i32,
    out_channels: S.i32,
    in_d: S.i32,
    in_h: S.i32,
    in_w: S.i32,
    out_d: S.i32,
    out_h: S.i32,
    out_w: S.i32,
    kernel_d: S.i32,
    kernel_h: S.i32,
    kernel_w: S.i32,
    stride: S.i32,
    padding: S.i32,
    dilation: S.i32,
    groups: S.i32,
):
    """
    3D Convolution kernel with asymmetric kernel (kernel_d, kernel_h, kernel_w).
    Each thread computes one output element.

    Input/Output tensor layout: (batch, channels, D, H, W)
    Weight layout: (out_channels, in_channels_per_group, kernel_d, kernel_h, kernel_w)
    """
    tid = S.thread_id(0)
    block_idx = S.block_id(0)

    # Decode linear block index to (batch, out_channel, spatial)
    spatial_size = out_d * out_h * out_w

    # Linear index within the grid
    linear_idx = block_idx * BLOCK_SIZE + tid

    # Total elements to compute
    total_elements = batch_size * out_channels * spatial_size

    # Bounds check
    if linear_idx >= total_elements:
        return

    # Decode linear index to (n, oc, od, oh, ow)
    n = linear_idx // (out_channels * spatial_size)
    remaining = linear_idx % (out_channels * spatial_size)
    oc = remaining // spatial_size
    remaining = remaining % spatial_size

    od = remaining // (out_h * out_w)
    remaining = remaining % (out_h * out_w)
    oh = remaining // out_w
    ow = remaining % out_w

    # Compute input/output strides
    # Input: (batch, in_channels, D, H, W)
    in_stride_w = S.convert(1, S.i32)
    in_stride_h = in_stride_w * in_w
    in_stride_d = in_stride_h * in_h
    in_stride_c = in_stride_d * in_d
    in_stride_b = in_stride_c * in_channels

    in_layout = S.make_layout(
        (batch_size, in_channels, in_d, in_h, in_w),
        (in_stride_b, in_stride_c, in_stride_d, in_stride_h, in_stride_w)
    )
    input_tensor = S.make_tensor(input_ptr, S.bf16, in_layout)

    # Weight: (out_channels, in_channels // groups, kernel_d, kernel_h, kernel_w)
    in_channels_per_group = in_channels // groups
    out_channels_per_group = out_channels // groups
    group_idx = oc // out_channels_per_group
    in_channel_start = group_idx * in_channels_per_group

    w_stride_w = S.convert(1, S.i32)
    w_stride_h = w_stride_w * kernel_w
    w_stride_d = w_stride_h * kernel_h
    w_stride_ic = w_stride_d * kernel_d
    w_stride_oc = w_stride_ic * in_channels_per_group

    weight_layout = S.make_layout(
        (out_channels, in_channels_per_group, kernel_d, kernel_h, kernel_w),
        (w_stride_oc, w_stride_ic, w_stride_d, w_stride_h, w_stride_w)
    )
    weight_tensor = S.make_tensor(weight_ptr, S.bf16, weight_layout)

    # Output: (batch, out_channels, out_d, out_h, out_w)
    out_stride_w = S.convert(1, S.i32)
    out_stride_h = out_stride_w * out_w
    out_stride_d = out_stride_h * out_h
    out_stride_c = out_stride_d * out_d
    out_stride_b = out_stride_c * out_channels

    out_layout = S.make_layout(
        (batch_size, out_channels, out_d, out_h, out_w),
        (out_stride_b, out_stride_c, out_stride_d, out_stride_h, out_stride_w)
    )
    output_tensor = S.make_tensor(output_ptr, S.bf16, out_layout)

    # Compute convolution
    acc = S.convert(0.0, S.f32)

    # Loop over input channels in this group
    for ic_local in S.range(in_channels_per_group):
        ic = in_channel_start + ic_local

        # Loop over kernel dimensions (D, H, W)
        for kd in S.range(kernel_d):
            # Compute input D position
            id_in = od * stride + kd * dilation - padding

            # Bounds check for D
            if id_in < 0 or id_in >= in_d:
                continue

            for kh in S.range(kernel_h):
                # Compute input H position
                ih = oh * stride + kh * dilation - padding

                # Bounds check for H
                if ih < 0 or ih >= in_h:
                    continue

                for kw in S.range(kernel_w):
                    # Compute input W position
                    iw = ow * stride + kw * dilation - padding

                    # Bounds check for W
                    if iw < 0 or iw >= in_w:
                        continue

                    # Load input and weight values
                    in_val = S.convert(input_tensor[n, ic, id_in, ih, iw], S.f32)
                    w_val = S.convert(weight_tensor[oc, ic_local, kd, kh, kw], S.f32)

                    # Accumulate
                    acc = acc + in_val * w_val

    # Store result
    output_tensor[n, oc, od, oh, ow] = S.convert(acc, S.bf16)


class ModelNew(torch.nn.Module):
    """
    Optimized 3D convolution using Substrate DSL.
    Handles asymmetric kernel (kernel_size, kernel_size, 1).

    Input/Output tensor layout follows PyTorch convention: (batch, channels, D, H, W)
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, dilation: int = 1,
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # Weight tensor shape: (out_channels, in_channels // groups, kernel_d, kernel_h, kernel_w)
        # For kernel_size=(kernel_size, kernel_size, 1):
        #   kernel_d = kernel_size, kernel_h = kernel_size, kernel_w = 1
        self.weight = nn.Parameter(torch.empty(
            out_channels, in_channels // groups, kernel_size, kernel_size, 1
        ))

        if bias:
            self.bias_param = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias_param', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias_param is not None:
            fan_in = self.in_channels // self.groups * self.kernel_size ** 2
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_param, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is contiguous and in BF16
        x = x.contiguous()
        original_dtype = x.dtype
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)

        batch_size, in_channels, in_d, in_h, in_w = x.shape

        # Kernel size: (kernel_size, kernel_size, 1) = (kernel_d, kernel_h, kernel_w)
        kernel_d = self.kernel_size
        kernel_h = self.kernel_size
        kernel_w = 1

        # Compute output dimensions
        out_d = (in_d + 2 * self.padding - self.dilation * (kernel_d - 1) - 1) // self.stride + 1
        out_h = (in_h + 2 * self.padding - self.dilation * (kernel_h - 1) - 1) // self.stride + 1
        out_w = (in_w + 2 * self.padding - self.dilation * (kernel_w - 1) - 1) // self.stride + 1

        # Create output tensor
        output = torch.empty(
            (batch_size, self.out_channels, out_d, out_h, out_w),
            dtype=torch.bfloat16, device=x.device
        )

        # Ensure weight is contiguous and in BF16
        weight = self.weight.data.contiguous()
        if weight.dtype != torch.bfloat16:
            weight = weight.to(torch.bfloat16)

        # Calculate grid size
        total_elements = batch_size * self.out_channels * out_d * out_h * out_w
        grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

        # Launch kernel
        conv3d_asymmetric_kernel[lambda: ((grid_size, 1, 1), (BLOCK_SIZE, 1, 1))](
            x, weight, output,
            batch_size, in_channels, self.out_channels,
            in_d, in_h, in_w,
            out_d, out_h, out_w,
            kernel_d, kernel_h, kernel_w,
            self.stride, self.padding, self.dilation, self.groups
        )

        # Add bias if needed
        if self.bias_param is not None:
            output = output + self.bias_param.view(1, -1, 1, 1, 1)

        # Convert back to original dtype if needed
        if original_dtype != torch.bfloat16:
            output = output.to(original_dtype)

        return output
