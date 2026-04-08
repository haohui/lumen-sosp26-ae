import torch
import torch.nn as nn
import math
import substrate
import substrate.language as S


TILE_H: S.constexpr = 8
TILE_W: S.constexpr = 8


@substrate.jit
def fused_kernel(
    input_ptr: S.Pointer(S.bf16),
    weight_ptr: S.Pointer(S.bf16),
    bias_ptr: S.Pointer(S.bf16),
    output_ptr: S.Pointer(S.bf16),
    batch_size: S.i32,
    in_channels: S.i32,
    out_channels: S.i32,
    in_height: S.i32,
    in_width: S.i32,
    out_height: S.i32,
    out_width: S.i32,
    kernel_h: S.i32,
    kernel_w: S.i32,
    stride_h: S.i32,
    stride_w: S.i32,
    add_value: S.constexpr,
    multiply_value: S.constexpr,
):
    """
    Fused ConvTranspose2d + epilogue kernel with bias support.
    """
    bc_block = S.block_id(0)
    n = bc_block // out_channels
    c_out = bc_block % out_channels

    tile_hw_linear = S.block_id(1)
    tiles_w = (out_width + TILE_W - 1) // TILE_W

    tile_h_idx = tile_hw_linear // tiles_w
    tile_w_idx = tile_hw_linear % tiles_w

    tid_h = S.thread_id(0)
    tid_w = S.thread_id(1)

    h_out = tile_h_idx * TILE_H + tid_h
    w_out = tile_w_idx * TILE_W + tid_w

    if n >= batch_size or c_out >= out_channels or h_out >= out_height or w_out >= out_width:
        return

    # Layouts
    input_layout = S.make_layout(
        (batch_size, in_channels, in_height, in_width),
        (in_channels * in_height * in_width, in_height * in_width, in_width, 1)
    )
    input_tensor = S.make_tensor(input_ptr, S.bf16, input_layout)

    weight_layout = S.make_layout(
        (in_channels, out_channels, kernel_h, kernel_w),
        (out_channels * kernel_h * kernel_w, kernel_h * kernel_w, kernel_w, 1)
    )
    weight_tensor = S.make_tensor(weight_ptr, S.bf16, weight_layout)

    bias_layout = S.make_layout((out_channels,), (1,))
    bias_tensor = S.make_tensor(bias_ptr, S.bf16, bias_layout)

    output_layout = S.make_layout(
        (batch_size, out_channels, out_height, out_width),
        (out_channels * out_height * out_width, out_height * out_width, out_width, 1)
    )
    output_tensor = S.make_tensor(output_ptr, S.bf16, output_layout)

    # FP32 accumulation for transposed convolution
    acc = S.convert(0.0, S.f32)

    # Main computation loop for ConvTranspose2d
    # Each output position (h_out, w_out) accumulates contributions from all input channels
    # and kernel positions that map to this output location
    for kh in S.range(kernel_h):
        h_in_scaled = h_out - kh
        if h_in_scaled >= 0:
            h_in_rem = h_in_scaled % stride_h
            if h_in_rem == 0:
                h_in = h_in_scaled // stride_h
                if h_in < in_height:
                    for kw in S.range(kernel_w):
                        w_in_scaled = w_out - kw
                        if w_in_scaled >= 0:
                            w_in_rem = w_in_scaled % stride_w
                            if w_in_rem == 0:
                                w_in = w_in_scaled // stride_w
                                if w_in < in_width:
                                    for c_in in S.range(in_channels):
                                        in_bf16 = input_tensor[n, c_in, h_in, w_in]
                                        w_bf16 = weight_tensor[c_in, c_out, kh, kw]
                                        in_f32 = S.convert(in_bf16, S.f32)
                                        w_f32 = S.convert(w_bf16, S.f32)
                                        acc = acc + in_f32 * w_f32

    # Add bias
    bias_f32 = S.convert(bias_tensor[c_out], S.f32)
    acc = acc + bias_f32

    # Epilogue: add, min(0), gelu, multiply
    add_f32 = S.convert(add_value, S.f32)
    mult_f32 = S.convert(multiply_value, S.f32)
    zero = S.convert(0.0, S.f32)
    half = S.convert(0.5, S.f32)
    one = S.convert(1.0, S.f32)
    inv_sqrt2 = S.convert(0.7071067811865475, S.f32)

    # add_value
    acc = acc + add_f32
    # torch.min(x, torch.tensor(0.0)) - clamp to <= 0
    if acc > zero:
        acc = zero
    # GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    erf_val = S.erf(acc * inv_sqrt2)
    acc = half * acc * (one + erf_val)
    # multiply_value
    acc = acc * mult_f32

    output_tensor[n, c_out, h_out, w_out] = S.convert(acc, S.bf16)


class ModelNew(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int, add_value: float, multiply_value: float):
        super(ModelNew, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.add_value = add_value
        self.multiply_value = multiply_value

        # PyTorch ConvTranspose2d weight shape: (in_channels, out_channels, kH, kW)
        self.weight = nn.Parameter(torch.empty(
            in_channels, out_channels, kernel_size, kernel_size
        ))

        # ConvTranspose2d has bias=True by default
        self.bias = nn.Parameter(torch.empty(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            # For ConvTranspose2d, fan_in = out_channels * kernel_size^2
            # This matches PyTorch's _calculate_fan_in_and_fan_out for transposed conv
            fan_in = self.out_channels * self.kernel_size * self.kernel_size
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        batch_size, in_channels, in_height, in_width = x.shape

        out_height = (in_height - 1) * self.stride + self.kernel_size
        out_width = (in_width - 1) * self.stride + self.kernel_size

        output = torch.zeros((batch_size, self.out_channels, out_height, out_width),
                            dtype=x.dtype, device=x.device)
        weight = self.weight.data.contiguous()
        bias = self.bias.data.contiguous()

        tiles_h = (out_height + TILE_H - 1) // TILE_H
        tiles_w = (out_width + TILE_W - 1) // TILE_W

        grid = (batch_size * self.out_channels, tiles_h * tiles_w, 1)
        block = (TILE_H, TILE_W, 1)

        fused_kernel[lambda: (grid, block)](
            x, weight, bias, output,
            batch_size, self.in_channels, self.out_channels,
            in_height, in_width,
            out_height, out_width,
            self.kernel_size, self.kernel_size,
            self.stride, self.stride,
            self.add_value, self.multiply_value
        )

        return output
