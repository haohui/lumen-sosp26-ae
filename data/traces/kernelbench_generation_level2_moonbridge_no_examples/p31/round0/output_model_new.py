import struct

import torch
import torch.nn as nn

import avelang
import avelang.language as al

BLOCK_SIZE = 256


def _f32_to_bits(f: float) -> int:
    return struct.unpack("<i", struct.pack("<f", f))[0]


@avelang.jit
def post_process_kernel(
    data_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    total_elems: al.i32,
    C_out: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    constant_val_bits: al.i32,
    scale_bits: al.i32,
):
    data_layout = al.make_layout((total_elems,), (1,))
    data = al.make_tensor(data_ptr, al.bf16, data_layout)

    bias_layout = al.make_layout((C_out, 1, 1), (1, 1, 1))
    bias = al.make_tensor(bias_ptr, al.bf16, bias_layout)

    out_layout = al.make_layout((total_elems,), (1,))
    out = al.make_tensor(output_ptr, al.bf16, out_layout)

    tid = al.thread_id(0)
    bid = al.block_id(0)
    block_dim = al.block_dim(0)
    idx = bid * block_dim + tid

    constant_val = al.bitcast(constant_val_bits, al.f32)
    scale = al.bitcast(scale_bits, al.f32)

    if idx < total_elems:
        hw = H_out * W_out
        c_idx = (idx // hw) % C_out

        val = al.convert(data[idx], al.f32)
        if val > constant_val:
            val = constant_val
        b = al.convert(bias[c_idx, 0, 0], al.f32)
        val = val + b
        val = val * scale
        out[idx] = al.convert(val, al.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.constant_value = constant_value
        self.scaling_factor = scaling_factor
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv(x)

        N, C_out, H_out, W_out = x.shape
        total_elems = N * C_out * H_out * W_out

        x = x.contiguous()
        bias_bf16 = self.bias.contiguous()
        out = torch.empty_like(x)

        grid = (total_elems + BLOCK_SIZE - 1) // BLOCK_SIZE

        post_process_kernel[lambda: ((grid, 1, 1), (BLOCK_SIZE, 1, 1))](
            x, bias_bf16, out,
            total_elems, C_out, H_out, W_out,
            _f32_to_bits(self.constant_value),
            _f32_to_bits(self.scaling_factor),
        )

        return out
