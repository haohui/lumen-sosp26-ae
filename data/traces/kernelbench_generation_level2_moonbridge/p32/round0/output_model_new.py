import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE: al.constexpr = 256


@avelang.jit
def conv_scale_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    batch_size: al.i32,
    in_channels: al.i32,
    out_channels: al.i32,
    kernel_size: al.i32,
    H_in: al.i32,
    W_in: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    scale_factor: al.f32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    global_id = bid * BLOCK_SIZE + tid
    total_outputs = batch_size * out_channels * H_out * W_out

    if global_id < total_outputs:
        tmp = global_id
        w = tmp % W_out
        tmp = tmp // W_out
        h = tmp % H_out
        tmp = tmp // H_out
        oc = tmp % out_channels
        b = tmp // out_channels

        layout_in = al.make_layout(
            (batch_size, in_channels, H_in, W_in),
            (in_channels * H_in * W_in, H_in * W_in, W_in, 1),
        )
        input_t = al.make_tensor(input_ptr, al.bf16, layout_in)

        layout_w = al.make_layout(
            (out_channels, in_channels, kernel_size, kernel_size),
            (in_channels * kernel_size * kernel_size, kernel_size * kernel_size, kernel_size, 1),
        )
        weight_t = al.make_tensor(weight_ptr, al.bf16, layout_w)

        bp = al.make_tensor(bias_ptr, al.bf16, al.make_layout((out_channels,), (1,)))

        acc = al.convert(bp[oc], al.f32)

        for ic in al.range(in_channels):
            for kh in al.range(kernel_size):
                for kw in al.range(kernel_size):
                    val_in = al.convert(input_t[b, ic, h + kh, w + kw], al.f32)
                    val_w = al.convert(weight_t[oc, ic, kh, kw], al.f32)
                    acc = acc + val_in * val_w

        result = acc * scale_factor

        layout_out = al.make_layout(
            (batch_size, out_channels, H_out, W_out),
            (out_channels * H_out * W_out, H_out * W_out, W_out, 1),
        )
        output_t = al.make_tensor(output_ptr, al.bf16, layout_out)
        output_t[b, oc, h, w] = al.convert(result, al.bf16)


@avelang.jit
def min_reduction_kernel(
    input_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    batch_size: al.i32,
    out_channels: al.i32,
    H_out: al.i32,
    W_out: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    global_id = bid * BLOCK_SIZE + tid
    total_outputs = batch_size * H_out * W_out

    if global_id < total_outputs:
        b = global_id // (H_out * W_out)
        rem = global_id - b * (H_out * W_out)
        h = rem // W_out
        w = rem - h * W_out

        layout_in = al.make_layout(
            (batch_size, out_channels, H_out, W_out),
            (out_channels * H_out * W_out, H_out * W_out, W_out, 1),
        )
        input_t = al.make_tensor(input_ptr, al.bf16, layout_in)

        current_min = input_t[b, 0, h, w]

        for oc in al.range(1, out_channels):
            val = input_t[b, oc, h, w]
            current_min = val if val < current_min else current_min

        layout_out = al.make_layout(
            (batch_size, 1, H_out, W_out),
            (H_out * W_out, H_out * W_out, W_out, 1),
        )
        output_t = al.make_tensor(output_ptr, al.bf16, layout_out)
        output_t[b, 0, h, w] = current_min


def avelang_conv_min(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    scale_factor: float,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = x.contiguous().to(dtype=torch.bfloat16)
    weight_bf16 = weight.contiguous().to(dtype=torch.bfloat16)
    bias_bf16 = bias.contiguous().to(dtype=torch.bfloat16)

    batch_size = x_bf16.shape[0]
    in_channels = x_bf16.shape[1]
    H_in = x_bf16.shape[2]
    W_in = x_bf16.shape[3]
    out_channels = weight_bf16.shape[0]
    kernel_h = weight_bf16.shape[2]
    kernel_w = weight_bf16.shape[3]
    H_out = H_in - kernel_h + 1
    W_out = W_in - kernel_w + 1

    conv_out = torch.empty(
        (batch_size, out_channels, H_out, W_out),
        dtype=torch.bfloat16,
        device=x_bf16.device,
    )

    total_conv = batch_size * out_channels * H_out * W_out
    num_blocks_conv = (total_conv + BLOCK_SIZE - 1) // BLOCK_SIZE

    conv_scale_kernel[lambda: ((num_blocks_conv, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_bf16,
        weight_bf16,
        bias_bf16,
        conv_out,
        batch_size,
        in_channels,
        out_channels,
        kernel_h,
        H_in,
        W_in,
        H_out,
        W_out,
        scale_factor,
    )

    final_out = torch.empty(
        (batch_size, 1, H_out, W_out),
        dtype=torch.bfloat16,
        device=x_bf16.device,
    )

    total_min = batch_size * H_out * W_out
    num_blocks_min = (total_min + BLOCK_SIZE - 1) // BLOCK_SIZE

    min_reduction_kernel[lambda: ((num_blocks_min, 1, 1), (BLOCK_SIZE, 1, 1))](
        conv_out,
        final_out,
        batch_size,
        out_channels,
        H_out,
        W_out,
    )

    return final_out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor

    def forward(self, x):
        result = avelang_conv_min(
            x,
            self.conv.weight,
            self.conv.bias,
            self.scale_factor,
        )
        return result.to(x.dtype)
