import torch
import torch.nn as nn
import math
import avelang
import avelang.language as al

NUM_THREADS = 256
OC_PER_THREAD = 8


@avelang.jit
def conv_transpose2d_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    B: al.i32,
    C_in: al.i32,
    C_out: al.i32,
    H_in: al.i32,
    W_in: al.i32,
    H_out: al.i32,
    W_out: al.i32,
):
    one = al.convert(1, al.i32)
    zero_i32 = al.convert(0, al.i32)
    eight = al.convert(8, al.i32)
    three = al.convert(3, al.i32)
    seven = al.convert(7, al.i32)

    in_stride_b = C_in * H_in * W_in
    in_stride_c = H_in * W_in
    in_stride_h = W_in
    in_layout = al.make_layout(
        (B, C_in, H_in, W_in),
        (in_stride_b, in_stride_c, in_stride_h, one),
    )
    in_t = al.make_tensor(input_ptr, al.bf16, in_layout)

    w_stride_ic = C_out * three * seven
    w_stride_oc = three * seven
    w_stride_kh = seven
    w_layout = al.make_layout(
        (C_in, C_out, three, seven),
        (w_stride_ic, w_stride_oc, w_stride_kh, one),
    )
    w_t = al.make_tensor(weight_ptr, al.bf16, w_layout)

    out_stride_b = C_out * H_out * W_out
    out_stride_oc = H_out * W_out
    out_stride_h = W_out
    out_layout = al.make_layout(
        (B, C_out, H_out, W_out),
        (out_stride_b, out_stride_oc, out_stride_h, one),
    )
    out_t = al.make_tensor(output_ptr, al.bf16, out_layout)

    tid = al.thread_id(0)
    bid = al.block_id(0)
    num_threads = al.block_dim(0)
    idx = bid * num_threads + tid

    oc_groups = C_out // eight
    spatial_per_batch = H_out * W_out
    elems_per_batch = oc_groups * spatial_per_batch

    if idx < B * elems_per_batch:
        b = idx // elems_per_batch
        rem = idx % elems_per_batch
        oc_group = rem // spatial_per_batch
        hw_idx = rem % spatial_per_batch
        h = hw_idx // W_out
        w = hw_idx % W_out
        oc_base = oc_group * eight

        acc = al.make_local((8,), al.f32)
        for i in al.range(8):
            acc[i] = al.convert(0.0, al.f32)

        for ic in al.range(C_in):
            for kh in al.range(3):
                in_h = h - kh
                if in_h >= zero_i32:
                    if in_h < H_in:
                        for kw in al.range(7):
                            in_w = w - kw
                            if in_w >= zero_i32:
                                if in_w < W_in:
                                    in_val = al.convert(in_t[b, ic, in_h, in_w], al.f32)
                                    for oc_loc in al.range(8):
                                        g_oc = oc_base + oc_loc
                                        w_val = al.convert(w_t[ic, g_oc, kh, kw], al.f32)
                                        acc[oc_loc] = acc[oc_loc] + in_val * w_val

        for oc_loc in al.range(8):
            g_oc = oc_base + oc_loc
            out_t[b, g_oc, h, w] = al.convert(acc[oc_loc], al.bf16)


def avelang_conv_transpose2d(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    B, C_in, H_in, W_in = x.shape
    C_in_w, C_out, kH, kW = weight.shape

    assert kH == 3 and kW == 7

    H_out = H_in + kH - 1
    W_out = W_in + kW - 1

    x_bf16 = x.contiguous().to(torch.bfloat16)
    w_bf16 = weight.contiguous().to(torch.bfloat16)
    out_bf16 = torch.empty(B, C_out, H_out, W_out, dtype=torch.bfloat16, device=x.device)

    oc_groups = C_out // OC_PER_THREAD
    total_threads = B * oc_groups * H_out * W_out
    grid_size = (total_threads + NUM_THREADS - 1) // NUM_THREADS

    conv_transpose2d_kernel[lambda: ((grid_size, 1, 1), (NUM_THREADS, 1, 1))](
        x_bf16.data_ptr(),
        w_bf16.data_ptr(),
        out_bf16.data_ptr(),
        B, C_in, C_out, H_in, W_in, H_out, W_out,
    )

    return out_bf16


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

        self.weight = nn.Parameter(
            torch.empty(in_channels, out_channels, kernel_size[0], kernel_size[1])
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight.to(device=x.device, dtype=torch.float32)
        return avelang_conv_transpose2d(x, weight)
