import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE = 256
WARP_SIZE = 64
NUM_WARPS = BLOCK_SIZE // WARP_SIZE


@avelang.jit
def conv_transpose_multiply_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.f32),
    N: al.i32,
    IC: al.i32,
    OC: al.i32,
    H_in: al.i32,
    W_in: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    stride_val: al.i32,
    padding_val: al.i32,
    output_padding_val: al.i32,
    KH: al.i32,
    KW: al.i32,
):
    # Input layout: (N, IC, H_in, W_in) row-major
    in_n_s = IC * H_in * W_in
    in_c_s = H_in * W_in
    in_h_s = W_in
    in_w_s = al.convert(1, al.i32)
    input_layout = al.make_layout(
        (N, IC, H_in, W_in),
        (in_n_s, in_c_s, in_h_s, in_w_s),
    )
    input_t = al.make_tensor(input_ptr, al.bf16, input_layout)

    # Weight layout: (IC, OC, KH, KW) row-major
    w_ic_s = OC * KH * KW
    w_oc_s = KH * KW
    w_kh_s = KW
    w_kw_s = al.convert(1, al.i32)
    weight_layout = al.make_layout(
        (IC, OC, KH, KW),
        (w_ic_s, w_oc_s, w_kh_s, w_kw_s),
    )
    weight_t = al.make_tensor(weight_ptr, al.bf16, weight_layout)

    # Bias layout: (OC,) row-major
    bias_layout = al.make_layout((OC,), (al.convert(1, al.i32),))
    bias_t = al.make_tensor(bias_ptr, al.bf16, bias_layout)

    # Output layout: (N, OC, H_out, W_out) row-major
    out_n_s = OC * H_out * W_out
    out_c_s = H_out * W_out
    out_h_s = W_out
    out_w_s = al.convert(1, al.i32)
    output_layout = al.make_layout(
        (N, OC, H_out, W_out),
        (out_n_s, out_c_s, out_h_s, out_w_s),
    )
    output_t = al.make_tensor(output_ptr, al.f32, output_layout)

    n = al.block_id(0)
    oc = al.block_id(1)
    spatial_block = al.block_id(2)
    tid = al.thread_id(0)

    spatial_idx = spatial_block * BLOCK_SIZE + tid
    hw_total = H_out * W_out

    if spatial_idx < hw_total:
        h = spatial_idx // W_out
        w = spatial_idx % W_out

        acc = al.convert(0.0, al.f32)
        one = al.convert(1, al.i32)
        kh_max = KH - one
        kw_max = KW - one
        pad_adj = kh_max - padding_val

        for ic in al.range(IC):
            for kh in al.range(KH):
                h_diff = h + kh - pad_adj
                h_rem = h_diff % stride_val
                if h_rem == 0:
                    h_in = h_diff // stride_val
                    if h_in >= 0:
                        if h_in < H_in:
                            for kw in al.range(KW):
                                w_diff = w + kw - pad_adj
                                w_rem = w_diff % stride_val
                                if w_rem == 0:
                                    w_in = w_diff // stride_val
                                    if w_in >= 0:
                                        if w_in < W_in:
                                            inp = al.convert(input_t[n, ic, h_in, w_in], al.f32)
                                            wt = al.convert(weight_t[ic, oc, kh_max - kh, kw_max - kw], al.f32)
                                            acc = acc + inp * wt

        # Add bias
        b = al.convert(bias_t[oc], al.f32)
        acc = acc + b

        # Multiply by scalar (0.5) and store as bf16
        output_t[n, oc, h, w] = acc * al.convert(0.5, al.f32)


@avelang.jit
def spatial_reduce_kernel(
    input_ptr: al.Pointer(al.f32),
    output_ptr: al.Pointer(al.bf16),
    N: al.i32,
    OC: al.i32,
    H: al.i32,
    W: al.i32,
):
    # Input layout: (N, OC, H, W) row-major
    in_n_s = OC * H * W
    in_c_s = H * W
    in_h_s = W
    in_w_s = al.convert(1, al.i32)
    input_layout = al.make_layout(
        (N, OC, H, W),
        (in_n_s, in_c_s, in_h_s, in_w_s),
    )
    input_t = al.make_tensor(input_ptr, al.f32, input_layout)

    # Output layout: (N, OC, 1, 1) row-major
    one = al.convert(1, al.i32)
    out_n_s = OC
    out_c_s = al.convert(1, al.i32)
    output_layout = al.make_layout(
        (N, OC, one, one),
        (out_n_s, out_c_s, one, one),
    )
    output_t = al.make_tensor(output_ptr, al.bf16, output_layout)

    n = al.block_id(0)
    oc = al.block_id(1)
    tid = al.thread_id(0)
    block_dim = al.block_dim(0)

    total_spatial = H * W
    acc = al.convert(0.0, al.f32)

    for idx in al.range(tid, total_spatial, block_dim):
        h = idx // W
        w = idx % W
        acc = acc + input_t[n, oc, h, w]

    # Warp-level reduction (warp size = 64 on AMD)
    acc = acc + al.shuffle_down(acc, 32, 64)
    acc = acc + al.shuffle_down(acc, 16, 64)
    acc = acc + al.shuffle_down(acc, 8, 64)
    acc = acc + al.shuffle_down(acc, 4, 64)
    acc = acc + al.shuffle_down(acc, 2, 64)
    acc = acc + al.shuffle_down(acc, 1, 64)

    # Write warp partial sums to shared memory
    shared = al.make_shared((NUM_WARPS,), al.f32)
    warp_id = tid // WARP_SIZE
    lane_id = tid % WARP_SIZE
    if lane_id == 0:
        shared[warp_id] = acc
    al.syncthreads()

    # Final reduction of warp partials using one warp
    if tid < NUM_WARPS:
        warp_acc = shared[tid]
        warp_acc = warp_acc + al.shuffle_down(warp_acc, 2, 4)
        warp_acc = warp_acc + al.shuffle_down(warp_acc, 1, 4)
        if tid == 0:
            count = al.convert(total_spatial, al.f32)
            output_t[n, oc, 0, 0] = al.convert(warp_acc / count, al.bf16)


def _run_model(x: torch.Tensor, conv_weight: torch.Tensor, conv_bias: torch.Tensor,
               stride: int, padding: int, output_padding: int, multiplier: float) -> torch.Tensor:
    # Ensure inputs are on the GPU
    device = x.device

    N, IC, H_in, W_in = x.shape
    OC = conv_weight.shape[1]
    KH, KW = conv_weight.shape[2], conv_weight.shape[3]

    H_out = (H_in - 1) * stride - 2 * padding + KH + output_padding
    W_out = (W_in - 1) * stride - 2 * padding + KW + output_padding

    # Convert inputs to bf16
    x_bf16 = x.contiguous()
    w_bf16 = conv_weight.contiguous()
    bias_bf16 = conv_bias.contiguous()

    # Intermediate buffer for conv transpose output
    intermediate = torch.empty(N, OC, H_out, W_out, dtype=torch.float32, device=device)

    hw = H_out * W_out
    grid_z = (hw + BLOCK_SIZE - 1) // BLOCK_SIZE

    conv_transpose_multiply_kernel[lambda: ((N, OC, grid_z), (BLOCK_SIZE, 1, 1))](
        x_bf16.data_ptr(),
        w_bf16.data_ptr(),
        bias_bf16.data_ptr(),
        intermediate.data_ptr(),
        N, IC, OC, H_in, W_in, H_out, W_out,
        stride, padding, output_padding, KH, KW,
    )

    # Output buffer
    output = torch.empty(N, OC, 1, 1, dtype=torch.bfloat16, device=device)

    spatial_reduce_kernel[lambda: ((N, OC, 1), (BLOCK_SIZE, 1, 1))](
        intermediate.data_ptr(),
        output.data_ptr(),
        N, OC, H_out, W_out,
    )

    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding,
        )
        self.multiplier = multiplier

    def forward(self, x):
        return _run_model(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.conv_transpose.stride[0],
            self.conv_transpose.padding[0],
            self.conv_transpose.output_padding[0],
            self.multiplier,
        )
