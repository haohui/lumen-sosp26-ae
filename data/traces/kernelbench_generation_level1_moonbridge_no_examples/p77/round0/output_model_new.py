import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def conv_transpose3d_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.f32),
    B: al.i32,
    IC: al.i32,
    OC: al.i32,
    D_in: al.i32,
    H_in: al.i32,
    W_in: al.i32,
    D_out: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    K: al.i32,
    stride: al.i32,
    padding: al.i32,
    dilation: al.i32,
    TOTAL_SPATIAL: al.constexpr,
    BLOCK_SIZE: al.constexpr,
    NUM_SPATIAL_BLOCKS: al.constexpr,
):
    # Create 1D tensor views over the flat buffers
    x_total = B * IC * D_in * H_in * W_in
    x_layout = al.make_layout((x_total,), (1,))
    x = al.make_tensor(x_ptr, al.bf16, x_layout)

    w_total = IC * OC * K * K * K
    w_layout = al.make_layout((w_total,), (1,))
    w_weights = al.make_tensor(w_ptr, al.bf16, w_layout)

    out_total = B * OC * D_out * H_out * W_out
    out_layout = al.make_layout((out_total,), (1,))
    out = al.make_tensor(out_ptr, al.f32, out_layout)

    # Decode block ID to (oc, spatial_block) and batch
    block_linear = al.convert(al.block_id(0), al.i32)
    b = al.convert(al.block_id(1), al.i32)
    oc = block_linear % OC
    spatial_block = block_linear / OC

    tid = al.convert(al.thread_id(0), al.i32)
    blk_i32 = al.convert(BLOCK_SIZE, al.i32)
    spatial_idx = spatial_block * blk_i32 + tid

    if spatial_idx < TOTAL_SPATIAL:
        # Flat spatial index -> (d_pos, h_pos, w_pos)
        hw_area = H_out * W_out
        d_pos = spatial_idx / hw_area
        rem_d = spatial_idx % hw_area
        h_pos = rem_d / W_out
        w_pos = rem_d % W_out

        # Precompute strides for indexing
        x_stride_ic = D_in * H_in * W_in
        x_stride_d = H_in * W_in
        w_oc_stride = K * K * K
        w_ic_stride = OC * w_oc_stride
        out_stride_oc = D_out * H_out * W_out
        out_stride_d = H_out * W_out

        zero_i32 = al.convert(0, al.i32)
        one_i32 = al.convert(1, al.i32)
        k3 = K * K * K
        k2 = K * K

        x_b_offset = b * IC * x_stride_ic
        out_idx = b * OC * out_stride_oc + oc * out_stride_oc + d_pos * out_stride_d + h_pos * W_out + w_pos

        # Read initial FP32 accumulator value (zero from torch.zeros)
        acc = out[out_idx]

        for ic_raw in al.range(IC):
            ic = al.convert(ic_raw, al.i32)
            w_ic_base = ic * w_ic_stride + oc * w_oc_stride
            x_ic_offset = x_b_offset + ic * x_stride_ic

            for k_idx_raw in al.range(k3):
                k_idx = al.convert(k_idx_raw, al.i32)
                kw = k_idx % K
                rem_kw = k_idx / K
                kh = rem_kw % K
                kd = rem_kw / K

                # Compute candidate input position for this kernel element
                d_temp = d_pos + padding - kd * dilation
                h_temp = h_pos + padding - kh * dilation
                w_temp = w_pos + padding - kw * dilation

                # Validate: position must be non-negative and stride-aligned
                valid = one_i32
                d_rem = d_temp % stride
                if d_temp < zero_i32:
                    valid = zero_i32
                if d_rem != zero_i32:
                    valid = zero_i32
                if valid != zero_i32:
                    h_rem = h_temp % stride
                    if h_temp < zero_i32:
                        valid = zero_i32
                    if h_rem != zero_i32:
                        valid = zero_i32
                    if valid != zero_i32:
                        w_rem = w_temp % stride
                        if w_temp < zero_i32:
                            valid = zero_i32
                        if w_rem != zero_i32:
                            valid = zero_i32
                        if valid != zero_i32:
                            d_in = d_temp / stride
                            h_in = h_temp / stride
                            w_in = w_temp / stride
                            # Bounds check
                            if d_in < D_in:
                                if h_in < H_in:
                                    if w_in < W_in:
                                        x_idx = x_ic_offset + d_in * x_stride_d + h_in * W_in + w_in
                                        w_idx = w_ic_base + kd * k2 + kh * K + kw
                                        x_val = al.convert(x[x_idx], al.f32)
                                        w_val = al.convert(w_weights[w_idx], al.f32)
                                        acc = acc + x_val * w_val

        # Write accumulated result
        out[out_idx] = acc


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super().__init__()
        self.conv = nn.ConvTranspose3d(
            in_channels, out_channels,
            kernel_size=(kernel_size, kernel_size, kernel_size),
            stride=stride, padding=padding, dilation=dilation, bias=bias,
        )
        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.conv.weight.data

        B, IC, D_in, H_in, W_in = x.shape
        OC = self.out_channels
        K = self.conv.kernel_size[0]

        D_out = (D_in - 1) * self.stride - 2 * self.padding + self.dilation * (K - 1) + 1
        H_out = (H_in - 1) * self.stride - 2 * self.padding + self.dilation * (K - 1) + 1
        W_out = (W_in - 1) * self.stride - 2 * self.padding + self.dilation * (K - 1) + 1

        TOTAL_SPATIAL = D_out * H_out * W_out
        BLOCK_SIZE = 256
        NUM_SPATIAL_BLOCKS = (TOTAL_SPATIAL + BLOCK_SIZE - 1) // BLOCK_SIZE

        x = x.contiguous()
        weight = weight.contiguous()

        # Use FP32 accumulation buffer to avoid repeated BF16 precision loss
        out_f32 = torch.zeros(B, OC, D_out, H_out, W_out, dtype=torch.float32, device=x.device)

        grid_x = NUM_SPATIAL_BLOCKS * OC
        grid_y = B

        conv_transpose3d_kernel[lambda: ((grid_x, grid_y, 1), (BLOCK_SIZE, 1, 1))](
            x.to(torch.bfloat16), weight.to(torch.bfloat16), out_f32,
            B, IC, OC,
            D_in, H_in, W_in,
            D_out, H_out, W_out,
            K,
            self.stride, self.padding, self.dilation,
            TOTAL_SPATIAL, BLOCK_SIZE, NUM_SPATIAL_BLOCKS,
        )

        out = out_f32.to(torch.bfloat16)

        if self.bias:
            out = out + self.conv.bias.data.to(torch.bfloat16).view(1, -1, 1, 1, 1)

        return out


# Test code
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = 3
depth = 16
height = 32
width = 32
stride = 2
padding = 1
dilation = 2

def get_inputs():
    x = torch.rand(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]
