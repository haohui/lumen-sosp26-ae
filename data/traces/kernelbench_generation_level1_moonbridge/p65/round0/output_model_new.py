import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE = 256
TILE_OC = 8
SPATIAL_PER_THREAD = 4


@avelang.jit
def conv_transpose2d_kernel(
    in_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    IC: al.i32,
    OC: al.i32,
    H: al.i32,
    W: al.i32,
    KH: al.i32,
    KW: al.i32,
    H_OUT: al.i32,
    W_OUT: al.i32,
):
    tid = al.thread_id(0)
    block_x = al.block_id(0)
    oc_block = al.block_id(1)
    b = al.block_id(2)

    base_spatial = (block_x * BLOCK_SIZE + tid) * SPATIAL_PER_THREAD
    oc_start = oc_block * TILE_OC
    total_spatial = H_OUT * W_OUT

    in_tensor = al.make_tensor(
        in_ptr, al.bf16,
        al.make_layout((B, IC, H, W), (IC * H * W, H * W, W, 1))
    )
    w_tensor = al.make_tensor(
        w_ptr, al.bf16,
        al.make_layout((IC, OC, KH, KW), (OC * KH * KW, KH * KW, KW, 1))
    )
    out_tensor = al.make_tensor(
        out_ptr, al.bf16,
        al.make_layout((B, OC, H_OUT, W_OUT), (OC * H_OUT * W_OUT, H_OUT * W_OUT, W_OUT, 1))
    )

    zero = al.convert(0, al.i32)
    num_acc = SPATIAL_PER_THREAD * TILE_OC
    acc = al.make_local((num_acc,), al.f32)
    for idx in al.range(num_acc):
        acc[idx] = al.convert(0.0, al.f32)

    # Local weight buffer for current (ic, kh, kw) slice
    w_local = al.make_local((TILE_OC,), al.f32)

    for ic in al.range(IC):
        for kh in al.range(KH):
            for kw in al.range(KW):
                # Load weight for all TILE_OC channels into local buffer
                for t in al.range(TILE_OC):
                    oc = oc_start + t
                    if oc < OC:
                        w_local[t] = al.convert(w_tensor[ic, oc, kh, kw], al.f32)

                # Process all spatial positions using the cached weight
                for s in al.range(SPATIAL_PER_THREAD):
                    spatial_idx = base_spatial + s
                    if spatial_idx < total_spatial:
                        h_out = spatial_idx // W_OUT
                        w_out = spatial_idx % W_OUT
                        h_in = h_out - kh
                        w_in = w_out - kw
                        if h_in >= zero and h_in < H and w_in >= zero and w_in < W:
                            in_val = al.convert(in_tensor[b, ic, h_in, w_in], al.f32)
                            for t in al.range(TILE_OC):
                                oc = oc_start + t
                                if oc < OC:
                                    acc[s * TILE_OC + t] = acc[s * TILE_OC + t] + in_val * w_local[t]

    # Write output
    for s in al.range(SPATIAL_PER_THREAD):
        spatial_idx = base_spatial + s
        if spatial_idx < total_spatial:
            h_out = spatial_idx // W_OUT
            w_out = spatial_idx % W_OUT
            for t in al.range(TILE_OC):
                oc = oc_start + t
                if oc < OC:
                    out_tensor[b, oc, h_out, w_out] = al.convert(acc[s * TILE_OC + t], al.bf16)


def avelang_conv_transpose2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    stride: int = 1,
    padding: int = 0,
    output_padding: int = 0,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = x
    if not x_bf16.is_cuda:
        x_bf16 = x_bf16.cuda()
    if x_bf16.dtype != torch.bfloat16:
        x_bf16 = x_bf16.to(dtype=torch.bfloat16)
    if not x_bf16.is_contiguous():
        x_bf16 = x_bf16.contiguous()

    w_bf16 = weight
    if not w_bf16.is_cuda:
        w_bf16 = w_bf16.cuda()
    if w_bf16.dtype != torch.bfloat16:
        w_bf16 = w_bf16.to(dtype=torch.bfloat16)
    if not w_bf16.is_contiguous():
        w_bf16 = w_bf16.contiguous()

    B, IC, H, W = x_bf16.shape
    w_IC, OC, KH, KW = w_bf16.shape

    H_out = (H - 1) * stride - 2 * padding + KH + output_padding
    W_out = (W - 1) * stride - 2 * padding + KW + output_padding

    out = torch.empty((B, OC, H_out, W_out), device=x_bf16.device, dtype=torch.bfloat16)

    total_spatial = H_out * W_out
    total_threads_per_grid_x = BLOCK_SIZE * SPATIAL_PER_THREAD
    grid_x = (total_spatial + total_threads_per_grid_x - 1) // total_threads_per_grid_x
    grid_y = (OC + TILE_OC - 1) // TILE_OC
    grid = (grid_x, grid_y, B)

    conv_transpose2d_kernel[lambda: (grid, (BLOCK_SIZE, 1, 1))](
        x_bf16, w_bf16, out,
        B, IC, OC, H, W, KH, KW, H_out, W_out,
    )

    return out


class ModelNew(nn.Module):
    """
    Performs a transposed 2D convolution with a square input and an asymmetric kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (height, width).
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int or tuple, optional): Padding applied to the input. Defaults to 0.
        output_padding (int or tuple, optional): Additional size added to one side of the output shape. Defaults to 0.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return avelang_conv_transpose2d(
            x, self.conv_transpose2d.weight,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
        )


# Test code
batch_size = 8
in_channels = 64
out_channels = 64
kernel_size = (3, 7)
width = 512
height = 512

def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
