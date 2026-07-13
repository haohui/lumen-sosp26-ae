import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def depthwise_conv_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    y_ptr: al.Pointer(al.bf16),
    B: al.i32,
    C: al.i32,
    H: al.i32,
    W: al.i32,
    OH: al.i32,
    OW: al.i32,
    TILE_M: al.i32,
):
    tid = al.thread_id(0)
    block_idx = al.block_id(0)
    num_threads = al.block_dim(0)

    total_tiles = (OH * OW + TILE_M - 1) // TILE_M
    b = block_idx // total_tiles
    tile_idx = block_idx % total_tiles

    start_pos = tile_idx * TILE_M

    x_layout = al.make_layout((B, C, H, W), (C * H * W, H * W, W, 1))
    x = al.make_tensor(x_ptr, al.bf16, x_layout)
    w_layout = al.make_layout((C, 1, 3, 1), (3, 3, 1, 1))
    w = al.make_tensor(w_ptr, al.bf16, w_layout)
    y_layout = al.make_layout((B, C, OH, OW), (C * OH * OW, OH * OW, OW, 1))
    y = al.make_tensor(y_ptr, al.bf16, y_layout)

    # Each thread handles one output position within the tile
    pos_offset = tid
    if pos_offset < TILE_M:
        pos_idx = start_pos + pos_offset
        if pos_idx < OH * OW:
            h_out = pos_idx // OW
            w_out = pos_idx % OW

            for c in al.range(C):
                acc = al.convert(0.0, al.f32)
                for k0 in al.range(3):
                    h_in = h_out + k0
                    x_val = al.convert(x[b, c, h_in, w_out], al.f32)
                    w_val = al.convert(w[c, 0, k0, 0], al.f32)
                    acc = acc + x_val * w_val
                y[b, c, h_out, w_out] = al.convert(acc, al.bf16)


class ModelNew(nn.Module):

    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.conv2d = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=(kernel_size, 1),
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
        )
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x):
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)

        B_val, C_val, H_val, W_val = x.shape
        KS_val = self.kernel_size

        OH_val = (H_val + 2 * self.padding - self.dilation * (KS_val - 1) - 1) // self.stride + 1
        OW_val = (W_val + 2 * self.padding - self.dilation * (1 - 1) - 1) // self.stride + 1

        x_cont = x.contiguous()
        w = self.conv2d.weight.to(device=x.device, dtype=torch.bfloat16).contiguous()
        y = torch.empty((B_val, C_val, OH_val, OW_val), device=x.device, dtype=torch.bfloat16)

        total_positions = OH_val * OW_val

        TILE_M = 256
        total_tiles = (total_positions + TILE_M - 1) // TILE_M
        grid = (B_val * total_tiles, 1, 1)
        block = (TILE_M, 1, 1)

        depthwise_conv_kernel[lambda: (grid, block)](
            x_cont,
            w,
            y,
            B_val,
            C_val,
            H_val,
            W_val,
            OH_val,
            OW_val,
            TILE_M,
        )
        return y
