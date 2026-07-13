import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE = 256
KH = 3
KW = 3
TILE_H = 16
TILE_W = 16
TILE_IN_H = TILE_H + KH - 1
TILE_IN_W = TILE_W + KW - 1
TILE_SIZE = TILE_IN_H * TILE_IN_W


@avelang.jit
def depthwise_conv2d_tiled_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    C: al.i32,
    H: al.i32,
    W: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    stride: al.i32,
):
    tid = al.thread_id(0)
    block_w = al.block_id(0)
    block_h = al.block_id(1)
    block_bc = al.block_id(2)

    b_idx = block_bc // C
    c_idx = block_bc - b_idx * C

    ty = tid // TILE_W
    tx = tid - ty * TILE_W

    h_out = block_h * TILE_H + ty
    w_out = block_w * TILE_W + tx

    # --- load input tile into shared memory ---
    shm_in = al.make_shared((TILE_SIZE,), al.bf16)

    h_tile_start = block_h * TILE_H * stride
    w_tile_start = block_w * TILE_W * stride

    x = al.make_tensor(
        x_ptr, al.bf16,
        al.make_layout((B, C, H, W), (C * H * W, H * W, W, al.convert(1, al.i32))),
    )

    for i in al.range(tid, TILE_SIZE, BLOCK_SIZE):
        r = i // TILE_IN_W
        c = i - r * TILE_IN_W
        h_in = h_tile_start + r
        w_in = w_tile_start + c
        if h_in >= 0 and h_in < H and w_in >= 0 and w_in < W:
            shm_in[i] = x[b_idx, c_idx, h_in, w_in]
        else:
            shm_in[i] = al.convert(0.0, al.bf16)

    al.syncthreads()

    # --- compute convolution from shared memory ---
    if h_out < H_out and w_out < W_out:
        w_t = al.make_tensor(
            w_ptr, al.bf16,
            al.make_layout((C, KH, KW), (KH * KW, KW, al.convert(1, al.i32))),
        )

        acc = al.convert(0.0, al.f32)
        for kh in al.range(KH):
            shm_row = ty * stride + kh
            base = shm_row * TILE_IN_W + tx * stride
            for kw in al.range(KW):
                x_val = al.convert(shm_in[base + kw], al.f32)
                w_val = al.convert(w_t[c_idx, kh, kw], al.f32)
                acc = acc + x_val * w_val

        out = al.make_tensor(
            out_ptr, al.bf16,
            al.make_layout((B, C, H_out, W_out), (C * H_out * W_out, H_out * W_out, W_out, al.convert(1, al.i32))),
        )
        out[b_idx, c_idx, h_out, w_out] = al.convert(acc, al.bf16)


@avelang.jit
def depthwise_conv2d_simple_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    C: al.i32,
    H: al.i32,
    W: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    stride: al.i32,
    padding: al.i32,
    has_bias: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    idx = bid * BLOCK_SIZE + tid
    total = B * C * H_out * W_out

    if idx < total:
        wh = W_out * H_out
        bc = idx // wh
        b_idx = bc // C
        c_idx = bc - b_idx * C
        hw = idx - bc * wh
        h_out = hw // W_out
        w_out = hw - h_out * W_out

        x = al.make_tensor(
            x_ptr, al.bf16,
            al.make_layout((B, C, H, W), (C * H * W, H * W, W, al.convert(1, al.i32))),
        )
        w = al.make_tensor(
            w_ptr, al.bf16,
            al.make_layout((C, KH, KW), (KH * KW, KW, al.convert(1, al.i32))),
        )
        out = al.make_tensor(
            out_ptr, al.bf16,
            al.make_layout((B, C, H_out, W_out), (C * H_out * W_out, H_out * W_out, W_out, al.convert(1, al.i32))),
        )

        acc = al.convert(0.0, al.f32)
        for kh in al.range(KH):
            h_in = h_out * stride + kh - padding
            if h_in >= 0 and h_in < H:
                for kw in al.range(KW):
                    w_in = w_out * stride + kw - padding
                    if w_in >= 0 and w_in < W:
                        x_val = al.convert(x[b_idx, c_idx, h_in, w_in], al.f32)
                        w_val = al.convert(w[c_idx, kh, kw], al.f32)
                        acc = acc + x_val * w_val

        if has_bias != 0:
            b_tensor = al.make_tensor(
                b_ptr, al.bf16,
                al.make_layout((C,), (al.convert(1, al.i32),)),
            )
            b_val = al.convert(b_tensor[c_idx], al.f32)
            acc = acc + b_val

        out[b_idx, c_idx, h_out, w_out] = al.convert(acc, al.bf16)


def avelang_depthwise_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: int,
    padding: int,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = x.contiguous().to(dtype=torch.bfloat16)
    w_bf16 = weight.contiguous().to(dtype=torch.bfloat16)

    B, C_in, H, W = x_bf16.shape
    C_out = w_bf16.shape[0]
    KH_val = w_bf16.shape[2]
    KW_val = w_bf16.shape[3]
    # weight in nn.Conv2d with groups is (out_channels, in_channels//groups, kH, kW)
    # when groups=in_channels, it's (out_channels, 1, kH, kW)
    # we squeeze out the singleton dimension for the kernel
    w_bf16 = w_bf16.reshape(C_out, KH_val, KW_val).contiguous()

    H_out = (H + 2 * padding - KH_val) // stride + 1
    W_out = (W + 2 * padding - KW_val) // stride + 1

    C = C_out

    out = torch.empty((B, C, H_out, W_out), device=x_bf16.device, dtype=torch.bfloat16)

    if bias is not None:
        b_bf16 = bias.contiguous().to(dtype=torch.bfloat16)
        has_bias_flag = 1
    else:
        b_bf16 = torch.empty((1,), device=x_bf16.device, dtype=torch.bfloat16)
        has_bias_flag = 0

    # Use tiled kernel with shared memory when stride==1 and padding==0
    if stride == 1 and padding == 0:
        num_blocks_w = (W_out + TILE_W - 1) // TILE_W
        num_blocks_h = (H_out + TILE_H - 1) // TILE_H
        num_blocks_bc = B * C

        depthwise_conv2d_tiled_kernel[lambda: ((num_blocks_w, num_blocks_h, num_blocks_bc), (BLOCK_SIZE, 1, 1))](
            x_bf16, w_bf16, out, B, C, H, W, H_out, W_out, stride,
        )
    else:
        total = B * C * H_out * W_out
        num_blocks = (total + BLOCK_SIZE - 1) // BLOCK_SIZE
        depthwise_conv2d_simple_kernel[lambda: ((num_blocks, 1, 1), (BLOCK_SIZE, 1, 1))](
            x_bf16, w_bf16, b_bf16, out,
            B, C, H, W, H_out, W_out, stride, padding, has_bias_flag,
        )
    return out


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._stride = stride
        self._padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.conv2d.weight.data
        bias = self.conv2d.bias.data if self.conv2d.bias is not None else None
        return avelang_depthwise_conv2d(
            x, weight, bias, self._stride, self._padding
        )


def get_inputs():
    x = torch.rand(64, 128, 256, 512)
    return [x]


def get_init_inputs():
    return [128, 128, 3, 1, 0]
