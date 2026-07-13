import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE = 256

W_TOTAL_MAX = 32 * 32 * 3 * 3 * 3
D_TILE = 8
H_TILE = 8
W_TILE = 32


@avelang.jit
def convt3d_kernel_v8(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C_in: al.i32,
    C_out: al.i32,
    D_in: al.i32,
    H_in: al.i32,
    W_in: al.i32,
    D_out: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    K: al.i32,
    stride: al.i32,
    padding: al.i32,
    groups: al.i32,
    d_tiles: al.i32,
    h_tiles: al.i32,
    w_tiles: al.i32,
    has_bias: al.i32,
    w_total: al.i32,
):
    tid = al.thread_id(0)
    bid_batch = al.block_id(0)
    bid_spatial = al.block_id(1)

    C_out_g = C_out // groups
    C_in_g = C_in // groups

    batch = bid_batch

    d_tile_idx = bid_spatial // (h_tiles * w_tiles)
    rem = bid_spatial - d_tile_idx * h_tiles * w_tiles
    h_tile_idx = rem // w_tiles
    w_tile_idx = rem - h_tile_idx * w_tiles

    d_start = d_tile_idx * D_TILE
    h_start = h_tile_idx * H_TILE
    w_start = w_tile_idx * W_TILE
    d_end = al.min(d_start + D_TILE, D_out)
    h_end = al.min(h_start + H_TILE, H_out)
    w_end = al.min(w_start + W_TILE, W_out)

    w_stride_cout = K * K * K
    w_stride_cin = C_out_g * w_stride_cout

    w_shm = al.make_shared((W_TOTAL_MAX,), al.bf16)
    w_flat = al.make_tensor(w_ptr, al.bf16, al.make_layout((w_total,), (1,)))
    for i in al.range(tid, w_total, BLOCK_SIZE):
        w_shm[i] = w_flat[i]
    al.syncthreads()

    x_stride_dh = H_in * W_in
    x_stride_cdh = D_in * x_stride_dh
    x_stride_ncdh = C_in * x_stride_cdh

    out_stride_cdh = D_out * H_out * W_out
    out_stride_dh = H_out * W_out

    x_total = N * C_in * D_in * H_in * W_in
    out_total = N * C_out * D_out * H_out * W_out
    x_flat = al.make_tensor(x_ptr, al.bf16, al.make_layout((x_total,), (1,)))
    out_flat = al.make_tensor(out_ptr, al.bf16, al.make_layout((out_total,), (1,)))

    zero_i32 = al.convert(0, al.i32)
    zero_f32 = al.convert(0.0, al.f32)

    num_d = d_end - d_start
    num_h = h_end - h_start
    num_w = w_end - w_start

    total_work = num_d * num_h * num_w * C_out_g

    for elem_idx in al.range(tid, total_work, BLOCK_SIZE):
        r = elem_idx
        c_off = r % C_out_g
        r = r // C_out_g
        wi = r % num_w
        r = r // num_w
        hi = r % num_h
        di = r // num_h

        d = d_start + di
        h = h_start + hi
        w = w_start + wi
        c_out = c_off

        w_cout_base = c_off * w_stride_cout
        out_idx = batch * C_out * out_stride_cdh + c_out * out_stride_cdh + d * out_stride_dh + h * W_out + w

        acc = zero_f32

        for c_in in al.range(C_in_g):
            x_c_base = batch * x_stride_ncdh + c_in * x_stride_cdh
            w_c_base = c_in * w_stride_cin + w_cout_base

            for kd in al.range(K):
                d_shifted = d + padding - kd
                d_in = d_shifted // stride
                d_rem = d_shifted - d_in * stride
                if (d_in >= zero_i32) and (d_in < D_in) and (d_rem == zero_i32):
                    x_d_base = x_c_base + d_in * x_stride_dh
                    for kh in al.range(K):
                        h_shifted = h + padding - kh
                        h_in = h_shifted // stride
                        h_rem = h_shifted - h_in * stride
                        if (h_in >= zero_i32) and (h_in < H_in) and (h_rem == zero_i32):
                            x_h_base = x_d_base + h_in * W_in
                            for kw in al.range(K):
                                w_shifted = w + padding - kw
                                w_in = w_shifted // stride
                                w_rem = w_shifted - w_in * stride
                                if (w_in >= zero_i32) and (w_in < W_in) and (w_rem == zero_i32):
                                    x_val = al.convert(x_flat[x_h_base + w_in], al.f32)
                                    w_idx = w_c_base + kd * K * K + kh * K + kw
                                    w_val = al.convert(w_shm[w_idx], al.f32)
                                    acc = acc + x_val * w_val

        if has_bias != zero_i32:
            b_flat = al.make_tensor(b_ptr, al.bf16, al.make_layout((C_out,), (1,)))
            acc = acc + al.convert(b_flat[c_out], al.f32)

        out_flat[out_idx] = al.convert(acc, al.bf16)


def _to_bf16_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_conv_transpose3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: int,
    padding: int,
    groups: int,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = _to_bf16_contiguous(x)
    w_bf16 = _to_bf16_contiguous(weight)

    N, C_in, D_in, H_in, W_in = x_bf16.shape
    w_C_in, C_out_g, K, _, _ = w_bf16.shape
    C_out = C_out_g * groups

    D_out = (D_in - 1) * stride - 2 * padding + K
    H_out = (H_in - 1) * stride - 2 * padding + K
    W_out = (W_in - 1) * stride - 2 * padding + K

    out = torch.empty((N, C_out, D_out, H_out, W_out), device=x_bf16.device, dtype=torch.bfloat16)

    has_bias_int = 1 if bias is not None else 0
    if bias is not None:
        b_bf16 = _to_bf16_contiguous(bias)
    else:
        b_bf16 = torch.empty(1, device=x_bf16.device, dtype=torch.bfloat16)

    w_total = w_bf16.numel()
    d_tiles = (D_out + D_TILE - 1) // D_TILE
    h_tiles = (H_out + H_TILE - 1) // H_TILE
    w_tiles = (W_out + W_TILE - 1) // W_TILE
    grid = (N, d_tiles * h_tiles * w_tiles, 1)

    convt3d_kernel_v8[lambda: (grid, (BLOCK_SIZE, 1, 1))](
        x_bf16, w_bf16, b_bf16, out,
        N, C_in, C_out,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        K, stride, padding, groups,
        d_tiles, h_tiles, w_tiles, has_bias_int,
        w_total,
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
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.conv = nn.ConvTranspose3d(
            in_channels, out_channels,
            kernel_size=(kernel_size, kernel_size, kernel_size),
            stride=stride, padding=padding,
            groups=groups, bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_conv_transpose3d(
            x, self.conv.weight, self.conv.bias,
            self.conv.stride[0], self.conv.padding[0], self.conv.groups,
        )


batch_size = 4
in_channels = 32
out_channels = 32
kernel_size = 3
depth = 32
height = 64
width = 128
stride = 2
padding = 1
groups = 4


def get_inputs():
    x = torch.rand(batch_size, in_channels, depth, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, groups]
