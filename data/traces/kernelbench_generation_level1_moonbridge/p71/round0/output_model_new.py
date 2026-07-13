import torch
import torch.nn as nn
import avelang
import avelang.language as al

TILE_H = 16
TILE_W = 16
THREADS = TILE_H * TILE_W  # 256
K_SIZE = 3
C_IN = 32
C_OUT = 32
WEIGHT_TOTAL = C_IN * C_OUT * K_SIZE * K_SIZE  # 9216
kk_co = K_SIZE * K_SIZE * C_OUT  # 288
k_co = K_SIZE * C_OUT  # 96


@avelang.jit
def conv_transpose2d_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    _in_channels: al.i32,
    _out_channels: al.i32,
    _height_in: al.i32,
    _width_in: al.i32,
    _height_out: al.i32,
    _width_out: al.i32,
):
    tid = al.thread_id(0)
    block_w = al.block_id(0)
    block_h = al.block_id(1)
    batch_idx = al.block_id(2)

    oh0 = block_h * TILE_H
    ow0 = block_w * TILE_W
    lh = tid // TILE_W
    lw = tid - lh * TILE_W
    oh = oh0 + lh
    ow = ow0 + lw

    valid_out = (oh < _height_out) and (ow < _width_out)

    # Load ALL weight into shared memory once.
    # Layout in shared memory: (ic, kh, kw, oc) — oc fastest-varying for stride-1 access.
    smem_weight = al.make_shared((WEIGHT_TOTAL,), al.bf16)
    w_global = al.make_tensor(
        weight_ptr, al.bf16,
        al.make_layout(
            (C_IN, C_OUT, K_SIZE, K_SIZE),
            (C_OUT * K_SIZE * K_SIZE, K_SIZE * K_SIZE, K_SIZE, 1),
        ),
    )
    for wi in al.range(tid, WEIGHT_TOTAL, THREADS):
        w_ic = wi // kk_co
        w_rest = wi - w_ic * kk_co
        w_kh = w_rest // k_co
        w_rest2 = w_rest - w_kh * k_co
        w_kw = w_rest2 // C_OUT
        w_oc = w_rest2 - w_kw * C_OUT
        smem_weight[wi] = w_global[w_ic, w_oc, w_kh, w_kw]

    al.syncthreads()

    if not valid_out:
        return

    # Global tensor references
    in_tensor = al.make_tensor(
        input_ptr, al.bf16,
        al.make_layout(
            (8, C_IN, _height_in, _width_in),
            (C_IN * _height_in * _width_in, _height_in * _width_in, _width_in, 1),
        ),
    )
    out_tensor = al.make_tensor(
        output_ptr, al.bf16,
        al.make_layout(
            (8, C_OUT, _height_out, _width_out),
            (C_OUT * _height_out * _width_out, _height_out * _width_out, _width_out, 1),
        ),
    )

    zero_i32 = al.convert(0, al.i32)
    zero_f32 = al.convert(0.0, al.f32)

    # 32 FP32 accumulators, one per output channel
    acc = al.make_local((C_OUT,), al.f32)
    for o in al.range(C_OUT):
        acc[o] = zero_f32

    # Iterate over input channels, reading each input element once and
    # accumulating across all 32 output channels using stride-1 weight access
    for ic in al.range(C_IN):
        ic_w_base = ic * kk_co

        for kh in al.range(K_SIZE):
            ih = oh - kh
            ih_valid = (ih >= zero_i32) and (ih < _height_in)

            if ih_valid:
                for kw in al.range(K_SIZE):
                    iw = ow - kw
                    iw_valid = (iw >= zero_i32) and (iw < _width_in)

                    if iw_valid:
                        in_val = al.convert(
                            in_tensor[batch_idx, ic, ih, iw], al.f32
                        )
                        w_base = ic_w_base + kh * k_co + kw * C_OUT

                        for oc in al.range(C_OUT):
                            w_val = al.convert(smem_weight[w_base + oc], al.f32)
                            acc[oc] = acc[oc] + in_val * w_val

    for oc in al.range(C_OUT):
        out_tensor[batch_idx, oc, oh, ow] = al.convert(acc[oc], al.bf16)


def avelang_conv_transpose2d(
    x: torch.Tensor, weight: torch.Tensor,
) -> torch.Tensor:
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA/HIP device."

    x_bf16 = x.contiguous().to(torch.bfloat16)
    w_bf16 = weight.contiguous().to(torch.bfloat16)

    N, C_in, H_in, W_in = x_bf16.shape
    C_in_w, C_out, KH, KW = w_bf16.shape

    assert C_in == C_in_w, f"Input channels mismatch: {C_in} vs {C_in_w}"
    assert KH == K_SIZE and KW == K_SIZE, f"Kernel size must be {K_SIZE}"

    H_out = (H_in - 1) * 1 - 2 * 0 + K_SIZE + 0
    W_out = (W_in - 1) * 1 - 2 * 0 + K_SIZE + 0

    out = torch.empty(
        (N, C_out, H_out, W_out), device=x_bf16.device, dtype=torch.bfloat16
    )

    num_h_tiles = (H_out + TILE_H - 1) // TILE_H
    num_w_tiles = (W_out + TILE_W - 1) // TILE_W

    grid = (num_w_tiles, num_h_tiles, N)

    conv_transpose2d_kernel[lambda: (grid, (THREADS, 1, 1))](
        x_bf16, w_bf16, out,
        C_in, C_out, H_in, W_in, H_out, W_out,
    )

    return out


class ModelNew(nn.Module):
    """
    Performs a transposed 2D convolution. Optimized with AveLang BF16 kernel.
    """

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
        self.conv_transpose2d = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding,
            output_padding=output_padding, groups=groups, bias=bias,
        )
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self._bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.conv_transpose2d.weight.data
        bias = self.conv_transpose2d.bias

        if (
            self.groups != 1
            or bias is not None
            or self.stride != 1
            or self.padding != 0
            or self.output_padding != 0
        ):
            return self.conv_transpose2d(x)

        result_bf16 = avelang_conv_transpose2d(x, weight)
        return result_bf16.to(x.dtype)


def get_inputs():
    x = torch.rand(8, 32, 512, 1024)
    return [x]


def get_init_inputs():
    return [32, 32, 3]
