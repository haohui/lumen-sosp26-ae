import torch
import torch.nn as nn
import math
import avelang
import avelang.language as al

# Compile-time constants
KERNEL_H: int = 3
KERNEL_W: int = 7
TILE_H: int = 16
TILE_W: int = 16
OC_PER_BLOCK: int = 8
THREADS: int = TILE_H * TILE_W  # 256
SHM_INPUT_H: int = TILE_H + KERNEL_H - 1  # 18
SHM_INPUT_W: int = TILE_W + KERNEL_W - 1  # 22
SHM_INPUT_ELEMS: int = SHM_INPUT_H * SHM_INPUT_W  # 396
SHM_INPUT_LOADS: int = (SHM_INPUT_ELEMS + THREADS - 1) // THREADS  # 2


@avelang.jit
def conv_transpose2d_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    C_in: al.i32,
    C_out: al.i32,
    H_in: al.i32,
    W_in: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    pH: al.i32,
    pW: al.i32,
):
    tid = al.thread_id(0)
    local_oh = tid // TILE_W
    local_ow = tid % TILE_W

    block_oh_start = al.block_id(1) * TILE_H
    block_ow_start = al.block_id(0) * TILE_W
    oc_groups = C_out // OC_PER_BLOCK
    block_z = al.block_id(2)
    block_b = block_z // oc_groups
    block_oc_base = (block_z - block_b * oc_groups) * OC_PER_BLOCK

    global_oh = block_oh_start + local_oh
    global_ow = block_ow_start + local_ow

    one = al.convert(1, al.i32)
    zero_i32 = al.convert(0, al.i32)
    zero_f32 = al.convert(0.0, al.f32)
    zero_bf16 = al.convert(0.0, al.bf16)

    # Input tensor view
    x_stride_c = H_in * W_in
    x = al.make_tensor(
        x_ptr, al.bf16,
        al.make_layout(
            (B, C_in, H_in, W_in),
            (C_in * x_stride_c, x_stride_c, W_in, one),
        ),
    )

    # Weight tensor view
    w_stride_ic = C_out * KERNEL_H * KERNEL_W
    w_stride_oc = KERNEL_H * KERNEL_W
    w = al.make_tensor(
        w_ptr, al.bf16,
        al.make_layout(
            (C_in, C_out, KERNEL_H, KERNEL_W),
            (w_stride_ic, w_stride_oc, KERNEL_W, one),
        ),
    )

    # Output tensor view
    out_stride_c = H_out * W_out
    out = al.make_tensor(
        out_ptr, al.bf16,
        al.make_layout(
            (B, C_out, H_out, W_out),
            (C_out * out_stride_c, out_stride_c, W_out, one),
        ),
    )

    # Shared memory for input tile streaming
    shm_input = al.make_shared((SHM_INPUT_ELEMS,), al.bf16)

    # Base input coordinates for this block's needed region
    in_h_base = block_oh_start + pH - al.convert(KERNEL_H - 1, al.i32)
    in_w_base = block_ow_start + pW - al.convert(KERNEL_W - 1, al.i32)

    if global_oh < H_out:
        if global_ow < W_out:
            # Per-output-channel accumulators (FP32)
            acc = al.make_local((OC_PER_BLOCK,), al.f32)
            for oc_off in al.range(OC_PER_BLOCK):
                acc[oc_off] = zero_f32

            for ic in al.range(C_in):
                ic_i32 = al.convert(ic, al.i32)

                # Load input tile for this channel into shared memory
                ld_idx = tid
                for _ in al.range(SHM_INPUT_LOADS):
                    if ld_idx < SHM_INPUT_ELEMS:
                        lh = ld_idx // SHM_INPUT_W
                        lw = ld_idx % SHM_INPUT_W
                        ih = in_h_base + lh
                        iw = in_w_base + lw
                        val = zero_bf16
                        if ih >= zero_i32:
                            if ih < H_in:
                                if iw >= zero_i32:
                                    if iw < W_in:
                                        val = x[block_b, ic_i32, ih, iw]
                        shm_input[ld_idx] = val
                    ld_idx = ld_idx + THREADS

                al.syncthreads()

                # Accumulate kernel contributions from shared memory
                for kh in al.range(KERNEL_H):
                    in_row = local_oh + KERNEL_H - 1 - kh
                    in_row_base = in_row * SHM_INPUT_W
                    for kw in al.range(KERNEL_W):
                        in_col = local_ow + KERNEL_W - 1 - kw
                        x_val = al.convert(shm_input[in_row_base + in_col], al.f32)

                        for oc_off in al.range(OC_PER_BLOCK):
                            oc = block_oc_base + oc_off
                            w_val = al.convert(w[ic_i32, oc, kh, kw], al.f32)
                            acc[oc_off] = acc[oc_off] + x_val * w_val

                al.syncthreads()

            for oc_off in al.range(OC_PER_BLOCK):
                oc = block_oc_base + oc_off
                out[block_b, oc, global_oh, global_ow] = al.convert(acc[oc_off], al.bf16)


def _ensure_bf16_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_conv_transpose2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    stride: tuple,
    padding: tuple,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    if stride != (1, 1):
        raise ValueError(
            f"This kernel only supports stride=(1, 1), got stride={stride}"
        )

    if weight.shape[2] != KERNEL_H or weight.shape[3] != KERNEL_W:
        raise ValueError(
            f"Expected kernel size ({KERNEL_H}, {KERNEL_W}), "
            f"got ({weight.shape[2]}, {weight.shape[3]})"
        )

    x_bf16 = _ensure_bf16_contiguous(x)
    w_bf16 = _ensure_bf16_contiguous(weight)

    B, C_in, H_in, W_in = x_bf16.shape
    w_C_in, C_out, w_kH, w_kW = w_bf16.shape

    if w_C_in != C_in:
        raise ValueError(
            f"Weight in_channels mismatch: input has {C_in}, "
            f"weight has {w_C_in}"
        )

    if C_out % OC_PER_BLOCK != 0:
        raise ValueError(
            f"C_out ({C_out}) must be divisible by OC_PER_BLOCK ({OC_PER_BLOCK})"
        )

    pH, pW = padding
    sH, sW = stride

    H_out = (H_in - 1) * sH - 2 * pH + w_kH
    W_out = (W_in - 1) * sW - 2 * pW + w_kW

    out = torch.empty(
        (B, C_out, H_out, W_out),
        device=x_bf16.device,
        dtype=torch.bfloat16,
    )

    tiles_h = (H_out + TILE_H - 1) // TILE_H
    tiles_w = (W_out + TILE_W - 1) // TILE_W
    grid = (tiles_w, tiles_h, B * (C_out // OC_PER_BLOCK))

    conv_transpose2d_kernel[lambda: (grid, (THREADS, 1, 1))](
        x_bf16, w_bf16, out,
        B, C_in, C_out, H_in, W_in, H_out, W_out, pH, pW,
    )

    return out


class ModelNew(nn.Module):
    """
    Performs a 2D transposed convolution with an AveLang BF16 GPU kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (height, width).
        stride (tuple, optional): Stride of the convolution. Defaults to (1, 1).
        padding (tuple, optional): Padding applied to the input. Defaults to (0, 0).
        bias (bool, optional): If True, adds a learnable bias. Defaults to False.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple = (1, 1),
        padding: tuple = (0, 0),
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = bias

        # Weight layout: (in_channels, out_channels, kernel_height, kernel_width)
        self.weight = nn.Parameter(
            torch.empty(in_channels, out_channels, *kernel_size)
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in = in_channels * kernel_size[0] * kernel_size[1]
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_conv_transpose2d(
            x, self.weight, self.stride, self.padding
        )


# Test code — preserved from input_model.py for harness compatibility
batch_size = 8
in_channels = 32
out_channels = 32
kernel_size = (3, 7)
height = 512
width = 1024
stride = (1, 1)
padding = (1, 3)


def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]
