import torch
import torch.nn as nn
import avelang
import avelang.language as al

TILE_H = 16
TILE_W = 16
THREADS = TILE_H * TILE_W
TILE_OC = 16
TILE_IC = 16
KERNEL_H = 3
KERNEL_W = 3
INPUT_WINDOW_H = TILE_H + KERNEL_H - 1
INPUT_WINDOW_W = TILE_W + KERNEL_W - 1
ELEMENTS_TO_LOAD = INPUT_WINDOW_H * INPUT_WINDOW_W
SHM_LOAD_SIZE = TILE_IC * ELEMENTS_TO_LOAD


@avelang.jit
def conv_fused_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    mult_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    C_in: al.i32,
    C_out: al.i32,
    H: al.i32,
    W: al.i32,
    H_out: al.i32,
    W_out: al.i32,
):
    tid = al.thread_id(0)
    block_w = al.block_id(0)
    block_h = al.block_id(1)
    block_b_ocg = al.block_id(2)

    oc_groups = (C_out + TILE_OC - 1) // TILE_OC
    b = block_b_ocg // oc_groups
    oc_group = block_b_ocg - b * oc_groups
    oc_base = oc_group * TILE_OC

    thread_h = tid // TILE_W
    thread_w = tid - thread_h * TILE_W

    h_out = block_h * TILE_H + thread_h
    w_out = block_w * TILE_W + thread_w

    valid_spatial = (h_out < H_out) and (w_out < W_out)

    layout_x = al.make_layout((B, C_in, H, W), (C_in * H * W, H * W, W, 1))
    x = al.make_tensor(x_ptr, al.bf16, layout_x)

    layout_w = al.make_layout(
        (C_out, C_in, KERNEL_H, KERNEL_W),
        (C_in * KERNEL_H * KERNEL_W, KERNEL_H * KERNEL_W, KERNEL_W, 1),
    )
    w = al.make_tensor(w_ptr, al.bf16, layout_w)

    layout_bias = al.make_layout((C_out,), (1,))
    bias = al.make_tensor(bias_ptr, al.bf16, layout_bias)

    layout_mult = al.make_layout((C_out, 1, 1), (1, 1, 1))
    mult = al.make_tensor(mult_ptr, al.bf16, layout_mult)

    layout_out = al.make_layout(
        (B, C_out, H_out, W_out),
        (C_out * H_out * W_out, H_out * W_out, W_out, 1),
    )
    out = al.make_tensor(out_ptr, al.bf16, layout_out)

    shm_in = al.make_shared((TILE_IC, INPUT_WINDOW_H, INPUT_WINDOW_W), al.bf16)

    acc = al.make_local((TILE_OC,), al.f32)
    for i in al.range(TILE_OC):
        oc = oc_base + i
        if valid_spatial and (oc < C_out):
            acc[i] = al.convert(bias[oc], al.f32)
        else:
            acc[i] = al.convert(0.0, al.f32)

    zero_bf16 = al.convert(0.0, al.bf16)

    for ic_base in al.range(0, C_in, TILE_IC):
        for idx in al.range(tid, SHM_LOAD_SIZE, THREADS):
            ic_tile = idx // ELEMENTS_TO_LOAD
            spatial_idx = idx - ic_tile * ELEMENTS_TO_LOAD
            lh = spatial_idx // INPUT_WINDOW_W
            lw = spatial_idx - lh * INPUT_WINDOW_W
            ic = ic_base + ic_tile
            ih = block_h * TILE_H + lh
            iw = block_w * TILE_W + lw
            if (ic < C_in) and (ih < H) and (iw < W):
                shm_in[ic_tile, lh, lw] = x[b, ic, ih, iw]
            else:
                shm_in[ic_tile, lh, lw] = zero_bf16

        al.syncthreads()

        if valid_spatial:
            for ic_tile in al.range(TILE_IC):
                ic = ic_base + ic_tile
                if ic < C_in:
                    for kh in al.range(KERNEL_H):
                        for kw in al.range(KERNEL_W):
                            input_val = al.convert(
                                shm_in[ic_tile, thread_h + kh, thread_w + kw], al.f32
                            )
                            for i in al.range(TILE_OC):
                                oc = oc_base + i
                                if oc < C_out:
                                    weight_val = al.convert(
                                        w[oc, ic, kh, kw], al.f32
                                    )
                                    acc[i] = acc[i] + input_val * weight_val

        al.syncthreads()

    if valid_spatial:
        for i in al.range(TILE_OC):
            oc = oc_base + i
            if oc < C_out:
                val = acc[i]
                mult_val = al.convert(mult[oc, 0, 0], al.f32)
                val = val * mult_val

                negative_slope = al.convert(0.01, al.f32)
                zero_f32 = al.convert(0.0, al.f32)
                if val < zero_f32:
                    val = val * negative_slope

                half = al.convert(0.5, al.f32)
                one = al.convert(1.0, al.f32)
                sqrt_2_over_pi = al.convert(0.7978845608028654, al.f32)
                coeff = al.convert(0.044715, al.f32)

                x3 = val * val * val
                inner = sqrt_2_over_pi * (val + coeff * x3)
                tanh_val = al.tanh(inner)
                gelu_val = half * val * (one + tanh_val)

                out[b, oc, h_out, w_out] = al.convert(gelu_val, al.bf16)


def _prepare_bf16_cuda_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_conv_fused(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    multiplier: torch.Tensor,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = _prepare_bf16_cuda_contiguous(x)
    w_bf16 = _prepare_bf16_cuda_contiguous(weight)
    bias_bf16 = _prepare_bf16_cuda_contiguous(bias)
    mult_bf16 = _prepare_bf16_cuda_contiguous(multiplier)

    B, C_in, H, W = x_bf16.shape
    C_out, C_in_w, K_h, K_w = w_bf16.shape
    if C_in_w != C_in or K_h != KERNEL_H or K_w != KERNEL_W:
        raise ValueError(
            f"Weight shape mismatch: expected ({C_out}, {C_in}, {KERNEL_H}, {KERNEL_W}), "
            f"got {w_bf16.shape}"
        )

    H_out = H - KERNEL_H + 1
    W_out = W - KERNEL_W + 1

    out = torch.empty(
        (B, C_out, H_out, W_out),
        device=x_bf16.device,
        dtype=torch.bfloat16,
    )

    W_tiles = (W_out + TILE_W - 1) // TILE_W
    H_tiles = (H_out + TILE_H - 1) // TILE_H
    OC_groups = (C_out + TILE_OC - 1) // TILE_OC
    grid = (W_tiles, H_tiles, B * OC_groups)

    conv_fused_kernel[lambda: (grid, (THREADS, 1, 1))](
        x_bf16,
        w_bf16,
        bias_bf16,
        mult_bf16,
        out,
        B,
        C_in,
        C_out,
        H,
        W,
        H_out,
        W_out,
    )
    return out


class ModelNew(nn.Module):
    """
    Model that performs a convolution, multiplies by a learnable scalar,
    applies LeakyReLU, and then GELU -- optimized with AveLang fused kernel.
    """

    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        return avelang_conv_fused(
            x,
            self.conv.weight,
            self.conv.bias,
            self.multiplier,
        )


batch_size = 64
in_channels = 64
out_channels = 64
height, width = 256, 256
kernel_size = 3
multiplier_shape = (out_channels, 1, 1)


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, multiplier_shape]
