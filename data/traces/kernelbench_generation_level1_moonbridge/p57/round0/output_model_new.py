import torch
import torch.nn as nn
import avelang
import avelang.language as al

# Test-code constants
BATCH_SIZE = 8
IN_CHANNELS = 64
OUT_CHANNELS = 64
KERNEL_SIZE = 3
HEIGHT = 1024
WIDTH = 1024
STRIDE = 1
PADDING = 0

H_OUT = (HEIGHT - 1) * STRIDE - 2 * PADDING + KERNEL_SIZE  # 1026
W_OUT = (WIDTH - 1) * STRIDE - 2 * PADDING + KERNEL_SIZE    # 1026

BLOCK_H = 16
BLOCK_W = 16


@avelang.jit
def conv_transpose2d_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
):
    tid_h = al.thread_id(0)
    tid_w = al.thread_id(1)
    block_h = al.block_id(0)
    block_w = al.block_id(1)
    batch = al.block_id(2)

    oh = block_h * al.convert(16, al.i32) + tid_h
    ow = block_w * al.convert(16, al.i32) + tid_w

    total_x = al.convert(BATCH_SIZE * IN_CHANNELS * HEIGHT * WIDTH, al.i32)
    total_w = al.convert(IN_CHANNELS * OUT_CHANNELS * KERNEL_SIZE * KERNEL_SIZE, al.i32)
    total_o = al.convert(BATCH_SIZE * OUT_CHANNELS * H_OUT * W_OUT, al.i32)

    x = al.make_tensor(x_ptr, al.bf16, al.make_layout((total_x,), (1,)))
    w = al.make_tensor(w_ptr, al.bf16, al.make_layout((total_w,), (1,)))
    out = al.make_tensor(out_ptr, al.bf16, al.make_layout((total_o,), (1,)))

    hout_v = al.convert(H_OUT, al.i32)
    wout_v = al.convert(W_OUT, al.i32)

    # Allocate shared memory and local accumulators unconditionally
    shm_w = al.make_shared((64, 64), al.bf16)
    acc = al.make_local((64,), al.f32)
    zero_f32 = al.convert(0.0, al.f32)
    for _i in al.range(64):
        acc[_i] = zero_f32

    tid_flat = tid_h * al.convert(16, al.i32) + tid_w
    zero = al.convert(0, al.i32)
    H_val = al.convert(HEIGHT, al.i32)
    W_val = al.convert(WIDTH, al.i32)
    ic_limit = al.convert(64, al.i32)

    x_bs = al.convert(IN_CHANNELS * HEIGHT * WIDTH, al.i32)
    x_is = al.convert(HEIGHT * WIDTH, al.i32)
    w_is = al.convert(OUT_CHANNELS * KERNEL_SIZE * KERNEL_SIZE, al.i32)
    w_os = al.convert(KERNEL_SIZE * KERNEL_SIZE, al.i32)
    w_ks = al.convert(KERNEL_SIZE, al.i32)
    o_bs = al.convert(OUT_CHANNELS * H_OUT * W_OUT, al.i32)
    o_os = al.convert(H_OUT * W_OUT, al.i32)
    o_hs = al.convert(W_OUT, al.i32)

    # All threads participate in the main loop including weight loads and barriers.
    # Only in-bounds threads execute the compute and writeback.
    in_bounds = al.convert(0, al.i32)
    if oh < hout_v:
        if ow < wout_v:
            in_bounds = al.convert(1, al.i32)

    for kh in al.range(3):
        for kw in al.range(3):
            # Cooperative load: threads 0-63 each load one IC row
            if tid_flat < ic_limit:
                _ic = tid_flat
                for _oc in al.range(64):
                    w_off = _ic * w_is + _oc * w_os + kh * w_ks + kw
                    shm_w[_ic, _oc] = w[w_off]
            al.syncthreads()

            if in_bounds != 0:
                h_in = oh - kh
                w_in = ow - kw

                if h_in >= zero:
                    if h_in < H_val:
                        if w_in >= zero:
                            if w_in < W_val:
                                x_base = batch * x_bs + h_in * W_val + w_in
                                for ic in al.range(64):
                                    x_off = x_base + ic * x_is
                                    val = al.convert(x[x_off], al.f32)
                                    for oc in al.range(64):
                                        w_val = al.convert(shm_w[ic, oc], al.f32)
                                        acc[oc] = acc[oc] + val * w_val

            al.syncthreads()

    # Writeback only for in-bounds threads
    if in_bounds != 0:
        o_base = batch * o_bs + oh * o_hs + ow
        for _oc_out in al.range(64):
            o_off = o_base + _oc_out * o_os
            out[o_off] = al.convert(acc[_oc_out], al.bf16)


def _to_bf16_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_conv_transpose2d(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = _to_bf16_contiguous(x)
    w_bf16 = _to_bf16_contiguous(weight)

    B, IC, H, W = x_bf16.shape
    w_IC, OC, KH, KW = w_bf16.shape

    if w_IC != IC:
        raise ValueError(f"Weight IC mismatch: {w_IC} vs {IC}")

    H_out_val = (H - 1) * STRIDE - 2 * PADDING + KH
    W_out_val = (W - 1) * STRIDE - 2 * PADDING + KW

    out = torch.empty((B, OC, H_out_val, W_out_val), device=x_bf16.device, dtype=torch.bfloat16)

    grid_h = (H_out_val + BLOCK_H - 1) // BLOCK_H
    grid_w = (W_out_val + BLOCK_W - 1) // BLOCK_W
    grid = (grid_h, grid_w, B)
    block = (BLOCK_H, BLOCK_W, 1)

    conv_transpose2d_kernel[lambda: (grid, block)](
        x_bf16,
        w_bf16,
        out,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_conv_transpose2d(x, self.conv_transpose2d.weight)


# Test code
batch_size = 8
in_channels = 64
out_channels = 64
kernel_size = 3
height = 1024
width = 1024

def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
