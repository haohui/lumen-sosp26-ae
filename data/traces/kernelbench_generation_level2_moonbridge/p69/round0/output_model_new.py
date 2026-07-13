import torch
import torch.nn as nn
import avelang
import avelang.language as al

# ── Problem dimensions ────────────────────────────────────────────────────────
IN_CHANNELS = 8
OUT_CHANNELS = 64
HEIGHT = 128
WIDTH = 128
KERNEL_SIZE = 3
OUT_HEIGHT = HEIGHT - KERNEL_SIZE + 1   # 126
OUT_WIDTH = WIDTH - KERNEL_SIZE + 1     # 126

# ── Tiling ────────────────────────────────────────────────────────────────────
TILE_H = 14
TILE_W = 14
IN_TILE_H = TILE_H + KERNEL_SIZE - 1  # 16
IN_TILE_W = TILE_W + KERNEL_SIZE - 1  # 16
BLOCK_SIZE = 256
IN_ELEMENTS = IN_TILE_H * IN_TILE_W * IN_CHANNELS   # 4096
W_ELEMENTS = OUT_CHANNELS * IN_CHANNELS * KERNEL_SIZE * KERNEL_SIZE  # 4608
TILE_OUT_ELEMS = TILE_H * TILE_W * OUT_CHANNELS  # 12544


@avelang.jit
def conv2d_hardswish_relu_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    OC: al.i32,
    H: al.i32,
    W: al.i32,
    OH: al.i32,
    OW: al.i32,
):
    tid = al.thread_id(0)
    block_oh = al.block_id(0)
    block_ow = al.block_id(1)
    batch_idx = al.block_id(2)

    oh_base = block_oh * TILE_H
    ow_base = block_ow * TILE_W

    # ── Shared memory for input tile ──────────────────────────────────────────
    shm_in = al.make_shared((IN_TILE_H, IN_TILE_W, IN_CHANNELS), al.bf16)
    layout_in = al.make_layout((B, IN_CHANNELS, H, W), (IN_CHANNELS * H * W, H * W, W, 1))
    x = al.make_tensor(x_ptr, al.bf16, layout_in)

    # Cooperative load: 4096 / 256 = 16 rounds
    idx = tid
    for _ in al.range((IN_ELEMENTS + BLOCK_SIZE - 1) // BLOCK_SIZE):
        if idx < IN_ELEMENTS:
            ih = idx // (IN_TILE_W * IN_CHANNELS)
            rem = idx - ih * IN_TILE_W * IN_CHANNELS
            iw = rem // IN_CHANNELS
            ic_idx = rem - iw * IN_CHANNELS
            shm_in[ih, iw, ic_idx] = x[batch_idx, ic_idx, oh_base + ih, ow_base + iw]
        idx = idx + BLOCK_SIZE
    al.syncthreads()

    # ── Shared memory for weight ─────────────────────────────────────────────
    shm_w = al.make_shared((OUT_CHANNELS, IN_CHANNELS, KERNEL_SIZE, KERNEL_SIZE), al.bf16)
    layout_w_flat = al.make_layout((W_ELEMENTS,), (1,))
    w_flat = al.make_tensor(w_ptr, al.bf16, layout_w_flat)

    # Cooperative load: 4608 / 256 = 18 rounds
    idx = tid
    for _ in al.range((W_ELEMENTS + BLOCK_SIZE - 1) // BLOCK_SIZE):
        if idx < W_ELEMENTS:
            oc_idx = idx // (IN_CHANNELS * KERNEL_SIZE * KERNEL_SIZE)
            rem = idx - oc_idx * IN_CHANNELS * KERNEL_SIZE * KERNEL_SIZE
            ic_idx = rem // (KERNEL_SIZE * KERNEL_SIZE)
            rem2 = rem - ic_idx * KERNEL_SIZE * KERNEL_SIZE
            kh_idx = rem2 // KERNEL_SIZE
            kw_idx = rem2 - kh_idx * KERNEL_SIZE
            shm_w[oc_idx, ic_idx, kh_idx, kw_idx] = w_flat[idx]
        idx = idx + BLOCK_SIZE
    al.syncthreads()

    # ── Compute output elements ──────────────────────────────────────────────
    zero_f32 = al.convert(0.0, al.f32)
    three_f32 = al.convert(3.0, al.f32)
    six_f32 = al.convert(6.0, al.f32)

    layout_out = al.make_layout((B, OC, OH, OW), (OC * OH * OW, OH * OW, OW, 1))
    g_out = al.make_tensor(out_ptr, al.bf16, layout_out)

    layout_b = al.make_layout((OC,), (1,))
    g_bias = al.make_tensor(bias_ptr, al.bf16, layout_b)

    elem_idx = tid
    for _ in al.range((TILE_OUT_ELEMS + BLOCK_SIZE - 1) // BLOCK_SIZE):
        if elem_idx < TILE_OUT_ELEMS:
            oc = elem_idx // (TILE_H * TILE_W)
            rem_sp = elem_idx - oc * TILE_H * TILE_W
            th = rem_sp // TILE_W
            tw = rem_sp - th * TILE_W

            acc = zero_f32
            ic = al.convert(0, al.i32)
            for _ic in al.range(IN_CHANNELS):
                kh = al.convert(0, al.i32)
                for _kh in al.range(KERNEL_SIZE):
                    kw = al.convert(0, al.i32)
                    for _kw in al.range(KERNEL_SIZE):
                        in_val = al.convert(shm_in[th + kh, tw + kw, ic], al.f32)
                        w_val = al.convert(shm_w[oc, ic, kh, kw], al.f32)
                        acc = acc + in_val * w_val
                        kw = kw + 1
                    kh = kh + 1
                ic = ic + 1

            # Add bias
            acc = acc + al.convert(g_bias[oc], al.f32)

            # HardSwish: x * relu6(x + 3) / 6
            tmp = acc + three_f32
            if tmp < zero_f32:
                tmp = zero_f32
            if tmp > six_f32:
                tmp = six_f32
            hs = acc * tmp / six_f32
            # ReLU: max(0, hs)
            if hs < zero_f32:
                hs = zero_f32

            g_out[batch_idx, oc, oh_base + th, ow_base + tw] = al.convert(hs, al.bf16)

        elem_idx = elem_idx + BLOCK_SIZE


def _prepare_bf16_cuda_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_conv2d_fused(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    B: int,
    OC: int,
    OH: int,
    OW: int,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = _prepare_bf16_cuda_contiguous(x)
    w_bf16 = weight.contiguous().to(dtype=torch.bfloat16)
    b_bf16 = bias.contiguous().to(dtype=torch.bfloat16)

    out = torch.empty((B, OC, OH, OW), dtype=torch.bfloat16, device=x_bf16.device)

    grid_oh = OH // TILE_H
    grid_ow = OW // TILE_W
    grid = (grid_oh, grid_ow, B)
    conv2d_hardswish_relu_kernel[lambda: (grid, (BLOCK_SIZE, 1, 1))](
        x_bf16, w_bf16, b_bf16, out,
        B, OC, HEIGHT, WIDTH, OH, OW,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized Conv2d + HardSwish + ReLU using AveLang DSL.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        OH = x.shape[2] - KERNEL_SIZE + 1
        OW = x.shape[3] - KERNEL_SIZE + 1
        result = avelang_conv2d_fused(
            x,
            self.conv.weight.data,
            self.conv.bias.data,
            B, OUT_CHANNELS, OH, OW,
        )
        return result


def get_inputs():
    return [torch.rand(128, IN_CHANNELS, HEIGHT, WIDTH)]


def get_init_inputs():
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE]
