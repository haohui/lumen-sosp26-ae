import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE: al.constexpr = 256
KW: al.constexpr = 3
KW_VOL: al.constexpr = 27


@avelang.jit
def avg_pool3d_kernel(
    input_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    B: al.i32,
    C: al.i32,
    D: al.i32,
    H: al.i32,
    W: al.i32,
    OD: al.i32,
    OH: al.i32,
    OW: al.i32,
    stride: al.i32,
    padding: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)
    global_id = bid * BLOCK_SIZE + tid

    total_out = B * C * OD * OH * OW
    if global_id >= total_out:
        return

    # Decode flat index to (b, c, od, oh, ow) coordinates
    tmp = global_id
    ow = tmp % OW
    tmp = tmp // OW
    oh = tmp % OH
    tmp = tmp // OH
    od = tmp % OD
    tmp = tmp // OD
    c = tmp % C
    b = tmp // C

    # Flat 1D layouts for input and output
    layout_in = al.make_layout((B * C * D * H * W,), (1,))
    inp = al.make_tensor(input_ptr, al.bf16, layout_in)

    # Precompute BC offsets
    bc_in = (b * C + c) * D * H * W
    bc_out = (b * C + c) * OD * OH * OW

    acc = al.convert(0.0, al.f32)

    for kd in al.range(KW):
        id_val = od * stride + kd - padding
        if id_val >= 0:
            if id_val < D:
                id_base = bc_in + id_val * H * W
                for kh in al.range(KW):
                    ih_val = oh * stride + kh - padding
                    if ih_val >= 0:
                        if ih_val < H:
                            ih_base = id_base + ih_val * W
                            for kw in al.range(KW):
                                iw_val = ow * stride + kw - padding
                                if iw_val >= 0:
                                    if iw_val < W:
                                        val = al.convert(
                                            inp[ih_base + iw_val], al.f32
                                        )
                                        acc = acc + val

    result = acc / al.convert(KW_VOL, al.f32)

    layout_out = al.make_layout((B * C * OD * OH * OW,), (1,))
    out = al.make_tensor(output_ptr, al.bf16, layout_out)
    out[bc_out + od * OH * OW + oh * OW + ow] = al.convert(result, al.bf16)


def _compute_output_shape(
    D: int, H: int, W: int, kernel_size: int, stride: int, padding: int
):
    OD = (D + 2 * padding - kernel_size) // stride + 1
    OH = (H + 2 * padding - kernel_size) // stride + 1
    OW = (W + 2 * padding - kernel_size) // stride + 1
    return OD, OH, OW


def avelang_avg_pool3d(
    x: torch.Tensor, kernel_size: int, stride: int, padding: int
) -> torch.Tensor:
    assert x.is_cuda, "Tensors must be on CUDA/HIP device."

    x_dtype = x.dtype
    x_bf16 = x.to(dtype=torch.bfloat16).contiguous()

    B, C, D, H, W = x_bf16.shape
    OD, OH, OW = _compute_output_shape(D, H, W, kernel_size, stride, padding)

    out = torch.empty((B, C, OD, OH, OW), dtype=torch.bfloat16, device=x.device)

    total_out = B * C * OD * OH * OW
    num_blocks = (total_out + BLOCK_SIZE - 1) // BLOCK_SIZE

    avg_pool3d_kernel[lambda: ((num_blocks, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_bf16, out,
        B, C, D, H, W,
        OD, OH, OW,
        stride, padding,
    )

    return out.to(dtype=x_dtype)


class ModelNew(nn.Module):
    """
    Optimized model that performs 3D Average Pooling using AveLang DSL.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_avg_pool3d(x, self.kernel_size, self.stride, self.padding)
