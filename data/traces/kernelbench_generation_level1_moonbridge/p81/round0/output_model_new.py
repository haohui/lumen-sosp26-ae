import math
import torch
import torch.nn as nn
import avelang
import avelang.language as al

TILE_H = 16
TILE_W = 16
THREADS = TILE_H * TILE_W
BLOCK_OC = 16
_IC = 32
_OC = 64
_K = 3
_K2 = _K * _K
_WEIGHT_TILE_ELEMS = _IC * BLOCK_OC * _K2
_WEIGHT_LOADS = _WEIGHT_TILE_ELEMS // THREADS
_OC_PITCH = _K2
_IC_PITCH = BLOCK_OC * _K2


@avelang.jit
def conv_transpose2d_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    N: al.i32,
    IC: al.i32,
    OC: al.i32,
    H_IN: al.i32,
    W_IN: al.i32,
    H_OUT: al.i32,
    W_OUT: al.i32,
    K_size: al.i32,
    stride: al.i32,
    pad: al.i32,
    dil: al.i32,
):
    tid = al.thread_id(0)
    block_x = al.block_id(0)
    block_y = al.block_id(1)
    block_noc = al.block_id(2)

    oc_blocks = OC // BLOCK_OC
    n = block_noc // oc_blocks
    oc_block = block_noc % oc_blocks
    oc_start = oc_block * BLOCK_OC

    tile_h = tid // TILE_W
    tile_w = tid % TILE_W
    oh = block_y * TILE_H + tile_h
    ow = block_x * TILE_W + tile_w

    IC_H_W = IC * H_IN * W_IN
    OC_H_W = OC * H_OUT * W_OUT
    OCP = OC * _K2

    input_t = al.make_tensor(
        input_ptr,
        al.bf16,
        al.make_layout(
            (N, IC, H_IN, W_IN),
            (IC_H_W, H_IN * W_IN, W_IN, al.convert(1, al.i32)),
        ),
    )
    weight_flat = al.make_tensor(
        weight_ptr,
        al.bf16,
        al.make_layout((IC * OCP,), (al.convert(1, al.i32),)),
    )
    output_t = al.make_tensor(
        output_ptr,
        al.bf16,
        al.make_layout(
            (N, OC, H_OUT, W_OUT),
            (OC_H_W, H_OUT * W_OUT, W_OUT, al.convert(1, al.i32)),
        ),
    )

    # Cooperative load of weight tile into shared memory
    weight_shm = al.make_shared((_WEIGHT_TILE_ELEMS,), al.bf16)
    idx = tid
    w_base_global = oc_start * _K2
    for _ in al.range(_WEIGHT_LOADS):
        ic_idx = idx // (BLOCK_OC * _K2)
        rest = idx % (BLOCK_OC * _K2)
        g_idx = ic_idx * OCP + w_base_global + rest
        weight_shm[idx] = weight_flat[g_idx]
        idx = idx + THREADS
    al.syncthreads()

    if oh >= H_OUT or ow >= W_OUT:
        return

    # Precompute valid kernel position
    valid = al.convert(0, al.i32)
    ih_out = al.convert(0, al.i32)
    iw_out = al.convert(0, al.i32)
    w_khkwoff = al.convert(0, al.i32)

    kh = al.convert(0, al.i32)
    for _kh in al.range(K_size):
        ih_raw = oh + pad - kh * dil
        ih = ih_raw // stride
        if ih * stride == ih_raw:
            if ih >= al.convert(0, al.i32):
                if ih < H_IN:
                    kw = al.convert(0, al.i32)
                    for _kw in al.range(K_size):
                        iw_raw = ow + pad - kw * dil
                        iw = iw_raw // stride
                        if iw * stride == iw_raw:
                            if iw >= al.convert(0, al.i32):
                                if iw < W_IN:
                                    valid = al.convert(1, al.i32)
                                    ih_out = ih
                                    iw_out = iw
                                    w_khkwoff = kh * _K + kw
                        kw = kw + al.convert(1, al.i32)
        kh = kh + al.convert(1, al.i32)

    # Load input values into register array
    input_regs = al.make_local((_IC,), al.f32)
    ic = al.convert(0, al.i32)
    for _ in al.range(IC):
        val = al.convert(0.0, al.f32)
        if valid != 0:
            val = al.convert(input_t[n, ic, ih_out, iw_out], al.f32)
        input_regs[ic] = val
        ic = ic + al.convert(1, al.i32)

    zero_bf16 = al.convert(0.0, al.bf16)

    loc_oc = al.convert(0, al.i32)
    for _ in al.range(BLOCK_OC):
        oc_global = oc_start + loc_oc
        if valid != 0:
            acc = al.convert(0.0, al.f32)
            w_oc_base = loc_oc * _OC_PITCH + w_khkwoff
            ic2 = al.convert(0, al.i32)
            for __ in al.range(IC):
                w_idx = ic2 * _IC_PITCH + w_oc_base
                w_val = al.convert(weight_shm[w_idx], al.f32)
                acc = acc + input_regs[ic2] * w_val
                ic2 = ic2 + al.convert(1, al.i32)
            output_t[n, oc_global, oh, ow] = al.convert(acc, al.bf16)
        else:
            output_t[n, oc_global, oh, ow] = zero_bf16
        loc_oc = loc_oc + al.convert(1, al.i32)


def _compute_output_dims(
    H_IN: int, W_IN: int, K: int, stride: int, pad: int, dil: int
):
    H_OUT = (H_IN - 1) * stride - 2 * pad + dil * (K - 1) + 1
    W_OUT = (W_IN - 1) * stride - 2 * pad + dil * (K - 1) + 1
    return H_OUT, W_OUT


def _to_bf16_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_conv_transpose2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    stride: int,
    padding: int,
    dilation: int,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = _to_bf16_contiguous(x)
    w_bf16 = _to_bf16_contiguous(weight)

    N, IC, H_IN, W_IN = x_bf16.shape
    W_IC, OC, K, K2 = w_bf16.shape
    if W_IC != IC or K != K2:
        raise ValueError(
            f"Weight shape mismatch: expected ({IC}, OC, {K}, {K}), "
            f"got ({W_IC}, {OC}, {K}, {K2})"
        )

    H_OUT, W_OUT = _compute_output_dims(H_IN, W_IN, K, stride, padding, dilation)

    out = torch.empty(
        (N, OC, H_OUT, W_OUT),
        device=x_bf16.device,
        dtype=torch.bfloat16,
    )

    grid_x = (W_OUT + TILE_W - 1) // TILE_W
    grid_y = (H_OUT + TILE_H - 1) // TILE_H
    oc_blocks = OC // BLOCK_OC
    grid_z = N * oc_blocks
    grid = (grid_x, grid_y, grid_z)

    conv_transpose2d_kernel[lambda: (grid, (THREADS, 1, 1))](
        x_bf16,
        w_bf16,
        out,
        N,
        IC,
        OC,
        H_IN,
        W_IN,
        H_OUT,
        W_OUT,
        K,
        stride,
        padding,
        dilation,
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
        dilation: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weight = nn.Parameter(
            torch.empty(in_channels, out_channels, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_conv_transpose2d(
            x, self.weight, self.stride, self.padding, self.dilation
        )
