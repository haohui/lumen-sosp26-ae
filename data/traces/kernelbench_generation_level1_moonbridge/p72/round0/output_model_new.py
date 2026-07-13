import torch
import torch.nn as nn
import math
import avelang
import avelang.language as al

BLOCK_SIZE: al.constexpr = 256
SPAT_PER_BLOCK: al.constexpr = 8


@avelang.jit
def conv_transpose3d_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    IC: al.i32,
    OC: al.i32,
    IC_G: al.i32,
    OC_G: al.i32,
    D_IN: al.i32,
    H_IN: al.i32,
    W_IN: al.i32,
    D_OUT: al.i32,
    H_OUT: al.i32,
    W_OUT: al.i32,
    KD: al.i32,
    KH: al.i32,
    KW: al.i32,
    SD: al.i32,
    SH: al.i32,
    SW: al.i32,
    PD: al.i32,
    PH: al.i32,
    PW: al.i32,
    NUM_SPAT_BLOCKS: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    # Decompose block id into (batch, spatial_block)
    spat_block = bid % NUM_SPAT_BLOCKS
    n = bid // NUM_SPAT_BLOCKS

    # Thread mapping: oc = tid % OC, spat_local = tid // OC
    # Threads in the same warp share input spatial position → coalesced input loads
    oc = tid % OC
    spat_local = tid // OC

    # Global spatial position for this thread
    spat_global = spat_block * SPAT_PER_BLOCK + spat_local
    total_spatial = D_OUT * H_OUT * W_OUT

    if spat_global < total_spatial:
        # Decompose spatial index to 3D
        s1 = spat_global // W_OUT
        ow = spat_global - s1 * W_OUT
        s2 = s1 // H_OUT
        oh = s1 - s2 * H_OUT
        od = s2

        g = oc // OC_G
        oc_g = oc - g * OC_G
        ic_start = g * IC_G

        acc = al.convert(0.0, al.f32)

        # Flat views for global memory access
        x_total = B * IC * D_IN * H_IN * W_IN
        w_total = IC * OC_G * KD * KH * KW
        x = al.make_tensor(x_ptr, al.bf16, al.make_layout((x_total,), (1,)))
        w = al.make_tensor(w_ptr, al.bf16, al.make_layout((w_total,), (1,)))

        # Input strides
        x_stride_n = IC * D_IN * H_IN * W_IN
        x_stride_ic = D_IN * H_IN * W_IN
        x_stride_d = H_IN * W_IN
        x_stride_h = W_IN

        # Weight strides
        w_stride_ic = OC_G * KD * KH * KW
        w_stride_oc_g = KD * KH * KW
        w_stride_kd = KH * KW
        w_stride_kh = KW

        x_base_n = n * x_stride_n

        for ic_g in al.range(IC_G):
            ic = ic_start + ic_g
            x_base = x_base_n + ic * x_stride_ic
            w_base_ic = ic * w_stride_ic + oc_g * w_stride_oc_g

            for kd in al.range(KD):
                d_check = od + PD - kd
                if d_check >= 0:
                    id_val = d_check // SD
                    d_rem = d_check - id_val * SD
                    if d_rem == 0:
                        if id_val < D_IN:
                            x_off_d = x_base + id_val * x_stride_d
                            w_off_kd = w_base_ic + kd * w_stride_kd

                            for kh in al.range(KH):
                                h_check = oh + PH - kh
                                if h_check >= 0:
                                    ih_val = h_check // SH
                                    h_rem = h_check - ih_val * SH
                                    if h_rem == 0:
                                        if ih_val < H_IN:
                                            x_off_h = x_off_d + ih_val * x_stride_h
                                            w_off_kh = w_off_kd + kh * w_stride_kh

                                            for kw in al.range(KW):
                                                w_check = ow + PW - kw
                                                if w_check >= 0:
                                                    iw_val = w_check // SW
                                                    w_rem = w_check - iw_val * SW
                                                    if w_rem == 0:
                                                        if iw_val < W_IN:
                                                            x_val = al.convert(x[x_off_h + iw_val], al.f32)
                                                            w_val = al.convert(w[w_off_kh + kw], al.f32)
                                                            acc = acc + x_val * w_val

        # Write result using 5D output layout
        out = al.make_tensor(out_ptr, al.bf16, al.make_layout(
            (B, OC, D_OUT, H_OUT, W_OUT),
            (OC * D_OUT * H_OUT * W_OUT, D_OUT * H_OUT * W_OUT, H_OUT * W_OUT, W_OUT, 1)
        ))
        out[n, oc, od, oh, ow] = al.convert(acc, al.bf16)


def _prepare_bf16_cuda(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_conv_transpose3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    stride: tuple,
    padding: tuple,
    output_padding: tuple,
    groups: int,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device required for AveLang kernels.")

    B, IC, D_IN, H_IN, W_IN = x.shape
    OC_G = weight.shape[1]
    KD, KH, KW = weight.shape[2], weight.shape[3], weight.shape[4]
    OC = OC_G * groups

    SD, SH, SW = stride
    PD, PH, PW = padding
    OPD, OPH, OPW = output_padding
    IC_G = IC // groups

    D_OUT = (D_IN - 1) * SD - 2 * PD + KD + OPD
    H_OUT = (H_IN - 1) * SH - 2 * PH + KH + OPH
    W_OUT = (W_IN - 1) * SW - 2 * PW + KW + OPW

    x_bf16 = _prepare_bf16_cuda(x)
    w_bf16 = _prepare_bf16_cuda(weight)

    out = torch.empty((B, OC, D_OUT, H_OUT, W_OUT), device=x_bf16.device, dtype=torch.bfloat16)

    total_spatial = D_OUT * H_OUT * W_OUT
    num_spat_blocks = (total_spatial + SPAT_PER_BLOCK - 1) // SPAT_PER_BLOCK
    num_blocks = B * num_spat_blocks

    conv_transpose3d_kernel[lambda: ((num_blocks, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_bf16,
        w_bf16,
        out,
        B,
        IC,
        OC,
        IC_G,
        OC_G,
        D_IN,
        H_IN,
        W_IN,
        D_OUT,
        H_OUT,
        W_OUT,
        KD,
        KH,
        KW,
        SD,
        SH,
        SW,
        PD,
        PH,
        PW,
        num_spat_blocks,
    )
    return out


class ModelNew(nn.Module):
    """
    Performs a 3D transposed convolution operation with asymmetric input and kernel,
    and optional stride, using an optimized AveLang GPU kernel.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple = (1, 1, 1),
        padding: tuple = (0, 0, 0),
        output_padding: tuple = (0, 0, 0),
        groups: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

        out_cpg = out_channels // groups
        self.weight = nn.Parameter(torch.empty(in_channels, out_cpg, *kernel_size))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self._init_parameters()

    def _init_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_conv_transpose3d(
            x,
            self.weight,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
        )
