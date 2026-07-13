import math
import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_THREADS = 256
TD = 2
TH = 3
TW = 4
IC_TILE = 8
OC_TILE = 16

_KD = 3
_KH = 5
_KW = 7

INPUT_D = TD + _KD - 1
INPUT_H = TH + _KH - 1
INPUT_W = TW + _KW - 1
INPUT_REGION = INPUT_D * INPUT_H * INPUT_W
SHM_IN_ELEMS = IC_TILE * INPUT_REGION
SHM_W_ELEMS = IC_TILE * OC_TILE * _KD * _KH * _KW
MAX_OUT_PER_BLOCK = TD * TH * TW * OC_TILE
MAX_ELEMS_PER_THREAD = (MAX_OUT_PER_BLOCK + BLOCK_THREADS - 1) // BLOCK_THREADS


@avelang.jit
def conv_transpose3d_tiled_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    N: al.i32,
    IC: al.i32,
    OC: al.i32,
    D_in: al.i32,
    H_in: al.i32,
    W_in: al.i32,
    KD: al.i32,
    KH: al.i32,
    KW: al.i32,
    D_out: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    stride_d: al.i32,
    stride_h: al.i32,
    stride_w: al.i32,
    pad_d: al.i32,
    pad_h: al.i32,
    pad_w: al.i32,
    dil_d: al.i32,
    dil_h: al.i32,
    dil_w: al.i32,
    outpad_d: al.i32,
    outpad_h: al.i32,
    outpad_w: al.i32,
    groups: al.i32,
):
    tid = al.thread_id(0)
    bx = al.block_id(0)
    n = al.block_id(1)

    one = al.convert(1, al.i32)
    zero_i = al.convert(0, al.i32)
    zero_f = al.convert(0.0, al.f32)

    num_tiles_d = (D_out + TD - one) // TD
    num_tiles_h = (H_out + TH - one) // TH
    num_tiles_w = (W_out + TW - one) // TW
    num_tiles_oc = (OC + OC_TILE - one) // OC_TILE

    tmp = bx
    oc_tile = tmp % num_tiles_oc
    tmp = tmp // num_tiles_oc
    w_tile = tmp % num_tiles_w
    tmp = tmp // num_tiles_w
    h_tile = tmp % num_tiles_h
    d_tile = tmp // num_tiles_h

    oc_start = oc_tile * OC_TILE
    w_start = w_tile * TW
    h_start = h_tile * TH
    d_start = d_tile * TD

    cur_td = TD
    d_end = d_start + TD
    if d_end > D_out:
        cur_td = D_out - d_start
    cur_th = TH
    h_end = h_start + TH
    if h_end > H_out:
        cur_th = H_out - h_start
    cur_tw = TW
    w_end = w_start + TW
    if w_end > W_out:
        cur_tw = W_out - w_start
    cur_oc = OC_TILE
    if oc_start + OC_TILE > OC:
        cur_oc = OC - oc_start

    in_d_start = d_start - KD + one
    in_h_start = h_start - KH + one
    in_w_start = w_start - KW + one
    in_d_end = d_start + TD - one
    in_h_end = h_start + TH - one
    in_w_end = w_start + TW - one

    if in_d_start < zero_i:
        in_d_start = zero_i
    if in_h_start < zero_i:
        in_h_start = zero_i
    if in_w_start < zero_i:
        in_w_start = zero_i
    if in_d_end >= D_in:
        in_d_end = D_in - one
    if in_h_end >= H_in:
        in_h_end = H_in - one
    if in_w_end >= W_in:
        in_w_end = W_in - one

    reg_d = in_d_end - in_d_start + one
    reg_h = in_h_end - in_h_start + one
    reg_w = in_w_end - in_w_start + one
    if reg_d < zero_i:
        reg_d = zero_i
    if reg_h < zero_i:
        reg_h = zero_i
    if reg_w < zero_i:
        reg_w = zero_i

    dhw = D_in * H_in * W_in
    hw_in = H_in * W_in
    oc_kd_kh_kw = OC * KD * KH * KW
    kd_kh_kw = KD * KH * KW
    kh_kw = KH * KW
    out_dhw = D_out * H_out * W_out
    out_hw = H_out * W_out
    ic_dhw = IC * dhw

    input_1d = al.make_tensor(input_ptr, al.bf16, al.make_layout((N * ic_dhw,), (1,)))
    weight_1d = al.make_tensor(weight_ptr, al.bf16, al.make_layout((IC * oc_kd_kh_kw,), (1,)))
    output_1d = al.make_tensor(output_ptr, al.bf16, al.make_layout((N * OC * out_dhw,), (1,)))

    shm_in = al.make_shared((SHM_IN_ELEMS,), al.bf16, 128)
    shm_w = al.make_shared((SHM_W_ELEMS,), al.bf16, 128)

    num_elems = cur_td * cur_th * cur_tw * cur_oc
    elems_per_thread = (num_elems + BLOCK_THREADS - one) // BLOCK_THREADS

    acc = al.make_local((MAX_ELEMS_PER_THREAD,), al.f32)
    for i in al.range(MAX_ELEMS_PER_THREAD):
        acc[i] = zero_f

    ic_tiles = (IC + IC_TILE - one) // IC_TILE
    total_in_reg = reg_d * reg_h * reg_w

    for ict in al.range(ic_tiles):
        ic_base = ict * IC_TILE
        cur_ic = IC_TILE
        ic_end = ic_base + IC_TILE
        if ic_end > IC:
            cur_ic = IC - ic_base

        # Cooperative load of input region into shared memory
        total_in_load = cur_ic * total_in_reg
        idx = tid
        for _ in al.range((total_in_load + BLOCK_THREADS - one) // BLOCK_THREADS):
            if idx < total_in_load:
                lic = idx // total_in_reg
                rest = idx % total_in_reg
                ld = rest // (reg_h * reg_w)
                rest2 = rest % (reg_h * reg_w)
                lh = rest2 // reg_w
                lw = rest2 % reg_w
                gic = ic_base + lic
                gd = in_d_start + ld
                gh = in_h_start + lh
                gw = in_w_start + lw
                in_val = input_1d[n * ic_dhw + gic * dhw + gd * hw_in + gh * W_in + gw]
                shm_idx = lic * total_in_reg + ld * (reg_h * reg_w) + lh * reg_w + lw
                shm_in[shm_idx] = in_val
            idx = idx + BLOCK_THREADS

        # Cooperative load of weight tile into shared memory
        w_oc_kd_kh_kw = cur_oc * _KD * _KH * _KW
        w_kd_kh_kw_val = _KD * _KH * _KW
        total_w_load = cur_ic * w_oc_kd_kh_kw
        idx = tid
        for _ in al.range((total_w_load + BLOCK_THREADS - one) // BLOCK_THREADS):
            if idx < total_w_load:
                lic = idx // w_oc_kd_kh_kw
                rest = idx % w_oc_kd_kh_kw
                loc = rest // w_kd_kh_kw_val
                rest2 = rest % w_kd_kh_kw_val
                ldk = rest2 // kh_kw
                rest3 = rest2 % kh_kw
                lhk = rest3 // _KW
                lwk = rest3 % _KW
                gic = ic_base + lic
                goc = oc_start + loc
                if goc < OC:
                    w_val = weight_1d[gic * oc_kd_kh_kw + goc * kd_kh_kw + ldk * kh_kw + lhk * _KW + lwk]
                    w_shm_idx = lic * w_oc_kd_kh_kw + loc * w_kd_kh_kw_val + ldk * kh_kw + lhk * _KW + lwk
                    shm_w[w_shm_idx] = w_val
            idx = idx + BLOCK_THREADS

        al.syncthreads()

        # Compute contributions
        for e in al.range(elems_per_thread):
            elem_idx = tid * elems_per_thread + e
            if elem_idx < num_elems:
                tmp_e = elem_idx
                local_oc = tmp_e % cur_oc
                tmp_e = tmp_e // cur_oc
                local_w = tmp_e % cur_tw
                tmp_e = tmp_e // cur_tw
                local_h = tmp_e % cur_th
                local_d = tmp_e // cur_th
                god = d_start + local_d
                goh = h_start + local_h
                gow = w_start + local_w
                part = zero_f
                for lic in al.range(cur_ic):
                    w_lic_base = lic * w_oc_kd_kh_kw + local_oc * w_kd_kh_kw_val
                    in_lic_base = lic * total_in_reg
                    for ldk in al.range(_KD):
                        gd_in = god - ldk
                        sd = gd_in - in_d_start
                        if sd >= zero_i:
                            if sd < reg_d:
                                in_depth_base = in_lic_base + sd * (reg_h * reg_w)
                                w_depth_base = w_lic_base + ldk * kh_kw
                                for lhk in al.range(_KH):
                                    gh_in = goh - lhk
                                    sh = gh_in - in_h_start
                                    if sh >= zero_i:
                                        if sh < reg_h:
                                            in_h_base = in_depth_base + sh * reg_w
                                            w_h_base = w_depth_base + lhk * _KW
                                            for lwk in al.range(_KW):
                                                gw_in = gow - lwk
                                                sw = gw_in - in_w_start
                                                if sw >= zero_i:
                                                    if sw < reg_w:
                                                        in_val = al.convert(shm_in[in_h_base + sw], al.f32)
                                                        w_val = al.convert(shm_w[w_h_base + lwk], al.f32)
                                                        part = part + in_val * w_val
                acc[e] = acc[e] + part

        al.syncthreads()

    # Writeback to global memory
    for e in al.range(elems_per_thread):
        elem_idx = tid * elems_per_thread + e
        if elem_idx < num_elems:
            tmp_e = elem_idx
            local_oc = tmp_e % cur_oc
            tmp_e = tmp_e // cur_oc
            local_w = tmp_e % cur_tw
            tmp_e = tmp_e // cur_tw
            local_h = tmp_e % cur_th
            local_d = tmp_e // cur_th
            god = d_start + local_d
            goh = h_start + local_h
            gow = w_start + local_w
            gooc = oc_start + local_oc
            out_idx = n * OC * out_dhw + gooc * out_dhw + god * out_hw + goh * W_out + gow
            output_1d[out_idx] = al.convert(acc[e], al.bf16)


def _prepare_bf16_cuda_contiguous(t: torch.Tensor) -> torch.Tensor:
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
    dilation: tuple,
    groups: int,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")
    x_bf16 = _prepare_bf16_cuda_contiguous(x)
    w_bf16 = _prepare_bf16_cuda_contiguous(weight)
    N, IC, D_in, H_in, W_in = x_bf16.shape
    w_IC, OC_per_group, KD, KH, KW = w_bf16.shape
    OC = OC_per_group * groups
    if w_IC != IC:
        raise ValueError(f"Weight input channels mismatch: input has IC={IC}, weight first dim={w_IC}")
    if KD != _KD or KH != _KH or KW != _KW:
        raise ValueError(f"Kernel optimized for size ({_KD},{_KH},{_KW}), got ({KD},{KH},{KW})")
    stride_d, stride_h, stride_w = stride
    pad_d, pad_h, pad_w = padding
    outpad_d, outpad_h, outpad_w = output_padding
    dil_d, dil_h, dil_w = dilation
    D_out = (D_in - 1) * stride_d - 2 * pad_d + dil_d * (KD - 1) + outpad_d + 1
    H_out = (H_in - 1) * stride_h - 2 * pad_h + dil_h * (KH - 1) + outpad_h + 1
    W_out = (W_in - 1) * stride_w - 2 * pad_w + dil_w * (KW - 1) + outpad_w + 1
    out = torch.empty((N, OC, D_out, H_out, W_out), device=x_bf16.device, dtype=torch.bfloat16)
    num_tiles_d = (D_out + TD - 1) // TD
    num_tiles_h = (H_out + TH - 1) // TH
    num_tiles_w = (W_out + TW - 1) // TW
    num_tiles_oc = (OC + OC_TILE - 1) // OC_TILE
    num_blocks_x = num_tiles_w * num_tiles_h * num_tiles_d * num_tiles_oc
    num_blocks_y = N
    grid = (num_blocks_x, num_blocks_y, 1)
    conv_transpose3d_tiled_kernel[lambda: (grid, (BLOCK_THREADS, 1, 1))](
        x_bf16, w_bf16, out,
        N, IC, OC, D_in, H_in, W_in, KD, KH, KW,
        D_out, H_out, W_out,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w, dil_d, dil_h, dil_w,
        outpad_d, outpad_h, outpad_w, groups,
    )
    return out


class ModelNew(nn.Module):
    """
    Performs a transposed 3D convolution operation with asymmetric input and kernel sizes
    using an optimized AveLang GPU kernel with shared-memory tiling.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple,
                 stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0),
                 output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.dilation = (1, 1, 1)
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / (fan_in**0.5)
                nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_conv_transpose3d(x, self.weight, self.stride, self.padding,
                                        self.output_padding, self.dilation, self.groups)


batch_size = 16
in_channels = 32
out_channels = 16
kernel_size = (3, 5, 7)
depth_in = 16
height_in = 32
width_in = 64


def get_inputs():
    x = torch.rand(batch_size, in_channels, depth_in, height_in, width_in)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
