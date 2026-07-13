import torch
import torch.nn as nn
import avelang
import avelang.language as al

OH_TILE = 8
OW_TILE = 8
THREADS = OH_TILE * OW_TILE
INP_H = OH_TILE + 2
INP_W = OW_TILE + 2
INP_ELEMS = INP_H * INP_W
DEPTH_SLICES = 3
INP_TOTAL = INP_ELEMS * DEPTH_SLICES


@avelang.jit
def conv3d_mish_tanh_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.f32),
    output_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C_in: al.i32,
    C_out: al.i32,
    D: al.i32,
    H: al.i32,
    W: al.i32,
    OD: al.i32,
    OH: al.i32,
    OW: al.i32,
):
    in_sN = C_in * D * H * W
    in_sC = D * H * W
    in_sD = H * W
    in_sH = W

    w_sOC = C_in * 27
    w_sIC = 27
    w_sKD = 9
    w_sKH = 3

    out_sN = C_out * OD * OH * OW
    out_sC = OD * OH * OW
    out_sD = OH * OW
    out_sH = OW

    in_layout = al.make_layout(
        (N, C_in, D, H, W),
        (in_sN, in_sC, in_sD, in_sH, 1),
    )
    input_t = al.make_tensor(input_ptr, al.bf16, in_layout)

    w_layout = al.make_layout(
        (C_out, C_in, 3, 3, 3),
        (w_sOC, w_sIC, w_sKD, w_sKH, 1),
    )
    weight_t = al.make_tensor(weight_ptr, al.bf16, w_layout)

    b_layout = al.make_layout((C_out,), (1,))
    bias_t = al.make_tensor(bias_ptr, al.f32, b_layout)

    out_layout = al.make_layout(
        (N, C_out, OD, OH, OW),
        (out_sN, out_sC, out_sD, out_sH, 1),
    )
    output_t = al.make_tensor(output_ptr, al.bf16, out_layout)

    acc_smem = al.make_shared((THREADS,), al.f32)
    inp_smem = al.make_shared((INP_TOTAL,), al.bf16)

    grid_oh = al.block_id(1)
    grid_ow = al.block_id(2)
    tid = al.thread_id(0)
    oh_local = tid // OW_TILE
    ow_local = tid % OW_TILE

    tile_h = al.convert(OH_TILE, al.i32)
    tile_w = al.convert(OW_TILE, al.i32)
    oh = grid_oh * tile_h + oh_local
    ow = grid_ow * tile_w + ow_local

    oh_start = grid_oh * tile_h
    ow_start = grid_ow * tile_w

    inp_w = al.convert(INP_W, al.i32)
    inp_elems = al.convert(INP_ELEMS, al.i32)
    inp_total = al.convert(INP_TOTAL, al.i32)

    valid = (oh < OH) and (ow < OW)

    flat_idx = al.block_id(0)
    batch = flat_idx // (C_out * OD)
    tmp = flat_idx % (C_out * OD)
    oc = tmp // OD
    od = tmp % OD

    if valid:
        bval = bias_t[oc]
        acc_smem[tid] = bval

    for ic in al.range(C_in):
        for load_i in al.range(5):
            idx = tid * 5 + load_i
            if idx < inp_total:
                d_slice = idx // inp_elems
                local_idx = idx % inp_elems
                s_row = local_idx // inp_w
                s_col = local_idx % inp_w
                g_row = oh_start + s_row
                g_col = ow_start + s_col
                if (g_row < H) and (g_col < W):
                    inp_smem[idx] = input_t[batch, ic, od + d_slice, g_row, g_col]

        al.syncthreads()

        if valid:
            for kd in al.range(3):
                for kh in al.range(3):
                    for kw in al.range(3):
                        s_off = kd * inp_elems + (oh_local + kh) * inp_w + (ow_local + kw)
                        in_val = al.convert(inp_smem[s_off], al.f32)
                        w_val = al.convert(weight_t[oc, ic, kd, kh, kw], al.f32)
                        acc_smem[tid] = acc_smem[tid] + in_val * w_val

        al.syncthreads()

    if valid:
        val = acc_smem[tid]
        one = al.convert(1.0, al.f32)
        sp = al.log(one + al.exp(val))
        mish_val = val * al.tanh(sp)
        result = al.tanh(mish_val)
        output_t[batch, oc, od, oh, ow] = al.convert(result, al.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        assert x.is_cuda, "Input tensor must be on CUDA/HIP device"

        weight = self.conv.weight.data
        bias = self.conv.bias.data.float()
        weight_bf16 = weight.to(torch.bfloat16)

        N, C_in, D, H, W = x.shape
        C_out = weight_bf16.shape[0]
        KD, KH, KW = self.kernel_size, self.kernel_size, self.kernel_size
        OD = D - KD + 1
        OH = H - KH + 1
        OW = W - KW + 1

        out_conv = torch.empty(
            N, C_out, OD, OH, OW, device=x.device, dtype=torch.bfloat16
        )

        grid_x = N * C_out * OD
        grid_y = (OH + OH_TILE - 1) // OH_TILE
        grid_z = (OW + OW_TILE - 1) // OW_TILE

        x_bf16 = x.to(torch.bfloat16)
        conv3d_mish_tanh_kernel[lambda: ((grid_x, grid_y, grid_z), (THREADS, 1, 1))](
            x_bf16,
            weight_bf16,
            bias,
            out_conv,
            N,
            C_in,
            C_out,
            D,
            H,
            W,
            OD,
            OH,
            OW,
        )

        return out_conv
