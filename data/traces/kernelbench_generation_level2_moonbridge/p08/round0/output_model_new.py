import torch
import torch.nn as nn
import avelang
import avelang.language as al

batch_size   = 128
in_channels  = 8
out_channels = 16
depth = 16
height = 64
width = 64
kernel_size = (3, 3, 3)
divisor = 2.0
pool_size = (2, 2, 2)
bias_shape = (out_channels, 1, 1, 1)
sum_dim = 1

BLOCK_SIZE_POST = 256
MAX_OC = 16

TILE_H = 4
TILE_W = 4
TILE_OC = 16
IN_HALO_H = TILE_H + 2
IN_HALO_W = TILE_W + 2
IN_HALO_D = 3
IN_SHM_ELEMS = in_channels * IN_HALO_D * IN_HALO_H * IN_HALO_W
W_SHM_ELEMS = out_channels * in_channels * 3 * 3 * 3
BLOCK_SIZE_CONV = TILE_OC * TILE_H * TILE_W


@avelang.jit
def conv3d_div_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    conv_bias_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    IC: al.i32,
    OC: al.i32,
    D: al.i32,
    H: al.i32,
    W: al.i32,
    OD: al.i32,
    OH: al.i32,
    OW: al.i32,
    num_h_blocks: al.i32,
    num_w_blocks: al.i32,
):
    tid = al.thread_id(0)
    block_hw = al.block_id(0)
    block_d = al.block_id(1)
    block_b = al.block_id(2)

    block_h = block_hw // num_w_blocks
    block_w = block_hw - block_h * num_w_blocks

    h_start = block_h * TILE_H
    w_start = block_w * TILE_W

    oh_local = tid // TILE_W % TILE_H
    ow_local = tid % TILE_W
    oc_local = tid // (TILE_H * TILE_W)

    oh_out = h_start + oh_local
    ow_out = w_start + ow_local
    od_out = block_d

    valid = (oh_out < OH) and (ow_out < OW)

    x_s0 = IC * D * H * W
    x_s1 = D * H * W
    x_s2 = H * W
    x_s3 = W
    x = al.make_tensor(x_ptr, al.bf16, al.make_layout(
        (B, IC, D, H, W),
        (x_s0, x_s1, x_s2, x_s3, al.convert(1, al.i32)),
    ))

    w_s0 = IC * 3 * 3 * 3
    w_s1 = 3 * 3 * 3
    w_s2 = 3 * 3
    w_s3 = 3
    w = al.make_tensor(w_ptr, al.bf16, al.make_layout(
        (OC, IC, 3, 3, 3),
        (w_s0, w_s1, w_s2, w_s3, al.convert(1, al.i32)),
    ))

    cb = al.make_tensor(conv_bias_ptr, al.bf16, al.make_layout(
        (OC,), (al.convert(1, al.i32),),
    ))

    out_s0 = OC * OD * OH * OW
    out_s1 = OD * OH * OW
    out_s2 = OH * OW
    out_s3 = OW
    out = al.make_tensor(out_ptr, al.bf16, al.make_layout(
        (B, OC, OD, OH, OW),
        (out_s0, out_s1, out_s2, out_s3, al.convert(1, al.i32)),
    ))

    shm_in = al.make_shared((IN_SHM_ELEMS,), al.bf16)
    shm_w = al.make_shared((W_SHM_ELEMS,), al.bf16)

    for load_idx in al.range(tid, IN_SHM_ELEMS, BLOCK_SIZE_CONV):
        lw = load_idx % IN_HALO_W
        tmp1 = load_idx // IN_HALO_W
        lh = tmp1 % IN_HALO_H
        tmp2 = tmp1 // IN_HALO_H
        ld = tmp2 % IN_HALO_D
        ic = tmp2 // IN_HALO_D

        g_h = h_start + lh
        g_w = w_start + lw
        g_h_c = g_h
        if g_h >= H:
            g_h_c = H - al.convert(1, al.i32)
        g_w_c = g_w
        if g_w >= W:
            g_w_c = W - al.convert(1, al.i32)

        shm_in[load_idx] = x[block_b, ic, block_d + ld, g_h_c, g_w_c]

    for load_idx in al.range(tid, W_SHM_ELEMS, BLOCK_SIZE_CONV):
        kw = load_idx % 3
        tmp1 = load_idx // 3
        kh = tmp1 % 3
        tmp2 = tmp1 // 3
        kd = tmp2 % 3
        tmp3 = tmp2 // 3
        ic = tmp3 % IC
        oc = tmp3 // IC
        shm_w[load_idx] = w[oc, ic, kd, kh, kw]

    al.syncthreads()

    if valid:
        acc = al.convert(0.0, al.f32)

        for ic in al.range(IC):
            for kd in al.range(3):
                for kh in al.range(3):
                    for kw in al.range(3):
                        in_idx = ic * (IN_HALO_D * IN_HALO_H * IN_HALO_W)
                        in_idx = in_idx + kd * (IN_HALO_H * IN_HALO_W)
                        in_idx = in_idx + (oh_local + kh) * IN_HALO_W
                        in_idx = in_idx + (ow_local + kw)

                        w_idx = oc_local * (IC * 27) + ic * 27 + kd * 9 + kh * 3 + kw

                        x_f32 = al.convert(shm_in[in_idx], al.f32)
                        w_f32 = al.convert(shm_w[w_idx], al.f32)
                        acc = acc + x_f32 * w_f32

        cb_f32 = al.convert(cb[oc_local], al.f32)
        acc = acc + cb_f32
        acc = acc / al.convert(2.0, al.f32)
        out[block_b, oc_local, od_out, oh_out, ow_out] = al.convert(acc, al.bf16)


@avelang.jit
def maxpool_avgpool_bias_sum_kernel(
    conv_ptr: al.Pointer(al.bf16),
    extra_bias_ptr: al.Pointer(al.bf16),
    final_out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    OC: al.i32,
    OD: al.i32,
    OH: al.i32,
    OW: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    if bid < B:
        conv_s0 = OC * OD * OH * OW
        conv_s1 = OD * OH * OW
        conv_s2 = OH * OW
        conv_s3 = OW
        conv = al.make_tensor(conv_ptr, al.bf16, al.make_layout(
            (B, OC, OD, OH, OW),
            (conv_s0, conv_s1, conv_s2, conv_s3, al.convert(1, al.i32)),
        ))

        smem = al.make_shared((BLOCK_SIZE_POST, MAX_OC), al.f32)

        for c in al.range(OC):
            smem[tid, c] = al.convert(0.0, al.f32)

        MD = OD // 2
        MH = OH // 2
        MW = OW // 2
        total_positions = OC * MD * MH * MW

        for pos in al.range(tid, total_positions, BLOCK_SIZE_POST):
            mw = pos % MW
            d1 = pos // MW
            mh = d1 % MH
            d2 = d1 // MH
            md = d2 % MD
            c = d2 // MD

            v0 = al.convert(conv[bid, c, md * 2, mh * 2, mw * 2], al.f32)
            v1 = al.convert(conv[bid, c, md * 2, mh * 2, mw * 2 + 1], al.f32)
            v2 = al.convert(conv[bid, c, md * 2, mh * 2 + 1, mw * 2], al.f32)
            v3 = al.convert(conv[bid, c, md * 2, mh * 2 + 1, mw * 2 + 1], al.f32)
            v4 = al.convert(conv[bid, c, md * 2 + 1, mh * 2, mw * 2], al.f32)
            v5 = al.convert(conv[bid, c, md * 2 + 1, mh * 2, mw * 2 + 1], al.f32)
            v6 = al.convert(conv[bid, c, md * 2 + 1, mh * 2 + 1, mw * 2], al.f32)
            v7 = al.convert(conv[bid, c, md * 2 + 1, mh * 2 + 1, mw * 2 + 1], al.f32)

            max_val = v0
            if v1 > max_val:
                max_val = v1
            if v2 > max_val:
                max_val = v2
            if v3 > max_val:
                max_val = v3
            if v4 > max_val:
                max_val = v4
            if v5 > max_val:
                max_val = v5
            if v6 > max_val:
                max_val = v6
            if v7 > max_val:
                max_val = v7

            smem[tid, c] = smem[tid, c] + max_val

        al.syncthreads()

        if tid < 128:
            for c in al.range(OC):
                smem[tid, c] = smem[tid, c] + smem[tid + 128, c]
        al.syncthreads()
        if tid < 64:
            for c in al.range(OC):
                smem[tid, c] = smem[tid, c] + smem[tid + 64, c]
        al.syncthreads()
        if tid < 32:
            for c in al.range(OC):
                smem[tid, c] = smem[tid, c] + smem[tid + 32, c]
        al.syncthreads()
        if tid < 16:
            for c in al.range(OC):
                smem[tid, c] = smem[tid, c] + smem[tid + 16, c]
        al.syncthreads()
        if tid < 8:
            for c in al.range(OC):
                smem[tid, c] = smem[tid, c] + smem[tid + 8, c]
        al.syncthreads()
        if tid < 4:
            for c in al.range(OC):
                smem[tid, c] = smem[tid, c] + smem[tid + 4, c]
        al.syncthreads()
        if tid < 2:
            for c in al.range(OC):
                smem[tid, c] = smem[tid, c] + smem[tid + 2, c]
        al.syncthreads()
        if tid < 1:
            for c in al.range(OC):
                smem[0, c] = smem[0, c] + smem[1, c]

        al.syncthreads()

        if tid == 0:
            extra_bias = al.make_tensor(extra_bias_ptr, al.bf16, al.make_layout(
                (OC,), (al.convert(1, al.i32),),
            ))
            total_spatial = MD * MH * MW
            total_f32 = al.convert(total_spatial, al.f32)
            final_sum = al.convert(0.0, al.f32)

            for c in al.range(OC):
                avg = smem[0, c] / total_f32
                eb_f32 = al.convert(extra_bias[c], al.f32)
                final_sum = final_sum + avg + eb_f32

            final_t = al.make_tensor(final_out_ptr, al.bf16, al.make_layout(
                (B,), (al.convert(1, al.i32),),
            ))
            final_t[bid] = al.convert(final_sum, al.bf16)


def _to_bf16_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_conv_pipeline(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_bias: torch.Tensor,
    extra_bias: torch.Tensor,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = _to_bf16_contiguous(x)
    w_bf16 = _to_bf16_contiguous(weight)
    cb_bf16 = _to_bf16_contiguous(conv_bias)
    eb_bf16 = _to_bf16_contiguous(extra_bias)

    B_val, IC_val, D_val, H_val, W_val = x_bf16.shape
    OC_val = w_bf16.shape[0]
    OD_val = D_val - 2
    OH_val = H_val - 2
    OW_val = W_val - 2

    num_h_blocks = (OH_val + TILE_H - 1) // TILE_H
    num_w_blocks = (OW_val + TILE_W - 1) // TILE_W
    num_hw_blocks = num_h_blocks * num_w_blocks

    conv_out = torch.empty(
        (B_val, OC_val, OD_val, OH_val, OW_val),
        device=x_bf16.device,
        dtype=torch.bfloat16,
    )

    conv3d_div_kernel[lambda: ((int(num_hw_blocks), int(OD_val), int(B_val)), (BLOCK_SIZE_CONV, 1, 1))](
        x_bf16,
        w_bf16,
        cb_bf16,
        conv_out,
        int(B_val),
        int(IC_val),
        int(OC_val),
        int(D_val),
        int(H_val),
        int(W_val),
        int(OD_val),
        int(OH_val),
        int(OW_val),
        int(num_h_blocks),
        int(num_w_blocks),
    )

    final_out = torch.empty(
        (int(B_val),),
        device=x_bf16.device,
        dtype=torch.bfloat16,
    )

    maxpool_avgpool_bias_sum_kernel[lambda: ((int(B_val), 1, 1), (BLOCK_SIZE_POST, 1, 1))](
        conv_out,
        eb_bf16,
        final_out,
        int(B_val),
        int(OC_val),
        int(OD_val),
        int(OH_val),
        int(OW_val),
    )

    return final_out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim

    def forward(self, x):
        result_flat = avelang_conv_pipeline(
            x,
            self.conv.weight.data,
            self.conv.bias.data,
            self.bias.data,
        )
        return result_flat.to(dtype=x.dtype).view(x.shape[0], 1, 1, 1)


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim]
