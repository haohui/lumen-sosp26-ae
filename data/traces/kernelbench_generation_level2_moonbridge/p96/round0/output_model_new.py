import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE: al.constexpr = 256


@avelang.jit
def fused_ct_scale_maxpool_kernel(
    x_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    mid_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C_in: al.i32,
    C_out: al.i32,
    D_in: al.i32,
    H_in: al.i32,
    W_in: al.i32,
    D_mid: al.i32,
    H_mid: al.i32,
    W_mid: al.i32,
    stride_c: al.i32,
    stride_d: al.i32,
    stride_h: al.i32,
    total_cells: al.i32,
    scale_val: al.f32,
):
    tid = al.thread_id(0)
    gid = al.block_id(0) * BLOCK_SIZE + tid

    if gid < total_cells:
        s_co = C_out * stride_c
        b = gid // s_co
        r = gid - b * s_co
        c_out = r // stride_c
        r2 = r - c_out * stride_c
        d_mp = r2 // stride_d
        r3 = r2 - d_mp * stride_d
        h_mp = r3 // stride_h
        w_mp = r3 - h_mp * stride_h

        one = al.convert(1, al.i32)
        x_1d = al.make_tensor(x_ptr, al.bf16, al.make_layout((N * C_in * D_in * H_in * W_in,), (one,)))
        w_1d = al.make_tensor(weight_ptr, al.bf16, al.make_layout((C_in * C_out * 27,), (one,)))
        bias_1d = al.make_tensor(bias_ptr, al.bf16, al.make_layout((C_out,), (one,)))
        mid_1d = al.make_tensor(mid_ptr, al.bf16, al.make_layout((N * C_out * stride_c,), (one,)))

        in_stride_b = C_in * D_in * H_in * W_in
        in_stride_c = D_in * H_in * W_in
        in_stride_d = H_in * W_in
        in_stride_h = W_in
        w_stride_cin = C_out * 27
        w_stride_cout = 27
        w_stride_kd = 9
        w_stride_kh = 3

        mid_offset = b * C_out * stride_c + c_out * stride_c + d_mp * stride_d + h_mp * stride_h + w_mp

        large_neg = al.convert(-1.0e30, al.f32)
        max_val = large_neg

        b_in_base = b * in_stride_b

        for dd in al.range(2):
            do_num = 2 * d_mp + dd
            for dh in al.range(2):
                ho_num = 2 * h_mp + dh
                for dw in al.range(2):
                    wo_num = 2 * w_mp + dw

                    acc = al.convert(bias_1d[c_out], al.f32)

                    for c_in in al.range(C_in):
                        c_in_base = b_in_base + c_in * in_stride_c
                        w_cin_base = c_in * w_stride_cin + c_out * w_stride_cout

                        for kd in al.range(3):
                            di_num = do_num + 1 - kd
                            if di_num >= 0:
                                di = di_num // 2
                                di_rem = di_num - di * 2
                                if not di_rem:
                                    if di < D_in:
                                        in_idx_d = c_in_base + di * in_stride_d
                                        w_idx_d = w_cin_base + kd * w_stride_kd

                                        for kh in al.range(3):
                                            hi_num = ho_num + 1 - kh
                                            if hi_num >= 0:
                                                hi = hi_num // 2
                                                hi_rem = hi_num - hi * 2
                                                if not hi_rem:
                                                    if hi < H_in:
                                                        in_idx_h = in_idx_d + hi * in_stride_h
                                                        w_idx_h = w_idx_d + kh * w_stride_kh

                                                        for kw in al.range(3):
                                                            wi_num = wo_num + 1 - kw
                                                            if wi_num >= 0:
                                                                wi = wi_num // 2
                                                                wi_rem = wi_num - wi * 2
                                                                if not wi_rem:
                                                                    if wi < W_in:
                                                                        in_idx = in_idx_h + wi
                                                                        w_idx = w_idx_h + kw
                                                                        in_val = al.convert(x_1d[in_idx], al.f32)
                                                                        w_val = al.convert(w_1d[w_idx], al.f32)
                                                                        acc = acc + in_val * w_val

                    acc = acc * scale_val
                    if acc > max_val:
                        max_val = acc

        mid_1d[mid_offset] = al.convert(max_val, al.bf16)


@avelang.jit
def avgpool_clamp_kernel(
    mid_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C_out: al.i32,
    D_mid: al.i32,
    H_mid: al.i32,
    W_mid: al.i32,
    spatial_size: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    if bid < N * C_out:
        n_idx = bid // C_out
        c_idx = bid - n_idx * C_out

        smem = al.make_shared((BLOCK_SIZE,), al.f32)

        one = al.convert(1, al.i32)
        mid_total = N * C_out * D_mid * H_mid * W_mid
        mid_1d = al.make_tensor(mid_ptr, al.bf16, al.make_layout((mid_total,), (one,)))

        base = n_idx * C_out * spatial_size + c_idx * spatial_size

        local_sum = al.convert(0.0, al.f32)

        for i in al.range(tid, spatial_size, BLOCK_SIZE):
            idx = base + i
            val = al.convert(mid_1d[idx], al.f32)
            local_sum = local_sum + val

        smem[tid] = local_sum
        al.syncthreads()

        if tid < 128:
            smem[tid] = smem[tid] + smem[tid + 128]
        al.syncthreads()
        if tid < 64:
            smem[tid] = smem[tid] + smem[tid + 64]
        al.syncthreads()
        if tid < 32:
            smem[tid] = smem[tid] + smem[tid + 32]
        al.syncthreads()
        if tid < 16:
            smem[tid] = smem[tid] + smem[tid + 16]
        al.syncthreads()
        if tid < 8:
            smem[tid] = smem[tid] + smem[tid + 8]
        al.syncthreads()
        if tid < 4:
            smem[tid] = smem[tid] + smem[tid + 4]
        al.syncthreads()
        if tid < 2:
            smem[tid] = smem[tid] + smem[tid + 2]
        al.syncthreads()
        if tid < 1:
            smem[tid] = smem[tid] + smem[tid + 1]

        if tid == 0:
            total = smem[0]
            count_f = al.convert(spatial_size, al.f32)
            avg = total / count_f

            zero_f = al.convert(0.0, al.f32)
            one_f = al.convert(1.0, al.f32)
            if avg < zero_f:
                avg = zero_f
            if avg > one_f:
                avg = one_f

            out_1d = al.make_tensor(out_ptr, al.bf16, al.make_layout((N * C_out,), (one,)))
            out_1d[bid] = al.convert(avg, al.bf16)


def avelang_fused_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    N, C_in, D_in, H_in, W_in = x.shape
    C_out = weight.shape[1]
    kD, kH, kW = weight.shape[2], weight.shape[3], weight.shape[4]
    conv_stride = 2
    padding = 1

    D_out = (D_in - 1) * conv_stride + kD - 2 * padding
    H_out = (H_in - 1) * conv_stride + kH - 2 * padding
    W_out = (W_in - 1) * conv_stride + kW - 2 * padding

    mp_kernel = 2
    D_mid = D_out // mp_kernel
    H_mid = H_out // mp_kernel
    W_mid = W_out // mp_kernel

    stride_h = W_mid
    stride_d = H_mid * W_mid
    stride_c = D_mid * H_mid * W_mid

    total_cells = N * C_out * stride_c
    num_blocks = (total_cells + BLOCK_SIZE - 1) // BLOCK_SIZE

    x_bf16 = x.contiguous().to(dtype=torch.bfloat16)
    w_bf16 = weight.contiguous().to(dtype=torch.bfloat16)
    b_bf16 = bias.contiguous().to(dtype=torch.bfloat16)

    mid = torch.empty((N, C_out, D_mid, H_mid, W_mid), dtype=torch.bfloat16, device=x.device)
    mid_flat = mid.contiguous().view(-1)

    scale_f32 = float(scale)

    fused_ct_scale_maxpool_kernel[lambda: ((num_blocks, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_bf16, w_bf16, b_bf16, mid_flat,
        N, C_in, C_out, D_in, H_in, W_in,
        D_mid, H_mid, W_mid,
        stride_c, stride_d, stride_h,
        total_cells, scale_f32,
    )

    spatial_size = D_mid * H_mid * W_mid
    out_flat = torch.empty((N * C_out), dtype=torch.bfloat16, device=x.device)

    num_blocks_avg = N * C_out

    avgpool_clamp_kernel[lambda: ((num_blocks_avg, 1, 1), (BLOCK_SIZE, 1, 1))](
        mid_flat, out_flat,
        N, C_out, D_mid, H_mid, W_mid,
        spatial_size,
    )

    return out_flat.view(N, C_out, 1, 1, 1)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale = scale
        self.maxpool_kernel_size = maxpool_kernel_size
        self.clamp_min = 0
        self.clamp_max = 1

    def forward(self, x):
        weight = self.conv_transpose.weight.data
        bias = self.conv_transpose.bias.data if self.conv_transpose.bias is not None else torch.zeros(
            self.conv_transpose.out_channels, device=x.device, dtype=x.dtype
        )
        return avelang_fused_forward(x, weight, bias, self.scale)


batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
scale = 0.5
maxpool_kernel_size = 2


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size]
