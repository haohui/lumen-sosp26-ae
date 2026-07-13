import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE: al.constexpr = 256


@avelang.jit
def conv3d_bf16_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C_in: al.i32,
    C_out: al.i32,
    D: al.i32,
    H: al.i32,
    W: al.i32,
    kD: al.i32,
    kH: al.i32,
    kW: al.i32,
    total_elems: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)
    BLK = al.convert(256, al.i32)
    gid = bid * BLK + tid

    one = al.convert(1, al.i32)
    stride_xn = C_in * D * H * W
    stride_xc = D * H * W
    stride_xd = H * W
    stride_xh = W
    layout_in = al.make_layout(
        (N, C_in, D, H, W),
        (stride_xn, stride_xc, stride_xd, stride_xh, one),
    )
    x = al.make_tensor(x_ptr, al.bf16, layout_in)

    stride_wn = C_in * kD * kH * kW
    stride_wc = kD * kH * kW
    stride_wd = kH * kW
    stride_wh = kW
    layout_w = al.make_layout(
        (C_out, C_in, kD, kH, kW),
        (stride_wn, stride_wc, stride_wd, stride_wh, one),
    )
    w = al.make_tensor(w_ptr, al.bf16, layout_w)

    layout_b = al.make_layout((C_out,), (one,))
    b = al.make_tensor(b_ptr, al.bf16, layout_b)

    stride_out_n = C_out * D * H * W
    stride_out_c = D * H * W
    stride_out_d = H * W
    stride_out_h = W
    layout_out = al.make_layout(
        (N, C_out, D, H, W),
        (stride_out_n, stride_out_c, stride_out_d, stride_out_h, one),
    )
    out = al.make_tensor(out_ptr, al.bf16, layout_out)

    if gid < total_elems:
        stride_n = C_out * D * H * W
        stride_c = D * H * W
        stride_d = H * W
        stride_h = W

        n = gid // stride_n
        rem = gid - n * stride_n
        co = rem // stride_c
        rem = rem - co * stride_c
        d_out = rem // stride_d
        rem = rem - d_out * stride_d
        h_out = rem // stride_h
        w_out = rem - h_out * stride_h

        acc = al.convert(0.0, al.f32)

        for ci in al.range(C_in):
            for kd_idx in al.range(kD):
                in_d = d_out + kd_idx
                for kh_idx in al.range(kH):
                    in_h = h_out + kh_idx
                    for kw_idx in al.range(kW):
                        in_w = w_out + kw_idx
                        x_val = al.convert(x[n, ci, in_d, in_h, in_w], al.f32)
                        w_val = al.convert(w[co, ci, kd_idx, kh_idx, kw_idx], al.f32)
                        acc = acc + x_val * w_val

        b_val = al.convert(b[co], al.f32)
        acc = acc + b_val
        out[n, co, d_out, h_out, w_out] = al.convert(acc, al.bf16)


@avelang.jit
def groupnorm_reduce_bf16_kernel(
    x_ptr: al.Pointer(al.bf16),
    partial_sum_ptr: al.Pointer(al.f32),
    partial_sq_ptr: al.Pointer(al.f32),
    N: al.i32,
    C: al.i32,
    D: al.i32,
    H: al.i32,
    W: al.i32,
    num_groups: al.i32,
    C_per_group: al.i32,
    elems_per_group: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    n = bid // num_groups
    g = bid - n * num_groups

    if n < N:
        smem_sum = al.make_shared((256,), al.f32)
        smem_sq = al.make_shared((256,), al.f32)

        stride_in_n = C * D * H * W
        stride_in_c = D * H * W
        stride_in_d = H * W
        stride_in_h = W
        one = al.convert(1, al.i32)
        layout_in = al.make_layout(
            (N, C, D, H, W),
            (stride_in_n, stride_in_c, stride_in_d, stride_in_h, one),
        )
        x = al.make_tensor(x_ptr, al.bf16, layout_in)

        local_sum = al.convert(0.0, al.f32)
        local_sq = al.convert(0.0, al.f32)

        BLK = al.convert(256, al.i32)
        sp = D * H * W

        for idx in al.range(tid, elems_per_group, BLK):
            cg = idx // sp
            rem = idx - cg * sp
            d_idx = rem // (H * W)
            rem = rem - d_idx * (H * W)
            h_idx = rem // W
            w_idx = rem - h_idx * W

            c_global = g * C_per_group + cg
            val = al.convert(x[n, c_global, d_idx, h_idx, w_idx], al.f32)
            local_sum = local_sum + val
            local_sq = local_sq + val * val

        smem_sum[tid] = local_sum
        smem_sq[tid] = local_sq
        al.syncthreads()

        if tid < 128:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 128]
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 128]
        al.syncthreads()
        if tid < 64:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 64]
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 64]
        al.syncthreads()
        if tid < 32:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 32]
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 32]
        al.syncthreads()
        if tid < 16:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 16]
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 16]
        al.syncthreads()
        if tid < 8:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 8]
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 8]
        al.syncthreads()
        if tid < 4:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 4]
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 4]
        al.syncthreads()
        if tid < 2:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 2]
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 2]
        al.syncthreads()
        if tid < 1:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 1]
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 1]

        if tid == 0:
            layout_ps = al.make_layout((N, num_groups), (num_groups, one))
            ps = al.make_tensor(partial_sum_ptr, al.f32, layout_ps)
            psq = al.make_tensor(partial_sq_ptr, al.f32, layout_ps)
            ps[n, g] = smem_sum[0]
            psq[n, g] = smem_sq[0]


@avelang.jit
def groupnorm_apply_bf16_kernel(
    x_ptr: al.Pointer(al.bf16),
    partial_sum_ptr: al.Pointer(al.f32),
    partial_sq_ptr: al.Pointer(al.f32),
    weight_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C: al.i32,
    D: al.i32,
    H: al.i32,
    W: al.i32,
    num_groups: al.i32,
    C_per_group: al.i32,
    elems_per_group: al.i32,
    eps: al.f32,
    min_val: al.f32,
    max_val: al.f32,
    total_elems: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)
    BLK = al.convert(256, al.i32)
    gid = bid * BLK + tid

    if gid < total_elems:
        stride_n = C * D * H * W
        stride_c = D * H * W
        stride_d = H * W
        stride_h = W

        n = gid // stride_n
        rem = gid - n * stride_n
        c = rem // stride_c
        rem = rem - c * stride_c
        d_out = rem // stride_d
        rem = rem - d_out * stride_d
        h_out = rem // stride_h
        w_out = rem - h_out * stride_h

        one = al.convert(1, al.i32)

        layout_in = al.make_layout(
            (N, C, D, H, W),
            (stride_n, stride_c, stride_d, stride_h, one),
        )
        x = al.make_tensor(x_ptr, al.bf16, layout_in)

        layout_ps = al.make_layout((N, num_groups), (num_groups, one))
        ps = al.make_tensor(partial_sum_ptr, al.f32, layout_ps)
        psq = al.make_tensor(partial_sq_ptr, al.f32, layout_ps)

        layout_stats = al.make_layout((C,), (one,))
        wt = al.make_tensor(weight_ptr, al.bf16, layout_stats)
        bt = al.make_tensor(bias_ptr, al.bf16, layout_stats)

        layout_out = al.make_layout(
            (N, C, D, H, W),
            (stride_n, stride_c, stride_d, stride_h, one),
        )
        out = al.make_tensor(out_ptr, al.bf16, layout_out)

        g = c // C_per_group
        group_sum = ps[n, g]
        group_sq = psq[n, g]

        elems_f32 = al.convert(elems_per_group, al.f32)
        mean = group_sum / elems_f32
        var = group_sq / elems_f32 - mean * mean
        rstd = al.convert(1.0, al.f32) / al.sqrt(var + eps)

        x_val = al.convert(x[n, c, d_out, h_out, w_out], al.f32)
        w_val = al.convert(wt[c], al.f32)
        b_val = al.convert(bt[c], al.f32)

        result = (x_val - mean) * rstd
        result = result * w_val + b_val

        if result > min_val:
            result = min_val

        if result < min_val:
            result = min_val
        if result > max_val:
            result = max_val

        out[n, c, d_out, h_out, w_out] = al.convert(result, al.bf16)


def _to_bf16_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    return t.contiguous().to(dtype=torch.bfloat16)


def avelang_conv3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    assert x.is_cuda, "Input must be on CUDA/HIP device."
    N, C_in, D, H, W = x.shape
    C_out = weight.shape[0]
    kD, kH, kW = weight.shape[2], weight.shape[3], weight.shape[4]
    pad = 0

    x_bf16 = _to_bf16_contiguous(x)
    w_bf16 = _to_bf16_contiguous(weight)

    if bias is not None:
        b_bf16 = _to_bf16_contiguous(bias)
    else:
        b_bf16 = torch.zeros(C_out, device=x.device, dtype=torch.bfloat16)

    D_out = D + 2 * pad - kD + 1
    H_out = H + 2 * pad - kH + 1
    W_out = W + 2 * pad - kW + 1

    total_elems = N * C_out * D_out * H_out * W_out
    grid = (total_elems + 256 - 1) // 256

    out = torch.empty((N, C_out, D_out, H_out, W_out), device=x.device, dtype=torch.bfloat16)

    conv3d_bf16_kernel[lambda: ((grid, 1, 1), (256, 1, 1))](
        x_bf16, w_bf16, b_bf16, out,
        N, C_in, C_out, D, H, W,
        kD, kH, kW,
        total_elems,
    )
    return out


def avelang_groupnorm_min_clamp(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_groups: int,
    min_value: float,
    max_value: float,
) -> torch.Tensor:
    assert x.is_cuda, "Input must be on CUDA/HIP device."
    N, C, D, H, W = x.shape
    C_per_group = C // num_groups
    elems_per_group = C_per_group * D * H * W

    x_bf16 = _to_bf16_contiguous(x)
    w_bf16 = _to_bf16_contiguous(weight)
    b_bf16 = _to_bf16_contiguous(bias)

    eps = 1e-5
    min_val = float(min_value)
    max_val = float(max_value)

    partial_sum = torch.empty((N, num_groups), dtype=torch.float32, device=x.device)
    partial_sq = torch.empty((N, num_groups), dtype=torch.float32, device=x.device)

    reduce_grid = N * num_groups
    groupnorm_reduce_bf16_kernel[lambda: ((reduce_grid, 1, 1), (256, 1, 1))](
        x_bf16, partial_sum, partial_sq,
        N, C, D, H, W, num_groups, C_per_group, elems_per_group,
    )

    total_elems = N * C * D * H * W
    apply_grid = (total_elems + 256 - 1) // 256

    out = torch.empty_like(x_bf16)

    groupnorm_apply_bf16_kernel[lambda: ((apply_grid, 1, 1), (256, 1, 1))](
        x_bf16, partial_sum, partial_sq, w_bf16, b_bf16, out,
        N, C, D, H, W, num_groups, C_per_group, elems_per_group,
        eps, min_val, max_val, total_elems,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, min_value, max_value, dropout_p):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.min_value = min_value
        self.max_value = max_value
        self.groups = groups
        _ = dropout_p

    def forward(self, x):
        conv_out = avelang_conv3d(x, self.conv.weight, self.conv.bias)
        out = avelang_groupnorm_min_clamp(
            conv_out,
            self.norm.weight,
            self.norm.bias,
            self.groups,
            self.min_value,
            self.max_value,
        )
        return out


batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 64, 64
kernel_size = 3
groups = 8
min_value = 0.0
max_value = 1.0
dropout_p = 0.2


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, groups, min_value, max_value, dropout_p]
