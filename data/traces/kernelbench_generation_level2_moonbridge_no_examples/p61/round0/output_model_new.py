import torch
import torch.nn as nn
import avelang
import avelang.language as al


# ============================================================
# Kernel 1: ConvTranspose3d + ReLU (fused)
# ============================================================

@avelang.jit
def conv_transpose3d_relu_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    total_elements: al.i32,
    STRIDE_N: al.constexpr,
    STRIDE_C: al.constexpr,
    STRIDE_D: al.constexpr,
    STRIDE_H: al.constexpr,
    X_STRIDE_N: al.constexpr,
    X_STRIDE_C: al.constexpr,
    X_STRIDE_D: al.constexpr,
    X_STRIDE_H: al.constexpr,
    W_STRIDE_0: al.constexpr,
    W_STRIDE_1: al.constexpr,
    W_STRIDE_2: al.constexpr,
    W_STRIDE_3: al.constexpr,
    N: al.constexpr,
    C_in: al.constexpr,
    C_out: al.constexpr,
    D_in: al.constexpr,
    H_in: al.constexpr,
    W_in: al.constexpr,
    D_out: al.constexpr,
    H_out: al.constexpr,
    W_out: al.constexpr,
    K: al.constexpr,
    pad: al.constexpr,
):
    """Per-element ConvTranspose3d + ReLU with constexpr strides/dims."""

    idx = al.thread_id(0) + al.block_id(0) * al.block_dim(0)

    if idx < total_elements:
        n = idx // STRIDE_N
        rem = idx % STRIDE_N
        oc = rem // STRIDE_C
        rem = rem % STRIDE_C
        d_out = rem // STRIDE_D
        rem = rem % STRIDE_D
        h_out = rem // STRIDE_H
        w_out = rem % STRIDE_H

        x_layout = al.make_layout(
            (N, C_in, D_in, H_in, W_in),
            (X_STRIDE_N, X_STRIDE_C, X_STRIDE_D, X_STRIDE_H, 1),
        )
        x = al.make_tensor(x_ptr, al.bf16, x_layout)

        w_layout = al.make_layout(
            (C_in, C_out, K, K, K),
            (W_STRIDE_0, W_STRIDE_1, W_STRIDE_2, W_STRIDE_3, 1),
        )
        w = al.make_tensor(w_ptr, al.bf16, w_layout)

        out_layout = al.make_layout(
            (N, C_out, D_out, H_out, W_out),
            (STRIDE_N, STRIDE_C, STRIDE_D, STRIDE_H, 1),
        )
        out = al.make_tensor(out_ptr, al.bf16, out_layout)

        acc = al.convert(0.0, al.f32)

        for ic in al.range(C_in):
            for kd in al.range(K):
                d_in = d_out + pad - kd
                if d_in >= 0 and d_in < D_in:
                    for kh in al.range(K):
                        h_in = h_out + pad - kh
                        if h_in >= 0 and h_in < H_in:
                            for kw in al.range(K):
                                w_in = w_out + pad - kw
                                if w_in >= 0 and w_in < W_in:
                                    x_val = al.convert(x[n, ic, d_in, h_in, w_in], al.f32)
                                    w_val = al.convert(w[ic, oc, kd, kh, kw], al.f32)
                                    acc = acc + x_val * w_val

        zero = al.convert(0.0, al.f32)
        if acc < zero:
            acc = zero

        out[n, oc, d_out, h_out, w_out] = al.convert(acc, al.bf16)


# ============================================================
# Kernel 2: GroupNorm (training mode, with affine)
# ============================================================

@avelang.jit
def group_norm_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    gamma_ptr: al.Pointer(al.bf16),
    beta_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C: al.i32,
    D: al.i32,
    H: al.i32,
    W: al.i32,
    num_groups: al.i32,
    BLOCK_SIZE: al.constexpr,
):
    """GroupNorm training mode. One block per (sample, group)."""

    sample_idx = al.block_id(0) // num_groups
    group_idx = al.block_id(0) % num_groups
    tid = al.thread_id(0)

    C_per_group = C // num_groups
    spatial_size = D * H * W
    group_size = C_per_group * spatial_size
    c_start = group_idx * C_per_group

    stride_n = C * D * H * W
    stride_c = D * H * W
    stride_d = H * W
    stride_h = W
    layout_5d = al.make_layout(
        (N, C, D, H, W),
        (stride_n, stride_c, stride_d, stride_h, 1),
    )
    x = al.make_tensor(x_ptr, al.bf16, layout_5d)
    out = al.make_tensor(out_ptr, al.bf16, layout_5d)

    gamma_layout = al.make_layout((C,), (1,))
    gamma = al.make_tensor(gamma_ptr, al.bf16, gamma_layout)
    beta = al.make_tensor(beta_ptr, al.bf16, gamma_layout)

    sum_val = al.convert(0.0, al.f32)
    sum_sq = al.convert(0.0, al.f32)

    num_iters = (group_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    for iter_idx in al.range(num_iters):
        i = tid + iter_idx * BLOCK_SIZE
        if i < group_size:
            c_local = i // spatial_size
            rem = i % spatial_size
            dpos = rem // stride_d
            rem2 = rem % stride_d
            hpos = rem2 // stride_h
            wpos = rem2 % stride_h
            c = c_start + c_local

            val = al.convert(x[sample_idx, c, dpos, hpos, wpos], al.f32)
            sum_val = sum_val + val
            sum_sq = sum_sq + val * val

    smem_sum = al.make_shared((BLOCK_SIZE,), al.f32)
    smem_sq = al.make_shared((BLOCK_SIZE,), al.f32)
    smem_sum[tid] = sum_val
    smem_sq[tid] = sum_sq
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
        smem_sum[0] = smem_sum[0] + smem_sum[1]
        smem_sq[0] = smem_sq[0] + smem_sq[1]
    al.syncthreads()

    total_sum = smem_sum[0]
    total_sq = smem_sq[0]

    count_f = al.convert(group_size, al.f32)
    mean = total_sum / count_f
    variance = total_sq / count_f - mean * mean

    eps_val = al.convert(1.0e-5, al.f32)
    inv_std = al.convert(1.0, al.f32) / al.sqrt(variance + eps_val)

    for iter_idx in al.range(num_iters):
        i = tid + iter_idx * BLOCK_SIZE
        if i < group_size:
            c_local = i // spatial_size
            rem = i % spatial_size
            dpos = rem // stride_d
            rem2 = rem % stride_d
            hpos = rem2 // stride_h
            wpos = rem2 % stride_h
            c = c_start + c_local

            val = al.convert(x[sample_idx, c, dpos, hpos, wpos], al.f32)
            norm_val = (val - mean) * inv_std

            g = al.convert(gamma[c], al.f32)
            b = al.convert(beta[c], al.f32)
            result = norm_val * g + b

            out[sample_idx, c, dpos, hpos, wpos] = al.convert(result, al.bf16)


# ============================================================
# Host-side launch helpers
# ============================================================

def _run_conv_transpose_relu(x_bf16, w_bf16, out_bf16, pad):
    """Launch ConvTranspose3d + ReLU kernel with constexpr strides."""
    N, C_in, D_in, H_in, W_in = x_bf16.shape
    C_out = out_bf16.shape[1]
    D_out = out_bf16.shape[2]
    H_out = out_bf16.shape[3]
    W_out = out_bf16.shape[4]
    K = w_bf16.shape[2]

    STRIDE_N = C_out * D_out * H_out * W_out
    STRIDE_C = D_out * H_out * W_out
    STRIDE_D = H_out * W_out
    STRIDE_H = W_out

    X_STRIDE_N = C_in * D_in * H_in * W_in
    X_STRIDE_C = D_in * H_in * W_in
    X_STRIDE_D = H_in * W_in
    X_STRIDE_H = W_in

    W_STRIDE_0 = C_out * K * K * K
    W_STRIDE_1 = K * K * K
    W_STRIDE_2 = K * K
    W_STRIDE_3 = K

    total_elements = N * C_out * D_out * H_out * W_out
    BLOCK_SIZE = 256
    grid = ((total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE, 1, 1)

    conv_transpose3d_relu_kernel[lambda: (grid, (BLOCK_SIZE, 1, 1))](
        x_bf16.data_ptr(),
        w_bf16.data_ptr(),
        out_bf16.data_ptr(),
        total_elements,
        STRIDE_N, STRIDE_C, STRIDE_D, STRIDE_H,
        X_STRIDE_N, X_STRIDE_C, X_STRIDE_D, X_STRIDE_H,
        W_STRIDE_0, W_STRIDE_1, W_STRIDE_2, W_STRIDE_3,
        N, C_in, C_out, D_in, H_in, W_in, D_out, H_out, W_out, K, pad,
    )


def _run_group_norm(x_bf16, gamma_bf16, beta_bf16, out_bf16, num_groups):
    """Launch GroupNorm kernel."""
    N, C, D, H, W = x_bf16.shape
    BLOCK_SIZE = 256
    grid = (N * num_groups, 1, 1)

    group_norm_kernel[lambda: (grid, (BLOCK_SIZE, 1, 1))](
        x_bf16.data_ptr(),
        out_bf16.data_ptr(),
        gamma_bf16.data_ptr(),
        beta_bf16.data_ptr(),
        N, C, D, H, W, num_groups,
        BLOCK_SIZE,
    )


# ============================================================
# ModelNew
# ============================================================

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, bias=bias
        )
        self.group_norm = nn.GroupNorm(
            num_groups=groups, num_channels=out_channels
        )

    def forward(self, x):
        x_bf16 = x.to(torch.bfloat16).contiguous()
        w_conv = self.conv_transpose.weight.data.to(torch.bfloat16).contiguous()

        N, C_in, D_in, H_in, W_in = x_bf16.shape
        C_out = w_conv.shape[1]
        K = w_conv.shape[2]
        pad = 0

        D_out = D_in + K - 1
        H_out = H_in + K - 1
        W_out = W_in + K - 1

        conv_out = torch.empty(
            N, C_out, D_out, H_out, W_out,
            dtype=torch.bfloat16, device=x.device,
        )
        _run_conv_transpose_relu(x_bf16, w_conv, conv_out, pad)

        gamma = self.group_norm.weight.data.to(torch.bfloat16).contiguous()
        beta = self.group_norm.bias.data.to(torch.bfloat16).contiguous()
        num_groups = self.group_norm.num_groups

        gn_out = torch.empty_like(conv_out)
        _run_group_norm(conv_out, gamma, beta, gn_out, num_groups)

        return gn_out
