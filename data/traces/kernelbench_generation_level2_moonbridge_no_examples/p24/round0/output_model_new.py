import torch
import torch.nn as nn
import avelang
import avelang.language as al

# ── Conv3D kernel ──────────────────────────────────────────────────────────────
# Flat grid: one block per (n, c_out, d_out) triple.
# Input tile (C_in, kD, H, W) and weight tile (C_in, kD, kH, kW) in shared memory.

@avelang.jit
def conv3d_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
):
    x_layout = al.make_layout((128, 3, 24, 32, 32), (3 * 24 * 32 * 32, 24 * 32 * 32, 32 * 32, 32, 1))
    x = al.make_tensor(x_ptr, al.bf16, x_layout)

    w_layout = al.make_layout((24, 3, 3, 3, 3), (3 * 3 * 3 * 3, 3 * 3 * 3, 3 * 3, 3, 1))
    w = al.make_tensor(w_ptr, al.bf16, w_layout)

    b_layout = al.make_layout((24,), (1,))
    b = al.make_tensor(b_ptr, al.bf16, b_layout)

    out_layout = al.make_layout((128, 24, 22, 30, 30), (24 * 22 * 30 * 30, 22 * 30 * 30, 30 * 30, 30, 1))
    out = al.make_tensor(out_ptr, al.bf16, out_layout)

    block_idx = al.block_id(0)
    tid = al.thread_id(0)
    block_dim = al.block_dim(0)

    n = block_idx // 528
    rem_n = block_idx % 528
    c = rem_n // 22
    d_out = rem_n % 22

    # ── shared memory ──
    in_shared = al.make_shared((3, 3, 32, 32), al.bf16)
    total_in = 3 * 3 * 32 * 32
    for idx in al.range(tid, total_in, block_dim):
        ci = idx // 3072
        rem0 = idx % 3072
        kd_val = rem0 // 1024
        rem1 = rem0 % 1024
        h_in = rem1 // 32
        w_in = rem1 % 32
        in_shared[ci, kd_val, h_in, w_in] = x[n, ci, d_out + kd_val, h_in, w_in]

    w_shared = al.make_shared((3, 3, 3, 3), al.bf16)
    total_w = 81
    for idx in al.range(tid, total_w, block_dim):
        ci = idx // 27
        rem_a = idx % 27
        kd_val = rem_a // 9
        rem_b = rem_a % 9
        kh = rem_b // 3
        kw_val = rem_b % 3
        w_shared[ci, kd_val, kh, kw_val] = w[c, ci, kd_val, kh, kw_val]

    al.syncthreads()

    bias_val = al.convert(b[c], al.f32)
    total_hw = 30 * 30

    for hw_idx in al.range(tid, total_hw, block_dim):
        h_out = hw_idx // 30
        w_out = hw_idx % 30

        acc = al.convert(0.0, al.f32)
        for ci in al.range(3):
            for kd_val in al.range(3):
                for kh in al.range(3):
                    for kw_val in al.range(3):
                        in_val = al.convert(in_shared[ci, kd_val, h_out + kh, w_out + kw_val], al.f32)
                        w_val = al.convert(w_shared[ci, kd_val, kh, kw_val], al.f32)
                        acc = acc + in_val * w_val
        acc = acc + bias_val
        out[n, c, d_out, h_out, w_out] = al.convert(acc, al.bf16)


# ── Min-reduction along dimension 2 (depth) ────────────────────────────────────

@avelang.jit
def min_reduce_dim2_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
):
    in_layout = al.make_layout((128, 24, 22, 30, 30), (24 * 22 * 30 * 30, 22 * 30 * 30, 30 * 30, 30, 1))
    x = al.make_tensor(x_ptr, al.bf16, in_layout)

    out_layout = al.make_layout((128, 24, 30, 30), (24 * 30 * 30, 30 * 30, 30, 1))
    out = al.make_tensor(out_ptr, al.bf16, out_layout)

    tid = al.thread_id(0)
    block_dim_val = al.block_dim(0)
    block_idx = al.block_id(0)

    total = 128 * 24 * 30 * 30
    idx = block_idx * block_dim_val + tid

    if idx < total:
        n = idx // 21600
        rem_n = idx % 21600
        c = rem_n // 900
        rem_c = rem_n % 900
        h = rem_c // 30
        w = rem_c % 30

        min_val = al.convert(x[n, c, 0, h, w], al.f32)
        for d in al.range(1, 22):
            val = al.convert(x[n, c, d, h, w], al.f32)
            if val < min_val:
                min_val = val

        out[n, c, h, w] = al.convert(min_val, al.bf16)


# ── Softmax along dimension 1 (channels) ───────────────────────────────────────

@avelang.jit
def softmax_dim1_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
):
    layout = al.make_layout((128, 24, 30, 30), (24 * 30 * 30, 30 * 30, 30, 1))
    x = al.make_tensor(x_ptr, al.bf16, layout)
    out = al.make_tensor(out_ptr, al.bf16, layout)

    tid = al.thread_id(0)
    block_dim_val = al.block_dim(0)
    block_idx = al.block_id(0)

    total = 128 * 30 * 30
    idx = block_idx * block_dim_val + tid

    if idx < total:
        n = idx // 900
        rem = idx % 900
        h = rem // 30
        w = rem % 30

        max_val = al.convert(x[n, 0, h, w], al.f32)
        for c in al.range(1, 24):
            val = al.convert(x[n, c, h, w], al.f32)
            if val > max_val:
                max_val = val

        sum_val = al.convert(0.0, al.f32)
        for c in al.range(24):
            val = al.convert(x[n, c, h, w], al.f32)
            exp_val = al.exp(val - max_val)
            sum_val = sum_val + exp_val

        for c in al.range(24):
            val = al.convert(x[n, c, h, w], al.f32)
            exp_val = al.exp(val - max_val)
            out[n, c, h, w] = al.convert(exp_val / sum_val, al.bf16)


# ── Host wrappers ──────────────────────────────────────────────────────────────

def avelang_conv3d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    N, C_in, D, H, W = x.shape
    C_out = weight.shape[0]
    kD, kH, kW = weight.shape[2], weight.shape[3], weight.shape[4]
    D_out = D - kD + 1
    H_out = H - kH + 1
    W_out = W - kW + 1
    out = torch.empty((N, C_out, D_out, H_out, W_out), dtype=torch.bfloat16, device=x.device)
    grid = (N * C_out * D_out, 1, 1)
    block = (256, 1, 1)
    conv3d_kernel[lambda: (grid, block)](x, weight, bias, out)
    return out


def avelang_min_reduce_dim2(x: torch.Tensor) -> torch.Tensor:
    N, C, D_red, H, W = x.shape
    out = torch.empty((N, C, H, W), dtype=torch.bfloat16, device=x.device)
    total = N * C * H * W
    block = (256, 1, 1)
    grid = ((total + 255) // 256, 1, 1)
    min_reduce_dim2_kernel[lambda: (grid, block)](x, out)
    return out


def avelang_softmax_dim1(x: torch.Tensor) -> torch.Tensor:
    N, C, H, W = x.shape
    out = torch.empty_like(x)
    total = N * H * W
    block = (256, 1, 1)
    grid = ((total + 255) // 256, 1, 1)
    softmax_dim1_kernel[lambda: (grid, block)](x, out)
    return out


# ── ModelNew ───────────────────────────────────────────────────────────────────

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.dim = dim

    def forward(self, x):
        x_bf16 = x.to(torch.bfloat16).contiguous()
        w_bf16 = self.conv.weight.data.to(torch.bfloat16).contiguous()
        b_bf16 = self.conv.bias.data.to(torch.bfloat16).contiguous()

        conv_out = avelang_conv3d(x_bf16, w_bf16, b_bf16)
        min_out = avelang_min_reduce_dim2(conv_out)
        softmax_out = avelang_softmax_dim1(min_out)
        return softmax_out
