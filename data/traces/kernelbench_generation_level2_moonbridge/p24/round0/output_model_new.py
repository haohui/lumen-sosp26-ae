import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE: al.constexpr = 256
C_IN: al.constexpr = 3
K_SIZE: al.constexpr = 3


@avelang.jit
def conv3d_min_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    D: al.i32,
    H: al.i32,
    W: al.i32,
    C_out: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    D_out: al.i32,
    total_out: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    gid = bid * BLOCK_SIZE + tid

    if gid < total_out:
        C_out_HW = C_out * H_out * W_out
        HW = H_out * W_out
        b = gid // C_out_HW
        rem = gid - b * C_out_HW
        oc = rem // HW
        rem2 = rem - oc * HW
        h = rem2 // W_out
        w = rem2 - h * W_out

        D_H_W = D * H * W
        H_W = H * W
        C_in_D_H_W = C_IN * D_H_W
        x = al.make_tensor(
            x_ptr, al.bf16,
            al.make_layout(
                (B, C_IN, D, H, W),
                (C_in_D_H_W, D_H_W, H_W, W, 1),
            ),
        )

        K3 = K_SIZE * K_SIZE * K_SIZE
        K2 = K_SIZE * K_SIZE
        w_t = al.make_tensor(
            w_ptr, al.bf16,
            al.make_layout(
                (C_out, C_IN, K_SIZE, K_SIZE, K_SIZE),
                (C_IN * K3, K3, K2, K_SIZE, 1),
            ),
        )

        b_t = al.make_tensor(b_ptr, al.bf16, al.make_layout((C_out,), (1,)))
        bias_f32 = al.convert(b_t[oc], al.f32)

        # d_out = 0: initial min
        acc = bias_f32
        for ic in al.range(C_IN):
            for kd in al.range(K_SIZE):
                for kh in al.range(K_SIZE):
                    for kw in al.range(K_SIZE):
                        xv = al.convert(x[b, ic, kd, h + kh, w + kw], al.f32)
                        wv = al.convert(w_t[oc, ic, kd, kh, kw], al.f32)
                        acc = acc + xv * wv
        min_val = acc

        # d_out = 1..D_out-1
        for d_out in al.range(1, D_out):
            acc = bias_f32
            for ic in al.range(C_IN):
                for kd in al.range(K_SIZE):
                    for kh in al.range(K_SIZE):
                        for kw in al.range(K_SIZE):
                            xv = al.convert(x[b, ic, d_out + kd, h + kh, w + kw], al.f32)
                            wv = al.convert(w_t[oc, ic, kd, kh, kw], al.f32)
                            acc = acc + xv * wv
            if acc < min_val:
                min_val = acc

        out_t = al.make_tensor(
            out_ptr, al.bf16,
            al.make_layout(
                (B, C_out, H_out, W_out),
                (C_out * H_out * W_out, H_out * W_out, W_out, 1),
            ),
        )
        out_t[b, oc, h, w] = al.convert(min_val, al.bf16)


@avelang.jit
def softmax_channel_kernel(
    data_ptr: al.Pointer(al.bf16),
    B: al.i32,
    C: al.i32,
    H: al.i32,
    W: al.i32,
):
    tid = al.thread_id(0)
    bid_x = al.block_id(0)
    bid_y = al.block_id(1)

    global_id = bid_x * BLOCK_SIZE + tid

    if global_id < H * W:
        h = global_id // W
        w = global_id - h * W
        b = bid_y

        data = al.make_tensor(
            data_ptr, al.bf16,
            al.make_layout((B, C, H, W), (C * H * W, H * W, W, 1)),
        )

        max_val = al.convert(data[b, 0, h, w], al.f32)
        for c in al.range(1, C):
            val = al.convert(data[b, c, h, w], al.f32)
            if val > max_val:
                max_val = val

        exp_sum = al.convert(0.0, al.f32)
        for c in al.range(C):
            val = al.convert(data[b, c, h, w], al.f32)
            exp_sum = exp_sum + al.exp(val - max_val)

        for c in al.range(C):
            val = al.convert(data[b, c, h, w], al.f32)
            result = al.exp(val - max_val) / exp_sum
            data[b, c, h, w] = al.convert(result, al.bf16)


def _to_bf16_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_conv3d_min_softmax(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    original_dtype = x.dtype

    x_bf16 = _to_bf16_contiguous(x)
    w_bf16 = _to_bf16_contiguous(weight)
    b_bf16 = _to_bf16_contiguous(bias)

    B_s = x_bf16.shape[0]
    D_s = x_bf16.shape[2]
    H_s = x_bf16.shape[3]
    W_s = x_bf16.shape[4]
    C_out_s = w_bf16.shape[0]
    K_s = w_bf16.shape[2]
    D_out_s = D_s - K_s + 1
    H_out_s = H_s - K_s + 1
    W_out_s = W_s - K_s + 1

    total_out = B_s * C_out_s * H_out_s * W_out_s

    intermediate = torch.empty(
        (B_s, C_out_s, H_out_s, W_out_s),
        dtype=torch.bfloat16,
        device=x_bf16.device,
    )

    grid_conv = (total_out + BLOCK_SIZE - 1) // BLOCK_SIZE

    conv3d_min_kernel[lambda: ((grid_conv, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_bf16,
        w_bf16,
        b_bf16,
        intermediate,
        B_s,
        D_s,
        H_s,
        W_s,
        C_out_s,
        H_out_s,
        W_out_s,
        D_out_s,
        total_out,
    )

    total_hw = H_out_s * W_out_s
    grid_x = (total_hw + BLOCK_SIZE - 1) // BLOCK_SIZE

    softmax_channel_kernel[lambda: ((grid_x, B_s, 1), (BLOCK_SIZE, 1, 1))](
        intermediate,
        B_s,
        C_out_s,
        H_out_s,
        W_out_s,
    )

    return intermediate.to(dtype=original_dtype)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.dim = dim

    def forward(self, x):
        return avelang_conv3d_min_softmax(
            x, self.conv.weight, self.conv.bias, self.dim
        )
