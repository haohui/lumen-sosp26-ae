import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_M = 16
BLOCK_N = 16
TILE_K = 16


@avelang.jit
def depthwise_conv_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    C: al.i32,
    H: al.i32,
    W: al.i32,
    KH: al.i32,
    KW: al.i32,
    stride: al.i32,
    padding: al.i32,
    dilation: al.i32,
    H_out: al.i32,
    W_out: al.i32,
):
    x_layout = al.make_layout((B, C, H, W), (C * H * W, H * W, W, 1))
    x = al.make_tensor(x_ptr, al.bf16, x_layout)

    w_layout = al.make_layout((C, 1, KH, KW), (KH * KW, KH * KW, KW, 1))
    w = al.make_tensor(w_ptr, al.bf16, w_layout)

    out_layout = al.make_layout((B, C, H_out, W_out), (C * H_out * W_out, H_out * W_out, W_out, 1))
    out = al.make_tensor(out_ptr, al.bf16, out_layout)

    c = al.block_id(0) * al.block_dim(0) + al.thread_id(0)
    h_out = al.block_id(1) * al.block_dim(1) + al.thread_id(1)
    w_out = al.block_id(2) * al.block_dim(2) + al.thread_id(2)

    if c < C:
        if h_out < H_out:
            if w_out < W_out:
                for b in al.range(B):
                    acc = al.convert(0.0, al.f32)
                    for kh in al.range(KH):
                        for kw in al.range(KW):
                            h_in = h_out * stride + kh * dilation - padding
                            w_in = w_out * stride + kw * dilation - padding
                            if h_in >= 0:
                                if h_in < H:
                                    if w_in >= 0:
                                        if w_in < W:
                                            x_val = al.convert(x[b, c, h_in, w_in], al.f32)
                                            w_val = al.convert(w[c, 0, kh, kw], al.f32)
                                            acc = acc + x_val * w_val
                    out[b, c, h_out, w_out] = al.convert(acc, al.bf16)


@avelang.jit
def pointwise_conv_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    M: al.i32,
    K: al.i32,
    N: al.i32,
    H_dim: al.i32,
    W_dim: al.i32,
):
    HW = H_dim * W_dim
    B_dim = M // HW

    # A = input (B, K, H, W)
    a_layout = al.make_layout((B_dim, K, H_dim, W_dim), (K * HW, HW, W_dim, 1))
    a = al.make_tensor(x_ptr, al.bf16, a_layout)

    # B = weight (N, K, 1, 1)
    b_layout = al.make_layout((N, K, 1, 1), (K, 1, 1, 1))
    b = al.make_tensor(w_ptr, al.bf16, b_layout)

    # C = output (B, N, H, W)
    c_layout = al.make_layout((B_dim, N, H_dim, W_dim), (N * HW, HW, W_dim, 1))
    c_out = al.make_tensor(out_ptr, al.bf16, c_layout)

    block_m = al.block_id(0)
    block_n = al.block_id(1)
    tid_m = al.thread_id(0)
    tid_n = al.thread_id(1)

    m = block_m * BLOCK_M + tid_m
    n = block_n * BLOCK_N + tid_n

    a_tile = al.make_shared((BLOCK_M, TILE_K), al.bf16)
    b_tile = al.make_shared((TILE_K, BLOCK_N), al.bf16)

    acc = al.convert(0.0, al.f32)

    for k_block in al.range(0, K, TILE_K):
        # Load a_tile: thread (tid_m, tid_n) loads from global at row block_m*BLOCK_M+tid_m, col k_block+tid_n
        a_m = block_m * BLOCK_M + tid_m
        a_k = k_block + tid_n
        if a_m < M:
            if a_k < K:
                a_b = a_m // HW
                a_r = a_m - a_b * HW
                a_h = a_r // W_dim
                a_w = a_r - a_h * W_dim
                a_tile[tid_m, tid_n] = a[a_b, a_k, a_h, a_w]
            else:
                a_tile[tid_m, tid_n] = al.convert(0.0, al.bf16)
        else:
            a_tile[tid_m, tid_n] = al.convert(0.0, al.bf16)

        # Load b_tile: only threads with tid_m < TILE_K participate
        if tid_m < TILE_K:
            b_k = k_block + tid_m
            b_n = block_n * BLOCK_N + tid_n
            if b_k < K:
                if b_n < N:
                    b_tile[tid_m, tid_n] = b[b_n, b_k, 0, 0]
                else:
                    b_tile[tid_m, tid_n] = al.convert(0.0, al.bf16)
            else:
                b_tile[tid_m, tid_n] = al.convert(0.0, al.bf16)

        al.syncthreads()

        for k_off in al.range(TILE_K):
            a_val = al.convert(a_tile[tid_m, k_off], al.f32)
            b_val = al.convert(b_tile[k_off, tid_n], al.f32)
            acc = acc + a_val * b_val

        al.syncthreads()

    if m < M:
        if n < N:
            s_b = m // HW
            s_r = m - s_b * HW
            s_h = s_r // W_dim
            s_w = s_r - s_h * W_dim
            c_out[s_b, n, s_h, s_w] = al.convert(acc, al.bf16)


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
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=in_channels, bias=bias,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        KH = self.kernel_size
        KW = self.kernel_size

        H_out = (H + 2 * self.padding - self.dilation * (KH - 1) - 1) // self.stride + 1
        W_out = (W + 2 * self.padding - self.dilation * (KW - 1) - 1) // self.stride + 1

        dw_w = self.depthwise.weight.data
        pw_w = self.pointwise.weight.data

        x = x.contiguous()
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)
        if dw_w.dtype != torch.bfloat16:
            dw_w = dw_w.to(torch.bfloat16)
        if pw_w.dtype != torch.bfloat16:
            pw_w = pw_w.to(torch.bfloat16)

        # Depthwise: 3D grid
        dw_out = torch.empty(B, C, H_out, W_out, dtype=torch.bfloat16, device=x.device)
        BLOCK_H = 16
        BLOCK_W = 16
        grid_h = (H_out + BLOCK_H - 1) // BLOCK_H
        grid_w = (W_out + BLOCK_W - 1) // BLOCK_W
        depthwise_conv_kernel[lambda: ((C, grid_h, grid_w), (1, BLOCK_H, BLOCK_W))](
            x, dw_w, dw_out,
            B, C, H, W, KH, KW,
            self.stride, self.padding, self.dilation,
            H_out, W_out,
        )

        # Pointwise: tiled GEMM with shared memory
        out = torch.empty(B, self.out_channels, H_out, W_out, dtype=torch.bfloat16, device=x.device)
        M = B * H_out * W_out
        K = C
        N_val = self.out_channels
        grid_m = (M + BLOCK_M - 1) // BLOCK_M
        grid_n = (N_val + BLOCK_N - 1) // BLOCK_N
        pointwise_conv_kernel[lambda: ((grid_m, grid_n, 1), (BLOCK_M, BLOCK_N, 1))](
            dw_out, pw_w, out,
            M, K, N_val, H_out, W_out,
        )

        return out
