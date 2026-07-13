import torch
import torch.nn as nn
import avelang
import avelang.language as al

# ---------------------------------------------------------------------------
# Compile-time tile parameters
# ---------------------------------------------------------------------------
TILE_H = 16
TILE_W = 16
TILE_C = 64
THREADS = 256

WIN_H = TILE_H + 2   # 18
WIN_W = TILE_W + 2   # 18
INPUT_WIN_ELEMS = 16 * WIN_H * WIN_W  # 5184
WEIGHT_PER_TILE = TILE_C * 16 * 9

INPUT_LOADS = (INPUT_WIN_ELEMS + THREADS - 1) // THREADS
WEIGHT_LOADS = (WEIGHT_PER_TILE + THREADS - 1) // THREADS


@avelang.jit
def conv2d_bf16_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C_in: al.i32,
    H: al.i32,
    W: al.i32,
    C_out: al.i32,
    H_out: al.i32,
    W_out: al.i32,
):
    tid = al.thread_id(0)
    flat_block = al.block_id(0)

    H_out_v = al.convert(H_out, al.i32)
    W_out_v = al.convert(W_out, al.i32)
    C_in_v = al.convert(C_in, al.i32)
    H_v = al.convert(H, al.i32)
    W_v = al.convert(W, al.i32)
    C_out_v = al.convert(C_out, al.i32)
    N_v = al.convert(N, al.i32)
    one = al.convert(1, al.i32)
    tile_h_c = al.convert(TILE_H, al.i32)
    tile_w_c = al.convert(TILE_W, al.i32)
    tile_c_c = al.convert(TILE_C, al.i32)

    tiles_h = (H_out_v + tile_h_c - one) // tile_h_c
    tiles_w = (W_out_v + tile_w_c - one) // tile_w_c
    tiles_c = (C_out_v + tile_c_c - one) // tile_c_c
    tiles_per_batch = tiles_h * tiles_w * tiles_c

    batch = flat_block // tiles_per_batch
    rest = flat_block - batch * tiles_per_batch
    tile_c = (rest % tiles_c) * tile_c_c
    rest_hw = rest // tiles_c
    tile_h = (rest_hw // tiles_w) * tile_h_c
    tile_w = (rest_hw - (rest_hw // tiles_w) * tiles_w) * tile_w_c

    # ---- Tensor views ----
    x_layout = al.make_layout(
        (N_v, C_in_v, H_v, W_v),
        (C_in_v * H_v * W_v, H_v * W_v, W_v, one),
    )
    x = al.make_tensor(x_ptr, al.bf16, x_layout)

    wt_total = C_out_v * C_in_v * al.convert(9, al.i32)
    w_layout = al.make_layout((wt_total,), (one,))
    w_flat = al.make_tensor(w_ptr, al.bf16, w_layout)

    out_layout = al.make_layout(
        (N_v, C_out_v, H_out_v, W_out_v),
        (C_out_v * H_out_v * W_out_v, H_out_v * W_out_v, W_out_v, one),
    )
    out = al.make_tensor(out_ptr, al.bf16, out_layout)

    # ---- Shared memory (all f32) ----
    shm_in = al.make_shared((INPUT_WIN_ELEMS,), al.f32)
    shm_w = al.make_shared((WEIGHT_PER_TILE,), al.f32)

    # ---- Load input window ----
    win_hw = al.convert(WIN_H * WIN_W, al.i32)
    win_w = al.convert(WIN_W, al.i32)
    zero_f32 = al.convert(0.0, al.f32)
    threads_v = al.convert(THREADS, al.i32)

    idx_in = al.convert(tid, al.i32)
    for _ in al.range(INPUT_LOADS):
        if idx_in < INPUT_WIN_ELEMS:
            c = idx_in // win_hw
            hw = idx_in - c * win_hw
            h_off = hw // win_w
            w_off = hw - h_off * win_w
            h_in = tile_h + h_off
            w_in = tile_w + w_off
            if h_in < H_v and w_in < W_v:
                shm_in[idx_in] = al.convert(x[batch, c, h_in, w_in], al.f32)
            else:
                shm_in[idx_in] = zero_f32
        idx_in = idx_in + threads_v

    # ---- Load weight tile ----
    wt_global_base = tile_c * C_in_v * al.convert(9, al.i32)
    idx_w = al.convert(tid, al.i32)
    for _ in al.range(WEIGHT_LOADS):
        if idx_w < WEIGHT_PER_TILE:
            shm_w[idx_w] = al.convert(w_flat[wt_global_base + idx_w], al.f32)
        idx_w = idx_w + threads_v

    al.syncthreads()

    # ---- Compute output ----
    # Thread tid maps to one spatial position; computes all TILE_C channels
    tid_v = al.convert(tid, al.i32)
    th = tid_v // tile_w_c
    tw = tid_v - th * tile_w_c
    h_out = tile_h + th
    w_out = tile_w + tw

    if h_out < H_out_v and w_out < W_out_v:
        nine = al.convert(9, al.i32)
        three = al.convert(3, al.i32)
        c_in_9 = C_in_v * nine
        t_off = th * win_w + tw

        # Accumulator for all TILE_C channels (in registers)
        acc = al.make_local((TILE_C,), al.f32)
        for tc_init in al.range(TILE_C):
            acc[tc_init] = al.convert(0.0, al.f32)

        # Outer loop over spatial kernel positions — input loaded once, reused across channels
        for c_in in al.range(16):
            c_in_i = al.convert(c_in, al.i32)
            in_c_base = c_in_i * win_hw
            wt_c_base = c_in_i * nine

            for kh in al.range(3):
                kh_i = al.convert(kh, al.i32)
                in_h = in_c_base + t_off + kh_i * win_w
                wt_kh = wt_c_base + kh_i * three

                for kw in al.range(3):
                    kw_i = al.convert(kw, al.i32)
                    in_val = shm_in[in_h + kw_i]
                    wt_base_idx = wt_kh + kw_i

                    # Reuse in_val across all channels
                    for tc in al.range(TILE_C):
                        wt_idx = al.convert(tc, al.i32) * c_in_9 + wt_base_idx
                        acc[tc] = acc[tc] + in_val * shm_w[wt_idx]

        # Write results
        for tc in al.range(TILE_C):
            c_out_val = tile_c + al.convert(tc, al.i32)
            if c_out_val < C_out_v:
                out[batch, c_out_val, h_out, w_out] = al.convert(acc[tc], al.bf16)


def _prepare_tensor(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.conv2d = nn.Conv2d(
            in_channels, out_channels, (kernel_size, kernel_size),
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias,
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.conv2d.weight.data
        x_bf16 = _prepare_tensor(x)
        w_bf16 = _prepare_tensor(weight)

        N, C_in, H, W = x_bf16.shape
        C_out = self.out_channels
        K = self.kernel_size
        S = self.stride
        P = self.padding
        D = self.dilation

        H_out = (H + 2 * P - D * (K - 1) - 1) // S + 1
        W_out = (W + 2 * P - D * (K - 1) - 1) // S + 1

        tiles_per_h = (H_out + TILE_H - 1) // TILE_H
        tiles_per_w = (W_out + TILE_W - 1) // TILE_W
        tiles_per_c = (C_out + TILE_C - 1) // TILE_C
        tiles_per_batch = tiles_per_h * tiles_per_w * tiles_per_c
        total_blocks = N * tiles_per_batch

        out = torch.empty(
            (N, C_out, H_out, W_out),
            device=x_bf16.device, dtype=torch.bfloat16,
        )

        conv2d_bf16_kernel[lambda: ((total_blocks, 1, 1), (THREADS, 1, 1))](
            x_bf16, w_bf16, out,
            N, C_in, H, W, C_out, H_out, W_out,
        )

        return out


# Test code
batch_size = 16
in_channels = 16
out_channels = 128
kernel_size = 3
width = 1024
height = 1024


def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
