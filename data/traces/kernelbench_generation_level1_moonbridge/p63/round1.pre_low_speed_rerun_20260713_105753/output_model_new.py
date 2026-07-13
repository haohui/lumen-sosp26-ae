import torch
import torch.nn as nn
import avelang
import avelang.language as al

# ── Compile-time geometry (tuned for the reference eval config) ──────
THREADS       = 256
TILE_H        = 16
TILE_W        = 16
KR            = 3                         # kernel rows
KS            = 3                         # kernel cols
C_IN          = 16                        # in_channels
K_OUT         = 128                       # out_channels
K_RED         = C_IN * KR * KS            # K reduction dim = 144
TIH           = TILE_H + KR - 1           # input tile height = 18
TIW           = TILE_W + KS - 1           # input tile width  = 18
TIW_HALF      = TIW // 2                  # 9
TIH_TIW_HALF  = TIH * TIW_HALF            # 162
IN_U32        = C_IN * TIH * TIW_HALF     # 2592
W_U32         = K_OUT * K_RED // 2        # 9216
W_ROWS        = W_U32 // 4                # 2304
TILE_OUT      = TILE_H * TILE_W * K_OUT   # 32768
BF16B         = 2


# ── AveLang kernels ─────────────────────────────────────────────────

@avelang.jit
def _load_input_to_shm(
    shm_u32: al.Tensor((IN_U32,), al.u32),
    x_ptr: al.Pointer(al.bf16),
    N: al.u32,
    H: al.u32,
    W: al.u32,
    n_idx: al.u32,
    h_tile: al.u32,
    w_tile: al.u32,
    tid: al.u32,
):
    cu = al.convert(C_IN, al.u32)
    tu = al.convert(IN_U32, al.u32)
    two = al.convert(2, al.u32)
    one = al.convert(1, al.u32)
    zero = al.convert(0, al.u32)
    tv = al.convert(THREADS, al.u32)
    bf = al.convert(BF16B, al.u32)
    tihx = al.convert(TIH_TIW_HALF, al.u32)
    tiwh = al.convert(TIW_HALF, al.u32)

    flat_len = N * cu * H * W
    x_flat = al.make_tensor(x_ptr, al.bf16, al.make_layout((flat_len,), (1,)))
    x_rsrc = al.amdgpu.make_rsrc(x_flat, flat_len * BF16B)

    idx = tid
    for _ in al.range((IN_U32 + THREADS - 1) // THREADS):
        if idx < tu:
            c = idx // tihx
            hw = idx % tihx
            h = hw // tiwh
            we = (hw % tiwh) * two

            load_h = h_tile + h
            if load_h >= H:
                load_h = H - one
            load_w = w_tile + we
            if load_w >= W:
                load_w = W - one

            off = ((n_idx * cu + c) * H + load_h) * W + load_w
            shm_u32[idx] = al.amdgpu.raw_buffer_load_x1(x_rsrc, zero, off * bf, 0)
        idx = idx + tv


@avelang.jit
def _load_weight_to_shm(
    shm_u32: al.Tensor((W_ROWS, 4), al.u32),
    w_ptr: al.Pointer(al.bf16),
    tid: al.u32,
):
    zero = al.convert(0, al.u32)
    bf = al.convert(BF16B, al.u32)
    tv = al.convert(THREADS, al.u32)
    row_limit = al.convert(W_ROWS, al.u32)

    w_flat = al.make_tensor(w_ptr, al.bf16,
        al.make_layout((K_OUT * K_RED,), (1,)))
    w_rsrc = al.amdgpu.make_rsrc(w_flat, K_OUT * K_RED * BF16B)

    row = tid
    for _ in al.range((W_ROWS + THREADS - 1) // THREADS):
        if row < row_limit:
            off = row * al.convert(16, al.u32)
            shm_u32[row] = al.amdgpu.raw_buffer_load_x4(w_rsrc, zero, off, 0)
        row = row + tv


@avelang.jit
def conv2d_bf16_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    N: al.u32,
    H: al.u32,
    W: al.u32,
    H_out: al.u32,
    W_out: al.u32,
):
    K_out_u32 = al.convert(K_OUT, al.u32)
    K_red_u32 = al.convert(K_RED, al.u32)
    RS_u32 = al.convert(KR * KS, al.u32)
    S_u32 = al.convert(KS, al.u32)
    one = al.convert(1, al.u32)
    zero_f32 = al.convert(0.0, al.f32)

    tid = al.thread_id(0)
    block_spatial = al.block_id(0)
    n_idx = al.block_id(1)

    tiles_per_row = (W_out + TILE_W - 1) // TILE_W
    tile_h = (block_spatial // tiles_per_row) * TILE_H
    tile_w = (block_spatial % tiles_per_row) * TILE_W

    # ── shared memory ───────────────────────────────────────────
    shm_iu32 = al.make_shared((IN_U32,), al.u32)
    shm_wu32 = al.make_shared((W_ROWS, 4), al.u32)

    _load_input_to_shm(shm_iu32, x_ptr, N, H, W, n_idx, tile_h, tile_w, tid)
    _load_weight_to_shm(shm_wu32, w_ptr, tid)
    al.syncthreads()

    shm_in = al.view(shm_iu32, al.Tensor((C_IN, TIH, TIW), al.bf16))
    shm_w  = al.view(shm_wu32, al.Tensor((K_OUT * K_RED,), al.bf16))

    out = al.make_tensor(out_ptr, al.bf16,
        al.make_layout((N, K_out_u32, H_out, W_out),
                       (K_out_u32 * H_out * W_out, H_out * W_out, W_out, 1)))

    total_out = al.convert(TILE_OUT, al.u32)
    tv = al.convert(THREADS, al.u32)
    tw = al.convert(TILE_W, al.u32)
    idx = tid

    for _ in al.range((TILE_OUT + THREADS - 1) // THREADS):
        if idx < total_out:
            k = idx % K_out_u32
            hw = idx // K_out_u32
            h_off = hw // tw
            w_off = hw % tw

            global_h = tile_h + h_off
            global_w = tile_w + w_off

            if global_h < H_out and global_w < W_out:
                acc = zero_f32
                w_base = k * K_red_u32
                for c in al.range(C_IN):
                    cu = al.convert(c, al.u32)
                    for r in al.range(KR):
                        ru = al.convert(r, al.u32)
                        for s in al.range(KS):
                            su = al.convert(s, al.u32)
                            iv = al.convert(shm_in[cu, h_off + ru, w_off + su], al.f32)
                            wv = al.convert(shm_w[w_base + cu * RS_u32 + ru * S_u32 + su], al.f32)
                            acc = acc + iv * wv
                out[n_idx, k, global_h, global_w] = al.convert(acc, al.bf16)
        idx = idx + tv


# ── Host wrapper ────────────────────────────────────────────────────

def avelang_conv2d(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if not x.is_cuda or not weight.is_cuda:
        raise RuntimeError("Inputs must be on CUDA/HIP device.")

    if x.shape[1] != C_IN or weight.shape[0] != K_OUT or weight.shape[2] != KR:
        return torch.nn.functional.conv2d(
            x, weight, stride=1, padding=0, dilation=1, groups=1)

    x_bf16 = x.contiguous().to(dtype=torch.bfloat16)
    weight_bf16 = weight.contiguous().to(dtype=torch.bfloat16)

    N_val, C_val, H_val, W_val = x_bf16.shape
    K_out_val, C_w, R_val, S_val = weight_bf16.shape
    assert C_val == C_w, "Channel count mismatch"

    H_out_val = H_val - R_val + 1
    W_out_val = W_val - S_val + 1
    K_red_val = C_val * R_val * S_val

    weight_flat = weight_bf16.reshape(K_out_val, K_red_val).contiguous()

    out = torch.empty((N_val, K_out_val, H_out_val, W_out_val),
                      device=x_bf16.device, dtype=torch.bfloat16)

    tiles_per_row = (W_out_val + TILE_W - 1) // TILE_W
    tiles_per_col = (H_out_val + TILE_H - 1) // TILE_H
    total_tiles = tiles_per_row * tiles_per_col

    conv2d_bf16_kernel[lambda: ((total_tiles, N_val, 1), (THREADS, 1, 1))](
        x_bf16, weight_flat, out, N_val, H_val, W_val, H_out_val, W_out_val)

    return out


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
        assert stride == 1 and padding == 0 and dilation == 1 and groups == 1, (
            "AveLang kernel supports stride=1 pad=0 dil=1 groups=1 only")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.conv2d = nn.Conv2d(
            in_channels, out_channels, (kernel_size, kernel_size),
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = avelang_conv2d(x, self.conv2d.weight)
        if self.conv2d.bias is not None:
            out = out + self.conv2d.bias.to(dtype=out.dtype).reshape(1, -1, 1, 1)
        return out

# ── Test harness compat ─────────────────────────────────────────────

batch_size   = 16
in_channels  = 16
out_channels = 128
kernel_size  = 3
width        = 1024
height       = 1024

def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
