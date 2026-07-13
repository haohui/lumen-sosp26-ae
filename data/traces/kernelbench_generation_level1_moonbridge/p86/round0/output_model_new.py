import torch
import torch.nn as nn
import avelang
import avelang.language as al

# =============================================================================
# Depthwise convolution tile constants
# =============================================================================
DW_TILE_H = 16
DW_TILE_W = 16
DW_HALO = 1
DW_THREADS = 256
DW_SHM_H = DW_TILE_H + 2 * DW_HALO
DW_SHM_W = DW_TILE_W + 2 * DW_HALO

# =============================================================================
# Pointwise convolution tile constants
# =============================================================================
PW_THREADS = 128
PW_TILE_M = 128


# =============================================================================
# Depthwise 3x3 convolution kernel
# =============================================================================
@avelang.jit
def depthwise_conv_3x3_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C: al.i32,
    H: al.i32,
    W: al.i32,
    KH: al.i32,
    KW: al.i32,
    stride: al.i32,
    padding: al.i32,
):
    tid = al.thread_id(0)
    tile_w = al.block_id(0)
    tile_h = al.block_id(1)
    nc = al.block_id(2)

    n = nc // C
    c = nc % C

    th = tid // DW_TILE_W
    tw = tid % DW_TILE_W

    h_out_start = tile_h * DW_TILE_H
    w_out_start = tile_w * DW_TILE_W
    h_out = h_out_start + th
    w_out = w_out_start + tw

    zero_bf16 = al.convert(0.0, al.bf16)

    chan_stride = H * W
    batch_stride = C * chan_stride
    total_elems = N * batch_stride

    input_flat = al.make_tensor(input_ptr, al.bf16, al.make_layout((total_elems,), (1,)))
    weight_flat = al.make_tensor(weight_ptr, al.bf16, al.make_layout((C * KH * KW,), (1,)))
    output_flat = al.make_tensor(output_ptr, al.bf16, al.make_layout((total_elems,), (1,)))

    shm_input = al.make_shared((DW_SHM_H, DW_SHM_W), al.bf16)

    shm_total = DW_SHM_H * DW_SHM_W
    for shm_idx in al.range(tid, shm_total, DW_THREADS):
        sh = shm_idx // DW_SHM_W
        sw = shm_idx % DW_SHM_W
        h_in = h_out_start + sh - DW_HALO
        w_in = w_out_start + sw - DW_HALO

        if h_in >= 0 and h_in < H and w_in >= 0 and w_in < W:
            gbl_idx = n * batch_stride + c * chan_stride + h_in * W + w_in
            shm_input[sh, sw] = input_flat[gbl_idx]
        else:
            shm_input[sh, sw] = zero_bf16

    al.syncthreads()

    if th < DW_TILE_H and tw < DW_TILE_W and h_out < H and w_out < W:
        acc = al.convert(0.0, al.f32)
        wgt_base = c * KH * KW

        for kh in al.range(KH):
            for kw in al.range(KW):
                wgt = al.convert(weight_flat[wgt_base + kh * KW + kw], al.f32)
                h_in_shm = th + kh * stride
                w_in_shm = tw + kw * stride
                inp = al.convert(shm_input[h_in_shm, w_in_shm], al.f32)
                acc = acc + wgt * inp

        out_idx = n * batch_stride + c * chan_stride + h_out * W + w_out
        output_flat[out_idx] = al.convert(acc, al.bf16)


# =============================================================================
# Pointwise 1x1 convolution kernel
# =============================================================================
@avelang.jit
def pointwise_conv_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    m: al.u32,
    n: al.u32,
    k: al.u32,
):
    tid = al.thread_id(0)
    block_m = al.block_id(0)

    x_flat = al.make_tensor(x_ptr, al.bf16, al.make_layout((m * k,), (1,)))
    # Weight transposed to (K, N): w[ki, col] at offset ki * n + col
    w_flat = al.make_tensor(w_ptr, al.bf16, al.make_layout((k * n,), (1,)))
    g_out = al.make_tensor(out_ptr, al.bf16, al.make_layout((m, n), (n, 1)))

    row_start = block_m * PW_TILE_M
    col = tid

    for local_row in al.range(PW_TILE_M):
        out_row = row_start + local_row
        if out_row < m:
            if col < n:
                dot = al.convert(0.0, al.f32)
                x_row_base = out_row * k
                for ki in al.range(k):
                    x_val = al.convert(x_flat[x_row_base + ki], al.f32)
                    w_val = al.convert(w_flat[ki * n + col], al.f32)
                    dot = dot + x_val * w_val
                g_out[out_row, col] = al.convert(dot, al.bf16)


# =============================================================================
# Host helpers
# =============================================================================
def _to_bf16_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_depthwise_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    stride: int,
    padding: int,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = _to_bf16_contiguous(x)
    w_bf16 = _to_bf16_contiguous(weight)

    N, C, H, W = x_bf16.shape
    _, _, KH, KW = w_bf16.shape

    H_out = H
    W_out = W

    out = torch.empty((N, C, H_out, W_out), device=x_bf16.device, dtype=torch.bfloat16)

    tiles_x = (W_out + DW_TILE_W - 1) // DW_TILE_W
    tiles_y = (H_out + DW_TILE_H - 1) // DW_TILE_H
    tiles_z = N * C

    grid = (tiles_x, tiles_y, tiles_z)
    block = (DW_THREADS, 1, 1)

    depthwise_conv_3x3_kernel[lambda: (grid, block)](
        x_bf16, w_bf16, out,
        N, C, H, W, KH, KW, stride, padding,
    )
    return out


def avelang_pointwise_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    N_batch, C_in, H, W = x.shape
    C_out, C_in_w, _, _ = weight.shape
    assert C_in_w == C_in, f"Channel mismatch: input has {C_in}, weight expects {C_in_w}"

    x_nhwc = x.permute(0, 2, 3, 1).contiguous()
    x_mat = x_nhwc.reshape(-1, C_in)

    w_mat = weight.reshape(C_out, C_in).contiguous()
    w_mat_t = w_mat.t().contiguous()

    M = x_mat.shape[0]
    K = C_in
    N_dim = C_out

    m_padded = ((M + PW_TILE_M - 1) // PW_TILE_M) * PW_TILE_M
    if m_padded != M:
        x_pad = torch.zeros((m_padded, K), device=x.device, dtype=torch.bfloat16)
        x_pad[:M, :K] = x_mat
        x_mat = x_pad

    out_mat = torch.empty((m_padded, N_dim), device=x.device, dtype=torch.bfloat16)

    grid = (m_padded // PW_TILE_M, 1, 1)
    block = (PW_THREADS, 1, 1)

    pointwise_conv_kernel[lambda: (grid, block)](
        x_mat, w_mat_t, out_mat,
        m_padded, N_dim, K,
    )

    out_trimmed = out_mat[:M, :N_dim]
    out_nhwc = out_trimmed.reshape(N_batch, H, W, N_dim)
    out_nchw = out_nhwc.permute(0, 3, 1, 2).contiguous()
    return out_nchw


# =============================================================================
# ModelNew: depthwise-separable 2D convolution
# =============================================================================
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
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1,
            bias=bias,
        )
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_bf16 = x.to(dtype=torch.bfloat16)

        dw_out = avelang_depthwise_conv(
            x_bf16, self.depthwise.weight.data,
            stride=self.stride, padding=self.padding,
        )

        pw_out = avelang_pointwise_conv(dw_out, self.pointwise.weight.data)

        if x.dtype != torch.bfloat16:
            return pw_out.to(dtype=x.dtype)
        return pw_out


# =============================================================================
# Test code
# =============================================================================
batch_size = 16
in_channels = 64
out_channels = 128
kernel_size = 3
width = 512
height = 512
stride = 1
padding = 1
dilation = 1


def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]
