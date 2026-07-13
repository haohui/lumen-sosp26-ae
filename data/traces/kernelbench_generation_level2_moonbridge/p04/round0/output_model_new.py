import torch
import torch.nn as nn
import avelang
import avelang.language as al

# =============================================================================
# Problem Constants
# =============================================================================
BATCH_SIZE = 64
IN_CHANNELS = 64
OUT_CHANNELS = 128
HEIGHT = 256
WIDTH = 256
KERNEL_SIZE = 3
PADDING = 0

# =============================================================================
# Tiling Constants
# =============================================================================
TILE_H = 16
TILE_W = 16
C_IN_TILE = 8
K = KERNEL_SIZE
HALO = K // 2  # halo for padding (0 in this case)
WIN_H = TILE_H + K - 1  # input window height (18)
WIN_W = TILE_W + K - 1  # input window width (18)
SHM_IN_SIZE = WIN_H * WIN_W * C_IN_TILE  # 18*18*8 = 2592
SHM_WT_SIZE = OUT_CHANNELS * C_IN_TILE * K * K  # 128*8*9 = 9216
SPATIAL_THREADS = TILE_H * TILE_W  # 256
BLOCK_SIZE_OPT = SPATIAL_THREADS  # 256


# =============================================================================
# Tiled Conv2d + Mish×2 Kernel
#
# Each block handles TILE_H×TILE_W spatial positions for one batch element.
# For each C_in tile, the input window and weight slice are staged into
# shared memory before accumulation across all C_out channels.
# =============================================================================
@avelang.jit
def conv_mish_tiled_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    b_dim: al.u32,
    c_in: al.u32,
    c_out: al.u32,
    h_dim: al.u32,
    w_dim: al.u32,
    h_out: al.u32,
    w_out: al.u32,
):
    tid = al.thread_id(0)  # spatial thread id [0, 255]
    block_h = al.block_id(0)
    block_w = al.block_id(1)
    block_b = al.block_id(2)

    # Output spatial position for this thread
    h_idx = block_h * TILE_H + (tid // TILE_W)
    w_idx = block_w * TILE_W + (tid % TILE_W)
    b_idx = block_b

    # Input window top-left corner
    h_start = block_h * TILE_H
    w_start = block_w * TILE_W

    # Shared memory for input window and weight slice
    shm_in = al.make_shared((SHM_IN_SIZE,), al.bf16)
    shm_wt = al.make_shared((SHM_WT_SIZE,), al.bf16)

    # Layouts for global tensors
    layout_x = al.make_layout(
        (b_dim, c_in, h_dim, w_dim),
        (c_in * h_dim * w_dim, h_dim * w_dim, w_dim, 1),
    )
    x = al.make_tensor(x_ptr, al.bf16, layout_x)

    layout_w = al.make_layout(
        (c_out, c_in, K, K),
        (c_in * K * K, K * K, K, 1),
    )
    w = al.make_tensor(w_ptr, al.bf16, layout_w)

    layout_b = al.make_layout((c_out,), (1,))
    bias = al.make_tensor(b_ptr, al.bf16, layout_b)

    # Output: NCHW layout
    layout_out = al.make_layout(
        (b_dim, c_out, h_out, w_out),
        (c_out * h_out * w_out, h_out * w_out, w_out, 1),
    )
    out = al.make_tensor(out_ptr, al.bf16, layout_out)

    # Accumulators per output channel (in registers)
    acc = al.make_local((OUT_CHANNELS,), al.f32)
    zero_f32 = al.convert(0.0, al.f32)
    for co in al.range(c_out):
        acc[co] = al.convert(bias[co], al.f32)

    c_in_tiles = c_in // C_IN_TILE

    for ct in al.range(c_in_tiles):
        ci_base = ct * C_IN_TILE

        # ---- Cooperative load: input window into shm_in ----
        # 2592 elements / 256 threads = 10 each, remainder 32
        zero_bf16 = al.convert(0.0, al.bf16)
        loads_per_in = SHM_IN_SIZE // SPATIAL_THREADS
        for i in al.range(loads_per_in):
            idx = tid * loads_per_in + i
            ci_local = idx % C_IN_TILE
            sp = idx // C_IN_TILE
            ky = sp // WIN_W
            kx = sp % WIN_W
            h_load = h_start + ky
            w_load = w_start + kx
            ci_global = ci_base + ci_local
            if h_load < h_dim:
                if w_load < w_dim:
                    shm_in[idx] = x[b_idx, ci_global, h_load, w_load]
                else:
                    shm_in[idx] = zero_bf16
            else:
                shm_in[idx] = zero_bf16
        rem_in = SHM_IN_SIZE - loads_per_in * SPATIAL_THREADS
        if tid < rem_in:
            idx = SPATIAL_THREADS * loads_per_in + tid
            ci_local = idx % C_IN_TILE
            sp = idx // C_IN_TILE
            ky = sp // WIN_W
            kx = sp % WIN_W
            h_load = h_start + ky
            w_load = w_start + kx
            ci_global = ci_base + ci_local
            if h_load < h_dim:
                if w_load < w_dim:
                    shm_in[idx] = x[b_idx, ci_global, h_load, w_load]
                else:
                    shm_in[idx] = zero_bf16
            else:
                shm_in[idx] = zero_bf16

        # ---- Cooperative load: weight slice into shm_wt ----
        loads_per_thread_wt = SHM_WT_SIZE // SPATIAL_THREADS
        for i in al.range(loads_per_thread_wt):
            idx = tid * loads_per_thread_wt + i
            co = idx // (C_IN_TILE * K * K)
            rem = idx - co * C_IN_TILE * K * K
            ci_local = rem // (K * K)
            rem2 = rem - ci_local * K * K
            ky = rem2 // K
            kx = rem2 - ky * K
            ci_global = ci_base + ci_local
            shm_wt[idx] = w[co, ci_global, ky, kx]

        al.syncthreads()

        # ---- Compute: accumulate for all C_out channels ----
        # Only threads within the valid output range participate
        if h_idx < h_out:
            if w_idx < w_out:
                for co in al.range(c_out):
                    for ci_local in al.range(C_IN_TILE):
                        for ky in al.range(K):
                            for kx in al.range(K):
                                h_in = h_idx - h_start + ky
                                w_in = w_idx - w_start + kx
                                in_idx = (h_in * WIN_W + w_in) * C_IN_TILE + ci_local
                                wt_idx = (co * C_IN_TILE + ci_local) * K * K + ky * K + kx
                                x_val = al.convert(shm_in[in_idx], al.f32)
                                w_val = al.convert(shm_wt[wt_idx], al.f32)
                                acc[co] = acc[co] + x_val * w_val

        al.syncthreads()

    # ---- Epilogue: Mish twice and write output ----
    if h_idx < h_out:
        if w_idx < w_out:
            one = al.convert(1.0, al.f32)
            for co in al.range(c_out):
                val = acc[co]
                sp = al.log(one + al.exp(val))
                val = val * al.tanh(sp)
                sp = al.log(one + al.exp(val))
                val = val * al.tanh(sp)
                out[b_idx, co, h_idx, w_idx] = al.convert(val, al.bf16)


# =============================================================================
# Host Helpers
# =============================================================================
def _prepare_bf16(t: torch.Tensor) -> torch.Tensor:
    if not t.is_cuda:
        t = t.cuda()
    if t.dtype != torch.bfloat16:
        t = t.to(dtype=torch.bfloat16)
    return t.contiguous()


def avelang_conv2d_mish(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    orig_dtype = x.dtype

    x_bf16 = _prepare_bf16(x)
    w_bf16 = _prepare_bf16(weight)
    b_bf16 = _prepare_bf16(bias)

    B = x_bf16.shape[0]
    C_in_val = x_bf16.shape[1]
    H_val = x_bf16.shape[2]
    W_val = x_bf16.shape[3]
    C_out_val = w_bf16.shape[0]
    H_out = H_val - KERNEL_SIZE + 1
    W_out = W_val - KERNEL_SIZE + 1

    if C_in_val % C_IN_TILE != 0:
        raise ValueError(f"C_in={C_in_val} must be divisible by C_IN_TILE={C_IN_TILE}")

    grid_h = (H_out + TILE_H - 1) // TILE_H
    grid_w = (W_out + TILE_W - 1) // TILE_W
    grid = (grid_h, grid_w, B)

    out = torch.empty(B, C_out_val, H_out, W_out, device=x_bf16.device, dtype=torch.bfloat16)
    conv_mish_tiled_kernel[lambda: (grid, (SPATIAL_THREADS, 1, 1))](
        x_bf16, w_bf16, b_bf16, out,
        B, C_in_val, C_out_val, H_val, W_val, H_out, W_out,
    )

    if orig_dtype != torch.bfloat16:
        out = out.to(dtype=orig_dtype)

    return out


# =============================================================================
# ModelNew
# =============================================================================
class ModelNew(nn.Module):
    """
    Optimized model: Conv2d + Mish + Mish using a tiled AveLang DSL kernel (BF16).
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        return avelang_conv2d_mish(x, self.conv.weight, self.conv.bias)


# =============================================================================
# Input / Init Functions
# =============================================================================
def get_inputs():
    return [torch.rand(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)]


def get_init_inputs():
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE]
