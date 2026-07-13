import torch
import torch.nn as nn
import avelang
import avelang.language as al

# Problem dimensions (matching input_model.py)
BATCH_SIZE = 16
IN_CHANNELS = 32
OUT_CHANNELS = 64
IN_D = 32
IN_H = 64
IN_W = 64
KERNEL_SIZE = 3
OUT_D = IN_D - KERNEL_SIZE + 1
OUT_H = IN_H - KERNEL_SIZE + 1
OUT_W = IN_W - KERNEL_SIZE + 1

# Tiling parameters
OC_TILE = 8
D_TILE = 4
H_TILE = 2
W_TILE = 2
IC_TILE = 8
THREADS = 128

# Derived tile dimensions
IN_TILE_D = D_TILE + KERNEL_SIZE - 1
IN_TILE_H = H_TILE + KERNEL_SIZE - 1
IN_TILE_W = W_TILE + KERNEL_SIZE - 1

# Shared memory element counts
SHM_IN_ELEMS = IC_TILE * IN_TILE_D * IN_TILE_H * IN_TILE_W
SHM_WT_ELEMS = OC_TILE * IC_TILE * KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE

# Block counts
D_BLOCKS = (OUT_D + D_TILE - 1) // D_TILE
H_BLOCKS = (OUT_H + H_TILE - 1) // H_TILE
W_BLOCKS = (OUT_W + W_TILE - 1) // W_TILE
HW_BLOCKS = H_BLOCKS * W_BLOCKS
SPATIAL_TILES = D_BLOCKS * HW_BLOCKS
IC_BLOCKS = IN_CHANNELS // IC_TILE

# Flattened strides and sizes
_IN_PER_B = IN_CHANNELS * IN_D * IN_H * IN_W
_IN_PER_C = IN_D * IN_H * IN_W
_IN_PER_D = IN_H * IN_W
_IN_PER_H = IN_W
_OUT_PER_B = OUT_CHANNELS * OUT_D * OUT_H * OUT_W
_OUT_PER_C = OUT_D * OUT_H * OUT_W
_OUT_PER_D = OUT_H * OUT_W
_OUT_PER_H = OUT_W
_WT_PER_C = IN_CHANNELS * KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE
_WT_PER_IC = KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE
_WT_PER_KD = KERNEL_SIZE * KERNEL_SIZE
_WT_PER_KH = KERNEL_SIZE
_SHM_IN_S_IC = IN_TILE_D * IN_TILE_H * IN_TILE_W
_SHM_IN_S_D = IN_TILE_H * IN_TILE_W
_SHM_IN_S_H = IN_TILE_W
_SHM_WT_S_OC = IC_TILE * KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE
_SHM_WT_S_IC = KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE

# Load iterations per thread
IN_LOADS = SHM_IN_ELEMS // THREADS
WT_LOADS = (SHM_WT_ELEMS + THREADS - 1) // THREADS

# Total kernel positions
TOTAL_KPOS = KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE


@avelang.jit
def conv3d_mish_tanh_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
):
    tid = al.thread_id(0)
    oc_block = al.block_id(0)
    spatial_block = al.block_id(1)
    b_id = al.block_id(2)

    # Decompose spatial_block into (d_block, h_block, w_block)
    hw_blocks = al.convert(HW_BLOCKS, al.i32)
    w_blocks = al.convert(W_BLOCKS, al.i32)
    d_block = spatial_block // hw_blocks
    rem = spatial_block - d_block * hw_blocks
    h_block = rem // w_blocks
    w_block = rem - h_block * w_blocks

    # Output tile bounds using al.min (no if-based clipping)
    oc_start = oc_block * al.convert(OC_TILE, al.i32)
    oc_end = al.min(oc_start + al.convert(OC_TILE, al.i32),
                    al.convert(OUT_CHANNELS, al.i32))

    d_start = d_block * al.convert(D_TILE, al.i32)
    d_end = al.min(d_start + al.convert(D_TILE, al.i32),
                   al.convert(OUT_D, al.i32))

    h_start = h_block * al.convert(H_TILE, al.i32)
    h_end = al.min(h_start + al.convert(H_TILE, al.i32),
                   al.convert(OUT_H, al.i32))

    w_start = w_block * al.convert(W_TILE, al.i32)
    w_end = al.min(w_start + al.convert(W_TILE, al.i32),
                   al.convert(OUT_W, al.i32))

    local_oc = oc_end - oc_start
    local_d = d_end - d_start
    local_h = h_end - h_start
    local_w = w_end - w_start

    local_d_hw = local_d * local_h * local_w
    local_hw = local_h * local_w
    valid_total = local_oc * local_d * local_h * local_w

    # Thread-to-output mapping
    oc_l = tid // local_d_hw
    rem1 = tid - oc_l * local_d_hw
    d_l = rem1 // local_hw
    rem2 = rem1 - d_l * local_hw
    h_l = rem2 // local_w
    w_l = rem2 - h_l * local_w

    is_active = tid < valid_total

    # Shared memory -- all threads participate in loading
    shm_in = al.make_shared((SHM_IN_ELEMS,), al.bf16)
    shm_wt = al.make_shared((SHM_WT_ELEMS,), al.bf16)

    # Global tensor views (flat 1D)
    in_total = al.convert(_IN_PER_B * BATCH_SIZE, al.i32)
    x_flat = al.make_tensor(x_ptr, al.bf16, al.make_layout((in_total,), (1,)))
    wt_total = al.convert(_WT_PER_C * OUT_CHANNELS, al.i32)
    w_flat = al.make_tensor(w_ptr, al.bf16, al.make_layout((wt_total,), (1,)))

    # Accumulator
    acc = al.convert(0.0, al.f32)

    # Pre-convert all compile-time constants
    c_ic_tile = al.convert(IC_TILE, al.i32)
    c_threads = al.convert(THREADS, al.i32)
    c_shm_in_s_ic = al.convert(_SHM_IN_S_IC, al.i32)
    c_shm_in_s_d = al.convert(_SHM_IN_S_D, al.i32)
    c_shm_in_s_h = al.convert(_SHM_IN_S_H, al.i32)
    c_in_per_c = al.convert(_IN_PER_C, al.i32)
    c_in_per_d = al.convert(_IN_PER_D, al.i32)
    c_in_per_h = al.convert(_IN_PER_H, al.i32)
    c_in_per_b = al.convert(_IN_PER_B, al.i32)
    c_in_d_m1 = al.convert(IN_D - 1, al.i32)
    c_in_h_m1 = al.convert(IN_H - 1, al.i32)
    c_in_w_m1 = al.convert(IN_W - 1, al.i32)
    c_shm_wt_s_oc = al.convert(_SHM_WT_S_OC, al.i32)
    c_shm_wt_s_ic = al.convert(_SHM_WT_S_IC, al.i32)
    c_wt_per_c = al.convert(_WT_PER_C, al.i32)
    c_wt_per_ic = al.convert(_WT_PER_IC, al.i32)
    c_wt_per_kd = al.convert(_WT_PER_KD, al.i32)
    c_wt_per_kh = al.convert(_WT_PER_KH, al.i32)
    c_shm_wt_elems = al.convert(SHM_WT_ELEMS, al.i32)

    # Loop over input channel tiles
    for ic_tile in al.range(IC_BLOCKS):
        ic_start = ic_tile * c_ic_tile

        # --- Load input tile into shared memory (all threads) ---
        load_idx = tid
        for _ in al.range(IN_LOADS):
            ic_l = load_idx // c_shm_in_s_ic
            rem_i = load_idx - ic_l * c_shm_in_s_ic
            d_li = rem_i // c_shm_in_s_d
            rem_i = rem_i - d_li * c_shm_in_s_d
            h_li = rem_i // c_shm_in_s_h
            w_li = rem_i - h_li * c_shm_in_s_h

            d_g_load = al.min(d_start + d_li, c_in_d_m1)
            h_g_load = al.min(h_start + h_li, c_in_h_m1)
            w_g_load = al.min(w_start + w_li, c_in_w_m1)

            g_off = (b_id * c_in_per_b +
                     (ic_start + ic_l) * c_in_per_c +
                     d_g_load * c_in_per_d +
                     h_g_load * c_in_per_h +
                     w_g_load)
            shm_in[load_idx] = x_flat[g_off]
            load_idx = load_idx + c_threads

        # --- Load weight tile into shared memory (all threads) ---
        load_idx = tid
        for _ in al.range(WT_LOADS):
            if load_idx < c_shm_wt_elems:
                oc_w_l = load_idx // c_shm_wt_s_oc
                rem_w = load_idx - oc_w_l * c_shm_wt_s_oc
                ic_w_l = rem_w // c_shm_wt_s_ic
                rem_w = rem_w - ic_w_l * c_shm_wt_s_ic
                kd_l = rem_w // c_wt_per_kd
                rem_w = rem_w - kd_l * c_wt_per_kd
                kh_l = rem_w // c_wt_per_kh
                kw_l = rem_w - kh_l * c_wt_per_kh

                oc_g_load = oc_start + oc_w_l
                ic_g_load = ic_start + ic_w_l
                w_off = (oc_g_load * c_wt_per_c +
                         ic_g_load * c_wt_per_ic +
                         kd_l * c_wt_per_kd +
                         kh_l * c_wt_per_kh +
                         kw_l)
                shm_wt[load_idx] = w_flat[w_off]
            load_idx = load_idx + c_threads

        al.syncthreads()

        # --- Compute partial accumulation (active threads only) ---
        if is_active:
            for k_idx in al.range(TOTAL_KPOS):
                kd = k_idx // c_wt_per_kd
                rem_k = k_idx - kd * c_wt_per_kd
                kh = rem_k // c_wt_per_kh
                kw = rem_k - kh * c_wt_per_kh

                in_d_idx = d_l + kd
                in_h_idx = h_l + kh
                in_w_idx = w_l + kw

                in_shm_off = (in_d_idx * c_shm_in_s_d +
                              in_h_idx * c_shm_in_s_h +
                              in_w_idx)

                wt_shm_base = (oc_l * c_shm_wt_s_oc +
                               kd * c_wt_per_kd +
                               kh * c_wt_per_kh +
                               kw)

                for ic_local in al.range(IC_TILE):
                    in_val = al.convert(
                        shm_in[ic_local * c_shm_in_s_ic + in_shm_off], al.f32)
                    wt_val = al.convert(
                        shm_wt[wt_shm_base + ic_local * c_shm_wt_s_ic], al.f32)
                    acc = acc + in_val * wt_val

        al.syncthreads()

    # --- Epilogue: bias, Mish, Tanh, write back ---
    if is_active:
        # Global output coordinates
        oc_g = oc_start + oc_l
        d_g = d_start + d_l
        h_g = h_start + h_l
        w_g = w_start + w_l

        b_out_c = al.convert(OUT_CHANNELS, al.i32)
        b_flat = al.make_tensor(b_ptr, al.bf16, al.make_layout((b_out_c,), (1,)))
        bias_val = al.convert(b_flat[oc_g], al.f32)
        result = acc + bias_val

        # Mish: x * tanh(softplus(x))
        one = al.convert(1.0, al.f32)
        softplus = al.log(one + al.exp(result))
        mish_val = result * al.tanh(softplus)

        # Tanh
        final = al.tanh(mish_val)

        # Write output
        out_total = al.convert(_OUT_PER_B * BATCH_SIZE, al.i32)
        out_flat = al.make_tensor(out_ptr, al.bf16, al.make_layout((out_total,), (1,)))
        out_per_b = al.convert(_OUT_PER_B, al.i32)
        out_per_c = al.convert(_OUT_PER_C, al.i32)
        out_per_d = al.convert(_OUT_PER_D, al.i32)
        out_per_h = al.convert(_OUT_PER_H, al.i32)
        out_off = (b_id * out_per_b +
                   oc_g * out_per_c +
                   d_g * out_per_d +
                   h_g * out_per_h +
                   w_g)
        out_flat[out_off] = al.convert(final, al.bf16)


def _prepare_bf16_cuda_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_conv3d_mish_tanh(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = _prepare_bf16_cuda_contiguous(x)
    w_bf16 = _prepare_bf16_cuda_contiguous(weight)
    b_bf16 = _prepare_bf16_cuda_contiguous(bias)

    batch, in_c, in_d, in_h, in_w = x_bf16.shape
    out_c, w_in_c, kd, kh, kw = w_bf16.shape

    expected_out_d = in_d - kd + 1
    expected_out_h = in_h - kh + 1
    expected_out_w = in_w - kw + 1

    if w_in_c != in_c:
        raise ValueError(f"Weight in_channels {w_in_c} != input channels {in_c}")

    out = torch.empty(
        (batch, out_c, expected_out_d, expected_out_h, expected_out_w),
        device=x_bf16.device,
        dtype=torch.bfloat16,
    )

    grid = (OUT_CHANNELS // OC_TILE, SPATIAL_TILES, batch)
    conv3d_mish_tanh_kernel[lambda: (grid, (THREADS, 1, 1))](
        x_bf16, w_bf16, b_bf16, out,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return avelang_conv3d_mish_tanh(x, self.conv.weight, self.conv.bias)


def get_inputs():
    return [torch.rand(BATCH_SIZE, IN_CHANNELS, IN_D, IN_H, IN_W)]


def get_init_inputs():
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE]
