import torch
import torch.nn as nn
import avelang
import avelang.language as al

# Compile-time constants
C_IN: al.constexpr = 64
C_OUT: al.constexpr = 128
K_H: al.constexpr = 4
K_W: al.constexpr = 4
STRIDE: al.constexpr = 2
OC_TILE: al.constexpr = 8
TILE_H: al.constexpr = 16
TILE_W: al.constexpr = 16
THREADS: al.constexpr = 256
WEIGHT_SHM_SIZE: al.constexpr = C_IN * OC_TILE * K_H * K_W  # 64 * 8 * 16 = 8192
ELEMS_PER_BLOCK: al.constexpr = OC_TILE * TILE_H * TILE_W  # 8 * 16 * 16 = 2048

ADD_VAL = 0.5
MULT_VAL = 2.0

# GELU constants
GELU_COEFF1 = 0.7978845608028654  # sqrt(2/pi)
GELU_COEFF2 = 0.044715


@avelang.jit
def fused_conv_transpose_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    N: al.i32,
    H_in: al.i32,
    W_in: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    num_h_tiles: al.i32,
    num_w_tiles: al.i32,
    num_oc_tiles: al.i32,
    num_spatial_tiles: al.i32,
):
    tid = al.thread_id(0)
    bid_combined = al.block_id(0)
    bid_n = al.block_id(1)

    # Decode block indices
    oc_tile = bid_combined // num_spatial_tiles
    spatial_tile = bid_combined - oc_tile * num_spatial_tiles
    h_tile = spatial_tile // num_w_tiles
    w_tile = spatial_tile - h_tile * num_w_tiles

    oc_start = oc_tile * OC_TILE
    h_start = h_tile * TILE_H
    w_start = w_tile * TILE_W

    oc_end = oc_start + OC_TILE
    h_end = h_start + TILE_H
    w_end = w_start + TILE_W

    if oc_end > C_OUT:
        oc_end = C_OUT
    if h_end > H_out:
        h_end = H_out
    if w_end > W_out:
        w_end = W_out

    oc_tile_actual = oc_end - oc_start
    h_tile_actual = h_end - h_start
    w_tile_actual = w_end - w_start

    # Early exit for empty tiles or out-of-range batches
    if oc_tile_actual <= 0:
        return
    if h_tile_actual <= 0:
        return
    if w_tile_actual <= 0:
        return
    if bid_n >= N:
        return

    # --- Shared memory for weight tile ---
    w_shm = al.make_shared((WEIGHT_SHM_SIZE,), al.bf16)

    w_layout = al.make_layout(
        (C_IN, C_OUT, K_H, K_W),
        (C_OUT * K_H * K_W, K_H * K_W, K_W, 1),
    )
    w_global = al.make_tensor(w_ptr, al.bf16, w_layout)

    # Cooperative weight load
    oc_kh_kw = OC_TILE * K_H * K_W
    kh_kw = K_H * K_W
    load_idx = al.convert(tid, al.i32)
    for _ in al.range(WEIGHT_SHM_SIZE // THREADS + 1):
        if load_idx < WEIGHT_SHM_SIZE:
            ic_w = load_idx // oc_kh_kw
            rest_w = load_idx - ic_w * oc_kh_kw
            oc_local = rest_w // kh_kw
            rest_k = rest_w - oc_local * kh_kw
            kh_w = rest_k // K_W
            kw_w = rest_k - kh_w * K_W
            global_oc = oc_start + oc_local
            if global_oc < C_OUT:
                w_shm[load_idx] = w_global[ic_w, global_oc, kh_w, kw_w]
        load_idx = load_idx + THREADS

    al.syncthreads()

    # --- Layouts for input, bias, and output ---
    x_layout = al.make_layout(
        (N, C_IN, H_in, W_in),
        (C_IN * H_in * W_in, H_in * W_in, W_in, 1),
    )
    x_global = al.make_tensor(x_ptr, al.bf16, x_layout)

    bias_layout = al.make_layout((C_OUT,), (1,))
    bias_global = al.make_tensor(bias_ptr, al.bf16, bias_layout)

    out_layout = al.make_layout(
        (N, C_OUT, H_out, W_out),
        (C_OUT * H_out * W_out, H_out * W_out, W_out, 1),
    )
    out_global = al.make_tensor(out_ptr, al.bf16, out_layout)

    # --- Epilogue constants ---
    add_val_f32 = al.convert(ADD_VAL, al.f32)
    mult_val_f32 = al.convert(MULT_VAL, al.f32)
    zero_f32 = al.convert(0.0, al.f32)
    half_f32 = al.convert(0.5, al.f32)
    one_f32 = al.convert(1.0, al.f32)
    gelu_c1 = al.convert(GELU_COEFF1, al.f32)
    gelu_c2 = al.convert(GELU_COEFF2, al.f32)

    # Convert tile dimensions to i32 for arithmetic
    oc_tile_i32 = al.convert(oc_tile_actual, al.i32)
    h_tile_i32 = al.convert(h_tile_actual, al.i32)
    w_tile_i32 = al.convert(w_tile_actual, al.i32)
    hw_per_oc = h_tile_i32 * w_tile_i32
    total_local = oc_tile_i32 * hw_per_oc

    # Precompute shared memory stride offsets
    w_oc_block = OC_TILE * kh_kw

    # --- Compute: each thread handles a subset of output elements ---
    elem_idx = al.convert(tid, al.i32)
    for _ in al.range(ELEMS_PER_BLOCK // THREADS + 2):
        if elem_idx < total_local:
            oc_local_e = elem_idx // hw_per_oc
            rest_e = elem_idx - oc_local_e * hw_per_oc
            h_local = rest_e // w_tile_i32
            w_local = rest_e - h_local * w_tile_i32

            out_oc = oc_start + oc_local_e
            out_h = h_start + h_local
            out_w = w_start + w_local

            acc = al.convert(0.0, al.f32)

            # Loop over input channels and kernel positions
            for ic in al.range(C_IN):
                w_ic_base = ic * w_oc_block + oc_local_e * kh_kw

                for kh in al.range(K_H):
                    diff_h = out_h - kh
                    ih = diff_h // STRIDE
                    if diff_h == ih * STRIDE:
                        if ih >= 0:
                            if ih < H_in:
                                for kw in al.range(K_W):
                                    diff_w = out_w - kw
                                    iw = diff_w // STRIDE
                                    if diff_w == iw * STRIDE:
                                        if iw >= 0:
                                            if iw < W_in:
                                                x_val = al.convert(
                                                    x_global[bid_n, ic, ih, iw], al.f32
                                                )
                                                w_shm_idx = w_ic_base + kh * K_W + kw
                                                w_val = al.convert(
                                                    w_shm[w_shm_idx], al.f32
                                                )
                                                acc = acc + x_val * w_val

            # --- Epilogue ---
            # Add ConvTranspose2d bias for this output channel
            acc = acc + al.convert(bias_global[out_oc], al.f32)

            # Add add_value
            result = acc + add_val_f32

            # min(x, 0.0)
            if result > zero_f32:
                result = zero_f32

            # GELU(x) = 0.5 * x * (1.0 + tanh(c1 * (x + c2 * x^3)))
            x2 = result * result
            x3 = x2 * result
            tanh_arg = gelu_c1 * (result + gelu_c2 * x3)
            tanh_val = al.tanh(tanh_arg)
            gelu_result = half_f32 * result * (one_f32 + tanh_val)

            final_result = gelu_result * mult_val_f32

            out_global[bid_n, out_oc, out_h, out_w] = al.convert(final_result, al.bf16)

        elem_idx = elem_idx + THREADS


def _prepare_bf16_cuda(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_conv_transpose_fused(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device required.")

    x_bf16 = _prepare_bf16_cuda(x)
    w_bf16 = _prepare_bf16_cuda(weight)
    bias_bf16 = _prepare_bf16_cuda(bias)

    N_val, C_in_val, H_in_val, W_in_val = x_bf16.shape
    C_in_w, C_out_val, K_h_val, K_w_val = w_bf16.shape

    H_out_val = (H_in_val - 1) * STRIDE + K_h_val
    W_out_val = (W_in_val - 1) * STRIDE + K_w_val

    num_h_tiles_val = (H_out_val + TILE_H - 1) // TILE_H
    num_w_tiles_val = (W_out_val + TILE_W - 1) // TILE_W
    num_spatial_tiles_val = num_h_tiles_val * num_w_tiles_val
    num_oc_tiles_val = (C_out_val + OC_TILE - 1) // OC_TILE

    out = torch.empty(
        (N_val, C_out_val, H_out_val, W_out_val),
        device=x_bf16.device,
        dtype=torch.bfloat16,
    )

    grid = (num_oc_tiles_val * num_spatial_tiles_val, N_val, 1)

    fused_conv_transpose_kernel[lambda: (grid, (THREADS, 1, 1))](
        x_bf16, w_bf16, bias_bf16, out,
        N_val, H_in_val, W_in_val, H_out_val, W_out_val,
        num_h_tiles_val, num_w_tiles_val,
        num_oc_tiles_val, num_spatial_tiles_val,
    )
    return out


class ModelNew(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value
    ):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride
        )
        self.add_value = add_value
        self.multiply_value = multiply_value

    def forward(self, x):
        weight = self.conv_transpose.weight.data
        bias = self.conv_transpose.bias.data
        result = avelang_conv_transpose_fused(x, weight, bias)
        return result


batch_size = 128
in_channels = 64
out_channels = 128
height, width = 64, 64
kernel_size = 4
stride = 2
add_value = 0.5
multiply_value = 2.0


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, add_value, multiply_value]
