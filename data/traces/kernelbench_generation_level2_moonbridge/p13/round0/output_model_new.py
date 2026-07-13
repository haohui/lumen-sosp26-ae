import torch
import torch.nn as nn
import avelang
import avelang.language as al

# Compile-time constants
C_IN: al.constexpr = 16
C_OUT: al.constexpr = 64
KD: al.constexpr = 3
KH: al.constexpr = 3
KW: al.constexpr = 3
PADDING: al.constexpr = 1
TILE_H: al.constexpr = 4
TILE_W: al.constexpr = 4
BLOCK_SIZE: al.constexpr = 256
SOFTMAX_BLOCK: al.constexpr = 64

# Derived
PATCH_H: al.constexpr = TILE_H + 2   # 6
PATCH_W: al.constexpr = TILE_W + 2   # 6
PATCH_SIZE: al.constexpr = PATCH_H * PATCH_W  # 36
NUM_OUTPUTS = TILE_H * TILE_W * C_OUT   # 1024
PER_THREAD = NUM_OUTPUTS // BLOCK_SIZE  # 4
WEIGHT_SIZE = C_IN * C_OUT * KD * KH * KW  # 27648
W_CI_STRIDE = C_OUT * KD * KH * KW      # 1728
W_CO_STRIDE = KD * KH * KW              # 27
SPATIAL_STRIDE = KH * KW                # 9
INPUT_PATCH_CI = C_IN * PATCH_SIZE     # 576


@avelang.jit
def conv_transpose_mean_bias_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_flat_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    B: al.i32,
    D: al.i32,
    H: al.i32,
    W: al.i32,
):
    tid = al.thread_id(0)
    block_b = al.block_id(0)
    block_h = al.block_id(1)
    block_w = al.block_id(2)

    if block_b >= B:
        return

    h_start = block_h * TILE_H
    w_start = block_w * TILE_W

    out_layout = al.make_layout((B, C_OUT, H, W), (C_OUT * H * W, H * W, W, 1))
    out = al.make_tensor(output_ptr, al.bf16, out_layout)

    in_layout = al.make_layout(
        (B, C_IN, D, H, W),
        (C_IN * D * H * W, D * H * W, H * W, W, 1),
    )
    inp = al.make_tensor(input_ptr, al.bf16, in_layout)

    bias_layout = al.make_layout((C_OUT,), (1,))
    bias = al.make_tensor(bias_ptr, al.bf16, bias_layout)

    weight_shm = al.make_shared((WEIGHT_SIZE,), al.bf16)
    w_flat_layout = al.make_layout((WEIGHT_SIZE,), (1,))
    w_flat = al.make_tensor(weight_flat_ptr, al.bf16, w_flat_layout)
    for w_i in al.range(tid, WEIGHT_SIZE, BLOCK_SIZE):
        weight_shm[w_i] = w_flat[w_i]
    al.syncthreads()

    input_shm = al.make_shared((INPUT_PATCH_CI,), al.bf16)

    zero_i32 = al.convert(0, al.i32)
    D_f32 = al.convert(D, al.f32)
    H_i32 = H
    W_i32 = W
    PATCH_OFF = al.convert(2, al.i32)

    for local_idx in al.range(PER_THREAD):
        global_idx = tid + local_idx * BLOCK_SIZE

        spatial_idx = global_idx // C_OUT
        co = global_idx - spatial_idx * C_OUT

        h_off = spatial_idx // TILE_W
        w_off = spatial_idx - h_off * TILE_W

        h_pos = h_start + h_off
        w_pos = w_start + w_off

        if h_pos >= H_i32 or w_pos >= W_i32:
            break

        accum = al.convert(0.0, al.f32)
        w_base = co * W_CO_STRIDE

        for d_idx in al.range(D):
            for patch_idx in al.range(tid, INPUT_PATCH_CI, BLOCK_SIZE):
                ci_load = patch_idx // PATCH_SIZE
                rem = patch_idx - ci_load * PATCH_SIZE
                ph = rem // PATCH_W
                pw = rem - ph * PATCH_W
                h_global = h_start + ph - al.convert(1, al.i32)
                w_global = w_start + pw - al.convert(1, al.i32)
                if h_global >= zero_i32 and h_global < H_i32 and w_global >= zero_i32 and w_global < W_i32:
                    input_shm[patch_idx] = inp[block_b, ci_load, d_idx, h_global, w_global]
                else:
                    input_shm[patch_idx] = al.convert(0.0, al.bf16)
            al.syncthreads()

            for ci in al.range(C_IN):
                w_ci_base = ci * W_CI_STRIDE + w_base
                in_ci_base = ci * PATCH_SIZE

                for kd in al.range(KD):
                    d_in = d_idx - kd + al.convert(PADDING, al.i32)
                    d_valid = al.convert(1, al.i32)
                    if d_in < zero_i32 or d_in >= D:
                        d_valid = al.convert(0, al.i32)

                    w_kd_base = w_ci_base + kd * SPATIAL_STRIDE

                    for kh in al.range(KH):
                        h_idx = h_off + PATCH_OFF - kh
                        h_valid = al.convert(1, al.i32)
                        if h_idx < zero_i32 or h_idx >= PATCH_H:
                            h_valid = al.convert(0, al.i32)

                        w_kh_base = w_kd_base + kh * KW
                        in_kh_base = in_ci_base + h_idx * PATCH_W

                        for kw in al.range(KW):
                            w_idx = w_off + PATCH_OFF - kw
                            w_valid = al.convert(1, al.i32)
                            if w_idx < zero_i32 or w_idx >= PATCH_W:
                                w_valid = al.convert(0, al.i32)

                            valid = d_valid
                            if h_valid == zero_i32:
                                valid = al.convert(0, al.i32)
                            if w_valid == zero_i32:
                                valid = al.convert(0, al.i32)

                            if valid != zero_i32:
                                in_val = al.convert(input_shm[in_kh_base + w_idx], al.f32)
                                w_val = al.convert(weight_shm[w_kh_base + kw], al.f32)
                                accum = accum + in_val * w_val

            al.syncthreads()

        mean_val = accum / D_f32
        biased = mean_val + al.convert(bias[co], al.f32)
        out[block_b, co, h_pos, w_pos] = al.convert(biased, al.bf16)


@avelang.jit
def softmax_tanh_scale_kernel(
    input_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    B: al.i32,
    H: al.i32,
    W: al.i32,
    scale: al.f32,
):
    tid = al.thread_id(0)
    block_b = al.block_id(0)
    block_h = al.block_id(1)
    block_w = al.block_id(2)

    if block_b >= B or block_h >= H or block_w >= W:
        return

    layout = al.make_layout((B, C_OUT, H, W), (C_OUT * H * W, H * W, W, 1))
    inp = al.make_tensor(input_ptr, al.bf16, layout)
    out = al.make_tensor(output_ptr, al.bf16, layout)

    smem_vals = al.make_shared((C_OUT,), al.f32)
    smem_reduce = al.make_shared((C_OUT,), al.f32)

    val = al.convert(inp[block_b, tid, block_h, block_w], al.f32)
    smem_vals[tid] = val
    smem_reduce[tid] = val
    al.syncthreads()

    if tid < 32:
        if smem_reduce[tid + 32] > smem_reduce[tid]:
            smem_reduce[tid] = smem_reduce[tid + 32]
    al.syncthreads()
    if tid < 16:
        if smem_reduce[tid + 16] > smem_reduce[tid]:
            smem_reduce[tid] = smem_reduce[tid + 16]
    al.syncthreads()
    if tid < 8:
        if smem_reduce[tid + 8] > smem_reduce[tid]:
            smem_reduce[tid] = smem_reduce[tid + 8]
    al.syncthreads()
    if tid < 4:
        if smem_reduce[tid + 4] > smem_reduce[tid]:
            smem_reduce[tid] = smem_reduce[tid + 4]
    al.syncthreads()
    if tid < 2:
        if smem_reduce[tid + 2] > smem_reduce[tid]:
            smem_reduce[tid] = smem_reduce[tid + 2]
    al.syncthreads()
    if tid < 1:
        if smem_reduce[tid + 1] > smem_reduce[tid]:
            smem_reduce[tid] = smem_reduce[tid + 1]
    al.syncthreads()
    max_val = smem_reduce[0]

    exp_val = al.exp(val - max_val)
    smem_vals[tid] = exp_val
    smem_reduce[tid] = exp_val
    al.syncthreads()

    if tid < 32:
        smem_reduce[tid] = smem_reduce[tid] + smem_reduce[tid + 32]
    al.syncthreads()
    if tid < 16:
        smem_reduce[tid] = smem_reduce[tid] + smem_reduce[tid + 16]
    al.syncthreads()
    if tid < 8:
        smem_reduce[tid] = smem_reduce[tid] + smem_reduce[tid + 8]
    al.syncthreads()
    if tid < 4:
        smem_reduce[tid] = smem_reduce[tid] + smem_reduce[tid + 4]
    al.syncthreads()
    if tid < 2:
        smem_reduce[tid] = smem_reduce[tid] + smem_reduce[tid + 2]
    al.syncthreads()
    if tid < 1:
        smem_reduce[tid] = smem_reduce[tid] + smem_reduce[tid + 1]
    al.syncthreads()
    sum_exp = smem_reduce[0]

    softmax_val = smem_vals[tid] / sum_exp
    tanh_val = al.tanh(softmax_val)
    result = tanh_val * scale
    out[block_b, tid, block_h, block_w] = al.convert(result, al.bf16)


def _prepare_bf16_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    combined_bias: torch.Tensor,
    scaling_factor: float,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device required.")

    B, C_in, D, H, W = x.shape

    x_bf16 = _prepare_bf16_contiguous(x)
    w_bf16 = _prepare_bf16_contiguous(weight)
    w_flat = w_bf16.reshape(-1).contiguous()
    b_bf16 = _prepare_bf16_contiguous(combined_bias)

    C_out = w_bf16.shape[1]
    intermediate = torch.empty((B, C_out, H, W), dtype=torch.bfloat16, device=x.device)

    grid_h = (H + TILE_H - 1) // TILE_H
    grid_w = (W + TILE_W - 1) // TILE_W

    conv_transpose_mean_bias_kernel[lambda: ((B, grid_h, grid_w), (BLOCK_SIZE, 1, 1))](
        x_bf16, w_flat, b_bf16, intermediate, B, D, H, W
    )

    output = torch.empty_like(intermediate)

    softmax_tanh_scale_kernel[lambda: ((B, H, W), (SOFTMAX_BLOCK, 1, 1))](
        intermediate, output, B, H, W, scaling_factor
    )

    return output.unsqueeze(2)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.explicit_bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        weight = self.conv_transpose.weight.data
        conv_bias = self.conv_transpose.bias.data
        explicit_b = self.explicit_bias.data.view(-1)
        total_bias = conv_bias + explicit_b
        return avelang_forward(x, weight, total_bias, self.scaling_factor)
