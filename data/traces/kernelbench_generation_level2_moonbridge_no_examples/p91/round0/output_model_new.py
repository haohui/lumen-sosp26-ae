import torch
import torch.nn as nn
import avelang
import avelang.language as al

# Problem constants
BATCH_SIZE = 128
IN_C = 64
OUT_C = 128
H_IN = 64
W_IN = 64
KERNEL = 4
STRIDE = 2
PAD = 1
OUT_PAD = 1
H_OUT = (H_IN - 1) * STRIDE - 2 * PAD + KERNEL + OUT_PAD  # 129
W_OUT = H_OUT  # 129
SCALING_FACTOR = 2.0

TILE_SIZE = 256
TILE_SIZE = 256


@avelang.jit
def conv_transpose2d_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    B: al.i32,
    IC: al.i32,
    OC: al.i32,
    H: al.i32,
    W: al.i32,
    OH: al.i32,
    OW: al.i32,
    K: al.i32,
    S: al.i32,
    P: al.i32,
    SPATIAL_SIZE: al.i32,
    TILE_SIZE: al.i32,
):
    input_layout = al.make_layout((B, IC, H, W), (IC * H * W, H * W, W, 1))
    input_t = al.make_tensor(input_ptr, al.bf16, input_layout)

    weight_layout = al.make_layout((IC, OC, K, K), (OC * K * K, K * K, K, 1))
    weight_t = al.make_tensor(weight_ptr, al.bf16, weight_layout)

    bias_layout = al.make_layout((OC,), (1,))
    bias_t = al.make_tensor(bias_ptr, al.bf16, bias_layout)

    output_layout = al.make_layout((B, OC, OH, OW), (OC * OH * OW, OH * OW, OW, 1))
    output_t = al.make_tensor(output_ptr, al.bf16, output_layout)

    block_spatial = al.block_id(0)
    block_bc = al.block_id(1)
    tid = al.thread_id(0)

    # Shared memory for weight slice: (IC, K, K) = (64, 4, 4)
    num_weight_elems = IC * K * K  # 64 * 4 * 4 = 1024
    weight_shared = al.make_shared((1024,), al.bf16)

    # Preload weight slice for this oc into shared memory
    oc_preload = block_bc % OC
    for w_idx in al.range(tid, num_weight_elems, TILE_SIZE):
        w_ic = w_idx // 16  # K * K = 16
        w_rest = w_idx % 16
        w_ky = w_rest // 4  # K = 4
        w_kx = w_rest % 4
        weight_shared[w_idx] = weight_t[w_ic, oc_preload, w_ky, w_kx]
    al.syncthreads()

    spatial_idx = block_spatial * TILE_SIZE + tid

    if spatial_idx < SPATIAL_SIZE:
        b = block_bc // OC
        oc = oc_preload

        if b < B:
            oh = spatial_idx // OW
            ow = spatial_idx % OW

            accum = al.convert(0.0, al.f32)

            ky_start = (oh + P) % S
            kx_start = (ow + P) % S

            for ic in al.range(IC):
                for ky in al.range(ky_start, K, S):
                    ih = (oh + P - ky) // S
                    if ih >= 0:
                        if ih < H:
                            for kx in al.range(kx_start, K, S):
                                iw = (ow + P - kx) // S
                                if iw >= 0:
                                    if iw < W:
                                        inp_val = al.convert(input_t[b, ic, ih, iw], al.f32)
                                        w_val = al.convert(weight_shared[ic * 16 + ky * 4 + kx], al.f32)
                                        accum = accum + inp_val * w_val

                bias_val = al.convert(bias_t[oc], al.f32)
                accum = accum + bias_val
                output_t[b, oc, oh, ow] = al.convert(accum, al.bf16)


@avelang.jit
def softmax_channel_kernel(
    input_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    B: al.i32,
    C: al.constexpr,
    H: al.i32,
    W: al.i32,
):
    input_layout = al.make_layout((B, C, H, W), (C * H * W, H * W, W, 1))
    input_t = al.make_tensor(input_ptr, al.bf16, input_layout)

    output_layout = al.make_layout((B, C, H, W), (C * H * W, H * W, W, 1))
    output_t = al.make_tensor(output_ptr, al.bf16, output_layout)

    block_hw = al.block_id(0)
    b = al.block_id(1)
    tid = al.thread_id(0)
    c = tid

    h = block_hw // W
    w = block_hw % W

    val = al.convert(input_t[b, c, h, w], al.f32)

    shared = al.make_shared((C,), al.f32)

    # --- Max reduction using shuffle + shared memory ---
    # Warp-level max reduction
    max_val = val
    other = al.shuffle_down(max_val, 32, 64)
    if other > max_val:
        max_val = other
    other = al.shuffle_down(max_val, 16, 64)
    if other > max_val:
        max_val = other
    other = al.shuffle_down(max_val, 8, 64)
    if other > max_val:
        max_val = other
    other = al.shuffle_down(max_val, 4, 64)
    if other > max_val:
        max_val = other
    other = al.shuffle_down(max_val, 2, 64)
    if other > max_val:
        max_val = other
    other = al.shuffle_down(max_val, 1, 64)
    if other > max_val:
        max_val = other

    # Cross-warp: lane 0 of each warp writes to shared
    warp_id = tid // 64
    lane_id = tid % 64
    if lane_id == 0:
        shared[warp_id] = max_val
    al.syncthreads()

    # Thread 0 combines warp results
    num_warps = C // 64
    if tid == 0:
        global_max = shared[0]
        for w in al.range(1, num_warps):
            if shared[w] > global_max:
                global_max = shared[w]
        shared[0] = global_max
    al.syncthreads()

    global_max = shared[0]

    # --- Sum reduction ---
    shifted = al.exp(val - global_max)

    # Warp-level sum reduction
    sum_val = shifted
    sum_val = sum_val + al.shuffle_down(sum_val, 32, 64)
    sum_val = sum_val + al.shuffle_down(sum_val, 16, 64)
    sum_val = sum_val + al.shuffle_down(sum_val, 8, 64)
    sum_val = sum_val + al.shuffle_down(sum_val, 4, 64)
    sum_val = sum_val + al.shuffle_down(sum_val, 2, 64)
    sum_val = sum_val + al.shuffle_down(sum_val, 1, 64)

    if lane_id == 0:
        shared[warp_id] = sum_val
    al.syncthreads()

    if tid == 0:
        global_sum = shared[0]
        for w in al.range(1, num_warps):
            global_sum = global_sum + shared[w]
        shared[0] = global_sum
    al.syncthreads()

    global_sum = shared[0]
    result = shifted / global_sum
    output_t[b, c, h, w] = al.convert(result, al.bf16)


@avelang.jit
def bias_scale_sigmoid_kernel(
    input_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    B: al.i32,
    C: al.i32,
    H: al.i32,
    W: al.i32,
    scale: al.constexpr,
    SPATIAL_SIZE: al.i32,
    TILE_SIZE: al.i32,
):
    input_layout = al.make_layout((B, C, H, W), (C * H * W, H * W, W, 1))
    input_t = al.make_tensor(input_ptr, al.bf16, input_layout)

    bias_layout = al.make_layout((C, 1, 1), (1, 1, 1))
    bias_t = al.make_tensor(bias_ptr, al.bf16, bias_layout)

    output_layout = al.make_layout((B, C, H, W), (C * H * W, H * W, W, 1))
    output_t = al.make_tensor(output_ptr, al.bf16, output_layout)

    block_spatial = al.block_id(0)
    block_bc = al.block_id(1)
    tid = al.thread_id(0)

    spatial_idx = block_spatial * TILE_SIZE + tid

    if spatial_idx < SPATIAL_SIZE:
        b = block_bc // C
        c = block_bc % C

        if b < B:
            h = spatial_idx // W
            w = spatial_idx % W

            val = al.convert(input_t[b, c, h, w], al.f32)
            bias_val = al.convert(bias_t[c, 0, 0], al.f32)
            scaled = (val + bias_val) * scale
            # sigmoid: 1 / (1 + exp(-x))
            neg_one = al.convert(-1.0, al.f32)
            one = al.convert(1.0, al.f32)
            sigmoid_val = one / (one + al.exp(neg_one * scaled))
            output_t[b, c, h, w] = al.convert(sigmoid_val, al.bf16)


def _make_contiguous(t):
    if not t.is_contiguous():
        t = t.contiguous()
    return t


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        # Extract conv parameters
        weight = self.conv_transpose.weight.data  # (IC, OC, K, K)
        conv_bias = self.conv_transpose.bias.data  # (OC,)
        extra_bias = self.bias.data  # (OC, 1, 1)

        B = x.shape[0]
        IC = x.shape[1]
        H_in = x.shape[2]
        W_in = x.shape[3]

        OC = weight.shape[1]
        K = weight.shape[2]
        S = self.conv_transpose.stride[0]
        P = self.conv_transpose.padding[0]
        OP = self.conv_transpose.output_padding[0]

        H_out = (H_in - 1) * S - 2 * P + K + OP
        W_out = (W_in - 1) * S - 2 * P + K + OP

        spatial_size = H_out * W_out
        spatial_tiles = (spatial_size + TILE_SIZE - 1) // TILE_SIZE
        bc_extent = B * OC

        # Ensure tensors are contiguous
        x = _make_contiguous(x)
        weight = _make_contiguous(weight)
        conv_bias = _make_contiguous(conv_bias)
        extra_bias = _make_contiguous(extra_bias)

        # Allocate intermediate buffers
        conv_out = torch.empty(B, OC, H_out, W_out, dtype=torch.bfloat16, device=x.device)

        # Launch conv kernel
        conv_transpose2d_kernel[lambda: (
            (spatial_tiles, bc_extent, 1),
            (TILE_SIZE, 1, 1),
        )](
            x, weight, conv_bias, conv_out,
            B, IC, OC, H_in, W_in, H_out, W_out,
            K, S, P, spatial_size, TILE_SIZE,
        )

        # Allocate softmax output buffer
        softmax_out = torch.empty(B, OC, H_out, W_out, dtype=torch.bfloat16, device=x.device)

        # Launch softmax kernel
        softmax_channel_kernel[lambda: (
            (H_out * W_out, B, 1),
            (OC, 1, 1),
        )](
            conv_out, softmax_out,
            B, OC, H_out, W_out,
        )

        # Allocate final output buffer
        final_out = torch.empty(B, OC, H_out, W_out, dtype=torch.bfloat16, device=x.device)

        # Launch bias+scale+sigmoid kernel
        bias_scale_sigmoid_kernel[lambda: (
            (spatial_tiles, bc_extent, 1),
            (TILE_SIZE, 1, 1),
        )](
            softmax_out, extra_bias, final_out,
            B, OC, H_out, W_out,
            float(self.scaling_factor),
            spatial_size, TILE_SIZE,
        )

        return final_out
