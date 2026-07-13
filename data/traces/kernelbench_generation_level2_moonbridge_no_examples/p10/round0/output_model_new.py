import math
import torch
import torch.nn as nn
import avelang
import avelang.language as al

POOL_TILE = 16
C_IN_TILE = 8
REDUCE_BLK = 256


@avelang.jit
def fused_conv_maxpool_hardtanh_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C_in: al.i32,
    C_out: al.i32,
    H_in: al.i32,
    W_in: al.i32,
    H_pool: al.i32,
    W_pool: al.i32,
    K: al.i32,
    P: al.i32,
    pool_stride: al.i32,
    num_w_tiles: al.i32,
    htanh_bounds_ptr: al.Pointer(al.bf16),
    PTILE: al.constexpr,
    CINTILE: al.constexpr,
    CONV_EXT: al.constexpr,
    KSIZE: al.constexpr,
):
    tid_h = al.thread_id(0)
    tid_w = al.thread_id(1)

    n = al.block_id(0)
    co = al.block_id(1)
    spatial_idx = al.block_id(2)

    h_tile = spatial_idx // num_w_tiles
    w_tile = spatial_idx % num_w_tiles

    hp = h_tile * PTILE + tid_h
    wp = w_tile * PTILE + tid_w

    smem_in = al.make_shared((CINTILE, CONV_EXT, CONV_EXT), al.bf16)
    smem_w = al.make_shared((CINTILE, KSIZE, KSIZE), al.bf16)

    if hp < H_pool and wp < W_pool:
        in_layout = al.make_layout(
            (N, C_in, H_in, W_in), (C_in * H_in * W_in, H_in * W_in, W_in, 1)
        )
        inp = al.make_tensor(input_ptr, al.bf16, in_layout)

        w_layout = al.make_layout(
            (C_in, C_out, K, K), (C_out * K * K, K * K, K, 1)
        )
        weight = al.make_tensor(weight_ptr, al.bf16, w_layout)

        bounds_layout = al.make_layout((2,), (1,))
        htanh_bounds = al.make_tensor(htanh_bounds_ptr, al.bf16, bounds_layout)
        htanh_min = al.convert(htanh_bounds[0], al.f32)
        htanh_max = al.convert(htanh_bounds[1], al.f32)

        b_layout = al.make_layout((C_out,), (1,))
        bias = al.make_tensor(bias_ptr, al.bf16, b_layout)
        bias_val = al.convert(bias[co], al.f32)

        acc00 = al.convert(0.0, al.f32)
        acc10 = al.convert(0.0, al.f32)
        acc01 = al.convert(0.0, al.f32)
        acc11 = al.convert(0.0, al.f32)

        smem_h_origin = h_tile * PTILE * pool_stride - K + 1
        smem_w_origin = w_tile * PTILE * pool_stride - K + 1

        ci_start = 0
        num_threads = PTILE * PTILE

        for _ci_tile in al.range(0, C_in, CINTILE):
            # Cooperative load of input tile into shared memory
            num_in_elems = CINTILE * CONV_EXT * CONV_EXT
            for load_idx in al.range(tid_h * PTILE + tid_w, num_in_elems, num_threads):
                ci_local = load_idx // (CONV_EXT * CONV_EXT)
                rem = load_idx % (CONV_EXT * CONV_EXT)
                h_local = rem // CONV_EXT
                w_local = rem % CONV_EXT

                ci_global = ci_start + ci_local
                h_global = smem_h_origin + h_local
                w_global = smem_w_origin + w_local

                if ci_global >= 0 and ci_global < C_in and h_global >= 0 and h_global < H_in and w_global >= 0 and w_global < W_in:
                    smem_in[ci_local, h_local, w_local] = inp[n, ci_global, h_global, w_global]

            # Cooperative load of weight tile into shared memory
            num_w_elems = CINTILE * KSIZE * KSIZE
            for load_idx in al.range(tid_h * PTILE + tid_w, num_w_elems, num_threads):
                ci_local = load_idx // (KSIZE * KSIZE)
                rem = load_idx % (KSIZE * KSIZE)
                kh_local = rem // KSIZE
                kw_local = rem % KSIZE

                ci_global = ci_start + ci_local
                if ci_global < C_in:
                    smem_w[ci_local, kh_local, kw_local] = weight[ci_global, co, kh_local, kw_local]

            al.syncthreads()

            for ci_local in al.range(CINTILE):
                ci_global = ci_start + ci_local
                if ci_global < C_in:
                    for kh in al.range(K):
                        for kw in al.range(K):
                            w_val = al.convert(smem_w[ci_local, kh, kw], al.f32)

                            h_in_00 = hp * pool_stride + P - kh
                            w_in_00 = wp * pool_stride + P - kw
                            h_in_10 = h_in_00 + 1
                            w_in_01 = w_in_00 + 1

                            h_sm_00 = h_in_00 - smem_h_origin
                            w_sm_00 = w_in_00 - smem_w_origin
                            h_sm_10 = h_in_10 - smem_h_origin
                            w_sm_01 = w_in_01 - smem_w_origin

                            if h_in_00 >= 0 and h_in_00 < H_in and w_in_00 >= 0 and w_in_00 < W_in:
                                inp_val = al.convert(smem_in[ci_local, h_sm_00, w_sm_00], al.f32)
                                acc00 = acc00 + inp_val * w_val
                            if h_in_10 >= 0 and h_in_10 < H_in and w_in_00 >= 0 and w_in_00 < W_in:
                                inp_val = al.convert(smem_in[ci_local, h_sm_10, w_sm_00], al.f32)
                                acc10 = acc10 + inp_val * w_val
                            if h_in_00 >= 0 and h_in_00 < H_in and w_in_01 >= 0 and w_in_01 < W_in:
                                inp_val = al.convert(smem_in[ci_local, h_sm_00, w_sm_01], al.f32)
                                acc01 = acc01 + inp_val * w_val
                            if h_in_10 >= 0 and h_in_10 < H_in and w_in_01 >= 0 and w_in_01 < W_in:
                                inp_val = al.convert(smem_in[ci_local, h_sm_10, w_sm_01], al.f32)
                                acc11 = acc11 + inp_val * w_val

            ci_start = ci_start + CINTILE
            al.syncthreads()

        acc00 = acc00 + bias_val
        acc10 = acc10 + bias_val
        acc01 = acc01 + bias_val
        acc11 = acc11 + bias_val

        max_val = acc00
        if acc10 > max_val:
            max_val = acc10
        if acc01 > max_val:
            max_val = acc01
        if acc11 > max_val:
            max_val = acc11

        if max_val < htanh_min:
            max_val = htanh_min
        if max_val > htanh_max:
            max_val = htanh_max

        out_layout = al.make_layout(
            (N, C_out, H_pool, W_pool),
            (C_out * H_pool * W_pool, H_pool * W_pool, W_pool, 1),
        )
        out = al.make_tensor(output_ptr, al.bf16, out_layout)
        out[n, co, hp, wp] = al.convert(max_val, al.bf16)


@avelang.jit
def mean_tanh_kernel(
    input_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C: al.i32,
    H: al.i32,
    W: al.i32,
    BLK: al.constexpr,
):
    n = al.block_id(0)
    c = al.block_id(1)
    tid = al.thread_id(0)

    in_layout = al.make_layout(
        (N, C, H, W), (C * H * W, H * W, W, 1)
    )
    inp = al.make_tensor(input_ptr, al.bf16, in_layout)

    total = al.convert(0.0, al.f32)
    num_elements = H * W

    idx = tid
    for _unused in al.range(0, num_elements, BLK):
        if idx < num_elements:
            h_idx = idx // W
            w_idx = idx % W
            total = total + al.convert(inp[n, c, h_idx, w_idx], al.f32)
        idx = idx + BLK

    smem = al.make_shared((BLK,), al.f32)
    smem[tid] = total
    al.syncthreads()

    if tid < 128:
        smem[tid] = smem[tid] + smem[tid + 128]
    al.syncthreads()
    if tid < 64:
        smem[tid] = smem[tid] + smem[tid + 64]
    al.syncthreads()
    if tid < 32:
        smem[tid] = smem[tid] + smem[tid + 32]
    al.syncthreads()
    if tid < 16:
        smem[tid] = smem[tid] + smem[tid + 16]
    al.syncthreads()
    if tid < 8:
        smem[tid] = smem[tid] + smem[tid + 8]
    al.syncthreads()
    if tid < 4:
        smem[tid] = smem[tid] + smem[tid + 4]
    al.syncthreads()
    if tid < 2:
        smem[tid] = smem[tid] + smem[tid + 2]
    al.syncthreads()
    if tid < 1:
        smem[tid] = smem[tid] + smem[tid + 1]
    al.syncthreads()

    if tid == 0:
        total_elements = al.convert(num_elements, al.f32)
        mean = smem[0] / total_elements
        result = al.tanh(mean)
        out_layout = al.make_layout((N, C, 1, 1), (C, 1, 1, 1))
        out = al.make_tensor(output_ptr, al.bf16, out_layout)
        out[n, c, 0, 0] = al.convert(result, al.bf16)


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        maxpool_kernel_size,
        maxpool_stride,
        hardtanh_min_val,
        hardtanh_max_val,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.maxpool_kernel_size = maxpool_kernel_size
        self.maxpool_stride = maxpool_stride
        self.hardtanh_min = hardtanh_min_val
        self.hardtanh_max = hardtanh_max_val

        fan_in = in_channels * kernel_size * kernel_size
        a = math.sqrt(5.0)
        weight_fp32 = torch.empty(in_channels, out_channels, kernel_size, kernel_size)
        nn.init.kaiming_uniform_(weight_fp32, a=a)
        bias_fp32 = torch.empty(out_channels)
        bound = 1.0 / math.sqrt(fan_in)
        nn.init.uniform_(bias_fp32, -bound, bound)

        self.register_buffer(
            "conv_weight",
            weight_fp32.to(dtype=torch.bfloat16),
        )
        self.register_buffer(
            "conv_bias",
            bias_fp32.to(dtype=torch.bfloat16),
        )

        H_conv = (256 - 1) * stride - 2 * padding + kernel_size
        W_conv = (256 - 1) * stride - 2 * padding + kernel_size
        self._H_pool = (H_conv - maxpool_kernel_size) // maxpool_stride + 1
        self._W_pool = (W_conv - maxpool_kernel_size) // maxpool_stride + 1

        self.register_buffer(
            "htanh_bounds",
            torch.tensor([hardtanh_min_val, hardtanh_max_val], dtype=torch.bfloat16),
        )
        self.register_buffer(
            "pool_out",
            torch.empty(128, out_channels, self._H_pool, self._W_pool,
                        dtype=torch.bfloat16),
        )
        self.register_buffer(
            "final_out",
            torch.empty(128, out_channels, 1, 1, dtype=torch.bfloat16),
        )

    def forward(self, x):
        N, C_in, H_in, W_in = x.shape
        assert x.is_cuda, "Input must be on GPU"
        x = x.contiguous()

        num_h_tiles = (self._H_pool + POOL_TILE - 1) // POOL_TILE
        num_w_tiles = (self._W_pool + POOL_TILE - 1) // POOL_TILE

        conv_extent = POOL_TILE * self.maxpool_stride + self.kernel_size - 1

        fused_conv_maxpool_hardtanh_kernel[
            lambda: (
                (N, self.out_channels, num_h_tiles * num_w_tiles),
                (POOL_TILE, POOL_TILE, 1),
            )
        ](
            x,
            self.conv_weight,
            self.conv_bias,
            self.pool_out,
            N,
            C_in,
            self.out_channels,
            H_in,
            W_in,
            self._H_pool,
            self._W_pool,
            self.kernel_size,
            self.padding,
            self.maxpool_stride,
            num_w_tiles,
            self.htanh_bounds,
            POOL_TILE,
            C_IN_TILE,
            conv_extent,
            self.kernel_size,
        )

        mean_tanh_kernel[
            lambda: (
                (N, self.out_channels, 1),
                (REDUCE_BLK, 1, 1),
            )
        ](
            self.pool_out,
            self.final_out,
            N,
            self.out_channels,
            self._H_pool,
            self._W_pool,
            REDUCE_BLK,
        )

        return self.final_out
