import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE: al.constexpr = 256
TILE_D: al.constexpr = 2
TILE_H: al.constexpr = 8
TILE_W: al.constexpr = 8
TILE_D_IN: al.constexpr = 4
TILE_H_IN: al.constexpr = 10
TILE_W_IN: al.constexpr = 10
SHM_IN_SIZE: al.constexpr = 1536
SHM_W_SIZE: al.constexpr = 1536


def _div_up(a: int, b: int) -> int:
    return (a + b - 1) // b


@avelang.jit
def _conv3d_tiled_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    B: al.i32,
    C_in: al.i32,
    C_out: al.i32,
    D_in: al.i32,
    H_in: al.i32,
    W_in: al.i32,
    K: al.i32,
    D_out: al.i32,
    H_out: al.i32,
    W_out: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    num_d_tiles = (D_out + TILE_D - 1) // TILE_D
    num_h_tiles = (H_out + TILE_H - 1) // TILE_H
    num_w_tiles = (W_out + TILE_W - 1) // TILE_W
    tiles_per_batch = num_d_tiles * num_h_tiles * num_w_tiles

    b_idx = bid // tiles_per_batch
    tile_idx = bid - b_idx * tiles_per_batch
    tile_d = tile_idx // (num_h_tiles * num_w_tiles)
    rem = tile_idx - tile_d * num_h_tiles * num_w_tiles
    tile_h = rem // num_w_tiles
    tile_w = rem - tile_h * num_w_tiles

    d_start = tile_d * TILE_D
    h_start = tile_h * TILE_H
    w_start = tile_w * TILE_W

    td_actual = TILE_D
    if d_start + TILE_D > D_out:
        td_actual = D_out - d_start
    th_actual = TILE_H
    if h_start + TILE_H > H_out:
        th_actual = H_out - h_start
    tw_actual = TILE_W
    if w_start + TILE_W > W_out:
        tw_actual = W_out - w_start

    shm_in = al.make_shared((SHM_IN_SIZE,), al.bf16)
    shm_w = al.make_shared((SHM_W_SIZE,), al.bf16)

    # Flat input view
    in_total = B * C_in * D_in * H_in * W_in
    layout_in_flat = al.make_layout((in_total,), (1,))
    inp = al.make_tensor(input_ptr, al.bf16, layout_in_flat)

    # Flat weight view
    w_total = C_out * C_in * K * K * K
    layout_w_flat = al.make_layout((w_total,), (1,))
    wgt = al.make_tensor(weight_ptr, al.bf16, layout_w_flat)

    # Bias
    layout_b_flat = al.make_layout((C_out,), (1,))
    bias_t = al.make_tensor(bias_ptr, al.bf16, layout_b_flat)

    # Cooperative load of weight into shared memory
    for idx in al.range(tid, w_total, BLOCK_SIZE):
        shm_w[idx] = wgt[idx]
    al.syncthreads()

    # Cooperative load of input tile into shared memory
    in_d_elems = TILE_D_IN
    in_h_elems = TILE_H_IN
    in_w_elems = TILE_W_IN
    in_tile_total = C_in * in_d_elems * in_h_elems * in_w_elems
    in_dhw = in_d_elems * in_h_elems * in_w_elems
    in_hw = in_h_elems * in_w_elems

    b_in_base = b_idx * (C_in * D_in * H_in * W_in)

    for idx in al.range(tid, in_tile_total, BLOCK_SIZE):
        ic = idx // in_dhw
        r = idx - ic * in_dhw
        id = r // in_hw
        r = r - id * in_hw
        ih = r // in_w_elems
        iw = r - ih * in_w_elems
        in_d = d_start + id
        in_h = h_start + ih
        in_w = w_start + iw
        if in_d < D_in and in_h < H_in and in_w < W_in:
            lin_idx = b_in_base + ic * (D_in * H_in * W_in) + in_d * (H_in * W_in) + in_h * W_in + in_w
            shm_in[idx] = inp[lin_idx]
        else:
            shm_in[idx] = al.convert(0.0, al.bf16)
    al.syncthreads()

    # Flat output view
    out_total = B * C_out * D_out * H_out * W_out
    layout_out_flat = al.make_layout((out_total,), (1,))
    out = al.make_tensor(output_ptr, al.bf16, layout_out_flat)

    # Each thread computes output elements from shared memory
    tile_out_total = td_actual * th_actual * tw_actual * C_out
    oc_stride = th_actual * tw_actual * C_out
    d_stride = tw_actual * C_out
    h_stride = C_out

    w_stride_k = K * K * K
    ic_w_stride = C_in * w_stride_k
    k_hw = K * K

    b_out_base = b_idx * (C_out * D_out * H_out * W_out)

    for out_idx in al.range(tid, tile_out_total, BLOCK_SIZE):
        local_d = out_idx // oc_stride
        rem = out_idx - local_d * oc_stride
        local_h = rem // d_stride
        rem = rem - local_h * d_stride
        local_w = rem // h_stride
        oc = rem - local_w * h_stride

        d = d_start + local_d
        h = h_start + local_h
        w = w_start + local_w

        if d < D_out and h < H_out and w < W_out:
            acc = al.convert(bias_t[oc], al.f32)
            wgt_oc = oc * ic_w_stride

            for ic in al.range(C_in):
                ic_shm = ic * in_dhw
                wgt_ic = wgt_oc + ic * w_stride_k
                for kd in al.range(K):
                    shm_d = ic_shm + (local_d + kd) * in_hw
                    wgt_kd = wgt_ic + kd * k_hw
                    for kh in al.range(K):
                        shm_dh = shm_d + (local_h + kh) * in_w_elems
                        wgt_kh = wgt_kd + kh * K
                        for kw in al.range(K):
                            inp_val = al.convert(shm_in[shm_dh + local_w + kw], al.f32)
                            w_val = al.convert(shm_w[wgt_kh + kw], al.f32)
                            acc = acc + inp_val * w_val

            out_lin = b_out_base + oc * (D_out * H_out * W_out) + d * (H_out * W_out) + h * W_out + w
            out[out_lin] = al.convert(acc, al.bf16)


@avelang.jit
def _softmax_kernel(
    input_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    B: al.i32,
    C: al.i32,
    D: al.i32,
    H: al.i32,
    W: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    total_positions = B * D * H * W
    global_id = bid * BLOCK_SIZE + tid

    if global_id < total_positions:
        hw = H * W
        dhw = D * hw

        b_idx = global_id // dhw
        rem = global_id - b_idx * dhw
        d = rem // hw
        rem = rem - d * hw
        h = rem // W
        w = rem - h * W

        in_total = B * C * D * H * W
        layout_flat = al.make_layout((in_total,), (1,))
        inp = al.make_tensor(input_ptr, al.bf16, layout_flat)
        out = al.make_tensor(output_ptr, al.bf16, layout_flat)

        b_base = b_idx * (C * D * H * W)
        spat_off = d * hw + h * W + w
        ch_stride = D * H * W
        base_idx = b_base + spat_off

        max_val = al.convert(inp[base_idx], al.f32)
        for c in al.range(1, C):
            val = al.convert(inp[base_idx + c * ch_stride], al.f32)
            if val > max_val:
                max_val = val

        exp_sum = al.convert(0.0, al.f32)
        for c in al.range(C):
            val = al.convert(inp[base_idx + c * ch_stride], al.f32)
            exp_sum = exp_sum + al.exp(val - max_val)

        inv_sum = al.convert(1.0, al.f32) / exp_sum
        for c in al.range(C):
            val = al.convert(inp[base_idx + c * ch_stride], al.f32)
            result = al.exp(val - max_val) * inv_sum
            out[base_idx + c * ch_stride] = al.convert(result, al.bf16)


@avelang.jit
def _maxpool3d_kernel(
    input_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    B: al.i32,
    C: al.i32,
    D_in: al.i32,
    H_in: al.i32,
    W_in: al.i32,
    D_out: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    pool_size: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    total_out = B * C * D_out * H_out * W_out
    global_id = bid * BLOCK_SIZE + tid

    if global_id < total_out:
        oc_stride = D_out * H_out * W_out
        d_stride = H_out * W_out
        h_stride = W_out

        b_idx = global_id // (C * oc_stride)
        rem = global_id - b_idx * C * oc_stride
        c = rem // oc_stride
        rem = rem - c * oc_stride
        d = rem // d_stride
        rem = rem - d * d_stride
        h = rem // h_stride
        w = rem - h * h_stride

        in_total = B * C * D_in * H_in * W_in
        layout_in_flat = al.make_layout((in_total,), (1,))
        inp = al.make_tensor(input_ptr, al.bf16, layout_in_flat)

        out_total = B * C * D_out * H_out * W_out
        layout_out_flat = al.make_layout((out_total,), (1,))
        out = al.make_tensor(output_ptr, al.bf16, layout_out_flat)

        b_base = b_idx * (C * D_in * H_in * W_in)
        c_off = c * (D_in * H_in * W_in)
        in_hw = H_in * W_in

        in_d_base = d * pool_size
        in_h_base = h * pool_size
        in_w_base = w * pool_size

        base_idx = b_base + c_off + in_d_base * in_hw + in_h_base * W_in + in_w_base

        max_val = al.convert(inp[base_idx], al.f32)

        for pd in al.range(pool_size):
            d_off = base_idx + pd * in_hw
            for ph in al.range(pool_size):
                h_off = d_off + ph * W_in
                for pw in al.range(pool_size):
                    w_off = h_off + pw
                    val = al.convert(inp[w_off], al.f32)
                    if val > max_val:
                        max_val = val

        out_idx = b_idx * (C * oc_stride) + c * oc_stride + d * d_stride + h * h_stride + w
        out[out_idx] = al.convert(max_val, al.bf16)


def _to_bf16_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_conv3d_softmax_maxpool(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = _to_bf16_contiguous(x)
    w_bf16 = _to_bf16_contiguous(weight)
    b_bf16 = _to_bf16_contiguous(bias)

    B, C_in, D_in, H_in, W_in = x_bf16.shape
    C_out, w_C_in, K, _, _ = w_bf16.shape

    D_out = D_in - K + 1
    H_out = H_in - K + 1
    W_out = W_in - K + 1

    # Tiled conv3d
    conv_out = torch.empty(
        (B, C_out, D_out, H_out, W_out),
        device=x_bf16.device,
        dtype=torch.bfloat16,
    )
    num_d_tiles = _div_up(D_out, TILE_D)
    num_h_tiles = _div_up(H_out, TILE_H)
    num_w_tiles = _div_up(W_out, TILE_W)
    total_tiles = B * num_d_tiles * num_h_tiles * num_w_tiles
    _conv3d_tiled_kernel[lambda: ((total_tiles, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_bf16, w_bf16, b_bf16, conv_out,
        B, C_in, C_out, D_in, H_in, W_in, K, D_out, H_out, W_out,
    )

    # Softmax
    softmax_out = torch.empty_like(conv_out)
    total_positions = B * D_out * H_out * W_out
    num_blocks_sm = _div_up(total_positions, BLOCK_SIZE)
    _softmax_kernel[lambda: ((num_blocks_sm, 1, 1), (BLOCK_SIZE, 1, 1))](
        conv_out, softmax_out, B, C_out, D_out, H_out, W_out,
    )

    # First maxpool
    pool_size = 2
    D1 = D_out // pool_size
    H1 = H_out // pool_size
    W1 = W_out // pool_size
    pool1_out = torch.empty(
        (B, C_out, D1, H1, W1),
        device=x_bf16.device,
        dtype=torch.bfloat16,
    )
    total_pool1 = B * C_out * D1 * H1 * W1
    num_blocks_p1 = _div_up(total_pool1, BLOCK_SIZE)
    _maxpool3d_kernel[lambda: ((num_blocks_p1, 1, 1), (BLOCK_SIZE, 1, 1))](
        softmax_out, pool1_out, B, C_out, D_out, H_out, W_out, D1, H1, W1, pool_size,
    )

    # Second maxpool
    D2 = D1 // pool_size
    H2 = H1 // pool_size
    W2 = W1 // pool_size
    pool2_out = torch.empty(
        (B, C_out, D2, H2, W2),
        device=x_bf16.device,
        dtype=torch.bfloat16,
    )
    total_pool2 = B * C_out * D2 * H2 * W2
    num_blocks_p2 = _div_up(total_pool2, BLOCK_SIZE)
    _maxpool3d_kernel[lambda: ((num_blocks_p2, 1, 1), (BLOCK_SIZE, 1, 1))](
        pool1_out, pool2_out, B, C_out, D1, H1, W1, D2, H2, W2, pool_size,
    )

    return pool2_out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pool_kernel_size = pool_kernel_size

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))

        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return avelang_conv3d_softmax_maxpool(x, self.weight, self.bias)
