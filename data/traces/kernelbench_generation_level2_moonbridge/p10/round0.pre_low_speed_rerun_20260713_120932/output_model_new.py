import torch
import torch.nn as nn
import avelang
import avelang.language as al

# Problem constants from input_model.py
BATCH_SIZE = 128
IN_CHANNELS = 64
OUT_CHANNELS = 64
HEIGHT = 256
WIDTH = 256
KERNEL_SIZE = 3
KH = 3
KW = 3
MAXPOOL_STRIDE = 2
PADDING = 1

# Derived
H_OUT = HEIGHT
W_OUT = WIDTH
MP_H = H_OUT // MAXPOOL_STRIDE  # 128
MP_W = W_OUT // MAXPOOL_STRIDE  # 128

# Tile dimensions (16x16 maxpool, 8 input channels per tile)
TILE_H = 16
TILE_W = 16
TILE_SIZE = TILE_H * TILE_W  # 256
C_TILE = 8
TILE_IN_H = TILE_H * 2 + 2  # 34
TILE_IN_W = TILE_W * 2 + 2  # 34
TOTAL_SHM = TILE_IN_H * TILE_IN_W * C_TILE  # 34*34*8 = 9248
WEIGHT_PER_CH = IN_CHANNELS * KH * KW  # 576

REDUCE_BLOCK = 256
MP_TOTAL = MP_H * MP_W  # 16384


@avelang.jit
def fused_convtranspose_maxpool_hardtanh_partial_mean_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    partial_sum_ptr: al.Pointer(al.f32),
    H: al.i32,
    W: al.i32,
    C_in: al.i32,
    C_out: al.i32,
    mp_h: al.i32,
    mp_w: al.i32,
    num_spatial_blocks: al.i32,
    num_col_blocks: al.i32,
):
    tid = al.thread_id(0)
    mp_col_block = al.block_id(0)
    mp_row_block = al.block_id(1)
    combined = al.block_id(2)

    batch_idx = combined // C_out
    ch_out = combined - batch_idx * C_out

    mp_row_start = mp_row_block * TILE_H
    mp_col_start = mp_col_block * TILE_W
    mp_local_row = tid // TILE_W
    mp_local_col = tid - mp_local_row * TILE_W

    mp_row = mp_row_start + mp_local_row
    mp_col = mp_col_start + mp_local_col

    local_partial = al.convert(0.0, al.f32)

    if mp_row < mp_h and mp_col < mp_w:
        acc00 = al.convert(0.0, al.f32)
        acc10 = al.convert(0.0, al.f32)
        acc01 = al.convert(0.0, al.f32)
        acc11 = al.convert(0.0, al.f32)

        input_layout = al.make_layout(
            (BATCH_SIZE, C_in, H, W),
            (C_in * H * W, H * W, W, 1),
        )
        inp = al.make_tensor(input_ptr, al.bf16, input_layout)

        weight_layout = al.make_layout(
            (C_in, C_out, KH, KW),
            (C_out * KH * KW, KH * KW, KW, 1),
        )
        wgt = al.make_tensor(weight_ptr, al.bf16, weight_layout)

        shm_in = al.make_shared((TILE_IN_H, TILE_IN_W, C_TILE), al.bf16)
        shm_wt = al.make_shared((WEIGHT_PER_CH,), al.bf16)

        zero_bf16 = al.convert(0.0, al.bf16)

        # Pre-load weights for this output channel into LDS
        for i in al.range(tid, WEIGHT_PER_CH, TILE_SIZE):
            c_idx = i // (KH * KW)
            k_idx = i - c_idx * (KH * KW)
            di = k_idx // KW
            dj = k_idx - di * KW
            shm_wt[i] = wgt[c_idx, ch_out, di, dj]
        al.syncthreads()

        # Pre-compute base shared-memory offsets for this thread
        base_row_0 = mp_local_row * 2 + 2
        base_row_1 = mp_local_row * 2 + 3
        base_col_0 = mp_local_col * 2 + 2
        base_col_1 = mp_local_col * 2 + 3

        for c_tile in al.range(0, C_in, C_TILE):
            # Cooperative load of input tile into shared memory
            for i in al.range(tid, TOTAL_SHM, TILE_SIZE):
                local_c = i % C_TILE
                tmp = i // C_TILE
                local_col = tmp % TILE_IN_W
                local_row = tmp // TILE_IN_W
                global_row = mp_row_start * 2 - 1 + local_row
                global_col = mp_col_start * 2 - 1 + local_col
                if global_row >= 0 and global_row < H and global_col >= 0 and global_col < W:
                    shm_in[local_row, local_col, local_c] = inp[batch_idx, c_tile + local_c, global_row, global_col]
                else:
                    shm_in[local_row, local_col, local_c] = zero_bf16
            al.syncthreads()

            for c_off in al.range(C_TILE):
                c_flat = (c_tile + c_off) * (KH * KW)
                for di in al.range(KH):
                    sr0 = base_row_0 - di
                    sr1 = base_row_1 - di
                    wb = c_flat + di * KW
                    for dj in al.range(KW):
                        w_val = al.convert(shm_wt[wb + dj], al.f32)
                        sc0 = base_col_0 - dj
                        sc1 = base_col_1 - dj

                        acc00 = acc00 + w_val * al.convert(shm_in[sr0, sc0, c_off], al.f32)
                        acc10 = acc10 + w_val * al.convert(shm_in[sr1, sc0, c_off], al.f32)
                        acc01 = acc01 + w_val * al.convert(shm_in[sr0, sc1, c_off], al.f32)
                        acc11 = acc11 + w_val * al.convert(shm_in[sr1, sc1, c_off], al.f32)

            al.syncthreads()

        # MaxPool 2x2
        max_val = acc00
        if acc10 > max_val:
            max_val = acc10
        if acc01 > max_val:
            max_val = acc01
        if acc11 > max_val:
            max_val = acc11

        # Add bias
        bias_layout = al.make_layout((C_out,), (1,))
        b = al.make_tensor(bias_ptr, al.bf16, bias_layout)
        max_val = max_val + al.convert(b[ch_out], al.f32)

        # Hardtanh
        hmin = al.convert(-1.0, al.f32)
        hmax = al.convert(1.0, al.f32)
        if max_val < hmin:
            max_val = hmin
        if max_val > hmax:
            max_val = hmax

        local_partial = max_val

    # Block-level tree reduction
    smem_red = al.make_shared((TILE_SIZE,), al.f32)
    smem_red[tid] = local_partial
    al.syncthreads()

    if tid < 128:
        smem_red[tid] = smem_red[tid] + smem_red[tid + 128]
    al.syncthreads()
    if tid < 64:
        smem_red[tid] = smem_red[tid] + smem_red[tid + 64]
    al.syncthreads()
    if tid < 32:
        smem_red[tid] = smem_red[tid] + smem_red[tid + 32]
    al.syncthreads()
    if tid < 16:
        smem_red[tid] = smem_red[tid] + smem_red[tid + 16]
    al.syncthreads()
    if tid < 8:
        smem_red[tid] = smem_red[tid] + smem_red[tid + 8]
    al.syncthreads()
    if tid < 4:
        smem_red[tid] = smem_red[tid] + smem_red[tid + 4]
    al.syncthreads()
    if tid < 2:
        smem_red[tid] = smem_red[tid] + smem_red[tid + 2]
    al.syncthreads()
    if tid < 1:
        smem_red[tid] = smem_red[tid] + smem_red[tid + 1]

    if tid == 0:
        block_idx = mp_row_block * num_col_blocks + mp_col_block
        ps_layout = al.make_layout(
            (BATCH_SIZE * C_out, num_spatial_blocks),
            (num_spatial_blocks, 1),
        )
        ps = al.make_tensor(partial_sum_ptr, al.f32, ps_layout)
        ps[batch_idx * C_out + ch_out, block_idx] = smem_red[0]


@avelang.jit
def reduce_partial_mean_tanh_kernel(
    partial_sum_ptr: al.Pointer(al.f32),
    output_ptr: al.Pointer(al.bf16),
    num_blocks: al.i32,
    C_out: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    batch_idx = bid // C_out
    ch_out = bid - batch_idx * C_out

    smem = al.make_shared((REDUCE_BLOCK,), al.f32)

    local_sum = al.convert(0.0, al.f32)

    ps_layout = al.make_layout(
        (BATCH_SIZE * C_out, num_blocks),
        (num_blocks, 1),
    )
    ps = al.make_tensor(partial_sum_ptr, al.f32, ps_layout)

    row = batch_idx * C_out + ch_out
    for i in al.range(tid, num_blocks, REDUCE_BLOCK):
        local_sum = local_sum + ps[row, i]

    smem[tid] = local_sum
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

    if tid == 0:
        N_f32 = al.convert(MP_TOTAL, al.f32)
        mean = smem[0] / N_f32
        result = al.tanh(mean)

        output_layout = al.make_layout(
            (BATCH_SIZE, C_out, 1, 1),
            (C_out, 1, 1, 1),
        )
        out = al.make_tensor(output_ptr, al.bf16, output_layout)
        out[batch_idx, ch_out, 0, 0] = al.convert(result, al.bf16)


def _prepare_bf16(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def _model_forward(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Input must be on CUDA/HIP device."

    x_bf16 = _prepare_bf16(x)
    w_bf16 = _prepare_bf16(weight)
    b_bf16 = _prepare_bf16(bias)

    B, C_in, H, W = x_bf16.shape
    C_out = w_bf16.shape[1]

    assert H == HEIGHT and W == WIDTH
    assert C_in == IN_CHANNELS
    assert C_out == OUT_CHANNELS

    mp_h = MP_H
    mp_w = MP_W

    grid_rows = (mp_h + TILE_H - 1) // TILE_H
    grid_cols = (mp_w + TILE_W - 1) // TILE_W
    num_spatial_blocks = grid_rows * grid_cols

    partial_sums = torch.empty(
        (B * C_out, num_spatial_blocks),
        device=x_bf16.device,
        dtype=torch.float32,
    )

    grid = (grid_cols, grid_rows, B * C_out)

    fused_convtranspose_maxpool_hardtanh_partial_mean_kernel[
        lambda: (grid, (TILE_SIZE, 1, 1))
    ](x_bf16, w_bf16, b_bf16, partial_sums, H, W, C_in, C_out, mp_h, mp_w, num_spatial_blocks, grid_cols)

    output = torch.empty(
        (B, C_out, 1, 1),
        device=x_bf16.device,
        dtype=torch.bfloat16,
    )

    reduce_grid = (B * C_out, 1, 1)
    reduce_partial_mean_tanh_kernel[
        lambda: (reduce_grid, (REDUCE_BLOCK, 1, 1))
    ](partial_sums, output, num_spatial_blocks, C_out)

    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding
        )

    def forward(self, x):
        return _model_forward(x, self.conv_transpose.weight, self.conv_transpose.bias)


def get_inputs():
    return [torch.rand(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)]


def get_init_inputs():
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, 1, 1, 2, 2, -1.0, 1.0]
