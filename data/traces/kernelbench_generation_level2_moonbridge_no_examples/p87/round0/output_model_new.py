import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def conv2d_mish_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C_IN: al.i32,
    C_OUT: al.i32,
    H_in: al.i32,
    W_in: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    num_tiles_h: al.i32,
    num_tiles_w: al.i32,
):
    batch = al.block_id(2)
    c_out_base = al.block_id(1) * 8
    spatial_id = al.block_id(0)

    tile_h = spatial_id // num_tiles_w
    tile_w = spatial_id % num_tiles_w

    row_local = al.thread_id(0)
    col_local = al.thread_id(1)

    TILE_H = 16
    TILE_W = 16

    row = tile_h * TILE_H + row_local
    col = tile_w * TILE_W + col_local

    in_base_h = tile_h * TILE_H
    in_base_w = tile_w * TILE_W

    # Shared memory: input tile for all channels (8 x 18 x 18)
    shared_input = al.make_shared((8, 18, 18), al.bf16)
    # Weight cache: 8 output channels x 8 input channels x 9 kernel positions
    shared_weight = al.make_shared((8, 8, 9), al.bf16)

    # 1D tensor views
    x_elems = N * C_IN * H_in * W_in
    x_layout = al.make_layout((x_elems,), (1,))
    x = al.make_tensor(x_ptr, al.bf16, x_layout)

    w_elems = C_OUT * C_IN * 9
    w_layout = al.make_layout((w_elems,), (1,))
    w = al.make_tensor(w_ptr, al.bf16, w_layout)

    b_elems = C_OUT
    b_layout = al.make_layout((b_elems,), (1,))
    b = al.make_tensor(b_ptr, al.bf16, b_layout)

    out_elems = N * C_OUT * H_out * W_out
    out_layout = al.make_layout((out_elems,), (1,))
    out = al.make_tensor(out_ptr, al.bf16, out_layout)

    x_n_offset = batch * C_IN * H_in * W_in
    tid = row_local * TILE_W + col_local

    # ----- Cooperative load of all input channels into shared memory -----
    idx1 = tid
    if idx1 < 324:
        r1 = idx1 // 18
        c1 = idx1 % 18
        in_h = in_base_h + r1
        in_w = in_base_w + c1
        if in_h < H_in and in_w < W_in:
            for c_in in al.range(C_IN):
                x_idx = x_n_offset + c_in * H_in * W_in + in_h * W_in + in_w
                shared_input[c_in, r1, c1] = x[x_idx]

    idx2 = tid + 256
    if idx2 < 324:
        r2 = idx2 // 18
        c2 = idx2 % 18
        in_h = in_base_h + r2
        in_w = in_base_w + c2
        if in_h < H_in and in_w < W_in:
            for c_in in al.range(C_IN):
                x_idx = x_n_offset + c_in * H_in * W_in + in_h * W_in + in_w
                shared_input[c_in, r2, c2] = x[x_idx]

    # ----- Cooperative load of weights: 8*8*9 = 576 values, 3 rounds -----
    STRIDE = C_IN * 9  # 72
    w_base = c_out_base * STRIDE

    w_idx1 = tid
    if w_idx1 < 576:
        ch1 = w_idx1 // STRIDE
        rem1 = w_idx1 % STRIDE
        ci1 = rem1 // 9
        k1 = rem1 % 9
        shared_weight[ch1, ci1, k1] = w[w_base + w_idx1]

    w_idx2 = tid + 256
    if w_idx2 < 576:
        ch2 = w_idx2 // STRIDE
        rem2 = w_idx2 % STRIDE
        ci2 = rem2 // 9
        k2 = rem2 % 9
        shared_weight[ch2, ci2, k2] = w[w_base + w_idx2]

    w_idx3 = tid + 512
    if w_idx3 < 576:
        ch3 = w_idx3 // STRIDE
        rem3 = w_idx3 % STRIDE
        ci3 = rem3 // 9
        k3 = rem3 % 9
        shared_weight[ch3, ci3, k3] = w[w_base + w_idx3]

    al.syncthreads()

    # ----- Compute for valid threads, 8 output channels per block -----
    if row < H_out and col < W_out:
        for ch in al.range(8):
            c_out = c_out_base + ch
            if c_out < C_OUT:
                acc = al.convert(0.0, al.f32)

                for c_in in al.range(8):
                    for kh in al.range(3):
                        for kw in al.range(3):
                            x_val = al.convert(shared_input[c_in, row_local + kh, col_local + kw], al.f32)
                            w_val = al.convert(shared_weight[ch, c_in, kh * 3 + kw], al.f32)
                            acc = acc + x_val * w_val

                acc = acc + al.convert(b[c_out], al.f32)

                # Mish activation
                exp_acc = al.exp(acc)
                softplus = al.log(al.convert(1.0, al.f32) + exp_acc)
                tanh_sp = al.tanh(softplus)
                result = acc * tanh_sp

                out_offset = batch * C_OUT * H_out * W_out + c_out * H_out * W_out + row * W_out + col
                out[out_offset] = al.convert(result, al.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        assert x.is_cuda, "Input must be on CUDA/HIP device"
        N, C_in, H_in, W_in = x.shape

        weight = self.conv.weight.data.to(device=x.device, dtype=torch.bfloat16).contiguous()

        sub_val = self.subtract_value_1 + self.subtract_value_2
        bias_f32 = self.conv.bias.data.to(device=x.device, dtype=torch.float32)
        adjusted_bias = (bias_f32 - sub_val).to(torch.bfloat16).contiguous()

        x_bf16 = x.to(torch.bfloat16).contiguous()

        TILE_H = 16
        TILE_W = 16
        CH_PER_BLOCK = 8

        KW = self.kernel_size
        H_out = H_in - KW + 1
        W_out = W_in - KW + 1

        num_tiles_h = (H_out + TILE_H - 1) // TILE_H
        num_tiles_w = (W_out + TILE_W - 1) // TILE_W
        num_ch_blocks = (self.out_channels + CH_PER_BLOCK - 1) // CH_PER_BLOCK

        out_bf16 = torch.empty(N, self.out_channels, H_out, W_out, dtype=torch.bfloat16, device=x.device)

        grid = (num_tiles_h * num_tiles_w, num_ch_blocks, N)
        block = (TILE_H, TILE_W, 1)

        conv2d_mish_kernel[lambda: (grid, block)](
            x_bf16, weight, adjusted_bias, out_bf16,
            N, C_in, self.out_channels, H_in, W_in, H_out, W_out,
            num_tiles_h, num_tiles_w,
        )

        return out_bf16
