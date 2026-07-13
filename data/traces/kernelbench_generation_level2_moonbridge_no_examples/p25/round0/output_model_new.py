import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def conv_partial_min_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    N: al.i32,
    H_IN: al.i32,
    W_IN: al.i32,
    H_OUT: al.i32,
    W_OUT: al.i32,
):
    bid0 = al.block_id(0)
    n = bid0 // 16
    g = bid0 % 16
    tile_h = al.block_id(1)
    tile_w = al.block_id(2)
    th = al.thread_id(0)
    tw = al.thread_id(1)

    h = tile_h * 16 + th
    w = tile_w * 16 + tw

    in_layout = al.make_layout((N, 16, H_IN, W_IN), (16 * H_IN * W_IN, H_IN * W_IN, W_IN, 1))
    input_t = al.make_tensor(input_ptr, al.bf16, in_layout)

    wt_layout = al.make_layout((64, 16, 3, 3), (16 * 9, 9, 3, 1))
    weight_t = al.make_tensor(weight_ptr, al.bf16, wt_layout)

    bias_layout = al.make_layout((64,), (1,))
    bias_t = al.make_tensor(bias_ptr, al.bf16, bias_layout)

    out_layout = al.make_layout((N, 16, H_OUT, W_OUT), (16 * H_OUT * W_OUT, H_OUT * W_OUT, W_OUT, 1))
    output_t = al.make_tensor(output_ptr, al.bf16, out_layout)

    # Shared memory for input tile: 16 x 18 x 18 = 5184 elements
    smem_in = al.make_shared((5184,), al.bf16)

    # Cooperative load: 256 threads, each ~21 elements
    tid = th * 16 + tw
    for idx in al.range(tid, 5184, 256):
        ic = idx // 324
        local_idx = idx % 324
        lh = local_idx // 18
        lw = local_idx % 18
        gh = tile_h * 16 + lh
        gw = tile_w * 16 + lw
        if gh < H_IN and gw < W_IN:
            smem_in[idx] = input_t[n, ic, gh, gw]
    al.syncthreads()

    if h < H_OUT and w < W_OUT:
        c_base = g * 4

        # Compute min over 4 output channels
        acc = al.convert(0.0, al.f32)
        for ic in al.range(16):
            for kh in al.range(3):
                for kw in al.range(3):
                    smem_idx = ic * 324 + (th + kh) * 18 + (tw + kw)
                    inp = al.convert(smem_in[smem_idx], al.f32)
                    wt = al.convert(weight_t[c_base, ic, kh, kw], al.f32)
                    acc = acc + inp * wt
        local_min = acc + al.convert(bias_t[c_base], al.f32)

        for c_off in al.range(1, 4):
            c_out = c_base + c_off
            acc = al.convert(0.0, al.f32)
            for ic in al.range(16):
                for kh in al.range(3):
                    for kw in al.range(3):
                        smem_idx = ic * 324 + (th + kh) * 18 + (tw + kw)
                        inp = al.convert(smem_in[smem_idx], al.f32)
                        wt = al.convert(weight_t[c_out, ic, kh, kw], al.f32)
                        acc = acc + inp * wt
            acc = acc + al.convert(bias_t[c_out], al.f32)
            if acc < local_min:
                local_min = acc

        output_t[n, g, h, w] = al.convert(local_min, al.bf16)


@avelang.jit
def final_min_tanh_kernel(
    input_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    N: al.i32,
    H: al.i32,
    W: al.i32,
):
    n = al.block_id(0)
    tile_h = al.block_id(1)
    tile_w = al.block_id(2)
    th = al.thread_id(0)
    tw = al.thread_id(1)

    h = tile_h * 16 + th
    w = tile_w * 16 + tw

    in_layout = al.make_layout((N, 16, H, W), (16 * H * W, H * W, W, 1))
    input_t = al.make_tensor(input_ptr, al.bf16, in_layout)

    out_layout = al.make_layout((N, 1, H, W), (H * W, H * W, W, 1))
    output_t = al.make_tensor(output_ptr, al.bf16, out_layout)

    if h < H and w < W:
        m = al.convert(input_t[n, 0, h, w], al.f32)
        for g in al.range(1, 16):
            val = al.convert(input_t[n, g, h, w], al.f32)
            if val < m:
                m = val
        val = al.tanh(m)
        val = al.tanh(val)
        output_t[n, 0, h, w] = al.convert(val, al.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.register_buffer("weight_bf16", conv.weight.data.clone().to(torch.bfloat16))
        self.register_buffer("bias_bf16", conv.bias.data.clone().to(torch.bfloat16))

    def forward(self, x):
        N, C, H, W = x.shape
        H_out = H - self.kernel_size + 1
        W_out = W - self.kernel_size + 1

        device = x.device
        x_bf16 = x.to(torch.bfloat16).contiguous()
        weight = self.weight_bf16.to(device).contiguous()
        bias = self.bias_bf16.to(device).contiguous()

        # Intermediate: partial mins, 16 groups -> (N, 16, H_out, W_out)
        partial = torch.empty(N, 16, H_out, W_out, dtype=torch.bfloat16, device=device)

        grid_h = (H_out + 15) // 16
        grid_w = (W_out + 15) // 16

        conv_partial_min_kernel[lambda: ((N * 16, grid_h, grid_w), (16, 16, 1))](
            x_bf16, weight, bias, partial,
            N, H, W, H_out, W_out,
        )

        # Final min + tanh: (N, 1, H_out, W_out)
        final_out = torch.empty(N, 1, H_out, W_out, dtype=torch.bfloat16, device=device)

        final_min_tanh_kernel[lambda: ((N, grid_h, grid_w), (16, 16, 1))](
            partial, final_out,
            N, H_out, W_out,
        )

        return final_out
