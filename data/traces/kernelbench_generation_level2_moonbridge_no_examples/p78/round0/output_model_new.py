import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def conv_transpose3d_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.f32),
    output_ptr: al.Pointer(al.bf16),
    B: al.i32, C_in: al.i32, C_out: al.i32,
    D_in: al.i32, H_in: al.i32, W_in: al.i32,
    D_out: al.i32, H_out: al.i32, W_out: al.i32,
    K: al.i32, stride: al.i32, padding: al.i32,
):
    b_oc = al.block_id(0)
    spatial_block = al.block_id(1)
    tid = al.thread_id(0)
    b = b_oc // C_out
    oc = b_oc % C_out
    BLK = al.block_dim(0)
    idx = spatial_block * BLK + tid
    total = D_out * H_out * W_out

    if idx < total:
        d = idx // (H_out * W_out)
        hw = idx % (H_out * W_out)
        h = hw // W_out
        w = hw % W_out

        in_s0 = C_in * D_in * H_in * W_in
        in_s1 = D_in * H_in * W_in
        in_s2 = H_in * W_in
        in_layout = al.make_layout(
            (B, C_in, D_in, H_in, W_in),
            (in_s0, in_s1, in_s2, W_in, 1),
        )
        in_t = al.make_tensor(input_ptr, al.bf16, in_layout)

        w_s0 = C_out * K * K * K
        w_s1 = K * K * K
        w_s2 = K * K
        w_layout = al.make_layout(
            (C_in, C_out, K, K, K),
            (w_s0, w_s1, w_s2, K, 1),
        )
        w_t = al.make_tensor(weight_ptr, al.bf16, w_layout)

        acc = al.convert(0.0, al.f32)

        for kd in al.range(K):
            tmp_d = d + padding - kd
            d_div = tmp_d // stride
            if d_div * stride == tmp_d:
                d_in = d_div
                if d_in >= 0:
                    if d_in < D_in:
                        for kh in al.range(K):
                            tmp_h = h + padding - kh
                            h_div = tmp_h // stride
                            if h_div * stride == tmp_h:
                                h_in = h_div
                                if h_in >= 0:
                                    if h_in < H_in:
                                        for kw in al.range(K):
                                            tmp_w = w + padding - kw
                                            w_div = tmp_w // stride
                                            if w_div * stride == tmp_w:
                                                w_in = w_div
                                                if w_in >= 0:
                                                    if w_in < W_in:
                                                        for ic in al.range(C_in):
                                                            iv = al.convert(in_t[b, ic, d_in, h_in, w_in], al.f32)
                                                            wv = al.convert(w_t[ic, oc, kd, kh, kw], al.f32)
                                                            acc = acc + iv * wv

        b_layout = al.make_layout((C_out,), (1,))
        b_t = al.make_tensor(bias_ptr, al.f32, b_layout)
        acc = acc + b_t[oc]

        out_s0 = C_out * D_out * H_out * W_out
        out_s1 = D_out * H_out * W_out
        out_s2 = H_out * W_out
        out_layout = al.make_layout(
            (B, C_out, D_out, H_out, W_out),
            (out_s0, out_s1, out_s2, W_out, 1),
        )
        out_t = al.make_tensor(output_ptr, al.bf16, out_layout)
        out_t[b, oc, d, h, w] = al.convert(acc, al.bf16)


@avelang.jit
def maxpool_sum_kernel(
    inter_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32, C: al.i32,
    D_inter: al.i32, H_inter: al.i32, W_inter: al.i32,
    D_out: al.i32, H_out: al.i32, W_out: al.i32,
    pool_size: al.i32,
):
    b = al.block_id(0)
    spatial_block = al.block_id(1)
    tid = al.thread_id(0)
    BLK = al.block_dim(0)
    idx = spatial_block * BLK + tid
    total = D_out * H_out * W_out

    if idx < total:
        df = idx // (H_out * W_out)
        hw = idx % (H_out * W_out)
        hf = hw // W_out
        wf = hw % W_out

        is0 = C * D_inter * H_inter * W_inter
        is1 = D_inter * H_inter * W_inter
        is2 = H_inter * W_inter
        in_layout = al.make_layout(
            (B, C, D_inter, H_inter, W_inter),
            (is0, is1, is2, W_inter, 1),
        )
        in_t = al.make_tensor(inter_ptr, al.bf16, in_layout)

        sum_acc = al.convert(0.0, al.f32)
        d_base = df * pool_size
        h_base = hf * pool_size
        w_base = wf * pool_size

        for c in al.range(C):
            fv = al.convert(in_t[b, c, d_base, h_base, w_base], al.f32)
            max_val = fv
            for kd in al.range(pool_size):
                d_cur = d_base + kd
                for kh in al.range(pool_size):
                    h_cur = h_base + kh
                    for kw in al.range(pool_size):
                        w_cur = w_base + kw
                        v = al.convert(in_t[b, c, d_cur, h_cur, w_cur], al.f32)
                        if v > max_val:
                            max_val = v
            sum_acc = sum_acc + max_val

        os0 = D_out * H_out * W_out
        os2 = H_out * W_out
        out_layout = al.make_layout(
            (B, 1, D_out, H_out, W_out),
            (os0, os0, os2, W_out, 1),
        )
        out_t = al.make_tensor(out_ptr, al.bf16, out_layout)
        out_t[b, 0, df, hf, wf] = al.convert(sum_acc, al.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding,
        )

    def forward(self, x):
        B, C_in, D_in, H_in, W_in = x.shape
        C_out = self.conv_transpose.out_channels
        K = int(self.conv_transpose.kernel_size[0])
        stride = int(self.conv_transpose.stride[0])
        padding = int(self.conv_transpose.padding[0])

        D_out = (D_in - 1) * stride - 2 * padding + (K - 1) + 1
        H_out = (H_in - 1) * stride - 2 * padding + (K - 1) + 1
        W_out = (W_in - 1) * stride - 2 * padding + (K - 1) + 1

        pool_size = 6
        D_final = (D_out - pool_size) // pool_size + 1
        H_final = (H_out - pool_size) // pool_size + 1
        W_final = (W_out - pool_size) // pool_size + 1

        device = x.device

        x_bf16 = x.contiguous().to(torch.bfloat16)
        w_bf16 = self.conv_transpose.weight.detach().contiguous().to(torch.bfloat16)
        b_f32 = self.conv_transpose.bias.detach().contiguous().float()

        inter = torch.empty(
            B, C_out, D_out, H_out, W_out,
            dtype=torch.bfloat16, device=device,
        )

        BLOCK = 256
        grid_x = int(B) * int(C_out)
        grid_y = (int(D_out) * int(H_out) * int(W_out) + BLOCK - 1) // BLOCK

        conv_transpose3d_kernel[lambda: ((grid_x, grid_y, 1), (BLOCK, 1, 1))](
            x_bf16.data_ptr(), w_bf16.data_ptr(), b_f32.data_ptr(), inter.data_ptr(),
            B, C_in, C_out, D_in, H_in, W_in,
            D_out, H_out, W_out, K, stride, padding,
        )

        out_bf16 = torch.empty(
            B, 1, D_final, H_final, W_final,
            dtype=torch.bfloat16, device=device,
        )

        grid_x2 = int(B)
        grid_y2 = (int(D_final) * int(H_final) * int(W_final) + BLOCK - 1) // BLOCK

        maxpool_sum_kernel[lambda: ((grid_x2, grid_y2, 1), (BLOCK, 1, 1))](
            inter.data_ptr(), out_bf16.data_ptr(),
            B, C_out, D_out, H_out, W_out,
            D_final, H_final, W_final, pool_size,
        )

        return out_bf16
