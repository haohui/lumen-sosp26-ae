import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def conv_transpose_3d_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    y_ptr: al.Pointer(al.f32),
    N: al.i32,
    C_in: al.i32,
    C_out: al.i32,
    D_in: al.i32,
    H_in: al.i32,
    W_in: al.i32,
    D_out: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    K: al.i32,
    SCALE: al.constexpr,
):
    n = al.block_id(0)
    oc = al.block_id(1)
    spat_blk = al.block_id(2)
    tid = al.thread_id(0)

    S_out = D_out * H_out * W_out
    gid = spat_blk * al.convert(256, al.i32) + tid

    if gid < S_out:
        od = gid // (H_out * W_out)
        remnant = gid - od * (H_out * W_out)
        oh = remnant // W_out
        ow = remnant - oh * W_out

        x_total = N * C_in * D_in * H_in * W_in
        x_layout = al.make_layout((x_total,), (al.convert(1, al.i32),))
        x = al.make_tensor(x_ptr, al.bf16, x_layout)

        w_total = C_in * C_out * K * K * K
        w_layout = al.make_layout((w_total,), (al.convert(1, al.i32),))
        w = al.make_tensor(w_ptr, al.bf16, w_layout)

        w_s_ic = C_out * K * K * K
        w_s_oc = K * K * K
        w_s_kd = K * K
        w_s_kh = K

        x_s_n = C_in * D_in * H_in * W_in
        x_s_ic = D_in * H_in * W_in
        x_s_id = H_in * W_in
        x_s_ih = W_in

        acc = al.convert(0.0, al.f32)
        zero = al.convert(0, al.i32)
        scale_f32 = al.convert(SCALE, al.f32)

        for ic in al.range(C_in):
            for kd in al.range(K):
                id_val = od - kd
                if id_val >= zero:
                    if id_val < D_in:
                        for kh in al.range(K):
                            ih_val = oh - kh
                            if ih_val >= zero:
                                if ih_val < H_in:
                                    for kw in al.range(K):
                                        iw_val = ow - kw
                                        if iw_val >= zero:
                                            if iw_val < W_in:
                                                x_idx = (
                                                    n * x_s_n
                                                    + ic * x_s_ic
                                                    + id_val * x_s_id
                                                    + ih_val * x_s_ih
                                                    + iw_val
                                                )
                                                w_idx = (
                                                    ic * w_s_ic
                                                    + oc * w_s_oc
                                                    + kd * w_s_kd
                                                    + kh * w_s_kh
                                                    + kw
                                                )
                                                xv = al.convert(x[x_idx], al.f32)
                                                wv = al.convert(w[w_idx], al.f32)
                                                acc = acc + xv * wv

        y_total = N * C_out * D_out * H_out * W_out
        y_layout = al.make_layout((y_total,), (al.convert(1, al.i32),))
        y = al.make_tensor(y_ptr, al.f32, y_layout)

        y_s_n = C_out * D_out * H_out * W_out
        y_s_oc = D_out * H_out * W_out
        y_s_od = H_out * W_out
        y_s_oh = W_out
        y_idx = n * y_s_n + oc * y_s_oc + od * y_s_od + oh * y_s_oh + ow
        y[y_idx] = scale_f32 * acc


class ModelNew(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1
    ):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor
        self.eps = eps
        self.momentum = momentum

        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        self.batch_norm = nn.BatchNorm3d(out_channels, eps=eps, momentum=momentum)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x):
        N, C_in, D_in, H_in, W_in = x.shape
        C_out = self.out_channels
        K = self.kernel_size

        D_out = (D_in - 1) + K
        H_out = (H_in - 1) + K
        W_out = (W_in - 1) + K
        S_out = D_out * H_out * W_out

        dev = x.device

        x_bf16 = x.contiguous().to(torch.bfloat16)
        w = self.conv_transpose.weight.detach().contiguous().to(torch.bfloat16)

        y_intermediate = torch.empty(
            N, C_out, D_out, H_out, W_out, dtype=torch.float32, device=dev
        )

        BLOCK = 256
        grid_spatial = (S_out + BLOCK - 1) // BLOCK

        conv_transpose_3d_kernel[lambda: ((N, C_out, grid_spatial), (BLOCK, 1, 1))](
            x_bf16.data_ptr(),
            w.data_ptr(),
            y_intermediate.data_ptr(),
            N,
            C_in,
            C_out,
            D_in,
            H_in,
            W_in,
            D_out,
            H_out,
            W_out,
            K,
            self.scale_factor,
        )

        bn = self.batch_norm
        y_bn = torch.nn.functional.batch_norm(
            y_intermediate.float(),
            bn.running_mean.float(),
            bn.running_var.float(),
            weight=bn.weight.float(),
            bias=bn.bias.float(),
            training=self.training,
            momentum=self.momentum,
            eps=self.eps,
        )
        y_pool = self.global_avg_pool(y_bn)
        return y_pool.to(torch.bfloat16)
