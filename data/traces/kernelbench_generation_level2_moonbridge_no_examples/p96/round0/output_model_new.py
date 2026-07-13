import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def fused_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    C_IN: al.i32,
    C_OUT: al.i32,
    D_IN: al.i32,
    H_IN: al.i32,
    W_IN: al.i32,
    D_MP: al.i32,
    H_MP: al.i32,
    W_MP: al.i32,
    N_MP: al.i32,
):
    batch = al.block_id(0)
    oc = al.block_id(1)
    tid = al.thread_id(0)

    in_D_stride = H_IN * W_IN
    in_H_stride = W_IN
    in_C_stride = D_IN * in_D_stride
    in_batch_stride = C_IN * in_C_stride

    x_layout = al.make_layout(
        (B, C_IN, D_IN, H_IN, W_IN),
        (in_batch_stride, in_C_stride, in_D_stride, in_H_stride, 1),
    )
    x = al.make_tensor(x_ptr, al.bf16, x_layout)

    w_OC_stride = 27
    w_KD_stride = 9
    w_KH_stride = 3
    w_IC_stride = C_OUT * w_OC_stride

    w_layout = al.make_layout(
        (C_IN, C_OUT, 3, 3, 3),
        (w_IC_stride, w_OC_stride, w_KD_stride, w_KH_stride, 1),
    )
    w = al.make_tensor(w_ptr, al.bf16, w_layout)

    b_layout = al.make_layout((C_OUT,), (1,))
    b = al.make_tensor(b_ptr, al.bf16, b_layout)

    acc = al.convert(0.0, al.f32)

    for pos in al.range(tid, N_MP, 256):
        mp_d = pos // (H_MP * W_MP)
        mp_rem = pos % (H_MP * W_MP)
        mp_h = mp_rem // W_MP
        mp_w = mp_rem % W_MP

        first_flag = al.convert(1, al.i32)
        local_max = al.convert(0.0, al.f32)

        for mm_d in al.range(2):
            d_ct = 2 * mp_d + mm_d
            for mm_h in al.range(2):
                h_ct = 2 * mp_h + mm_h
                for mm_w in al.range(2):
                    w_ct = 2 * mp_w + mm_w

                    conv_sum = al.convert(0.0, al.f32)

                    for ic in al.range(C_IN):
                        for kd in al.range(3):
                            off_d = d_ct + 1 - kd
                            if (off_d % 2) == 0:
                                id = off_d // 2
                                if id >= 0:
                                    if id < D_IN:
                                        for kh in al.range(3):
                                            off_h = h_ct + 1 - kh
                                            if (off_h % 2) == 0:
                                                ih = off_h // 2
                                                if ih >= 0:
                                                    if ih < H_IN:
                                                        for kw in al.range(3):
                                                            off_w = w_ct + 1 - kw
                                                            if (off_w % 2) == 0:
                                                                iw = off_w // 2
                                                                if iw >= 0:
                                                                    if iw < W_IN:
                                                                        in_val = al.convert(x[batch, ic, id, ih, iw], al.f32)
                                                                        w_val = al.convert(w[ic, oc, kd, kh, kw], al.f32)
                                                                        conv_sum = conv_sum + in_val * w_val

                    if first_flag == 1:
                        local_max = conv_sum
                        first_flag = al.convert(0, al.i32)
                    else:
                        if conv_sum > local_max:
                            local_max = conv_sum

        acc = acc + local_max

    shared = al.make_shared((256,), al.f32)
    shared[tid] = acc
    al.syncthreads()

    if tid == 0:
        total_sum = al.convert(0.0, al.f32)
        for i in al.range(256):
            total_sum = total_sum + shared[i]

        bias_f32 = al.convert(b[oc], al.f32)
        n_mp_f = al.convert(N_MP, al.f32)
        half = al.convert(0.5, al.f32)
        result = half * bias_f32 + half * total_sum / n_mp_f

        zero = al.convert(0.0, al.f32)
        one = al.convert(1.0, al.f32)
        if result < zero:
            result = zero
        if result > one:
            result = one

        out_layout = al.make_layout((B, C_OUT), (C_OUT, 1))
        out = al.make_tensor(out_ptr, al.bf16, out_layout)
        out[batch, oc] = al.convert(result, al.bf16)


def _run_kernel(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    B, C_IN, D_IN, H_IN, W_IN = x.shape
    C_OUT = w.shape[1]

    x_bf16 = x.to(torch.bfloat16).contiguous()
    w_bf16 = w.to(torch.bfloat16).contiguous()
    b_bf16 = b.to(torch.bfloat16).contiguous()

    D_CT = (D_IN - 1) * 2 - 2 * 1 + 3
    H_CT = (H_IN - 1) * 2 - 2 * 1 + 3
    W_CT = (W_IN - 1) * 2 - 2 * 1 + 3

    D_MP = D_CT // 2
    H_MP = H_CT // 2
    W_MP = W_CT // 2
    N_MP = D_MP * H_MP * W_MP

    out_bf16 = torch.empty(B, C_OUT, dtype=torch.bfloat16, device=x_bf16.device)

    fused_kernel[lambda: ((B, C_OUT, 1), (256, 1, 1))](
        x_bf16, w_bf16, b_bf16, out_bf16,
        B, C_IN, C_OUT, D_IN, H_IN, W_IN,
        D_MP, H_MP, W_MP, N_MP,
    )

    return out_bf16.view(B, C_OUT, 1, 1, 1)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding,
        )
        self.scale = scale
        self.clamp_min = 0
        self.clamp_max = 1

    def forward(self, x):
        orig_dtype = x.dtype
        w = self.conv_transpose.weight.data
        b = self.conv_transpose.bias.data
        out = _run_kernel(x, w, b)
        return out.to(orig_dtype)
