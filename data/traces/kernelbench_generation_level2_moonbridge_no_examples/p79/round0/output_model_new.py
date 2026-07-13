import torch
import torch.nn as nn
import avelang
import avelang.language as al


# ---------------------------------------------------------------------------
# Kernel 1: 3D convolution (bf16 output, matching reference)
# ---------------------------------------------------------------------------
@avelang.jit
def conv3d_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    N: al.i32,
    IC: al.i32,
    OC: al.i32,
    D_in: al.i32,
    H_in: al.i32,
    W_in: al.i32,
    D_out: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    KD: al.i32,
    KH: al.i32,
    KW: al.i32,
    PAD: al.i32,
    inp_stride_N: al.i32,
    inp_stride_IC: al.i32,
    inp_stride_D: al.i32,
    inp_stride_H: al.i32,
    out_stride_N: al.i32,
    out_stride_OC: al.i32,
    out_stride_D: al.i32,
    out_stride_H: al.i32,
    wgt_stride_OC: al.i32,
    wgt_stride_IC: al.i32,
    wgt_stride_KD: al.i32,
    wgt_stride_KH: al.i32,
):
    n = al.block_id(0) // D_out
    d = al.block_id(0) % D_out
    h = al.block_id(1)
    w = al.block_id(2)
    oc = al.thread_id(0)

    inp_total = N * IC * D_in * H_in * W_in
    inp = al.make_tensor(input_ptr, al.bf16, al.make_layout((inp_total,), (1,)))
    wgt_total = OC * IC * KD * KH * KW
    wgt = al.make_tensor(weight_ptr, al.bf16, al.make_layout((wgt_total,), (1,)))
    bias_t = al.make_tensor(bias_ptr, al.bf16, al.make_layout((OC,), (1,)))
    out_total = N * OC * D_out * H_out * W_out
    out = al.make_tensor(output_ptr, al.bf16, al.make_layout((out_total,), (1,)))

    acc = al.convert(0.0, al.f32)
    for ic in al.range(IC):
        for kd in al.range(KD):
            in_d = d + kd - PAD
            if in_d >= 0 and in_d < D_in:
                for kh in al.range(KH):
                    in_h = h + kh - PAD
                    if in_h >= 0 and in_h < H_in:
                        for kw in al.range(KW):
                            in_w = w + kw - PAD
                            if in_w >= 0 and in_w < W_in:
                                in_idx = n*inp_stride_N + ic*inp_stride_IC + in_d*inp_stride_D + in_h*inp_stride_H + in_w
                                wgt_idx = oc*wgt_stride_OC + ic*wgt_stride_IC + kd*wgt_stride_KD + kh*wgt_stride_KH + kw
                                acc = acc + al.convert(inp[in_idx], al.f32) * al.convert(wgt[wgt_idx], al.f32)

    acc = acc + al.convert(bias_t[oc], al.f32)
    out_idx = n*out_stride_N + oc*out_stride_OC + d*out_stride_D + h*out_stride_H + w
    out[out_idx] = al.convert(acc, al.bf16)


# ---------------------------------------------------------------------------
# Kernel 2: per-(sample,channel) mean and variance from bf16 input
# ---------------------------------------------------------------------------
@avelang.jit
def instancenorm_stats_kernel(
    input_ptr: al.Pointer(al.bf16),
    mean_ptr: al.Pointer(al.f32),
    var_ptr: al.Pointer(al.f32),
    N: al.i32,
    C: al.i32,
    D: al.i32,
    H: al.i32,
    W: al.i32,
    inp_stride_N: al.i32,
    inp_stride_C: al.i32,
    inp_stride_D: al.i32,
    inp_stride_H: al.i32,
):
    n = al.block_id(0)
    c = al.block_id(1)
    tid = al.thread_id(0)
    num_threads = al.block_dim(0)
    total_spatial = D * H * W

    inp_total = N * C * D * H * W
    inp = al.make_tensor(input_ptr, al.bf16, al.make_layout((inp_total,), (1,)))

    # -- Pass 1: compute mean -------------------------------------------------
    local_sum = al.convert(0.0, al.f32)
    for idx in al.range(tid, total_spatial, num_threads):
        d = idx // (H * W)
        rem = idx % (H * W)
        h = rem // W
        w = rem % W
        flat_idx = n*inp_stride_N + c*inp_stride_C + d*inp_stride_D + h*inp_stride_H + w
        val = al.convert(inp[flat_idx], al.f32)
        local_sum = local_sum + val

    smem = al.make_shared((256,), al.f32)
    smem[tid] = local_sum
    al.syncthreads()
    stride = 1
    for _ in al.range(8):
        if tid % (stride * 2) == 0:
            other = tid + stride
            if other < 256:
                smem[tid] = smem[tid] + smem[other]
        stride = stride * 2
        al.syncthreads()

    count = al.convert(total_spatial, al.f32)
    mean_val = smem[0] / count

    # broadcast mean to all threads
    smem[0] = mean_val
    al.syncthreads()
    mean_val = smem[0]

    # -- Pass 2: compute centered variance ------------------------------------
    local_sqdiff = al.convert(0.0, al.f32)
    for idx in al.range(tid, total_spatial, num_threads):
        d = idx // (H * W)
        rem = idx % (H * W)
        h = rem // W
        w = rem % W
        flat_idx = n*inp_stride_N + c*inp_stride_C + d*inp_stride_D + h*inp_stride_H + w
        diff = al.convert(inp[flat_idx], al.f32) - mean_val
        local_sqdiff = local_sqdiff + diff * diff

    smem[tid] = local_sqdiff
    al.syncthreads()
    stride = 1
    for _ in al.range(8):
        if tid % (stride * 2) == 0:
            other = tid + stride
            if other < 256:
                smem[tid] = smem[tid] + smem[other]
        stride = stride * 2
        al.syncthreads()

    var_val = smem[0] / count

    if tid == 0:
        mean_out = al.make_tensor(mean_ptr, al.f32, al.make_layout((N, C), (C, 1)))
        var_out = al.make_tensor(var_ptr, al.f32, al.make_layout((N, C), (C, 1)))
        mean_out[n, c] = mean_val
        var_out[n, c] = var_val


# ---------------------------------------------------------------------------
# Kernel 3: clamp → second-multiplier → max over channels (input already normalized)
# ---------------------------------------------------------------------------
@avelang.jit
def clamp_mult_max_kernel(
    inp_ptr: al.Pointer(al.bf16),
    multiplier_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C: al.i32,
    D_out: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    conv_stride_N: al.i32,
    conv_stride_C: al.i32,
    conv_stride_D: al.i32,
    conv_stride_H: al.i32,
    out_stride_N: al.i32,
    out_stride_D: al.i32,
    out_stride_H: al.i32,
    clamp_min: al.constexpr,
    clamp_max: al.constexpr,
):
    n = al.block_id(0) // D_out
    d = al.block_id(0) % D_out
    h = al.block_id(1)
    w = al.block_id(2)

    inp_total = N * C * D_out * H_out * W_out
    inp = al.make_tensor(inp_ptr, al.bf16, al.make_layout((inp_total,), (1,)))
    mult_t = al.make_tensor(multiplier_ptr, al.bf16, al.make_layout((C,), (1,)))
    out_total = N * D_out * H_out * W_out
    out = al.make_tensor(output_ptr, al.bf16, al.make_layout((out_total,), (1,)))

    # channel 0: initialise max
    inp_idx0 = n*conv_stride_N + d*conv_stride_D + h*conv_stride_H + w
    v0 = al.convert(inp[inp_idx0], al.f32)
    if v0 < clamp_min:
        v0 = clamp_min
    if v0 > clamp_max:
        v0 = clamp_max
    max_val = v0 * al.convert(mult_t[0], al.f32)

    for c in al.range(1, C):
        inp_idx = n*conv_stride_N + c*conv_stride_C + d*conv_stride_D + h*conv_stride_H + w
        v = al.convert(inp[inp_idx], al.f32)
        if v < clamp_min:
            v = clamp_min
        if v > clamp_max:
            v = clamp_max
        result = v * al.convert(mult_t[c], al.f32)
        if result > max_val:
            max_val = result

    out_idx = n*out_stride_N + d*out_stride_D + h*out_stride_H + w
    out[out_idx] = al.convert(max_val, al.bf16)


# ---------------------------------------------------------------------------
# Host wrapper
# ---------------------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size,
        multiplier_shape, clamp_min, clamp_max,
    ):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.instance_norm = nn.InstanceNorm3d(out_channels)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device

        weight = self.conv.weight.data.to(device=device, dtype=torch.bfloat16).contiguous()
        bias = (
            self.conv.bias.data.to(device=device, dtype=torch.bfloat16).contiguous()
            if self.conv.bias is not None
            else torch.zeros(self.conv.out_channels, dtype=torch.bfloat16, device=device)
        )
        multiplier = self.multiplier.data.to(device=device).contiguous().view(-1)
        multiplier_bf16 = multiplier.to(dtype=torch.bfloat16)

        N, IC, D_in, H_in, W_in = x.shape
        OC, _, KD, KH, KW = weight.shape
        PAD = self.conv.padding[0] if isinstance(self.conv.padding, tuple) else self.conv.padding
        x = x.contiguous()

        D_out = D_in - KD + 2 * PAD + 1
        H_out = H_in - KH + 2 * PAD + 1
        W_out = W_in - KW + 2 * PAD + 1

        inp_stride_N  = IC * D_in * H_in * W_in
        inp_stride_IC = D_in * H_in * W_in
        inp_stride_D  = H_in * W_in
        inp_stride_H  = W_in

        out_stride_N  = OC * D_out * H_out * W_out
        out_stride_OC = D_out * H_out * W_out
        out_stride_D  = H_out * W_out
        out_stride_H  = W_out

        wgt_stride_OC = IC * KD * KH * KW
        wgt_stride_IC = KD * KH * KW
        wgt_stride_KD = KH * KW
        wgt_stride_KH = KW

        # Conv + first multiplier (applied by host as bf16 multiply, matching reference)
        conv_raw = torch.empty(N, OC, D_out, H_out, W_out, dtype=torch.bfloat16, device=device)
        conv3d_kernel[lambda: ((N * D_out, H_out, W_out), (OC, 1, 1))](
            x, weight, bias, conv_raw,
            N, IC, OC, D_in, H_in, W_in, D_out, H_out, W_out,
            KD, KH, KW, PAD,
            inp_stride_N, inp_stride_IC, inp_stride_D, inp_stride_H,
            out_stride_N, out_stride_OC, out_stride_D, out_stride_H,
            wgt_stride_OC, wgt_stride_IC, wgt_stride_KD, wgt_stride_KH,
        )
        conv_out = conv_raw * multiplier_bf16.view(1, OC, 1, 1, 1)

        # InstanceNorm: use PyTorch for exact match
        normed = torch.nn.functional.instance_norm(
            conv_out.float(),
            running_mean=None, running_var=None,
            weight=None, bias=None,
            use_input_stats=True, momentum=0.1, eps=1e-5,
        )
        normed_bf16 = normed.to(torch.bfloat16)

        # Clamp + second multiplier + max in AveLang
        conv_stride_N = OC * D_out * H_out * W_out
        conv_stride_C = D_out * H_out * W_out
        conv_stride_D = H_out * W_out
        conv_stride_H = W_out

        output = torch.empty(N, D_out, H_out, W_out, dtype=torch.bfloat16, device=device)
        final_out_stride_N = D_out * H_out * W_out
        final_out_stride_D = H_out * W_out
        final_out_stride_H = W_out

        clamp_mult_max_kernel[lambda: ((N * D_out, H_out, W_out), (1, 1, 1))](
            normed_bf16, multiplier_bf16, output,
            N, OC, D_out, H_out, W_out,
            conv_stride_N, conv_stride_C, conv_stride_D, conv_stride_H,
            final_out_stride_N, final_out_stride_D, final_out_stride_H,
            self.clamp_min,
            self.clamp_max,
        )

        return output
