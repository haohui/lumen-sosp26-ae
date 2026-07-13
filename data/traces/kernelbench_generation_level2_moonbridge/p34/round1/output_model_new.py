import torch
import torch.nn as nn
import avelang
import avelang.language as al

W_NORM: al.constexpr = 64

GELU_SQRT2 = 1.4142135623730951


@avelang.jit
def fused_conv_ln_gelu_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    gamma_ptr: al.Pointer(al.bf16),
    beta_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    N: al.i32,
    Cin: al.i32,
    Cout: al.i32,
    D: al.i32,
    H: al.i32,
    W_in: al.i32,
    D_out: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    KD: al.i32,
    KH: al.i32,
    KW: al.i32,
    stride: al.i32,
    padding: al.i32,
    eps: al.f32,
    scaling_factor: al.f32,
):
    tid = al.thread_id(0)
    c_idx = al.block_id(0)
    dh_idx = al.block_id(1)
    n = al.block_id(2)

    d_idx = dh_idx // H_out
    h_idx = dh_idx - d_idx * H_out
    w_idx = tid

    # Conv computation
    x_layout = al.make_layout(
        (N, Cin, D, H, W_in),
        (Cin * D * H * W_in, D * H * W_in, H * W_in, W_in, 1),
    )
    x = al.make_tensor(x_ptr, al.bf16, x_layout)

    w_layout = al.make_layout(
        (Cin, Cout, KD, KH, KW),
        (Cout * KD * KH * KW, KD * KH * KW, KH * KW, KW, 1),
    )
    w = al.make_tensor(w_ptr, al.bf16, w_layout)

    b_layout = al.make_layout((Cout,), (1,))
    b = al.make_tensor(b_ptr, al.bf16, b_layout)

    param_layout = al.make_layout((W_out,), (1,))
    gamma = al.make_tensor(gamma_ptr, al.bf16, param_layout)
    beta_t = al.make_tensor(beta_ptr, al.bf16, param_layout)

    out_layout = al.make_layout(
        (N, Cout, D_out, H_out, W_out),
        (Cout * D_out * H_out * W_out, D_out * H_out * W_out, H_out * W_out, W_out, 1),
    )
    out = al.make_tensor(out_ptr, al.bf16, out_layout)

    smem = al.make_shared((W_NORM,), al.f32)
    smem_sq = al.make_shared((W_NORM,), al.f32)
    smem_val = al.make_shared((W_NORM,), al.f32)

    valid = n < N
    valid_c = c_idx < Cout
    valid_d = d_idx < D_out
    valid_h = h_idx < H_out
    valid_w = w_idx < W_out

    acc = al.convert(0.0, al.f32)
    if valid:
        if valid_c:
            if valid_d:
                if valid_h:
                    if valid_w:
                        acc = al.convert(b[c_idx], al.f32)
                        for ic in al.range(Cin):
                            for kd in al.range(KD):
                                idx_d = d_idx + padding - kd
                                if idx_d % stride == 0:
                                    id_val = idx_d // stride
                                    if id_val >= 0:
                                        if id_val < D:
                                            for kh in al.range(KH):
                                                idx_h = h_idx + padding - kh
                                                if idx_h % stride == 0:
                                                    ih_val = idx_h // stride
                                                    if ih_val >= 0:
                                                        if ih_val < H:
                                                            for kw in al.range(KW):
                                                                idx_w = w_idx + padding - kw
                                                                if idx_w % stride == 0:
                                                                    iw_val = idx_w // stride
                                                                    if iw_val >= 0:
                                                                        if iw_val < W_in:
                                                                            xv = al.convert(
                                                                                x[n, ic, id_val, ih_val, iw_val],
                                                                                al.f32,
                                                                            )
                                                                            wv = al.convert(
                                                                                w[ic, c_idx, kd, kh, kw],
                                                                                al.f32,
                                                                            )
                                                                            acc = acc + xv * wv

    smem[tid] = acc
    smem_sq[tid] = acc * acc
    smem_val[tid] = acc
    al.syncthreads()

    # Reduction to get sum and sum_sq
    if tid < 32:
        smem[tid] = smem[tid] + smem[tid + 32]
        smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 32]
    al.syncthreads()
    if tid < 16:
        smem[tid] = smem[tid] + smem[tid + 16]
        smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 16]
    al.syncthreads()
    if tid < 8:
        smem[tid] = smem[tid] + smem[tid + 8]
        smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 8]
    al.syncthreads()
    if tid < 4:
        smem[tid] = smem[tid] + smem[tid + 4]
        smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 4]
    al.syncthreads()
    if tid < 2:
        smem[tid] = smem[tid] + smem[tid + 2]
        smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 2]
    al.syncthreads()
    if tid < 1:
        smem[tid] = smem[tid] + smem[tid + 1]
        smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 1]
    al.syncthreads()

    slice_sum = smem[0]
    slice_sq = smem_sq[0]
    W_f32 = al.convert(W_out, al.f32)
    mean = slice_sum / W_f32
    var = slice_sq / W_f32 - mean * mean
    zero_f32 = al.convert(0.0, al.f32)
    if var < zero_f32:
        var = zero_f32
    rstd = al.convert(1.0, al.f32) / al.sqrt(var + eps)

    if valid:
        if valid_c:
            if valid_d:
                if valid_h:
                    if valid_w:
                        conv_val = smem_val[tid]
                        normalized = (conv_val - mean) * rstd

                        g_val = al.convert(gamma[w_idx], al.f32)
                        b_val = al.convert(beta_t[w_idx], al.f32)
                        # Simulate PyTorch's BF16 intermediate after affine transform
                        ln_bf16 = al.convert(normalized * g_val + b_val, al.bf16)
                        ln_out = al.convert(ln_bf16, al.f32)

                        half = al.convert(0.5, al.f32)
                        one = al.convert(1.0, al.f32)
                        sqrt2 = al.convert(GELU_SQRT2, al.f32)
                        erf_arg = ln_out / sqrt2
                        erf_val = al.erf(erf_arg)
                        gelu_val = half * ln_out * (one + erf_val)

                        result = gelu_val * scaling_factor
                        out[n, c_idx, d_idx, h_idx, w_idx] = al.convert(result, al.bf16)


def _to_bf16_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_conv_transpose_layernorm_gelu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    stride: int,
    padding: int,
    eps: float,
    scaling_factor: float,
) -> torch.Tensor:
    assert x.is_cuda, "Input must be on CUDA/HIP device."

    N, Cin, D, H, W_in = x.shape
    Cin_w, Cout, KD, KH, KW = weight.shape
    assert Cin_w == Cin, f"Weight in_channels {Cin_w} != input in_channels {Cin}"

    D_out = (D - 1) * stride - 2 * padding + KD
    H_out = (H - 1) * stride - 2 * padding + KH
    W_out = (W_in - 1) * stride - 2 * padding + KW

    x_bf16 = _to_bf16_contiguous(x)
    w_bf16 = _to_bf16_contiguous(weight)
    b_bf16 = _to_bf16_contiguous(bias)
    ln_w_bf16 = _to_bf16_contiguous(ln_weight)
    ln_b_bf16 = _to_bf16_contiguous(ln_bias)

    result = torch.empty(
        (N, Cout, D_out, H_out, W_out),
        dtype=torch.bfloat16,
        device=x.device,
    )

    dh_tiles = D_out * H_out

    fused_conv_ln_gelu_kernel[lambda: ((Cout, dh_tiles, N), (W_NORM, 1, 1))](
        x_bf16,
        w_bf16,
        b_bf16,
        ln_w_bf16,
        ln_b_bf16,
        result,
        N,
        Cin,
        Cout,
        D,
        H,
        W_in,
        D_out,
        H_out,
        W_out,
        KD,
        KH,
        KW,
        stride,
        padding,
        eps,
        scaling_factor,
    )

    return result


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias=True,
        eps=1e-5,
        scaling_factor=1.0,
    ):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.eps = eps
        self.scaling_factor = scaling_factor

        self.conv_weight = nn.Parameter(
            torch.empty(in_channels, out_channels, kernel_size, kernel_size, kernel_size)
        )
        if bias:
            self.conv_bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("conv_bias", None)

        self.ln_weight = nn.Parameter(torch.empty(out_channels))
        self.ln_bias = nn.Parameter(torch.empty(out_channels))

        self._init_parameters()

    def _init_parameters(self):
        nn.init.kaiming_uniform_(self.conv_weight, a=5 ** 0.5)
        if self.conv_bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.conv_weight)
            bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.conv_bias, -bound, bound)

        nn.init.ones_(self.ln_weight)
        nn.init.zeros_(self.ln_bias)

    def forward(self, x):
        bias = (
            self.conv_bias
            if self.conv_bias is not None
            else torch.zeros(self.out_channels, device=x.device, dtype=x.dtype)
        )

        orig_dtype = x.dtype
        result = avelang_conv_transpose_layernorm_gelu(
            x,
            self.conv_weight,
            bias,
            self.ln_weight,
            self.ln_bias,
            self.stride,
            self.padding,
            self.eps,
            self.scaling_factor,
        )
        return result.to(orig_dtype)
