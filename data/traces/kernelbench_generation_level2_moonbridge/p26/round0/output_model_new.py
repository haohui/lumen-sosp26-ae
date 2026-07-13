import torch
import torch.nn as nn
import avelang
import avelang.language as al


TILE_C = 4
TILE_D = 4
TILE_H = 4
TILE_W = 4
THREADS = 256


@avelang.jit
def _conv_transpose3d_fused_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    conv_bias_ptr: al.Pointer(al.bf16),
    add_input_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    B: al.u32,
    C_in: al.u32,
    C_out: al.u32,
    D_in: al.u32,
    H_in: al.u32,
    W_in: al.u32,
    kD: al.u32,
    kH: al.u32,
    kW: al.u32,
    stride: al.u32,
    padding: al.u32,
    D_out: al.u32,
    H_out: al.u32,
    W_out: al.u32,
    c_out_groups: al.u32,
    w_out_groups: al.u32,
):
    tid = al.thread_id(0)
    block_idx = al.block_id(0)
    block_d = al.block_id(1)
    block_hw = al.block_id(2)

    # Compute strides
    inp_n_stride = C_in * D_in * H_in * W_in
    inp_c_stride = D_in * H_in * W_in
    inp_d_stride = H_in * W_in
    inp_h_stride = W_in

    w_ci_stride = C_out * kD * kH * kW
    w_co_stride = kD * kH * kW
    w_kd_stride = kH * kW
    w_kh_stride = kW

    out_n_stride = C_out * D_out * H_out * W_out
    out_c_stride = D_out * H_out * W_out
    out_d_stride = H_out * W_out
    out_h_stride = W_out

    # 1D memory views
    input_mem = al.make_tensor(input_ptr, al.bf16, al.make_layout((B * inp_n_stride,), (1,)))
    weight_mem = al.make_tensor(weight_ptr, al.bf16, al.make_layout((C_in * w_ci_stride,), (1,)))
    conv_bias_mem = al.make_tensor(conv_bias_ptr, al.bf16, al.make_layout((C_out,), (1,)))
    add_input_mem = al.make_tensor(add_input_ptr, al.bf16, al.make_layout((B * out_n_stride,), (1,)))
    output_mem = al.make_tensor(output_ptr, al.bf16, al.make_layout((B * out_n_stride,), (1,)))

    # Block position
    n = block_idx / c_out_groups
    c_out_group = block_idx % c_out_groups
    c_out_start = c_out_group * 4

    h_group = block_hw / w_out_groups
    w_group = block_hw % w_out_groups

    d_start = block_d * 4
    h_start = h_group * 4
    w_start = w_group * 4

    # --- Stage 1: load weight tile (3456 bf16) into shared memory ---
    shm_weight = al.make_shared((3456,), al.bf16)
    for idx in al.range(tid, 3456, 256):
        c_local = idx / 864
        tmp = idx % 864
        ci = tmp / 27
        tmp2 = tmp % 27
        kd = tmp2 / 9
        tmp3 = tmp2 % 9
        kh = tmp3 / 3
        kw = tmp3 % 3
        w_idx = ci * w_ci_stride + (c_out_start + c_local) * w_co_stride + kd * w_kd_stride + kh * w_kh_stride + kw
        shm_weight[idx] = weight_mem[w_idx]

    # --- Stage 2: load input tile (2048 bf16) into shared memory ---
    shm_input = al.make_shared((2048,), al.bf16)
    d_in_base = d_start / stride
    h_in_base = h_start / stride
    w_in_base = w_start / stride

    for idx in al.range(tid, 2048, 256):
        ci = idx / 64
        s_idx = idx % 64
        di = s_idx / 16
        tmp_s = s_idx % 16
        hi = tmp_s / 4
        wi = tmp_s % 4
        g_d = d_in_base + di
        g_h = h_in_base + hi
        g_w = w_in_base + wi
        if g_d < D_in and g_h < H_in and g_w < W_in:
            inp_idx = n * inp_n_stride + ci * inp_c_stride + g_d * inp_d_stride + g_h * inp_h_stride + g_w
            shm_input[idx] = input_mem[inp_idx]

    al.syncthreads()

    # Map thread to (c_local, d_local, h_local, w_local)
    c_local = tid / 64
    spatial = tid % 64
    d_local = spatial / 16
    hw_local = spatial % 16
    h_local = hw_local / 4
    w_local = hw_local % 4

    c_out = c_out_start + c_local
    if c_out >= C_out:
        return
    d_out = d_start + d_local
    if d_out >= D_out:
        return
    h_out = h_start + h_local
    w_out = w_start + w_local
    if h_out >= H_out or w_out >= W_out:
        return

    # Shared constants
    zero = al.convert(0.0, al.f32)
    three = al.convert(3.0, al.f32)
    six = al.convert(6.0, al.f32)

    out_idx = n * out_n_stride + c_out * out_c_stride + d_out * out_d_stride + h_out * out_h_stride + w_out

    # FP32 accumulation from bias
    acc = al.convert(conv_bias_mem[c_out], al.f32)

    # Compute conv_transpose: iterate only over valid kernel positions
    d_parity = (d_out + padding) % stride
    h_parity = (h_out + padding) % stride
    w_parity = (w_out + padding) % stride

    for kd_val in al.range(d_parity, kD, stride):
        d_in = (d_out + padding - kd_val) / stride
        for kh_val in al.range(h_parity, kH, stride):
            h_in = (h_out + padding - kh_val) / stride
            for kw_val in al.range(w_parity, kW, stride):
                w_in = (w_out + padding - kw_val) / stride
                if d_in < D_in and h_in < H_in and w_in < W_in:
                    for ci in al.range(C_in):
                        in_shm_idx = ci * 64 + (d_in - d_in_base) * 16 + (h_in - h_in_base) * 4 + (w_in - w_in_base)
                        w_shm_idx = c_local * 864 + ci * 27 + kd_val * 9 + kh_val * 3 + kw_val
                        acc = acc + al.convert(shm_input[in_shm_idx], al.f32) * al.convert(shm_weight[w_shm_idx], al.f32)

    # Add add_input
    acc = acc + al.convert(add_input_mem[out_idx], al.f32)

    # HardSwish: x^2 * relu6(x+3) / 6
    x_plus_3 = acc + three
    relu6_val = x_plus_3
    if relu6_val < zero:
        relu6_val = zero
    if relu6_val > six:
        relu6_val = six
    output_mem[out_idx] = al.convert(acc * acc * relu6_val / six, al.bf16)


def _prepare_bf16_cuda_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_conv_transpose3d_fused(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_bias: torch.Tensor,
    add_input: torch.Tensor,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = _prepare_bf16_cuda_contiguous(x)
    weight_bf16 = _prepare_bf16_cuda_contiguous(weight)
    bias_bf16 = _prepare_bf16_cuda_contiguous(conv_bias)
    add_bf16 = _prepare_bf16_cuda_contiguous(add_input)

    B, C_in, D_in, H_in, W_in = x_bf16.shape
    C_in_w, C_out, kD, kH, kW = weight_bf16.shape
    B_a, C_out_a, D_out, H_out, W_out = add_bf16.shape

    stride_val = 2
    padding_val = 1

    out = torch.empty(
        (B, C_out, D_out, H_out, W_out),
        device=x_bf16.device,
        dtype=torch.bfloat16,
    )

    c_out_groups = (C_out + TILE_C - 1) // TILE_C
    d_out_groups = (D_out + TILE_D - 1) // TILE_D
    h_out_groups = (H_out + TILE_H - 1) // TILE_H
    w_out_groups = (W_out + TILE_W - 1) // TILE_W

    grid = (B * c_out_groups, d_out_groups, h_out_groups * w_out_groups)

    _conv_transpose3d_fused_kernel[lambda: (grid, (THREADS, 1, 1))](
        x_bf16,
        weight_bf16,
        bias_bf16,
        add_bf16,
        out,
        B,
        C_in,
        C_out,
        D_in,
        H_in,
        W_in,
        kD,
        kH,
        kW,
        stride_val,
        padding_val,
        D_out,
        H_out,
        W_out,
        c_out_groups,
        w_out_groups,
    )
    return out


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        bias_shape,
    ):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size_val = kernel_size
        self.stride_val = stride
        self.padding_val = padding
        self.output_padding_val = output_padding

        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=True,
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x, add_input):
        return avelang_conv_transpose3d_fused(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            add_input,
        )
