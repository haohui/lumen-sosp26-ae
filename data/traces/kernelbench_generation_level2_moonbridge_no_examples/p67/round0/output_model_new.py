import torch
import torch.nn as nn
import avelang
import avelang.language as al

TILE = 16
REDUCE_BLOCK = 256


@avelang.jit
def gelu_f32(x: al.f32) -> al.f32:
    sqrt_2_pi = al.convert(0.7978845608, al.f32)
    coeff = al.convert(0.044715, al.f32)
    half = al.convert(0.5, al.f32)
    one = al.convert(1.0, al.f32)
    x3 = x * x * x
    inner = sqrt_2_pi * (x + coeff * x3)
    return half * x * (one + al.tanh(inner))


@avelang.jit
def conv2d_3x3_gelu_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.f32),
    output_ptr: al.Pointer(al.f32),
    N: al.i32,
    C_in: al.i32,
    C_out: al.i32,
    H: al.i32,
    W: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    w_CO_stride: al.i32,
    w_CI_stride: al.i32,
    w_KH_stride: al.i32,
):
    bx = al.block_id(0)
    by = al.block_id(1)
    n = al.block_id(2)
    tx = al.thread_id(0)
    ty = al.thread_id(1)

    h_out = by * TILE + ty
    w_out = bx * TILE + tx

    in_spatial = H * W
    in_batch_stride = C_in * in_spatial
    out_spatial = H_out * W_out
    out_batch_stride = C_out * out_spatial

    total_in = N * in_batch_stride
    total_out = N * out_batch_stride

    in_flat = al.make_layout((total_in,), (1,))
    in_t = al.make_tensor(input_ptr, al.bf16, in_flat)

    w_total = C_out * w_CO_stride
    w_flat = al.make_layout((w_total,), (1,))
    w_t = al.make_tensor(weight_ptr, al.bf16, w_flat)

    b_layout = al.make_layout((C_out,), (1,))
    b_t = al.make_tensor(bias_ptr, al.f32, b_layout)

    out_flat = al.make_layout((total_out,), (1,))
    out_t = al.make_tensor(output_ptr, al.f32, out_flat)

    if h_out < H_out:
        if w_out < W_out:
            in_n_off = n * in_batch_stride
            out_n_off = n * out_batch_stride
            for cout in al.range(C_out):
                acc = b_t[cout]
                w_cout_off = cout * w_CO_stride
                out_c_off = cout * out_spatial
                for cin in al.range(C_in):
                    in_c_off = cin * in_spatial
                    w_cin_off = cin * w_CI_stride
                    for kh in al.range(3):
                        h_in = h_out + kh
                        in_h_off = h_in * W
                        w_kh_off = kh * w_KH_stride
                        for kw in al.range(3):
                            w_in = w_out + kw
                            in_idx = in_n_off + in_c_off + in_h_off + w_in
                            w_idx = w_cout_off + w_cin_off + w_kh_off + kw
                            in_val = al.convert(in_t[in_idx], al.f32)
                            w_val = al.convert(w_t[w_idx], al.f32)
                            acc = acc + in_val * w_val
                out_idx = out_n_off + out_c_off + h_out * W_out + w_out
                out_t[out_idx] = gelu_f32(acc)


@avelang.jit
def reduce_mean_kernel(
    input_ptr: al.Pointer(al.f32),
    output_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C_out: al.i32,
    H: al.i32,
    W: al.i32,
    total_elements: al.i32,
    out_stride: al.i32,
):
    n = al.block_id(0)
    c = al.block_id(1)
    tid = al.thread_id(0)

    in_spatial = H * W
    in_n_stride = C_out * in_spatial
    total_elems = N * in_n_stride

    in_flat = al.make_layout((total_elems,), (1,))
    in_t = al.make_tensor(input_ptr, al.f32, in_flat)

    out_flat = al.make_layout((N * out_stride,), (1,))
    out_t = al.make_tensor(output_ptr, al.bf16, out_flat)

    in_n_off = n * in_n_stride
    in_c_off = c * in_spatial

    acc = al.convert(0.0, al.f32)
    for h in al.range(H):
        w = tid
        if w < W:
            in_idx = in_n_off + in_c_off + h * W + w
            acc = acc + in_t[in_idx]

    smem = al.make_shared((REDUCE_BLOCK,), al.f32)
    smem[tid] = acc
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
    al.syncthreads()

    if tid < 1:
        total_f = al.convert(total_elements, al.f32)
        mean = smem[0] * al.amdgpu.rcp(total_f)
        out_idx = n * out_stride + c
        out_t[out_idx] = al.convert(mean, al.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        assert x.is_cuda, "Input must be on CUDA/HIP device."
        N, C_in, H, W = x.shape
        C_out = self.conv.out_channels
        KH, KW = self.conv.kernel_size

        H_out = H - KH + 1
        W_out = W - KW + 1

        x = x.contiguous()

        weight_bf16 = self.conv.weight.data.to(torch.bfloat16).contiguous()
        bias = self.conv.bias.data.contiguous()

        w_CO_stride = weight_bf16.stride(0)
        w_CI_stride = weight_bf16.stride(1)
        w_KH_stride = weight_bf16.stride(2)

        intermediate = torch.empty(N, C_out, H_out, W_out, dtype=torch.float32, device=x.device)

        grid_x = (W_out + TILE - 1) // TILE
        grid_y = (H_out + TILE - 1) // TILE

        conv2d_3x3_gelu_kernel[lambda: ((grid_x, grid_y, N), (TILE, TILE, 1))](
            x.data_ptr(),
            weight_bf16.data_ptr(),
            bias.data_ptr(),
            intermediate.data_ptr(),
            N,
            C_in,
            C_out,
            H,
            W,
            H_out,
            W_out,
            w_CO_stride,
            w_CI_stride,
            w_KH_stride,
        )

        out_tensor = torch.empty(N, C_out, dtype=torch.bfloat16, device=x.device)
        out_stride = out_tensor.stride(0)
        total_elements = H_out * W_out

        reduce_mean_kernel[lambda: ((N, C_out, 1), (REDUCE_BLOCK, 1, 1))](
            intermediate.data_ptr(),
            out_tensor.data_ptr(),
            N,
            C_out,
            H_out,
            W_out,
            total_elements,
            out_stride,
        )

        return out_tensor
