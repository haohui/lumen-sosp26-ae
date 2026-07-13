import torch
import torch.nn as nn
import avelang
import avelang.language as al

GROUP_OC = 128
GROUP_SPATIAL = 64
GROUP_IC = 16
THREADS = 256


@avelang.jit
def conv2d_bf16_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C_in: al.i32,
    C_out: al.i32,
    H: al.i32,
    W: al.i32,
    KH: al.i32,
    KW: al.i32,
    H_out: al.i32,
    W_out: al.i32,
):
    tid = al.thread_id(0)
    block_spatial = al.block_id(0)
    block_oc = al.block_id(1)
    oc_start = block_oc * GROUP_OC

    x_flat = al.make_tensor(x_ptr, al.bf16, al.make_layout((N * C_in * H * W,), (1,)))
    w_flat = al.make_tensor(w_ptr, al.bf16, al.make_layout((C_out * KH * KW * C_in,), (1,)))

    shm_weight = al.make_shared((GROUP_OC, GROUP_IC), al.bf16)
    shm_input = al.make_shared((GROUP_SPATIAL, GROUP_IC), al.bf16)

    acc = al.make_local((32,), al.f32)
    for i in al.range(32):
        acc[i] = al.convert(0.0, al.f32)

    H_out_W_out = H_out * W_out
    num_w = (GROUP_OC * GROUP_IC) // THREADS
    num_x = (GROUP_SPATIAL * GROUP_IC) // THREADS
    ic_tiles = C_in // GROUP_IC

    for kh in al.range(KH):
        for kw in al.range(KW):
            for ict in al.range(ic_tiles):
                ic_start = ict * GROUP_IC

                for li in al.range(num_w):
                    idx = tid * num_w + li
                    oc_local = idx // GROUP_IC
                    ic_local = idx - oc_local * GROUP_IC
                    oc = oc_start + oc_local
                    ic = ic_start + ic_local
                    w_idx = ((oc * KH + kh) * KW + kw) * C_in + ic
                    shm_weight[oc_local, ic_local] = w_flat[w_idx]

                for li in al.range(num_x):
                    idx = tid * num_x + li
                    s_local = idx // GROUP_IC
                    ic_local = idx - s_local * GROUP_IC
                    spatial_idx = block_spatial * GROUP_SPATIAL + s_local
                    n = spatial_idx // H_out_W_out
                    residual = spatial_idx - n * H_out_W_out
                    h_out = residual // W_out
                    w_out = residual - h_out * W_out
                    h_in = h_out + kh
                    w_in = w_out + kw
                    x_idx = ((n * C_in + ic_start + ic_local) * H + h_in) * W + w_in
                    shm_input[s_local, ic_local] = x_flat[x_idx]

                al.syncthreads()

                for i in al.range(32):
                    elem = tid * 32 + i
                    oc_local = elem // GROUP_SPATIAL
                    s_local = elem - oc_local * GROUP_SPATIAL
                    for j in al.range(GROUP_IC):
                        w_val = al.convert(shm_weight[oc_local, j], al.f32)
                        x_val = al.convert(shm_input[s_local, j], al.f32)
                        acc[i] = acc[i] + w_val * x_val

                al.syncthreads()

    out_flat = al.make_tensor(out_ptr, al.bf16, al.make_layout((N * C_out * H_out * W_out,), (1,)))
    for i in al.range(32):
        elem = tid * 32 + i
        oc_local = elem // GROUP_SPATIAL
        s_local = elem - oc_local * GROUP_SPATIAL
        oc = oc_start + oc_local
        spatial_idx = block_spatial * GROUP_SPATIAL + s_local
        n = spatial_idx // H_out_W_out
        residual = spatial_idx - n * H_out_W_out
        h_out = residual // W_out
        w_out = residual - h_out * W_out
        out_idx = ((n * C_out + oc) * H_out + h_out) * W_out + w_out
        out_flat[out_idx] = al.convert(acc[i], al.bf16)


def avelang_conv2d(x, weight, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1):
    x_bf16 = x.contiguous().to(dtype=torch.bfloat16)
    w_perm = weight.contiguous().to(dtype=torch.bfloat16).permute(0, 2, 3, 1).contiguous()

    N_val, C_in, H, W_in = x_bf16.shape
    C_out, KH, KW, _ = w_perm.shape

    H_out = (H + 2 * padding[0] - dilation[0] * (KH - 1) - 1) // stride[0] + 1
    W_out = (W_in + 2 * padding[1] - dilation[1] * (KW - 1) - 1) // stride[1] + 1

    out = torch.empty((N_val, C_out, H_out, W_out), device=x_bf16.device, dtype=torch.bfloat16)
    grid = ((N_val * H_out * W_out) // GROUP_SPATIAL, C_out // GROUP_OC, 1)
    conv2d_bf16_kernel[lambda: (grid, (THREADS, 1, 1))](
        x_bf16, w_perm, out, N_val, C_in, C_out, H, W_in, KH, KW, H_out, W_out
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        if bias:
            nn.init.uniform_(self.bias, -0.1, 0.1)

    def forward(self, x):
        r = avelang_conv2d(x, self.weight, self.stride, self.padding, self.dilation, self.groups)
        if self.bias is not None:
            r = r + self.bias.to(dtype=r.dtype, device=r.device).reshape(1, -1, 1, 1)
        return r.to(dtype=x.dtype)


batch_size = 8
in_channels = 64
out_channels = 128
kernel_size = (5, 7)
height = 512
width = 256

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
