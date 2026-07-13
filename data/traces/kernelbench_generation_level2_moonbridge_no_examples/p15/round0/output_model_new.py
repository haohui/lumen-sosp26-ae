import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def conv_transpose3d_kernel(
    x_ptr: al.Pointer(al.bf16), w_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16), y_ptr: al.Pointer(al.bf16),
    N: al.i32, C_in: al.i32, C_out: al.i32,
    D_in: al.i32, H_in: al.i32, W_in: al.i32,
    D_out: al.i32, H_out: al.i32, W_out: al.i32,
    K: al.i32, stride: al.i32, pad: al.i32,
):
    one = al.convert(1, al.i32)
    x = al.make_tensor(x_ptr, al.bf16, al.make_layout(
        (N, C_in, D_in, H_in, W_in), (C_in * D_in * H_in * W_in, D_in * H_in * W_in, H_in * W_in, W_in, one)))
    w = al.make_tensor(w_ptr, al.bf16, al.make_layout(
        (C_in, C_out, K, K, K), (C_out * K * K * K, K * K * K, K * K, K, one)))
    b = al.make_tensor(b_ptr, al.bf16, al.make_layout((C_out,), (one,)))
    y = al.make_tensor(y_ptr, al.bf16, al.make_layout(
        (N, C_out, D_out, H_out, W_out), (C_out * D_out * H_out * W_out, D_out * H_out * W_out, H_out * W_out, W_out, one)))
    flat_idx = al.block_id(0)
    n = flat_idx // C_out
    c_out = flat_idx % C_out
    tid = al.thread_id(0)
    total_spatial = D_out * H_out * W_out
    for idx in al.range(tid, total_spatial, al.convert(256, al.i32)):
        w_out = idx % W_out
        hw_rem = idx // W_out
        h_out = hw_rem % H_out
        d_out = hw_rem // H_out
        acc = al.convert(b[c_out], al.f32)
        for c_in in al.range(C_in):
            for kd in al.range(K):
                if d_out + pad >= kd:
                    d_target = d_out + pad - kd
                    d_in = d_target // stride
                    if d_in * stride == d_target:
                        if d_in < D_in:
                            for kh in al.range(K):
                                if h_out + pad >= kh:
                                    h_target = h_out + pad - kh
                                    h_in = h_target // stride
                                    if h_in * stride == h_target:
                                        if h_in < H_in:
                                            for kw in al.range(K):
                                                if w_out + pad >= kw:
                                                    w_target = w_out + pad - kw
                                                    w_in = w_target // stride
                                                    if w_in * stride == w_target:
                                                        if w_in < W_in:
                                                            x_val = al.convert(x[n, c_in, d_in, h_in, w_in], al.f32)
                                                            w_val = al.convert(w[c_in, c_out, kd, kh, kw], al.f32)
                                                            acc = acc + x_val * w_val
        y[n, c_out, d_out, h_out, w_out] = al.convert(acc, al.bf16)


@avelang.jit
def spatial_mean_subtract_kernel(
    x_ptr: al.Pointer(al.bf16), y_ptr: al.Pointer(al.bf16),
    N: al.i32, C: al.i32, D: al.i32, H: al.i32, W: al.i32,
):
    one = al.convert(1, al.i32)
    x = al.make_tensor(x_ptr, al.bf16, al.make_layout(
        (N, C, D, H, W), (C * D * H * W, D * H * W, H * W, W, one)))
    y = al.make_tensor(y_ptr, al.bf16, al.make_layout(
        (N, C, D, H, W), (C * D * H * W, D * H * W, H * W, W, one)))
    flat_idx = al.block_id(0)
    n = flat_idx // C
    c = flat_idx % C
    tid = al.thread_id(0)
    total_spatial = D * H * W
    shared_sum = al.make_shared((256,), al.f32)
    local_sum = al.convert(0, al.f32)
    for idx in al.range(tid, total_spatial, al.convert(256, al.i32)):
        w = idx % W
        hw_rem = idx // W
        h = hw_rem % H
        d = hw_rem // H
        local_sum = local_sum + al.convert(x[n, c, d, h, w], al.f32)
    shared_sum[tid] = local_sum
    al.syncthreads()
    if tid < 128:
        shared_sum[tid] = shared_sum[tid] + shared_sum[tid + 128]
    al.syncthreads()
    if tid < 64:
        shared_sum[tid] = shared_sum[tid] + shared_sum[tid + 64]
    al.syncthreads()
    if tid < 32:
        shared_sum[tid] = shared_sum[tid] + shared_sum[tid + 32]
    al.syncthreads()
    if tid < 16:
        shared_sum[tid] = shared_sum[tid] + shared_sum[tid + 16]
    al.syncthreads()
    if tid < 8:
        shared_sum[tid] = shared_sum[tid] + shared_sum[tid + 8]
    al.syncthreads()
    if tid < 4:
        shared_sum[tid] = shared_sum[tid] + shared_sum[tid + 4]
    al.syncthreads()
    if tid < 2:
        shared_sum[tid] = shared_sum[tid] + shared_sum[tid + 2]
    al.syncthreads()
    if tid < 1:
        shared_sum[tid] = shared_sum[tid] + shared_sum[tid + 1]
    al.syncthreads()
    zero_i32 = al.convert(0, al.i32)
    mean = shared_sum[zero_i32] / al.convert(total_spatial, al.f32)
    al.syncthreads()
    for idx in al.range(tid, total_spatial, al.convert(256, al.i32)):
        w = idx % W
        hw_rem = idx // W
        h = hw_rem % H
        d = hw_rem // H
        x_val = al.convert(x[n, c, d, h, w], al.f32)
        y[n, c, d, h, w] = al.convert(x_val - mean, al.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(ModelNew, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias
        )
        self.batch_norm = nn.BatchNorm3d(out_channels)
    
    def forward(self, x):
        N, C_in, D_in, H_in, W_in = x.shape
        C_out = self.out_channels
        K = self.kernel_size
        stride = self.stride
        pad = self.padding
        
        D_out = (D_in - 1) * stride - 2 * pad + K
        H_out = (H_in - 1) * stride - 2 * pad + K
        W_out = (W_in - 1) * stride - 2 * pad + K
        
        x_contig = x.contiguous()
        w = self.conv_transpose.weight.detach().contiguous()
        b = self.conv_transpose.bias
        if b is not None:
            b = b.detach().contiguous()
        else:
            b = torch.zeros(C_out, dtype=x_contig.dtype, device=x.device)
        
        conv_out = torch.empty(N, C_out, D_out, H_out, W_out, dtype=x_contig.dtype, device=x.device)
        grid_nc = N * C_out
        conv_transpose3d_kernel[lambda: ((grid_nc, 1, 1), (256, 1, 1))](
            x_contig, w, b, conv_out,
            N, C_in, C_out, D_in, H_in, W_in, D_out, H_out, W_out,
            K, stride, pad,
        )
        
        x_bn = self.batch_norm(conv_out)
        final_out = torch.empty_like(x_bn)
        spatial_mean_subtract_kernel[lambda: ((grid_nc, 1, 1), (256, 1, 1))](
            x_bn, final_out, N, C_out, D_out, H_out, W_out,
        )
        return final_out
