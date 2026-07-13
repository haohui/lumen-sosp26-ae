import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def spatial_reduce_kernel(
    x_ptr: al.Pointer(al.bf16),
    x_sum_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C_in: al.i32,
    H: al.i32,
    W: al.i32,
    n_stride: al.i32,
    c_stride: al.i32,
):
    """Reduce x over spatial dims H,W producing x_sum[n, c_in]."""
    block_id = al.block_id(0)
    n = block_id // C_in
    c_in = block_id % C_in
    base = n * n_stride + c_in * c_stride

    # 2D layout: (N*C_in, H*W) with row-major strides
    groups = N * C_in
    total_spatial = H * W
    x_layout = al.make_layout((groups, total_spatial), (c_stride, 1))
    x_2d = al.make_tensor(x_ptr, al.bf16, x_layout)

    # Output: flat (N*C_in,)
    sum_layout = al.make_layout((groups,), (1,))
    x_sum_flat = al.make_tensor(x_sum_ptr, al.bf16, sum_layout)

    tid = al.thread_id(0)

    # Each thread sums a strided chunk of the spatial dimension
    acc = al.convert(0.0, al.f32)
    for idx in al.range(tid, total_spatial, 256):
        val = al.convert(x_2d[block_id, idx], al.f32)
        acc = acc + val

    # Block-level reduction via shared memory
    shared = al.make_shared((256,), al.f32)
    shared[tid] = acc
    al.syncthreads()

    if tid < 128:
        shared[tid] = shared[tid] + shared[tid + 128]
    al.syncthreads()
    if tid < 64:
        shared[tid] = shared[tid] + shared[tid + 64]
    al.syncthreads()
    if tid < 32:
        shared[tid] = shared[tid] + shared[tid + 32]
    al.syncthreads()
    if tid < 16:
        shared[tid] = shared[tid] + shared[tid + 16]
    al.syncthreads()
    if tid < 8:
        shared[tid] = shared[tid] + shared[tid + 8]
    al.syncthreads()
    if tid < 4:
        shared[tid] = shared[tid] + shared[tid + 4]
    al.syncthreads()
    if tid < 2:
        shared[tid] = shared[tid] + shared[tid + 2]
    al.syncthreads()
    if tid < 1:
        shared[tid] = shared[tid] + shared[tid + 1]
    al.syncthreads()

    if tid == 0:
        x_sum_flat[block_id] = al.convert(shared[0], al.bf16)


@avelang.jit
def compute_lse_kernel(
    x_sum_ptr: al.Pointer(al.bf16),
    w_sum_ptr: al.Pointer(al.bf16),
    b_conv_ptr: al.Pointer(al.bf16),
    b_explicit_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    C_in: al.i32,
    hw_out: al.i32,
    C_OUT: al.constexpr,
):
    """Matmul x_sum @ w_sum, add biases, logsumexp over C_out, multiply by 10."""
    n = al.block_id(0)
    tid = al.thread_id(0)

    # scale = 1.0 / hw_out using AMDGPU reciprocal
    denom = al.convert(hw_out, al.f32)
    scale = al.amdgpu.rcp(denom)

    # x_sum flat layout: (N, C_in) row-major
    x_sum_total = (n + 1) * C_in
    x_sum_layout = al.make_layout((x_sum_total,), (1,))
    x_sum = al.make_tensor(x_sum_ptr, al.bf16, x_sum_layout)

    # w_sum flat layout: (C_in, C_out) row-major
    w_total = C_in * C_OUT
    w_layout = al.make_layout((w_total,), (1,))
    w_sum = al.make_tensor(w_sum_ptr, al.bf16, w_layout)

    # bias_conv flat: (C_out,)
    b_conv_layout = al.make_layout((C_OUT,), (1,))
    b_conv = al.make_tensor(b_conv_ptr, al.bf16, b_conv_layout)

    # bias_explicit flat: (C_out,)
    b_explicit_layout = al.make_layout((C_OUT,), (1,))
    b_explicit = al.make_tensor(b_explicit_ptr, al.bf16, b_explicit_layout)

    # output flat: (N,)
    N_max = n + 1
    out_layout = al.make_layout((N_max,), (1,))
    out = al.make_tensor(out_ptr, al.bf16, out_layout)

    x_base = n * C_in

    # Compute gap for this thread's output channel
    gap = al.convert(0.0, al.f32)
    if tid < C_OUT:
        acc = al.convert(0.0, al.f32)
        for c_in in al.range(C_in):
            xv = al.convert(x_sum[x_base + c_in], al.f32)
            wv = al.convert(w_sum[c_in * C_OUT + tid], al.f32)
            acc = acc + xv * wv
        gap = al.convert(b_conv[tid], al.f32) + scale * acc + al.convert(b_explicit[tid], al.f32)

    # Block reduction: logsumexp(gap)
    shared = al.make_shared((C_OUT,), al.f32)
    shared[tid] = gap
    al.syncthreads()

    # Max reduction
    if C_OUT >= 2:
        if tid < 64:
            if shared[tid + 64] > shared[tid]:
                shared[tid] = shared[tid + 64]
        al.syncthreads()
    if C_OUT >= 2:
        if tid < 32:
            if shared[tid + 32] > shared[tid]:
                shared[tid] = shared[tid + 32]
        al.syncthreads()
    if tid < 16:
        if shared[tid + 16] > shared[tid]:
            shared[tid] = shared[tid + 16]
    al.syncthreads()
    if tid < 8:
        if shared[tid + 8] > shared[tid]:
            shared[tid] = shared[tid + 8]
    al.syncthreads()
    if tid < 4:
        if shared[tid + 4] > shared[tid]:
            shared[tid] = shared[tid + 4]
    al.syncthreads()
    if tid < 2:
        if shared[tid + 2] > shared[tid]:
            shared[tid] = shared[tid + 2]
    al.syncthreads()
    if tid < 1:
        if shared[1] > shared[0]:
            shared[0] = shared[1]
    al.syncthreads()

    max_val = shared[0]

    # Compute exp(gap - max) and store back
    new_val = al.convert(0.0, al.f32)
    if tid < C_OUT:
        new_val = al.exp(gap - max_val)
    shared[tid] = new_val
    al.syncthreads()

    # Sum reduction
    if tid < 64:
        shared[tid] = shared[tid] + shared[tid + 64]
    al.syncthreads()
    if tid < 32:
        shared[tid] = shared[tid] + shared[tid + 32]
    al.syncthreads()
    if tid < 16:
        shared[tid] = shared[tid] + shared[tid + 16]
    al.syncthreads()
    if tid < 8:
        shared[tid] = shared[tid] + shared[tid + 8]
    al.syncthreads()
    if tid < 4:
        shared[tid] = shared[tid] + shared[tid + 4]
    al.syncthreads()
    if tid < 2:
        shared[tid] = shared[tid] + shared[tid + 2]
    al.syncthreads()
    if tid < 1:
        shared[tid] = shared[tid] + shared[tid + 1]
    al.syncthreads()

    if tid == 0:
        sum_exp = shared[0]
        lse = al.log(sum_exp) + max_val
        result = lse * al.convert(10.0, al.f32)
        out[n] = al.convert(result, al.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        w = self.conv_transpose.weight  # (C_in, C_out, K, K)
        b_conv = self.conv_transpose.bias  # (C_out,)
        b_explicit = self.bias  # (C_out, 1, 1)

        N, C_in, H, W = x.shape
        C_out = w.shape[1]
        K = w.shape[2]

        H_out = H + K - 1
        W_out = W + K - 1
        hw_out = H_out * W_out  # 514 * 514

        # Precompute sum of weight over kernel dims
        w_sum = w.sum(dim=(2, 3))  # (C_in, C_out)

        # Ensure contiguous
        x = x.contiguous()
        w_sum = w_sum.contiguous()
        b_conv = b_conv.contiguous()
        b_explicit_flat = b_explicit.view(-1).contiguous()

        n_stride = C_in * H * W
        c_stride = H * W

        # Allocate x_sum: (N, C_in)
        x_sum = torch.empty(N, C_in, dtype=torch.bfloat16, device=x.device)

        # Launch spatial reduce
        grid = (N * C_in, 1, 1)
        block = (256, 1, 1)
        spatial_reduce_kernel[lambda: (grid, block)](
            x.data_ptr(),
            x_sum.data_ptr(),
            N, C_in, H, W, n_stride, c_stride,
        )

        # Allocate result: (N,)
        result = torch.empty(N, dtype=torch.bfloat16, device=x.device)

        # Launch LSE kernel
        grid2 = (N, 1, 1)
        block2 = (C_out, 1, 1)
        compute_lse_kernel[lambda: (grid2, block2)](
            x_sum.data_ptr(),
            w_sum.data_ptr(),
            b_conv.data_ptr(),
            b_explicit_flat.data_ptr(),
            result.data_ptr(),
            C_in, hw_out, C_out,
        )

        return result.unsqueeze(1)  # (N, 1)
