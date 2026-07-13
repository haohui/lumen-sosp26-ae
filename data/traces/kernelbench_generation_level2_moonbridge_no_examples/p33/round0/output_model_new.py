import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def gemm_kernel(
    a_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    c_ptr: al.Pointer(al.bf16),
    M: al.i32,
    N: al.i32,
    K: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)
    bdim = al.block_dim(0)

    idx = bid * bdim + tid
    col = idx % N
    row = (idx - col) / N

    total_a = M * K
    a_layout = al.make_layout((total_a,), (1,))
    a = al.make_tensor(a_ptr, al.bf16, a_layout)

    total_b = K * N
    b_layout = al.make_layout((total_b,), (1,))
    b = al.make_tensor(b_ptr, al.bf16, b_layout)

    bias_layout = al.make_layout((N,), (1,))
    bias = al.make_tensor(bias_ptr, al.bf16, bias_layout)

    total_c = M * N
    c_layout = al.make_layout((total_c,), (1,))
    c = al.make_tensor(c_ptr, al.bf16, c_layout)

    if idx < total_c:
        acc = al.convert(0.0, al.f32)
        for k in al.range(K):
            a_val = al.convert(a[row * K + k], al.f32)
            b_val = al.convert(b[col * K + k], al.f32)
            acc = acc + a_val * b_val
        bias_val = al.convert(bias[col], al.f32)
        c[idx] = al.convert(acc + bias_val, al.bf16)


@avelang.jit
def scale_kernel(
    x_ptr: al.Pointer(al.bf16),
    s_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    rows: al.i32,
    cols: al.i32,
):
    tid = al.thread_id(0)
    bdim = al.block_dim(0)
    total = rows * cols

    x_layout = al.make_layout((total,), (1,))
    x = al.make_tensor(x_ptr, al.bf16, x_layout)
    s_layout = al.make_layout((cols,), (1,))
    s = al.make_tensor(s_ptr, al.bf16, s_layout)
    o_layout = al.make_layout((total,), (1,))
    o = al.make_tensor(out_ptr, al.bf16, o_layout)

    for idx in al.range(tid, total, bdim):
        c = idx % cols
        x_val = al.convert(x[idx], al.f32)
        s_val = al.convert(s[c], al.f32)
        o[idx] = al.convert(x_val * s_val, al.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)

    def forward(self, x):
        M = x.shape[0]
        K = x.shape[1]
        N = self.out_features

        x = x.contiguous()
        weight = self.gemm.weight.data.contiguous()
        bias = self.gemm.bias.data.contiguous()
        scale = self.scale.data.contiguous()

        gemm_out = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
        total_out = M * N
        grid = (total_out + 255) // 256
        gemm_kernel[lambda: ((grid, 1, 1), (256, 1, 1))](
            x, weight, bias, gemm_out, M, N, K
        )

        scaled_out = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
        total = M * N
        grid_total = (total + 255) // 256
        scale_kernel[lambda: ((grid_total, 1, 1), (256, 1, 1))](
            gemm_out, scale, scaled_out, M, N
        )

        return self.bn(scaled_out)
