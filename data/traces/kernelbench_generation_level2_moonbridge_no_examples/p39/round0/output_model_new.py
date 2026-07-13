import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def scale_kernel(
    x_ptr: al.Pointer(al.bf16),
    scale_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    M: al.i32,
    N: al.i32,
):
    block_m = al.block_id(0)
    block_n = al.block_id(1)
    tid_m = al.thread_id(0)
    tid_n = al.thread_id(1)
    row = block_m * 16 + tid_m
    col = block_n * 16 + tid_n
    if row < M:
        if col < N:
            xl = al.make_layout((M, N), (N, 1))
            xv = al.make_tensor(x_ptr, al.bf16, xl)
            sl = al.make_layout((N,), (1,))
            sv = al.make_tensor(scale_ptr, al.bf16, sl)
            ol = al.make_layout((M, N), (N, 1))
            ov = al.make_tensor(out_ptr, al.bf16, ol)
            val = al.convert(xv[row, col], al.f32)
            s = al.convert(sv[col], al.f32)
            ov[row, col] = al.convert(val * s, al.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)

    def forward(self, x):
        M = x.shape[0]
        N = self.gemm.weight.shape[0]
        device = x.device

        x_bf16 = x.contiguous().to(torch.bfloat16)
        s_bf16 = self.scale.to(torch.bfloat16)

        gemm_out = self.gemm(x_bf16).contiguous()
        scaled = torch.empty((M, N), device=device, dtype=torch.bfloat16)
        gm = (M + 15) // 16
        gn = (N + 15) // 16
        scale_kernel[lambda: ((gm, gn, 1), (16, 16, 1))](gemm_out, s_bf16, scaled, M, N)

        return self.bn(scaled)


batch_size = 16384
in_features = 4096
out_features = 4096
scale_shape = (out_features,)


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, scale_shape]
