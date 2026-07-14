import torch
import torch.nn as nn
import avelang
import avelang.language as al
SQRT_2 = 1.4142135623730951

def _launch():
    return ((1, 1, 1), (1, 1, 1))
BATCH_SIZE = 1024
IN_FEATURES = 8192
OUT_FEATURES = 8192
MAX_DIM = 1

@avelang.jit
def fused_kernel(X: al.Tensor((BATCH_SIZE, IN_FEATURES), al.bf16), W: al.Tensor((IN_FEATURES, OUT_FEATURES), al.bf16), BIAS0: al.Tensor((OUT_FEATURES,), al.bf16), Y: al.Tensor((BATCH_SIZE, 1), al.bf16)):
    zero = al.convert(0.0, al.f32)
    for i in al.range(BATCH_SIZE):
        max_v = al.convert(-1e+30, al.f32)
        for j in al.range(OUT_FEATURES):
            v = al.convert(0.0, al.f32)
            for kk in al.range(IN_FEATURES):
                v += al.convert(X[i, kk], al.f32) * al.convert(W[kk, j], al.f32)
            v += al.convert(BIAS0[j], al.f32)
            if v > max_v:
                max_v = v
        v = max_v - max_v
        v = al.convert(0.5, al.f32) * v * (al.convert(1.0, al.f32) + al.erf(v / al.convert(SQRT_2, al.f32)))
        Y[i, 0] = al.convert(v + zero, al.bf16)

class ModelNew(nn.Module):

    def __init__(self, in_features, out_features, max_dim):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.max_dim = max_dim

    def forward(self, x):
        if tuple(x.shape) != (BATCH_SIZE, IN_FEATURES) or x.dtype != torch.bfloat16 or self.max_dim != MAX_DIM:
            raise RuntimeError('This fused kernel only supports the benchmark input shape and dtype.')
        w_t = self.gemm.weight.t().to(device=x.device, dtype=x.dtype).contiguous()
        bias = self.gemm.bias.to(device=x.device, dtype=x.dtype).contiguous()
        y = torch.empty((BATCH_SIZE, 1), device=x.device, dtype=x.dtype)
        fused_kernel[_launch](x.contiguous(), w_t, bias, y)
        return y
