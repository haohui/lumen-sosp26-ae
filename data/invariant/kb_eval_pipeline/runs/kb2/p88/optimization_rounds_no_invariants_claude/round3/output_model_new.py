import torch
import torch.nn as nn
import substrate
import substrate.language as S

SQRT_2 = 1.4142135623730951
BATCH_SIZE = 1024
IN_FEATURES = 8192
OUT_FEATURES = 8192
NUM_GROUPS = 256
GROUP_SIZE = OUT_FEATURES // NUM_GROUPS
EPS = 1e-05

# Tiled GEMM with 32x32 tiles
TILE_SIZE = 32


def _launch():
    return ((BATCH_SIZE // TILE_SIZE, OUT_FEATURES // TILE_SIZE, 1), (TILE_SIZE, 1, 1))


@substrate.jit
def gemm_kernel(
    X: S.Tensor((BATCH_SIZE, IN_FEATURES), S.bf16),
    W: S.Tensor((IN_FEATURES, OUT_FEATURES), S.bf16),
    BIAS0: S.Tensor((OUT_FEATURES,), S.bf16),
    Y: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.bf16),
):
    bx = S.block_id(0)
    by = S.block_id(1)
    tid = S.thread_id(0)

    # Output tile position
    m_tile = bx * TILE_SIZE
    n_tile = by * TILE_SIZE

    # Each thread computes one row of the tile
    m = m_tile + tid

    # Compute all columns for this row
    for n in S.range(TILE_SIZE):
        global_n = n_tile + n

        # Accumulator
        acc = S.convert(0.0, S.f32)

        # Matrix multiplication
        for k in S.range(IN_FEATURES):
            x_val = S.convert(X[m, k], S.f32)
            w_val = S.convert(W[k, global_n], S.f32)
            acc = acc + x_val * w_val

        # Add bias and write output
        Y[m, global_n] = S.convert(acc + S.convert(BIAS0[global_n], S.f32), S.bf16)


@substrate.jit
def group_norm_silu_kernel(
    Y: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.bf16),
    GN_WEIGHT: S.Tensor((OUT_FEATURES,), S.bf16),
    GN_BIAS: S.Tensor((OUT_FEATURES,), S.bf16),
    MUL_WEIGHT: S.Tensor((OUT_FEATURES,), S.bf16),
):
    one = S.convert(1.0, S.f32)
    for i in S.range(BATCH_SIZE):
        for g in S.range(NUM_GROUPS):
            mean = S.convert(0.0, S.f32)
            for t in S.range(GROUP_SIZE):
                c = g * GROUP_SIZE + t
                mean += S.convert(Y[i, c], S.f32)
            mean = mean / S.convert(GROUP_SIZE, S.f32)
            var = S.convert(0.0, S.f32)
            for t in S.range(GROUP_SIZE):
                c = g * GROUP_SIZE + t
                d = S.convert(Y[i, c], S.f32) - mean
                var += d * d
            var = var / S.convert(GROUP_SIZE, S.f32)
            denom = S.sqrt(var + S.convert(EPS, S.f32))
            for t in S.range(GROUP_SIZE):
                c = g * GROUP_SIZE + t
                v = (S.convert(Y[i, c], S.f32) - mean) / denom
                v = v * S.convert(GN_WEIGHT[c], S.f32) + S.convert(GN_BIAS[c], S.f32)
                v = v * (one / (one + S.exp(-v)))
                v = v * S.convert(MUL_WEIGHT[c], S.f32)
                v = v * (one / (one + S.exp(-v)))
                Y[i, c] = S.convert(v, S.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape))

    def forward(self, x):
        if tuple(x.shape) != (BATCH_SIZE, IN_FEATURES) or x.dtype != torch.bfloat16 or self.group_norm.num_groups != NUM_GROUPS or (self.group_norm.eps != EPS) or (tuple(self.multiply_weight.shape) != (OUT_FEATURES,)):
            raise RuntimeError('This fused kernel only supports the benchmark input shape and dtype.')
        w_t = self.gemm.weight.t().to(device=x.device, dtype=x.dtype).contiguous()
        bias = self.gemm.bias.to(device=x.device, dtype=x.dtype).contiguous()
        gn_w = self.group_norm.weight.to(device=x.device, dtype=x.dtype).contiguous()
        gn_b = self.group_norm.bias.to(device=x.device, dtype=x.dtype).contiguous()
        mul = self.multiply_weight.to(device=x.device, dtype=x.dtype).contiguous()
        y = torch.empty((BATCH_SIZE, OUT_FEATURES), device=x.device, dtype=x.dtype)
        gemm_kernel[_launch](x.contiguous(), w_t, bias, y)
        group_norm_silu_kernel[lambda: ((BATCH_SIZE, 1, 1), (1, 1, 1))](y, gn_w, gn_b, mul)
        return y
