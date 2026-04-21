import torch
import torch.nn as nn
import substrate
import substrate.language as S

BATCH_SIZE = 1024
IN_FEATURES = 8192
OUT_FEATURES = 8192
NUM_GROUPS = 256
GROUP_SIZE = OUT_FEATURES // NUM_GROUPS  # 32
EPS = 1e-5


def _launch():
    grid_m = BATCH_SIZE // 32  # 32
    grid_n = OUT_FEATURES // 32  # 256
    return ((grid_m * grid_n, 1, 1), (64, 1, 1))


@substrate.jit
def fused_kernel(
    X: S.Tensor((BATCH_SIZE, IN_FEATURES // 4, 2), S.u32),
    # W_orig is the original weight (OUT_FEATURES x IN_FEATURES), NOT transposed
    W_orig: S.Tensor((OUT_FEATURES, IN_FEATURES // 4, 2), S.u32),
    BIAS0: S.Tensor((OUT_FEATURES,), S.bf16),
    GN_WEIGHT: S.Tensor((OUT_FEATURES,), S.bf16),
    GN_BIAS: S.Tensor((OUT_FEATURES,), S.bf16),
    MUL_WEIGHT: S.Tensor((OUT_FEATURES,), S.bf16),
    Y: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.bf16),
):
    lane = S.thread_id(0)
    bid = S.block_id(0)

    grid_n = OUT_FEATURES // 32
    tile_row = bid // grid_n
    tile_col = bid % grid_n

    m_base = tile_row * 32
    n_base = tile_col * 32

    acc = S.full((16,), 0.0, S.f32)

    a_row = m_base + (lane % 32)
    b_row = n_base + (lane % 32)

    # Create buffer resource descriptors with range for OOB-safe access
    # range is in bytes; when set, OOB loads return 0, OOB stores are discarded
    X_range = BATCH_SIZE * (IN_FEATURES // 4) * 2 * 4
    W_range = OUT_FEATURES * (IN_FEATURES // 4) * 2 * 4
    rsrc_X = S.amdgpu.make_rsrc(X, X_range)
    rsrc_W = S.amdgpu.make_rsrc(W_orig, W_range)

    # Pre-compute row byte offsets for X and W_orig
    # X shape: (BATCH_SIZE, IN_FEATURES//4, 2) of u32
    # X[a_row, k_idx, :] byte offset = a_row * (IN_FEATURES//4 * 2) * 4 + k_idx * 8
    a_row_byte_base = a_row * (IN_FEATURES // 4) * 2 * 4
    b_row_byte_base = b_row * (IN_FEATURES // 4) * 2 * 4

    # Software pipelining: K-loop unrolled by 2, interleaved loads and computes.
    # Uses raw_buffer_load_x2 with range to eliminate implicit bounds-check branches.
    for kk in S.range(IN_FEATURES // 16):
        k_off_0 = kk * 16
        k_off_1 = kk * 16 + 8

        # Compute k indices and byte offsets for X loads
        a_k_idx_0 = k_off_0 // 4 + (lane // 32)
        a_k_idx_1 = k_off_1 // 4 + (lane // 32)
        a_off_0 = a_row_byte_base + a_k_idx_0 * 8
        a_off_1 = a_row_byte_base + a_k_idx_1 * 8

        # Compute k indices and byte offsets for W loads
        b_k_idx_0 = k_off_0 // 4 + (lane // 32)
        b_k_idx_1 = k_off_1 // 4 + (lane // 32)
        b_off_0 = b_row_byte_base + b_k_idx_0 * 8
        b_off_1 = b_row_byte_base + b_k_idx_1 * 8

        # Load first sub-tile A and B using raw_buffer_load_x2 with range
        a_frag_0 = S.amdgpu.raw_buffer_load_x2(rsrc_X, a_off_0, 0, 0)
        b_frag_0 = S.amdgpu.raw_buffer_load_x2(rsrc_W, b_off_0, 0, 0)

        # Issue second sub-tile A load early (pipelined with view/compute)
        a_frag_1 = S.amdgpu.raw_buffer_load_x2(rsrc_X, a_off_1, 0, 0)

        # Compute first sub-tile (second A load in flight)
        m_a0 = S.view(a_frag_0, S.Tensor((1, 4, 1), S.bf16))
        m_b0 = S.view(b_frag_0, S.Tensor((1, 4, 1), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(m_a0[0], m_b0[0], acc)

        # Issue second sub-tile B load (pipelined with first MFMA)
        b_frag_1 = S.amdgpu.raw_buffer_load_x2(rsrc_W, b_off_1, 0, 0)

        # Compute second sub-tile (B load in flight)
        m_a1 = S.view(a_frag_1, S.Tensor((1, 4, 1), S.bf16))
        m_b1 = S.view(b_frag_1, S.Tensor((1, 4, 1), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(m_a1[0], m_b1[0], acc)

    one = S.convert(1.0, S.f32)
    col = n_base + (lane % 32)
    bias_v = S.convert(BIAS0[col], S.f32)

    vals = S.full((16,), 0.0, S.f32)
    for ai in S.range(16):
        vals[ai] = acc[ai] + bias_v

    # GroupNorm + SiLU + multiply + SiLU
    for ai in S.range(16):
        v = vals[ai]
        s = S.shuffle_xor(v, 16, 32)
        v = v + s
        s = S.shuffle_xor(v, 8, 32)
        v = v + s
        s = S.shuffle_xor(v, 4, 32)
        v = v + s
        s = S.shuffle_xor(v, 2, 32)
        v = v + s
        s = S.shuffle_xor(v, 1, 32)
        v = v + s
        mean = v / S.convert(GROUP_SIZE, S.f32)

        d = vals[ai] - mean
        v2 = d * d
        s = S.shuffle_xor(v2, 16, 32)
        v2 = v2 + s
        s = S.shuffle_xor(v2, 8, 32)
        v2 = v2 + s
        s = S.shuffle_xor(v2, 4, 32)
        v2 = v2 + s
        s = S.shuffle_xor(v2, 2, 32)
        v2 = v2 + s
        s = S.shuffle_xor(v2, 1, 32)
        v2 = v2 + s
        var = v2 / S.convert(GROUP_SIZE, S.f32)
        denom = S.sqrt(var + S.convert(EPS, S.f32))

        result = (vals[ai] - mean) / denom
        result = result * S.convert(GN_WEIGHT[col], S.f32) + S.convert(GN_BIAS[col], S.f32)
        result = result * (one / (one + S.exp(-result)))
        result = result * S.convert(MUL_WEIGHT[col], S.f32)
        result = result * (one / (one + S.exp(-result)))

        row = m_base + 8 * (ai // 4) + 4 * (lane // 32) + (ai % 4)
        Y[row, col] = S.convert(result, S.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape))

    def forward(self, x):
        if (
            tuple(x.shape) != (BATCH_SIZE, IN_FEATURES)
            or x.dtype != torch.bfloat16
            or self.group_norm.num_groups != NUM_GROUPS
            or self.group_norm.eps != EPS
            or tuple(self.multiply_weight.shape) != (OUT_FEATURES,)
        ):
            raise RuntimeError("This fused kernel only supports the benchmark input shape and dtype.")

        # Use original weight (NOT transposed) for B operand
        w_orig = self.gemm.weight.to(device=x.device, dtype=x.dtype).contiguous()
        bias = self.gemm.bias.to(device=x.device, dtype=x.dtype).contiguous()
        gn_w = self.group_norm.weight.to(device=x.device, dtype=x.dtype).contiguous()
        gn_b = self.group_norm.bias.to(device=x.device, dtype=x.dtype).contiguous()
        mul = self.multiply_weight.to(device=x.device, dtype=x.dtype).contiguous()

        x_4 = x.contiguous().view(torch.int32).reshape(BATCH_SIZE, IN_FEATURES // 4, 2)
        w_4 = w_orig.view(torch.int32).reshape(OUT_FEATURES, IN_FEATURES // 4, 2)

        y = torch.empty((BATCH_SIZE, OUT_FEATURES), device=x.device, dtype=x.dtype)
        fused_kernel[_launch](x_4, w_4, bias, gn_w, gn_b, mul, y)
        return y
