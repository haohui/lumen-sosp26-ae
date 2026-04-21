import torch
import torch.nn as nn
import substrate
import substrate.language as S

BATCH_SIZE = 16384
IN_FEATURES = 4096
OUT_FEATURES = 4096
EPS = 1e-5

WARP_SIZE = 64
NUM_WARPS = 4
THREADS = WARP_SIZE * NUM_WARPS

TILE_M = 64
TILE_N = 64
TILE_K = 16

MFMA_M = 32
MFMA_N = 32
MFMA_K = 8

X_BYTES = BATCH_SIZE * (IN_FEATURES // 2) * 4
W_BYTES = (IN_FEATURES // 4) * OUT_FEATURES * 2 * 4


def _launch():
    grid_m = (BATCH_SIZE + TILE_M - 1) // TILE_M
    grid_n = (OUT_FEATURES + TILE_N - 1) // TILE_N
    return ((grid_m * grid_n, 1, 1), (THREADS, 1, 1))


def pack_a_row_major(tensor_bf16):
    M, K = tensor_bf16.shape
    packed = torch.zeros(M, K // 2, device=tensor_bf16.device, dtype=torch.int32)
    for i in range(K // 2):
        lo = tensor_bf16[:, i * 2].view(torch.int16).int() & 0xFFFF
        hi = tensor_bf16[:, i * 2 + 1].view(torch.int16).int() & 0xFFFF
        packed[:, i] = (lo | (hi << 16)).int()
    return packed


def pack_b_for_mfma(B_bf16):
    K, N = B_bf16.shape
    num_k_chunks = K // 8
    total_rows = num_k_chunks * 2
    packed = torch.zeros(total_rows, N, 2, device=B_bf16.device, dtype=torch.int32)
    for k_chunk in range(num_k_chunks):
        k_base = k_chunk * 8
        for lane_group in range(2):
            k_start = k_base + lane_group * 4
            row_idx = k_chunk * 2 + lane_group
            for col in range(N):
                vals = B_bf16[k_start:k_start+4, col]
                lo0 = vals[0].view(torch.int16).int() & 0xFFFF
                hi0 = vals[1].view(torch.int16).int() & 0xFFFF
                lo1 = vals[2].view(torch.int16).int() & 0xFFFF
                hi1 = vals[3].view(torch.int16).int() & 0xFFFF
                packed[row_idx, col, 0] = (lo0 | (hi0 << 16))
                packed[row_idx, col, 1] = (lo1 | (hi1 << 16))
    return packed


@substrate.jit
def gemm_mfma_kernel(
    X: S.Tensor((BATCH_SIZE, IN_FEATURES // 2), S.u32),
    W: S.Tensor((IN_FEATURES // 4, OUT_FEATURES, 2), S.u32),
    BIAS: S.Tensor((OUT_FEATURES,), S.bf16),
    Y: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.bf16),
):
    block_idx = S.block_id(0)
    thread_idx = S.thread_id(0)

    grid_n = (OUT_FEATURES + TILE_N - 1) // TILE_N
    block_m = block_idx // grid_n
    block_n = block_idx % grid_n

    warp_id = thread_idx // WARP_SIZE
    lane_id = thread_idx % WARP_SIZE

    warp_m = warp_id // 2
    warp_n = warp_id % 2

    warp_m_start = block_m * TILE_M + warp_m * MFMA_M
    warp_n_start = block_n * TILE_N + warp_n * MFMA_N

    rsrc_X = S.amdgpu.make_rsrc(X, X_BYTES)
    rsrc_W = S.amdgpu.make_rsrc(W, W_BYTES)

    A_frag_shared = S.make_shared((THREADS, 2), S.u32)
    B_frag_shared = S.make_shared((THREADS, 2), S.u32)

    acc = S.full((16,), 0.0, S.f32)

    num_k_tiles = IN_FEATURES // TILE_K

    for k_tile in S.range(num_k_tiles):
        k_offset = k_tile * TILE_K

        for k_mfma in S.range(2):
            a_row = warp_m_start + (lane_id % 32)
            a_k_bf16_start = k_offset + k_mfma * MFMA_K + (lane_id // 32) * 4
            a_u32_col = a_k_bf16_start // 2

            a_byte_offset = (a_row * (IN_FEATURES // 2) + a_u32_col) * 4
            a_data = S.amdgpu.raw_buffer_load_x2(rsrc_X, a_byte_offset, 0, 0)
            a_data_u32 = S.view(a_data, S.Tensor((2,), S.u32))
            for u in S.range(2):
                A_frag_shared[thread_idx, u] = a_data_u32[u]

            b_col = warp_n_start + (lane_id % 32)
            b_k_bf16_start = k_offset + k_mfma * MFMA_K + (lane_id // 32) * 4
            b_row_idx = (b_k_bf16_start // 4)

            b_byte_offset = (b_row_idx * OUT_FEATURES * 2 + b_col * 2) * 4
            b_data = S.amdgpu.raw_buffer_load_x2(rsrc_W, b_byte_offset, 0, 0)
            b_data_u32 = S.view(b_data, S.Tensor((2,), S.u32))
            for u in S.range(2):
                B_frag_shared[thread_idx, u] = b_data_u32[u]

            S.syncthreads()

            m_a = S.view(A_frag_shared[thread_idx], S.Tensor((1, 4, 1), S.bf16))
            m_b = S.view(B_frag_shared[thread_idx], S.Tensor((1, 4, 1), S.bf16))

            acc = S.amdgpu.mfma_32x32x8_bf16_f32(m_a[0], m_b[0], acc)

    col = warp_n_start + (lane_id % 32)
    row_offset = 0 if lane_id < 32 else 4

    for j in S.range(16):
        row_in_warp = (j // 4) * 8 + (j % 4) + row_offset
        row = warp_m_start + row_in_warp

        val = acc[j]
        bias_val = S.convert(BIAS[col], S.f32)
        y_val = val + bias_val
        Y[row, col] = S.convert(y_val, S.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)

    def forward(self, x):
        x_packed = pack_a_row_major(x.contiguous())
        w_t = self.gemm.weight.t().to(device=x.device, dtype=x.dtype).contiguous()
        w_packed = pack_b_for_mfma(w_t)
        bias = self.gemm.bias.to(device=x.device, dtype=x.dtype).contiguous()

        y = torch.empty((BATCH_SIZE, OUT_FEATURES), device=x.device, dtype=x.dtype)

        gemm_mfma_kernel[_launch](x_packed, w_packed, bias, y)

        # Use PyTorch for scale and BN
        y = y * self.scale
        y = self.bn(y)

        return y
