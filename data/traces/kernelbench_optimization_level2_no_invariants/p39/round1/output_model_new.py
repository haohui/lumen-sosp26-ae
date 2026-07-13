import torch
import torch.nn as nn
import avelang
import avelang.language as al

_GEMM_BLOCK_M = 64
_GEMM_BLOCK_N = 64
_GEMM_BLOCK_K = 16
_WARP_M = 32
_WARP_N = 32
_NUM_WARPS = 4


@avelang.jit
def gemm_scale_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    scale_ptr: al.Pointer(al.bf16),
    y_out_ptr: al.Pointer(al.bf16),
    M: al.i32,
    K: al.i32,
    N: al.i32,
):
    x_layout = al.make_layout((M, K), (K, 1))
    w_layout = al.make_layout((K, N), (N, 1))
    yo_layout = al.make_layout((M, N), (N, 1))
    bias_layout = al.make_layout((N,), (1,))
    scale_layout = al.make_layout((N,), (1,))
    x = al.make_tensor(x_ptr, al.bf16, x_layout)
    w = al.make_tensor(w_ptr, al.bf16, w_layout)
    y_out = al.make_tensor(y_out_ptr, al.bf16, yo_layout)
    bias = al.make_tensor(bias_ptr, al.bf16, bias_layout)
    scale = al.make_tensor(scale_ptr, al.bf16, scale_layout)

    block_m = al.block_id(0)
    block_n = al.block_id(1)
    tid = al.thread_id(0)
    wid = tid // 64
    warp_m = wid // 2
    warp_n = wid % 2
    lane = tid % 64
    tile_m = block_m * _GEMM_BLOCK_M + warp_m * _WARP_M
    tile_n = block_n * _GEMM_BLOCK_N + warp_n * _WARP_N

    lane_row = lane // 8
    lane_col = lane % 8

    # Each thread computes 4x4 = 16 output elements
    acc = al.make_local((4, 4), al.f32)
    for ri in al.range(4):
        for ci in al.range(4):
            acc[ri, ci] = al.convert(0.0, al.f32)

    K_TILES = K // _GEMM_BLOCK_K

    # K unrolled by 2 for reduced branching
    for k_tile in al.range(0, K_TILES, 2):
        k0 = k_tile * _GEMM_BLOCK_K
        k1 = (k_tile + al.convert(1, al.i32)) * _GEMM_BLOCK_K

        # Process K-tile k0
        for ki in al.range(_GEMM_BLOCK_K):
            k_idx = k0 + ki
            for ri in al.range(4):
                a_row = tile_m + lane_row * 4 + ri
                a_val = al.convert(x[a_row, k_idx], al.f32)
                for ci in al.range(4):
                    b_col = tile_n + lane_col * 4 + ci
                    b_val = al.convert(w[k_idx, b_col], al.f32)
                    acc[ri, ci] = acc[ri, ci] + a_val * b_val

        # Process K-tile k1 (if exists)
        if k_tile + al.convert(1, al.i32) < K_TILES:
            for ki in al.range(_GEMM_BLOCK_K):
                k_idx = k1 + ki
                for ri in al.range(4):
                    a_row = tile_m + lane_row * 4 + ri
                    a_val = al.convert(x[a_row, k_idx], al.f32)
                    for ci in al.range(4):
                        b_col = tile_n + lane_col * 4 + ci
                        b_val = al.convert(w[k_idx, b_col], al.f32)
                        acc[ri, ci] = acc[ri, ci] + a_val * b_val

    # Writeback: bias + scale + convert to bf16
    for ri in al.range(4):
        for ci in al.range(4):
            out_r = tile_m + lane_row * 4 + ri
            out_c = tile_n + lane_col * 4 + ci
            val = acc[ri, ci] + al.convert(bias[out_c], al.f32)
            val = val * al.convert(scale[out_c], al.f32)
            y_out[out_r, out_c] = al.convert(val, al.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)

    def forward(self, x):
        M_val = x.shape[0]
        K_val = x.shape[1]
        N_val = self.gemm.out_features
        grid_m = (M_val + _GEMM_BLOCK_M - 1) // _GEMM_BLOCK_M
        grid_n = (N_val + _GEMM_BLOCK_N - 1) // _GEMM_BLOCK_N

        w_t = self.gemm.weight.t().to(device=x.device, dtype=x.dtype).contiguous()
        bias = self.gemm.bias.to(device=x.device, dtype=x.dtype).contiguous()
        scale_val = self.scale.to(device=x.device, dtype=x.dtype).contiguous()
        y = torch.empty((M_val, N_val), device=x.device, dtype=x.dtype)

        gemm_scale_kernel[lambda: ((grid_m, grid_n, 1), (_NUM_WARPS * 64, 1, 1))](
            x.contiguous(), w_t, bias, scale_val, y,
            M_val, K_val, N_val,
        )

        y = self.bn(y)
        return y
