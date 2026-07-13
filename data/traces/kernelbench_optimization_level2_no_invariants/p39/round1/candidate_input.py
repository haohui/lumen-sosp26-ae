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
_BN_THREADS = 256
_SCRATCH_PER_BLOCK = 4096


@avelang.jit
def gemm_scale_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    scratch_ptr: al.Pointer(al.f32),
    bias_ptr: al.Pointer(al.bf16),
    scale_ptr: al.Pointer(al.bf16),
    y_out_ptr: al.Pointer(al.bf16),
    M: al.i32,
    K: al.i32,
    N: al.i32,
    grid_n: al.i32,
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

    x_1d_layout = al.make_layout((M * K,), (1,))
    w_1d_layout = al.make_layout((K * N,), (1,))
    x_1d = al.view(x, al.bf16, x_1d_layout)
    w_1d = al.view(w, al.bf16, w_1d_layout)
    x_rsrc = al.amdgpu.make_rsrc(x_1d, M * K * 2)
    w_rsrc = al.amdgpu.make_rsrc(w_1d, K * N * 2)

    scratch_layout = al.make_layout((M * N * 4,), (1,))
    scratch = al.make_tensor(scratch_ptr, al.f32, scratch_layout)
    scratch_rsrc = al.amdgpu.make_rsrc(scratch, M * N * 4 * 4)

    block_m = al.block_id(0)
    block_n = al.block_id(1)
    tid = al.thread_id(0)
    wid = tid // 64
    warp_m = wid // 2
    warp_n = wid % 2
    lane = tid % 64
    tile_m = block_m * _GEMM_BLOCK_M + warp_m * _WARP_M
    tile_n = block_n * _GEMM_BLOCK_N + warp_n * _WARP_N

    block_base = (block_m * grid_n + block_n) * _SCRATCH_PER_BLOCK

    for sb in al.range(4):
        base = block_base + (wid * 4 + sb) * 256 + lane * 4
        for i in al.range(4):
            scratch[base + i] = al.convert(0.0, al.f32)

    lane_row = lane // 8
    lane_col = lane % 8

    K_TILES = K // _GEMM_BLOCK_K
    for k_tile in al.range(K_TILES):
        k_start = k_tile * _GEMM_BLOCK_K

        for sb_m in al.range(2):
            for sb_n in al.range(2):
                sb = sb_m * 2 + sb_n

                # Load A operand: 4 bf16 from X using contiguous-load mapping
                a_lin = lane * 4
                a_sb_row = a_lin // 16
                a_sb_col = a_lin % 16
                a_g_row = tile_m + sb_m * 16 + a_sb_row
                a_g_col = k_start + a_sb_col
                a_gbyte = (a_g_row * K + a_g_col) * 2
                a_vec = al.amdgpu.raw_buffer_load_x2(
                    x_rsrc,
                    al.convert(a_gbyte, al.i32),
                    al.convert(0, al.i32),
                    al.convert(0, al.i32),
                )

                # Load B operand: 4 bf16 from W
                b_lin = lane * 4
                b_sb_row = b_lin // 16
                b_sb_col = b_lin % 16
                b_g_row = k_start + b_sb_row
                b_g_col = tile_n + sb_n * 16 + b_sb_col
                b_gbyte = (b_g_row * N + b_g_col) * 2
                b_vec = al.amdgpu.raw_buffer_load_x2(
                    w_rsrc,
                    al.convert(b_gbyte, al.i32),
                    al.convert(0, al.i32),
                    al.convert(0, al.i32),
                )

                # Load C from scratchpad
                c_scratch_off = block_base + (wid * 4 + sb) * 256 + lane * 4
                c_sbyte = c_scratch_off * 4
                c_load = al.amdgpu.raw_buffer_load_x4(
                    scratch_rsrc,
                    al.convert(c_sbyte, al.i32),
                    al.convert(0, al.i32),
                    al.convert(0, al.i32),
                )
                c_vec = al.view(c_load, al.Tensor((4,), al.f32))

                # MFMA 16x16x16
                c_vec = al.amdgpu.mfma_16x16x16_bf16_f32(a_vec, b_vec, c_vec)

                # Store C back
                c_store = al.view(c_vec, al.Tensor((4,), al.u32))
                al.amdgpu.raw_buffer_store_x4(
                    c_store,
                    scratch_rsrc,
                    al.convert(c_sbyte, al.i32),
                    al.convert(0, al.i32),
                    al.convert(0, al.i32),
                )

    # Writeback with bias and scale
    for sb_m in al.range(2):
        for sb_n in al.range(2):
            sb = sb_m * 2 + sb_n
            base = block_base + (wid * 4 + sb) * 256 + lane * 4
            for r in al.range(2):
                for c in al.range(2):
                    acc_idx = r * 2 + c
                    out_r = tile_m + sb_m * 16 + lane_row * 2 + r
                    out_c = tile_n + sb_n * 16 + lane_col * 2 + c
                    val = scratch[base + acc_idx] + al.convert(bias[out_c], al.f32)
                    val = val * al.convert(scale[out_c], al.f32)
                    y_out[out_r, out_c] = al.convert(val, al.bf16)


@avelang.jit
def bn_kernel(
    y_ptr: al.Pointer(al.bf16),
    bn_weight_ptr: al.Pointer(al.bf16),
    bn_bias_ptr: al.Pointer(al.bf16),
    M: al.i32,
    N: al.i32,
):
    y_layout = al.make_layout((M, N), (N, 1))
    w_layout = al.make_layout((N,), (1,))
    b_layout = al.make_layout((N,), (1,))
    y = al.make_tensor(y_ptr, al.bf16, y_layout)
    bn_w = al.make_tensor(bn_weight_ptr, al.bf16, w_layout)
    bn_b = al.make_tensor(bn_bias_ptr, al.bf16, b_layout)

    col = al.block_id(0)
    tid = al.thread_id(0)
    smem = al.make_shared((_BN_THREADS,), al.f32)

    if col < N:
        bn_weight = al.convert(bn_w[col], al.f32)
        bn_bias_val = al.convert(bn_b[col], al.f32)

        partial_sum = al.convert(0.0, al.f32)
        for i in al.range(tid, M, _BN_THREADS):
            partial_sum = partial_sum + al.convert(y[i, col], al.f32)
        smem[tid] = partial_sum
        al.syncthreads()

        if tid < 128:
            smem[tid] = smem[tid] + smem[tid + 128]
        al.syncthreads()
        if tid < 64:
            smem[tid] = smem[tid] + smem[tid + 64]
        al.syncthreads()
        if tid < 32:
            smem[tid] = smem[tid] + smem[tid + 32]
        al.syncthreads()
        if tid < 32:
            val = smem[tid]
            val = val + al.shuffle_down(val, 16, 32)
            val = val + al.shuffle_down(val, 8, 32)
            val = val + al.shuffle_down(val, 4, 32)
            val = val + al.shuffle_down(val, 2, 32)
            val = val + al.shuffle_down(val, 1, 32)
            if tid == 0:
                smem[0] = val
        al.syncthreads()
        mean = smem[0] / al.convert(M, al.f32)

        partial_var = al.convert(0.0, al.f32)
        for i in al.range(tid, M, _BN_THREADS):
            diff = al.convert(y[i, col], al.f32) - mean
            partial_var = partial_var + diff * diff
        smem[tid] = partial_var
        al.syncthreads()

        if tid < 128:
            smem[tid] = smem[tid] + smem[tid + 128]
        al.syncthreads()
        if tid < 64:
            smem[tid] = smem[tid] + smem[tid + 64]
        al.syncthreads()
        if tid < 32:
            smem[tid] = smem[tid] + smem[tid + 32]
        al.syncthreads()
        if tid < 32:
            val = smem[tid]
            val = val + al.shuffle_down(val, 16, 32)
            val = val + al.shuffle_down(val, 8, 32)
            val = val + al.shuffle_down(val, 4, 32)
            val = val + al.shuffle_down(val, 2, 32)
            val = val + al.shuffle_down(val, 1, 32)
            if tid == 0:
                smem[0] = val
        al.syncthreads()
        var = smem[0] / al.convert(M, al.f32)
        denom = al.sqrt(var + al.convert(1e-5, al.f32))

        for i in al.range(tid, M, _BN_THREADS):
            val = (al.convert(y[i, col], al.f32) - mean) / denom
            val = val * bn_weight + bn_bias_val
            y[i, col] = al.convert(val, al.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)
        self._scratch = None

    def forward(self, x):
        M_val = x.shape[0]
        K_val = x.shape[1]
        N_val = self.gemm.out_features
        grid_m = (M_val + _GEMM_BLOCK_M - 1) // _GEMM_BLOCK_M
        grid_n = (N_val + _GEMM_BLOCK_N - 1) // _GEMM_BLOCK_N
        total_blocks = grid_m * grid_n
        scratch_size = total_blocks * _SCRATCH_PER_BLOCK

        if self._scratch is None or self._scratch.numel() < scratch_size:
            self._scratch = torch.empty(scratch_size, device=x.device, dtype=torch.float32)

        w_t = self.gemm.weight.t().to(device=x.device, dtype=x.dtype).contiguous()
        bias = self.gemm.bias.to(device=x.device, dtype=x.dtype).contiguous()
        scale_val = self.scale.to(device=x.device, dtype=x.dtype).contiguous()
        bn_w = self.bn.weight.to(device=x.device, dtype=x.dtype).contiguous()
        bn_b = self.bn.bias.to(device=x.device, dtype=x.dtype).contiguous()
        y = torch.empty((M_val, N_val), device=x.device, dtype=x.dtype)

        gemm_scale_kernel[lambda: ((grid_m, grid_n, 1), (_NUM_WARPS * 64, 1, 1))](
            x.contiguous(), w_t, self._scratch, bias, scale_val, y,
            M_val, K_val, N_val, grid_n,
        )

        bn_kernel[lambda: ((N_val, 1, 1), (_BN_THREADS, 1, 1))](
            y, bn_w, bn_b, M_val, N_val,
        )
        return y
