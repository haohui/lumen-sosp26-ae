import torch
import torch.nn as nn
import avelang
import avelang.language as al

TILE_SIZE = 4096
BLOCK_SIZE = 256


@avelang.jit
def bn_reduce_kernel(
    x_ptr: al.Pointer(al.bf16),
    sum_out_ptr: al.Pointer(al.f32),
    sq_out_ptr: al.Pointer(al.f32),
    N: al.i32,
    C: al.i32,
    H: al.i32,
    W: al.i32,
):
    block_idx = al.block_id(0)
    tid = al.thread_id(0)
    block_dim = al.block_dim(0)

    elems_per_ch = N * H * W
    tiles_per_ch = (elems_per_ch + TILE_SIZE - 1) // TILE_SIZE

    c = block_idx // tiles_per_ch
    tile_idx = block_idx - c * tiles_per_ch

    total_elems = N * C * H * W
    ch_stride = H * W
    batch_stride = C * H * W

    x_layout = al.make_layout((total_elems,), (1,))
    x = al.make_tensor(x_ptr, al.bf16, x_layout)

    out_layout = al.make_layout((C * tiles_per_ch,), (1,))
    sum_out = al.make_tensor(sum_out_ptr, al.f32, out_layout)
    sq_out = al.make_tensor(sq_out_ptr, al.f32, out_layout)

    sum_val = al.convert(0.0, al.f32)
    sq_val = al.convert(0.0, al.f32)

    start = tile_idx * TILE_SIZE
    end = start + TILE_SIZE
    if end > elems_per_ch:
        end = elems_per_ch

    for idx in al.range(start + tid, end, block_dim):
        n = idx // ch_stride
        rem = idx - n * ch_stride
        h = rem // W
        w = rem - h * W
        g_idx = n * batch_stride + c * ch_stride + h * W + w

        val = al.convert(x[g_idx], al.f32)
        sum_val = sum_val + val
        sq_val = sq_val + val * val

    # Warp-level reduction (warp size = 64 on AMD)
    sum_val = sum_val + al.shuffle_down(sum_val, 32, 64)
    sum_val = sum_val + al.shuffle_down(sum_val, 16, 64)
    sum_val = sum_val + al.shuffle_down(sum_val, 8, 64)
    sum_val = sum_val + al.shuffle_down(sum_val, 4, 64)
    sum_val = sum_val + al.shuffle_down(sum_val, 2, 64)
    sum_val = sum_val + al.shuffle_down(sum_val, 1, 64)

    sq_val = sq_val + al.shuffle_down(sq_val, 32, 64)
    sq_val = sq_val + al.shuffle_down(sq_val, 16, 64)
    sq_val = sq_val + al.shuffle_down(sq_val, 8, 64)
    sq_val = sq_val + al.shuffle_down(sq_val, 4, 64)
    sq_val = sq_val + al.shuffle_down(sq_val, 2, 64)
    sq_val = sq_val + al.shuffle_down(sq_val, 1, 64)

    # Cross-warp reduction via shared memory
    warp_id = tid // 64
    lane_id = tid - warp_id * 64

    shared_sum = al.make_shared((4,), al.f32)
    shared_sq = al.make_shared((4,), al.f32)

    if lane_id == 0:
        shared_sum[warp_id] = sum_val
        shared_sq[warp_id] = sq_val

    al.syncthreads()

    if warp_id == 0:
        if lane_id < 4:
            sum_val = shared_sum[lane_id]
            sq_val = shared_sq[lane_id]
        else:
            sum_val = al.convert(0.0, al.f32)
            sq_val = al.convert(0.0, al.f32)

        # Reduce 4 warp sums within warp 0
        sum_val = sum_val + al.shuffle_down(sum_val, 2, 4)
        sum_val = sum_val + al.shuffle_down(sum_val, 1, 4)

        sq_val = sq_val + al.shuffle_down(sq_val, 2, 4)
        sq_val = sq_val + al.shuffle_down(sq_val, 1, 4)

        if lane_id == 0:
            sum_out[block_idx] = sum_val
            sq_out[block_idx] = sq_val


@avelang.jit
def bn_normalize_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    gamma_ptr: al.Pointer(al.f32),
    beta_ptr: al.Pointer(al.f32),
    mean_ptr: al.Pointer(al.f32),
    inv_std_ptr: al.Pointer(al.f32),
    N: al.i32,
    C: al.i32,
    H: al.i32,
    W: al.i32,
):
    block_idx = al.block_id(0)
    tid = al.thread_id(0)
    block_dim = al.block_dim(0)

    elems_per_ch = N * H * W
    tiles_per_ch = (elems_per_ch + TILE_SIZE - 1) // TILE_SIZE

    c = block_idx // tiles_per_ch
    tile_idx = block_idx - c * tiles_per_ch

    total_elems = N * C * H * W
    ch_stride = H * W
    batch_stride = C * H * W

    x_layout = al.make_layout((total_elems,), (1,))
    x = al.make_tensor(x_ptr, al.bf16, x_layout)
    out = al.make_tensor(out_ptr, al.bf16, x_layout)

    param_layout = al.make_layout((C,), (1,))
    gamma = al.make_tensor(gamma_ptr, al.f32, param_layout)
    beta = al.make_tensor(beta_ptr, al.f32, param_layout)
    mean = al.make_tensor(mean_ptr, al.f32, param_layout)
    inv_std = al.make_tensor(inv_std_ptr, al.f32, param_layout)

    g = gamma[c]
    b = beta[c]
    m = mean[c]
    istd = inv_std[c]

    start = tile_idx * TILE_SIZE
    end = start + TILE_SIZE
    if end > elems_per_ch:
        end = elems_per_ch

    for idx in al.range(start + tid, end, block_dim):
        n = idx // ch_stride
        rem = idx - n * ch_stride
        h = rem // W
        w = rem - h * W
        g_idx = n * batch_stride + c * ch_stride + h * W + w

        val = al.convert(x[g_idx], al.f32)
        norm_val = (val - m) * istd
        result = g * norm_val + b
        out[g_idx] = al.convert(result, al.bf16)


class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.eps = 1e-5
        self.momentum = 0.1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape

        x_bf16 = x.to(torch.bfloat16).contiguous()

        elems_per_ch = N * H * W
        tiles_per_ch = (elems_per_ch + TILE_SIZE - 1) // TILE_SIZE
        total_tiles = C * tiles_per_ch

        grid = (total_tiles, 1, 1)
        block = (BLOCK_SIZE, 1, 1)

        if self.training:
            partial_sum = torch.zeros(total_tiles, dtype=torch.float32, device=x.device)
            partial_sq = torch.zeros(total_tiles, dtype=torch.float32, device=x.device)

            bn_reduce_kernel[lambda: (grid, block)](
                x_bf16, partial_sum, partial_sq,
                N, C, H, W,
            )

            partial_sum_2d = partial_sum.view(C, tiles_per_ch)
            partial_sq_2d = partial_sq.view(C, tiles_per_ch)

            sum_per_ch = partial_sum_2d.sum(dim=1)
            sq_per_ch = partial_sq_2d.sum(dim=1)

            count = float(elems_per_ch)
            mean = sum_per_ch / count
            var = sq_per_ch / count - mean * mean

            self.running_mean = (
                (1.0 - self.momentum) * self.running_mean.to(device=x.device)
                + self.momentum * mean.to(device=x.device)
            )
            self.running_var = (
                (1.0 - self.momentum) * self.running_var.to(device=x.device)
                + self.momentum * var.to(device=x.device)
            )
        else:
            mean = self.running_mean.to(device=x.device, dtype=torch.float32)
            var = self.running_var.to(device=x.device, dtype=torch.float32)

        inv_std = 1.0 / torch.sqrt(var + self.eps)

        # Convert parameters to FP32 for the kernel (they may be BF16 after model.to)
        weight_f32 = self.weight.to(dtype=torch.float32)
        bias_f32 = self.bias.to(dtype=torch.float32)

        out_bf16 = torch.empty_like(x_bf16)

        bn_normalize_kernel[lambda: (grid, block)](
            x_bf16, out_bf16,
            weight_f32, bias_f32,
            mean, inv_std,
            N, C, H, W,
        )

        return out_bf16.to(x.dtype)
