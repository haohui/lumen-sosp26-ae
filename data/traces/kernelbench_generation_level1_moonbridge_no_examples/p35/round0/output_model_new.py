import torch
import torch.nn as nn
import avelang
import avelang.language as al

NUM_CHUNKS = 32


@avelang.jit
def partial_reduce_kernel(
    x_ptr: al.Pointer(al.bf16),
    partial_sum_ptr: al.Pointer(al.f32),
    partial_sum_sq_ptr: al.Pointer(al.f32),
    N: al.i32,
    C: al.i32,
    G: al.i32,
    H: al.i32,
    W: al.i32,
):
    global_id = al.block_id(0)
    tid = al.thread_id(0)

    C_per_group = C // G
    H_W = H * W
    group_size = C_per_group * H_W
    total_elems = N * C * H_W
    chunk_size = group_size // al.convert(32, al.i32)

    num_partials = N * G * al.convert(32, al.i32)

    batch_idx = global_id // (G * al.convert(32, al.i32))
    group_idx = (global_id // al.convert(32, al.i32)) % G
    chunk_idx = global_id % al.convert(32, al.i32)

    chunk_start = chunk_idx * chunk_size

    layout_1d = al.make_layout((total_elems,), (1,))
    x = al.make_tensor(x_ptr, al.bf16, layout_1d)

    base = batch_idx * C * H_W + group_idx * C_per_group * H_W
    global_start = base + chunk_start

    # Accumulate sum and sum_sq over this chunk
    my_sum = al.convert(0.0, al.f32)
    my_sum_sq = al.convert(0.0, al.f32)

    for i in al.range(tid, chunk_size, 256):
        val_bf16 = x[global_start + i]
        val = al.convert(val_bf16, al.f32)
        my_sum = my_sum + val
        my_sum_sq = my_sum_sq + val * val

    # Warp-level butterfly reduction (256 threads = 4 warps of 64)
    my_sum = my_sum + al.shuffle_down(my_sum, 32, 64)
    my_sum = my_sum + al.shuffle_down(my_sum, 16, 64)
    my_sum = my_sum + al.shuffle_down(my_sum, 8, 64)
    my_sum = my_sum + al.shuffle_down(my_sum, 4, 64)
    my_sum = my_sum + al.shuffle_down(my_sum, 2, 64)
    my_sum = my_sum + al.shuffle_down(my_sum, 1, 64)

    my_sum_sq = my_sum_sq + al.shuffle_down(my_sum_sq, 32, 64)
    my_sum_sq = my_sum_sq + al.shuffle_down(my_sum_sq, 16, 64)
    my_sum_sq = my_sum_sq + al.shuffle_down(my_sum_sq, 8, 64)
    my_sum_sq = my_sum_sq + al.shuffle_down(my_sum_sq, 4, 64)
    my_sum_sq = my_sum_sq + al.shuffle_down(my_sum_sq, 2, 64)
    my_sum_sq = my_sum_sq + al.shuffle_down(my_sum_sq, 1, 64)

    # Cross-warp reduction via shared memory
    shared_sum = al.make_shared((4,), al.f32)
    shared_sum_sq = al.make_shared((4,), al.f32)

    warp_id = tid // al.convert(64, al.i32)
    lane_id = tid % al.convert(64, al.i32)

    if lane_id == al.convert(0, al.i32):
        shared_sum[warp_id] = my_sum
        shared_sum_sq[warp_id] = my_sum_sq

    al.syncthreads()

    if tid == al.convert(0, al.i32):
        block_sum = shared_sum[0] + shared_sum[1] + shared_sum[2] + shared_sum[3]
        block_sum_sq = shared_sum_sq[0] + shared_sum_sq[1] + shared_sum_sq[2] + shared_sum_sq[3]
        write_idx = batch_idx * G * al.convert(32, al.i32) + group_idx * al.convert(32, al.i32) + chunk_idx

        partial_layout = al.make_layout((num_partials,), (1,))
        ps = al.make_tensor(partial_sum_ptr, al.f32, partial_layout)
        pss = al.make_tensor(partial_sum_sq_ptr, al.f32, partial_layout)

        ps[write_idx] = block_sum
        pss[write_idx] = block_sum_sq


@avelang.jit
def group_norm_apply_kernel(
    x_ptr: al.Pointer(al.bf16),
    gamma_ptr: al.Pointer(al.bf16),
    beta_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    partial_sum_ptr: al.Pointer(al.f32),
    partial_sum_sq_ptr: al.Pointer(al.f32),
    N: al.i32,
    C: al.i32,
    G: al.i32,
    H: al.i32,
    W: al.i32,
    eps: al.f32,
):
    global_id = al.block_id(0)
    tid = al.thread_id(0)

    C_per_group = C // G
    H_W = H * W
    group_size = C_per_group * H_W
    total_elems = N * C * H_W
    chunk_size = group_size // al.convert(32, al.i32)

    num_partials = N * G * al.convert(32, al.i32)

    batch_idx = global_id // (G * al.convert(32, al.i32))
    group_idx = (global_id // al.convert(32, al.i32)) % G
    chunk_idx = global_id % al.convert(32, al.i32)

    chunk_start = chunk_idx * chunk_size

    # Create tensor views
    layout_1d = al.make_layout((total_elems,), (1,))
    x = al.make_tensor(x_ptr, al.bf16, layout_1d)
    out = al.make_tensor(out_ptr, al.bf16, layout_1d)

    param_layout = al.make_layout((C,), (1,))
    gamma = al.make_tensor(gamma_ptr, al.bf16, param_layout)
    beta = al.make_tensor(beta_ptr, al.bf16, param_layout)

    partial_layout = al.make_layout((num_partials,), (1,))
    ps = al.make_tensor(partial_sum_ptr, al.f32, partial_layout)
    pss = al.make_tensor(partial_sum_sq_ptr, al.f32, partial_layout)

    # Shared memory for mean/var broadcast
    shared_mean_var = al.make_shared((2,), al.f32)

    # Thread 0 aggregates all partial sums for this (batch, group)
    partial_base = batch_idx * G * al.convert(32, al.i32) + group_idx * al.convert(32, al.i32)

    if tid == al.convert(0, al.i32):
        total_sum = al.convert(0.0, al.f32)
        total_sum_sq = al.convert(0.0, al.f32)
        for p in al.range(0, al.convert(32, al.i32)):
            total_sum = total_sum + ps[partial_base + p]
            total_sum_sq = total_sum_sq + pss[partial_base + p]

        count = al.convert(group_size, al.f32)
        mean = total_sum / count
        var = total_sum_sq / count - mean * mean

        shared_mean_var[0] = mean
        shared_mean_var[1] = var

    al.syncthreads()

    mean = shared_mean_var[0]
    var = shared_mean_var[1]
    inv_std = al.convert(1.0, al.f32) / al.sqrt(var + eps)

    # Normalize this block's chunk
    base = batch_idx * C * H_W + group_idx * C_per_group * H_W
    global_start = base + chunk_start

    for i in al.range(tid, chunk_size, 256):
        val_bf16 = x[global_start + i]
        val = al.convert(val_bf16, al.f32)

        c_local = (chunk_start + i) // H_W
        c_global = group_idx * C_per_group + c_local

        g_val = al.convert(gamma[c_global], al.f32)
        b_val = al.convert(beta[c_global], al.f32)

        norm_val = (val - mean) * inv_std
        result = norm_val * g_val + b_val

        out[global_start + i] = al.convert(result, al.bf16)


class ModelNew(nn.Module):
    def __init__(self, num_features: int, num_groups: int):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.num_groups = num_groups
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gamma = self.gn.weight.data
        beta = self.gn.bias.data

        x = x.contiguous()
        N, C, H, W = x.shape
        G = self.num_groups

        x_bf16 = x.to(torch.bfloat16)
        gamma_bf16 = gamma.to(torch.bfloat16)
        beta_bf16 = beta.to(torch.bfloat16)

        num_partials = N * G * NUM_CHUNKS
        partial_sum = torch.empty(num_partials, dtype=torch.float32, device=x.device)
        partial_sum_sq = torch.empty(num_partials, dtype=torch.float32, device=x.device)

        # Kernel 1: compute partial sums per chunk
        partial_reduce_kernel[lambda: ((num_partials, 1, 1), (256, 1, 1))](
            x_bf16,
            partial_sum,
            partial_sum_sq,
            N,
            C,
            G,
            H,
            W,
        )

        out_bf16 = torch.empty_like(x_bf16)

        eps = 1e-5

        # Kernel 2: aggregate partials and apply normalization
        group_norm_apply_kernel[lambda: ((num_partials, 1, 1), (256, 1, 1))](
            x_bf16,
            gamma_bf16,
            beta_bf16,
            out_bf16,
            partial_sum,
            partial_sum_sq,
            N,
            C,
            G,
            H,
            W,
            eps,
        )

        return out_bf16
