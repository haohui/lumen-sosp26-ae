import torch
import torch.nn as nn
import substrate
import substrate.language as S

SQRT_2 = 1.4142135623730951

BATCH_SIZE = 1024
IN_FEATURES = 8192
OUT_FEATURES = 8192
POOL_KERNEL_SIZE = 16
POOLED_SIZE = OUT_FEATURES // POOL_KERNEL_SIZE
SCALE_FACTOR = 2.0

WARP_SIZE = 64
NUM_WARPS = 4
BLOCK_SIZE = WARP_SIZE * NUM_WARPS

BLOCKS = BATCH_SIZE

# Byte sizes for range parameter (in bytes)
X_RANGE = BATCH_SIZE * IN_FEATURES * 2  # bf16 = 2 bytes
W_RANGE = IN_FEATURES * OUT_FEATURES * 2
BIAS_RANGE = OUT_FEATURES * 2


def _launch():
    return ((BLOCKS, 1, 1), (BLOCK_SIZE, 1, 1))


@substrate.jit
def fused_kernel(
    X: S.Tensor((BATCH_SIZE, IN_FEATURES), S.bf16),
    W: S.Tensor((IN_FEATURES, OUT_FEATURES), S.bf16),
    BIAS0: S.Tensor((OUT_FEATURES,), S.bf16),
    Y: S.Tensor((BATCH_SIZE,), S.bf16),
):
    batch_idx = S.block_id(0)
    tid = S.thread_id(0)
    warp_id = tid // WARP_SIZE
    lane = tid % WARP_SIZE

    lds_max = S.make_shared((NUM_WARPS,), S.f32)

    pool_groups_per_warp = POOLED_SIZE // NUM_WARPS

    max_v = S.convert(-1e+30, S.f32)

    # Create resource descriptors with range for OOB handling
    rsrc_x = S.amdgpu.make_rsrc(X, X_RANGE)
    rsrc_w = S.amdgpu.make_rsrc(W, W_RANGE)
    rsrc_bias = S.amdgpu.make_rsrc(BIAS0, BIAS_RANGE)

    for pg in S.range(pool_groups_per_warp):
        pool_idx = warp_id * pool_groups_per_warp + pg
        j_start = pool_idx * POOL_KERNEL_SIZE

        total = S.convert(0.0, S.f32)

        for t in S.range(POOL_KERNEL_SIZE):
            j = j_start + t

            # Software pipelining: K-loop unrolled by 2 with prefetch
            k_per_thread = IN_FEATURES // WARP_SIZE

            acc = S.convert(0.0, S.f32)

            # Prefetch first 2 elements using raw_buffer_load_x4 with range
            k_idx_0 = lane * k_per_thread + 0
            k_idx_1 = lane * k_per_thread + 1

            # Load X values using raw_buffer_load_x4
            x_byte_offset = (batch_idx * IN_FEATURES + k_idx_0) * 2
            x_vec = S.amdgpu.raw_buffer_load_x4(rsrc_x, x_byte_offset, 0, 0)
            x_bf16_vec = S.view(x_vec, S.Tensor((8,), S.bf16))
            x_val_0 = S.convert(x_bf16_vec[0], S.f32)
            x_val_1 = S.convert(x_bf16_vec[1], S.f32)

            # Load W values using tensor indexing (strided access)
            w_val_0 = S.convert(W[k_idx_0, j], S.f32)
            w_val_1 = S.convert(W[k_idx_1, j], S.f32)

            # Process with software pipelining, K unrolled by 2
            for k_local in S.range(2, k_per_thread, 2):
                # Compute previous pair while prefetching next
                acc = acc + x_val_0 * w_val_0 + x_val_1 * w_val_1

                # Prefetch next pair - no branch needed, range handles OOB
                k_idx_cur = lane * k_per_thread + k_local
                k_idx_next = lane * k_per_thread + k_local + 1

                # Load X values using raw_buffer_load_x4 with range
                x_byte_offset = (batch_idx * IN_FEATURES + k_idx_cur) * 2
                x_vec = S.amdgpu.raw_buffer_load_x4(rsrc_x, x_byte_offset, 0, 0)
                x_bf16_vec = S.view(x_vec, S.Tensor((8,), S.bf16))
                x_val_0 = S.convert(x_bf16_vec[0], S.f32)
                x_val_1 = S.convert(x_bf16_vec[1], S.f32)

                # Load W values using tensor indexing
                w_val_0 = S.convert(W[k_idx_cur, j], S.f32)
                w_val_1 = S.convert(W[k_idx_next, j], S.f32)

            # Process last pair
            acc = acc + x_val_0 * w_val_0 + x_val_1 * w_val_1

            # Warp reduce
            acc = acc + S.shuffle_xor(acc, 1, WARP_SIZE)
            acc = acc + S.shuffle_xor(acc, 2, WARP_SIZE)
            acc = acc + S.shuffle_xor(acc, 4, WARP_SIZE)
            acc = acc + S.shuffle_xor(acc, 8, WARP_SIZE)
            acc = acc + S.shuffle_xor(acc, 16, WARP_SIZE)
            acc = acc + S.shuffle_xor(acc, 32, WARP_SIZE)

            if lane == 0:
                total += acc + S.convert(BIAS0[j], S.f32)

        if lane == 0:
            v = total / S.convert(POOL_KERNEL_SIZE, S.f32)
            v = S.convert(0.5, S.f32) * v * (S.convert(1.0, S.f32) + S.erf(v / S.convert(SQRT_2, S.f32)))
            v = v * S.convert(SCALE_FACTOR, S.f32)
            if v > max_v:
                max_v = v

    # Cross-warp reduction
    if lane == 0:
        lds_max[warp_id] = max_v
    S.syncthreads()

    if tid == 0:
        global_max = lds_max[0]
        if lds_max[1] > global_max:
            global_max = lds_max[1]
        if lds_max[2] > global_max:
            global_max = lds_max[2]
        if lds_max[3] > global_max:
            global_max = lds_max[3]
        Y[batch_idx] = S.convert(global_max, S.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.avg_pool = nn.AvgPool1d(kernel_size=pool_kernel_size)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = x.to(dtype=torch.bfloat16)
        w_t = self.matmul.weight.t().to(device=x.device, dtype=torch.bfloat16).contiguous()
        bias = self.matmul.bias.to(device=x.device, dtype=torch.bfloat16).contiguous()
        y = torch.empty((x.shape[0],), device=x.device, dtype=torch.bfloat16)
        fused_kernel[_launch](x.contiguous(), w_t, bias, y)
        return y
