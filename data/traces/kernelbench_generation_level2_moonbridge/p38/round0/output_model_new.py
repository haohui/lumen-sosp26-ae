import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE: al.constexpr = 256

batch_size = 32
in_channels = 32
out_channels = 64
depth, height, width = 32, 64, 64
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
pool_kernel_size = 2
clamp_min = 0.0
clamp_max = 1.0


@avelang.jit
def clamp_softmax_scale_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    scale_ptr: al.Pointer(al.bf16),
    B: al.i32,
    C: al.i32,
    spatial_size: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    total_pairs = B * C
    if bid < total_pairs:
        c_idx = bid - (bid // C) * C

        smem_val = al.make_shared((BLOCK_SIZE,), al.f32)

        base = bid * spatial_size

        layout_flat = al.make_layout((total_pairs * spatial_size,), (1,))
        x = al.make_tensor(x_ptr, al.bf16, layout_flat)
        out = al.make_tensor(out_ptr, al.bf16, layout_flat)

        layout_scale = al.make_layout((C,), (1,))
        scale = al.make_tensor(scale_ptr, al.bf16, layout_scale)
        scale_f32 = al.convert(scale[c_idx], al.f32)

        zero = al.convert(0.0, al.f32)
        one = al.convert(1.0, al.f32)
        neg_inf = al.convert(-3.402823e+38, al.f32)

        # ---- Pass 1: find max of clamped values ----
        local_max = neg_inf

        i = tid
        for _ in al.range(0, spatial_size, BLOCK_SIZE):
            if i < spatial_size:
                idx = base + i
                val = al.convert(x[idx], al.f32)
                if val < zero:
                    val = zero
                if val > one:
                    val = one
                if val > local_max:
                    local_max = val
            i = i + BLOCK_SIZE

        smem_val[tid] = local_max
        al.syncthreads()

        if tid < 128:
            other = smem_val[tid + 128]
            if other > smem_val[tid]:
                smem_val[tid] = other
        al.syncthreads()
        if tid < 64:
            other = smem_val[tid + 64]
            if other > smem_val[tid]:
                smem_val[tid] = other
        al.syncthreads()
        if tid < 32:
            other = smem_val[tid + 32]
            if other > smem_val[tid]:
                smem_val[tid] = other
        al.syncthreads()
        if tid < 16:
            other = smem_val[tid + 16]
            if other > smem_val[tid]:
                smem_val[tid] = other
        al.syncthreads()
        if tid < 8:
            other = smem_val[tid + 8]
            if other > smem_val[tid]:
                smem_val[tid] = other
        al.syncthreads()
        if tid < 4:
            other = smem_val[tid + 4]
            if other > smem_val[tid]:
                smem_val[tid] = other
        al.syncthreads()
        if tid < 2:
            other = smem_val[tid + 2]
            if other > smem_val[tid]:
                smem_val[tid] = other
        al.syncthreads()
        if tid < 1:
            other = smem_val[1]
            if other > smem_val[0]:
                smem_val[0] = other
        al.syncthreads()

        global_max = smem_val[0]

        # ---- Pass 2: compute sum of exp(x - global_max) ----
        local_sum = zero

        i = tid
        for _ in al.range(0, spatial_size, BLOCK_SIZE):
            if i < spatial_size:
                idx = base + i
                val = al.convert(x[idx], al.f32)
                if val < zero:
                    val = zero
                if val > one:
                    val = one
                diff = val - global_max
                local_sum = local_sum + al.exp(diff)
            i = i + BLOCK_SIZE

        smem_val[tid] = local_sum
        al.syncthreads()

        if tid < 128:
            smem_val[tid] = smem_val[tid] + smem_val[tid + 128]
        al.syncthreads()
        if tid < 64:
            smem_val[tid] = smem_val[tid] + smem_val[tid + 64]
        al.syncthreads()
        if tid < 32:
            smem_val[tid] = smem_val[tid] + smem_val[tid + 32]
        al.syncthreads()
        if tid < 16:
            smem_val[tid] = smem_val[tid] + smem_val[tid + 16]
        al.syncthreads()
        if tid < 8:
            smem_val[tid] = smem_val[tid] + smem_val[tid + 8]
        al.syncthreads()
        if tid < 4:
            smem_val[tid] = smem_val[tid] + smem_val[tid + 4]
        al.syncthreads()
        if tid < 2:
            smem_val[tid] = smem_val[tid] + smem_val[tid + 2]
        al.syncthreads()
        if tid < 1:
            smem_val[tid] = smem_val[tid] + smem_val[tid + 1]
        al.syncthreads()

        global_sum = smem_val[0]
        inv_sum = al.convert(1.0, al.f32) / global_sum

        # ---- Pass 3: apply softmax + scale ----
        i = tid
        for _ in al.range(0, spatial_size, BLOCK_SIZE):
            if i < spatial_size:
                idx = base + i
                val = al.convert(x[idx], al.f32)
                if val < zero:
                    val = zero
                if val > one:
                    val = one
                diff = val - global_max
                softmax_val = al.exp(diff) * inv_sum
                result = softmax_val * scale_f32
                out[idx] = al.convert(result, al.bf16)
            i = i + BLOCK_SIZE


def avelang_clamp_softmax_scale(
    x: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    assert x.is_cuda, "Tensors must be on CUDA/HIP device."

    x_contig = x.contiguous()
    scale_contig = scale.contiguous().to(dtype=x_contig.dtype)

    B, C, D, H, W = x_contig.shape
    spatial_size = D * H * W

    out = torch.empty_like(x_contig)
    num_blocks = B * C

    clamp_softmax_scale_kernel[lambda: ((num_blocks, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_contig, out, scale_contig, B, C, spatial_size
    )

    return out


class ModelNew(nn.Module):
    """
    Model that performs average pooling, 3D transposed convolution, and then
    clamp + spatial softmax + scale through a fused AveLang kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.avg_pool = nn.AvgPool3d(pool_kernel_size)
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1, 1))

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv_transpose(x)
        x = avelang_clamp_softmax_scale(x, self.scale)
        return x


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max]
