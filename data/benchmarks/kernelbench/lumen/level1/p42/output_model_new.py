import torch
import substrate
import substrate.language as S

BLOCK_SIZE: S.constexpr = 256
VEC_SIZE: S.constexpr = 8
U32_PER_VEC: S.constexpr = 4

BATCH_SIZE = 32
CHANNELS = 64
IN_HEIGHT = 512
IN_WIDTH = 512
OUT_HEIGHT = 511
OUT_WIDTH = 511
KERNEL_SIZE = 4
STRIDE = 1
PADDING = 1
DILATION = 1

INTERIOR_H_START = 1
INTERIOR_H = 509
MAIN_W_START = 1
MAIN_W_ELEMS = 496
MAIN_W_VECS = MAIN_W_ELEMS // VEC_SIZE
HEAD_W_ELEMS = 1
TAIL_W_START = 497
TAIL_W_ELEMS = 14
SIDE_W_ELEMS = HEAD_W_ELEMS + TAIL_W_ELEMS


@substrate.jit
def max_pool2d_vec_kernel(
    x_ptr: S.Pointer(S.bf16),
    out_ptr: S.Pointer(S.bf16),
    input_range_bytes: S.u32,
    output_range_bytes: S.u32,
):
    slot = S.block_id(0) * BLOCK_SIZE + S.thread_id(0)
    row_idx = slot // MAIN_W_VECS
    vec_idx = slot - row_idx * MAIN_W_VECS

    rows_per_batch = CHANNELS * INTERIOR_H
    b = row_idx // rows_per_batch
    rem = row_idx - b * rows_per_batch
    c = rem // INTERIOR_H
    oh = INTERIOR_H_START + (rem - c * INTERIOR_H)
    ow_base = MAIN_W_START + vec_idx * VEC_SIZE

    x_flat = S.make_tensor(x_ptr, S.bf16, S.make_layout((BATCH_SIZE * CHANNELS * IN_HEIGHT * IN_WIDTH,), (1,)))
    out_flat = S.make_tensor(out_ptr, S.bf16, S.make_layout((BATCH_SIZE * CHANNELS * OUT_HEIGHT * OUT_WIDTH,), (1,)))
    x_rsrc = S.amdgpu.make_rsrc(x_flat, input_range_bytes)
    out_rsrc = S.amdgpu.make_rsrc(out_flat, output_range_bytes)

    zero_u32 = S.convert(0, S.u32)
    max_vals = S.make_local((VEC_SIZE,), S.f32)
    result = S.make_local((VEC_SIZE,), S.bf16)

    first_pos = (((b * CHANNELS + c) * IN_HEIGHT + (oh - PADDING)) * IN_WIDTH + (ow_base - PADDING))
    first_vals = S.view(
        S.amdgpu.raw_buffer_load_x4(x_rsrc, S.convert(first_pos * 2, S.u32), zero_u32, 0),
        S.Tensor((VEC_SIZE,), S.bf16),
    )
    for j in S.range(VEC_SIZE):
        max_vals[j] = S.convert(first_vals[j], S.f32)

    for kw in S.range(1, KERNEL_SIZE):
        pos = (((b * CHANNELS + c) * IN_HEIGHT + (oh - PADDING)) * IN_WIDTH + (ow_base - PADDING + kw))
        vals = S.view(
            S.amdgpu.raw_buffer_load_x4(x_rsrc, S.convert(pos * 2, S.u32), zero_u32, 0),
            S.Tensor((VEC_SIZE,), S.bf16),
        )
        for j in S.range(VEC_SIZE):
            val_f32 = S.convert(vals[j], S.f32)
            max_vals[j] = val_f32 if val_f32 > max_vals[j] else max_vals[j]

    for kh in S.range(1, KERNEL_SIZE):
        ih = oh - PADDING + kh
        for kw in S.range(KERNEL_SIZE):
            pos = (((b * CHANNELS + c) * IN_HEIGHT + ih) * IN_WIDTH + (ow_base - PADDING + kw))
            vals = S.view(
                S.amdgpu.raw_buffer_load_x4(x_rsrc, S.convert(pos * 2, S.u32), zero_u32, 0),
                S.Tensor((VEC_SIZE,), S.bf16),
            )
            for j in S.range(VEC_SIZE):
                val_f32 = S.convert(vals[j], S.f32)
                max_vals[j] = val_f32 if val_f32 > max_vals[j] else max_vals[j]

    for j in S.range(VEC_SIZE):
        result[j] = S.convert(max_vals[j], S.bf16)

    out_pos = (((b * CHANNELS + c) * OUT_HEIGHT + oh) * OUT_WIDTH + ow_base)
    packed_out = S.view(result, S.Tensor((U32_PER_VEC,), S.u32))
    S.amdgpu.raw_buffer_store_x4(packed_out, out_rsrc, S.convert(out_pos * 2, S.u32), zero_u32, 0)


@substrate.jit
def max_pool2d_topbottom_kernel(
    x_ptr: S.Pointer(S.bf16),
    out_ptr: S.Pointer(S.bf16),
):
    gid = S.block_id(0) * BLOCK_SIZE + S.thread_id(0)
    plane = gid // (2 * OUT_WIDTH)
    local = gid - plane * (2 * OUT_WIDTH)
    which_row = local // OUT_WIDTH
    ow = local - which_row * OUT_WIDTH

    b = plane // CHANNELS
    c = plane - b * CHANNELS
    oh = S.convert(0, S.i32) if which_row == 0 else S.convert(OUT_HEIGHT - 1, S.i32)

    x_flat = S.make_tensor(x_ptr, S.bf16, S.make_layout((BATCH_SIZE * CHANNELS * IN_HEIGHT * IN_WIDTH,), (1,)))
    out_flat = S.make_tensor(out_ptr, S.bf16, S.make_layout((BATCH_SIZE * CHANNELS * OUT_HEIGHT * OUT_WIDTH,), (1,)))

    max_val = S.convert(-65504.0, S.f32)
    ih_base = oh - PADDING
    iw_base = ow - PADDING
    for kh in S.range(KERNEL_SIZE):
        ih = ih_base + kh
        for kw in S.range(KERNEL_SIZE):
            iw = iw_base + kw
            if ih >= 0 and ih < IN_HEIGHT and iw >= 0 and iw < IN_WIDTH:
                pos = (((b * CHANNELS + c) * IN_HEIGHT + ih) * IN_WIDTH + iw)
                val = S.convert(x_flat[pos], S.f32)
                max_val = val if val > max_val else max_val

    out_pos = (((b * CHANNELS + c) * OUT_HEIGHT + oh) * OUT_WIDTH + ow)
    out_flat[out_pos] = S.convert(max_val, S.bf16)


@substrate.jit
def max_pool2d_sidecols_kernel(
    x_ptr: S.Pointer(S.bf16),
    out_ptr: S.Pointer(S.bf16),
):
    gid = S.block_id(0) * BLOCK_SIZE + S.thread_id(0)
    plane = gid // (INTERIOR_H * SIDE_W_ELEMS)
    local = gid - plane * (INTERIOR_H * SIDE_W_ELEMS)
    oh = INTERIOR_H_START + local // SIDE_W_ELEMS
    side_col = local - (local // SIDE_W_ELEMS) * SIDE_W_ELEMS

    b = plane // CHANNELS
    c = plane - b * CHANNELS
    ow = side_col if side_col < HEAD_W_ELEMS else S.convert(TAIL_W_START, S.i32) + (side_col - HEAD_W_ELEMS)

    x_flat = S.make_tensor(x_ptr, S.bf16, S.make_layout((BATCH_SIZE * CHANNELS * IN_HEIGHT * IN_WIDTH,), (1,)))
    out_flat = S.make_tensor(out_ptr, S.bf16, S.make_layout((BATCH_SIZE * CHANNELS * OUT_HEIGHT * OUT_WIDTH,), (1,)))

    max_val = S.convert(-65504.0, S.f32)
    ih_base = oh - PADDING
    iw_base = ow - PADDING
    for kh in S.range(KERNEL_SIZE):
        ih = ih_base + kh
        for kw in S.range(KERNEL_SIZE):
            iw = iw_base + kw
            if ih >= 0 and ih < IN_HEIGHT and iw >= 0 and iw < IN_WIDTH:
                pos = (((b * CHANNELS + c) * IN_HEIGHT + ih) * IN_WIDTH + iw)
                val = S.convert(x_flat[pos], S.f32)
                max_val = val if val > max_val else max_val

    out_pos = (((b * CHANNELS + c) * OUT_HEIGHT + oh) * OUT_WIDTH + ow)
    out_flat[out_pos] = S.convert(max_val, S.bf16)


def substrate_max_pool2d(
    x: torch.Tensor,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA device"
    assert x.shape == (BATCH_SIZE, CHANNELS, IN_HEIGHT, IN_WIDTH), f"Expected shape ({BATCH_SIZE}, {CHANNELS}, {IN_HEIGHT}, {IN_WIDTH}), got {x.shape}"

    x = x.contiguous()
    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)

    out = torch.empty((BATCH_SIZE, CHANNELS, OUT_HEIGHT, OUT_WIDTH), dtype=torch.bfloat16, device=x.device)
    x_flat = x.view(-1)
    out_flat = out.view(-1)

    vec_blocks = (BATCH_SIZE * CHANNELS * INTERIOR_H * MAIN_W_VECS) // BLOCK_SIZE
    max_pool2d_vec_kernel[lambda: ((vec_blocks, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_flat,
        out_flat,
        x_flat.numel() * x_flat.element_size(),
        out_flat.numel() * out_flat.element_size(),
    )

    topbottom_blocks = (BATCH_SIZE * CHANNELS * 2 * OUT_WIDTH) // BLOCK_SIZE
    max_pool2d_topbottom_kernel[lambda: ((topbottom_blocks, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_flat,
        out_flat,
    )

    side_blocks = (BATCH_SIZE * CHANNELS * INTERIOR_H * SIDE_W_ELEMS) // BLOCK_SIZE
    max_pool2d_sidecols_kernel[lambda: ((side_blocks, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_flat,
        out_flat,
    )

    return out


class ModelNew(torch.nn.Module):
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if kernel_size != KERNEL_SIZE or stride != STRIDE or padding != PADDING or dilation != DILATION:
            raise NotImplementedError(
                f"This optimized kernel only supports kernel_size={KERNEL_SIZE}, stride={STRIDE}, "
                f"padding={PADDING}, dilation={DILATION}. "
                f"Got kernel_size={kernel_size}, stride={stride}, padding={padding}, dilation={dilation}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            x = x.cuda()
        return substrate_max_pool2d(x, self.kernel_size, self.stride, self.padding, self.dilation)
