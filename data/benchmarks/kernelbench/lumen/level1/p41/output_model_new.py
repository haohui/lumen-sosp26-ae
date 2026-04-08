import torch
import substrate
import substrate.language as S

BLOCK_SIZE: S.constexpr = 256
VEC_SIZE: S.constexpr = 8
U32_PER_VEC: S.constexpr = 4

BATCH_SIZE = 64
FEATURES = 192
SEQ_LEN = 65536
KERNEL_SIZE = 8
STRIDE = 1
PADDING = 4
DILATION = 3
OUT_SEQ_LEN = 65523
ROWS = BATCH_SIZE * FEATURES

MAIN_START = 4
MAIN_ELEMS = 65504
MAIN_VECS = MAIN_ELEMS // VEC_SIZE
TAIL_START = MAIN_START + MAIN_ELEMS
HEAD_ELEMS = MAIN_START
TAIL_ELEMS = OUT_SEQ_LEN - TAIL_START


@substrate.jit
def max_pool1d_vec_kernel(
    x_ptr: S.Pointer(S.bf16),
    out_ptr: S.Pointer(S.bf16),
    input_range_bytes: S.u32,
    output_range_bytes: S.u32,
):
    slot = S.block_id(0) * BLOCK_SIZE + S.thread_id(0)
    row = slot // MAIN_VECS
    vec_idx = slot - row * MAIN_VECS
    ow_base = MAIN_START + vec_idx * VEC_SIZE

    x_flat = S.make_tensor(x_ptr, S.bf16, S.make_layout((ROWS * SEQ_LEN,), (1,)))
    out_flat = S.make_tensor(out_ptr, S.bf16, S.make_layout((ROWS * OUT_SEQ_LEN,), (1,)))
    x_rsrc = S.amdgpu.make_rsrc(x_flat, input_range_bytes)
    out_rsrc = S.amdgpu.make_rsrc(out_flat, output_range_bytes)

    row_in_base = row * SEQ_LEN
    row_out_base = row * OUT_SEQ_LEN
    zero_u32 = S.convert(0, S.u32)
    max_vals = S.make_local((VEC_SIZE,), S.f32)
    result = S.make_local((VEC_SIZE,), S.bf16)

    first_byte = S.convert((row_in_base + ow_base - PADDING) * 2, S.u32)
    first_vals = S.view(
        S.amdgpu.raw_buffer_load_x4(x_rsrc, first_byte, zero_u32, 0),
        S.Tensor((VEC_SIZE,), S.bf16),
    )
    for j in S.range(VEC_SIZE):
        max_vals[j] = S.convert(first_vals[j], S.f32)

    for k in S.range(1, KERNEL_SIZE):
        byte_offset = S.convert((row_in_base + ow_base - PADDING + k * DILATION) * 2, S.u32)
        vals = S.view(
            S.amdgpu.raw_buffer_load_x4(x_rsrc, byte_offset, zero_u32, 0),
            S.Tensor((VEC_SIZE,), S.bf16),
        )
        for j in S.range(VEC_SIZE):
            val_f32 = S.convert(vals[j], S.f32)
            max_vals[j] = val_f32 if val_f32 > max_vals[j] else max_vals[j]

    for j in S.range(VEC_SIZE):
        result[j] = S.convert(max_vals[j], S.bf16)

    out_byte = S.convert((row_out_base + ow_base) * 2, S.u32)
    packed_out = S.view(result, S.Tensor((U32_PER_VEC,), S.u32))
    S.amdgpu.raw_buffer_store_x4(packed_out, out_rsrc, out_byte, zero_u32, 0)


@substrate.jit
def max_pool1d_scalar_range_kernel(
    x_ptr: S.Pointer(S.bf16),
    out_ptr: S.Pointer(S.bf16),
    ow_start: S.i32,
    outputs_per_row: S.i32,
):
    gid = S.block_id(0) * BLOCK_SIZE + S.thread_id(0)
    row = gid // outputs_per_row
    local_ow = gid - row * outputs_per_row
    ow = ow_start + local_ow

    x_flat = S.make_tensor(x_ptr, S.bf16, S.make_layout((ROWS * SEQ_LEN,), (1,)))
    out_flat = S.make_tensor(out_ptr, S.bf16, S.make_layout((ROWS * OUT_SEQ_LEN,), (1,)))

    row_in_base = row * SEQ_LEN
    row_out_base = row * OUT_SEQ_LEN
    max_val = S.convert(-65504.0, S.f32)

    for k in S.range(KERNEL_SIZE):
        input_pos = ow - PADDING + k * DILATION
        if input_pos >= 0 and input_pos < SEQ_LEN:
            val = S.convert(x_flat[row_in_base + input_pos], S.f32)
            max_val = val if val > max_val else max_val

    out_flat[row_out_base + ow] = S.convert(max_val, S.bf16)


def substrate_max_pool1d(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA device"
    assert x.shape == (BATCH_SIZE, FEATURES, SEQ_LEN), f"Expected shape ({BATCH_SIZE}, {FEATURES}, {SEQ_LEN}), got {x.shape}"

    x = x.contiguous()
    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)

    out = torch.empty((BATCH_SIZE, FEATURES, OUT_SEQ_LEN), dtype=torch.bfloat16, device=x.device)
    x_flat = x.view(-1)
    out_flat = out.view(-1)

    vec_blocks = (ROWS * MAIN_VECS) // BLOCK_SIZE
    max_pool1d_vec_kernel[lambda: ((vec_blocks, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_flat,
        out_flat,
        x_flat.numel() * x_flat.element_size(),
        out_flat.numel() * out_flat.element_size(),
    )

    head_blocks = (ROWS * HEAD_ELEMS) // BLOCK_SIZE
    max_pool1d_scalar_range_kernel[lambda: ((head_blocks, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_flat,
        out_flat,
        0,
        HEAD_ELEMS,
    )

    tail_blocks = (ROWS * TAIL_ELEMS) // BLOCK_SIZE
    max_pool1d_scalar_range_kernel[lambda: ((tail_blocks, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_flat,
        out_flat,
        TAIL_START,
        TAIL_ELEMS,
    )

    return out


class ModelNew(torch.nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices

        if kernel_size != KERNEL_SIZE or self.stride != STRIDE or padding != PADDING or dilation != DILATION:
            raise NotImplementedError(
                f"This optimized kernel only supports kernel_size={KERNEL_SIZE}, stride={STRIDE}, "
                f"padding={PADDING}, dilation={DILATION}. "
                f"Got kernel_size={kernel_size}, stride={self.stride}, padding={padding}, dilation={dilation}"
            )
        if return_indices:
            raise NotImplementedError("This kernel does not support return_indices=True")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            x = x.cuda()
        return substrate_max_pool1d(x)
