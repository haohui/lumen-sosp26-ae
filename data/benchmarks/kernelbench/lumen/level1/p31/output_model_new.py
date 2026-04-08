import torch
import substrate
import substrate.language as S

BLOCK_SIZE: S.constexpr = 256
VEC_SIZE: S.constexpr = 8
U32_PER_VEC: S.constexpr = 4


@substrate.jit
def elu_kernel(
    x_ptr: S.Pointer(S.bf16),
    out_ptr: S.Pointer(S.bf16),
    n_vectors: S.i32,
    range_bytes: S.u32,
    alpha: S.constexpr,
):
    idx = S.block_id(0) * BLOCK_SIZE + S.thread_id(0)
    bf16_layout = S.make_layout((n_vectors, VEC_SIZE), (VEC_SIZE, 1))
    u32_layout = S.make_layout((n_vectors, U32_PER_VEC), (U32_PER_VEC, 1))
    x_bf16 = S.make_tensor(x_ptr, S.bf16, bf16_layout)
    out_bf16 = S.make_tensor(out_ptr, S.bf16, bf16_layout)
    x_u32 = S.view(x_bf16, S.u32, u32_layout)
    out_u32 = S.view(out_bf16, S.u32, u32_layout)
    x_rsrc = S.amdgpu.make_rsrc(x_u32, range_bytes)
    out_rsrc = S.amdgpu.make_rsrc(out_u32, range_bytes)
    voffset = S.convert(idx * VEC_SIZE * 2, S.u32)
    zero_u32 = S.convert(0, S.u32)

    packed = S.amdgpu.raw_buffer_load_x4(x_rsrc, voffset, zero_u32, 0)
    val = S.view(packed, S.Tensor((VEC_SIZE,), S.bf16))
    result = S.make_local((VEC_SIZE,), S.bf16)
    zero = S.convert(0.0, S.bf16)
    one_f32 = S.convert(1.0, S.f32)
    alpha_f32 = S.convert(alpha, S.f32)

    for i in S.range(VEC_SIZE):
        lane = val[i]
        lane_f32 = S.convert(lane, S.f32)
        neg_val = S.convert(alpha_f32 * (S.exp(lane_f32) - one_f32), S.bf16)
        result[i] = lane if lane >= zero else neg_val

    packed_out = S.view(result, S.Tensor((U32_PER_VEC,), S.u32))
    S.amdgpu.raw_buffer_store_x4(packed_out, out_rsrc, voffset, zero_u32, 0)


@substrate.jit
def elu_scalar_kernel(
    x_ptr: S.Pointer(S.bf16),
    out_ptr: S.Pointer(S.bf16),
    n_elements: S.i32,
    alpha: S.constexpr,
):
    idx = S.thread_id(0)
    layout = S.make_layout((n_elements,), (1,))
    x = S.make_tensor(x_ptr, S.bf16, layout)
    out = S.make_tensor(out_ptr, S.bf16, layout)
    zero = S.convert(0.0, S.bf16)
    one_f32 = S.convert(1.0, S.f32)
    alpha_f32 = S.convert(alpha, S.f32)
    lane = x[idx]
    lane_f32 = S.convert(lane, S.f32)
    neg_val = S.convert(alpha_f32 * (S.exp(lane_f32) - one_f32), S.bf16)
    out[idx] = lane if lane >= zero else neg_val


def elu_substrate(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    assert x.is_cuda, "Tensors must be on CUDA/HIP device."

    orig_dtype = x.dtype
    x_contig = x.contiguous().to(torch.bfloat16)
    out = torch.empty_like(x_contig)
    x_flat = x_contig.view(-1)
    out_flat = out.view(-1)
    n = x_flat.numel()

    if n == 0:
        return out.to(orig_dtype)

    n_vectors = n // VEC_SIZE
    scalar_tail = n - n_vectors * VEC_SIZE
    offset = n_vectors * VEC_SIZE

    if n_vectors:
        vector_elems = n_vectors * VEC_SIZE
        grid_size = (n_vectors + BLOCK_SIZE - 1) // BLOCK_SIZE
        elu_kernel[lambda: ((grid_size, 1, 1), (BLOCK_SIZE, 1, 1))](
            x_flat.narrow(0, 0, vector_elems),
            out_flat.narrow(0, 0, vector_elems),
            n_vectors,
            vector_elems * 2,
            alpha,
        )

    if scalar_tail:
        elu_scalar_kernel[lambda: ((1, 1, 1), (scalar_tail, 1, 1))](
            x_flat.narrow(0, offset, scalar_tail),
            out_flat.narrow(0, offset, scalar_tail),
            scalar_tail,
            alpha,
        )

    return out.to(orig_dtype)


class ModelNew(torch.nn.Module):
    def __init__(self, alpha: float = 1.0):
        super(ModelNew, self).__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return elu_substrate(x, alpha=self.alpha)
