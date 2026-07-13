import torch
import torch.nn as nn
import avelang
import avelang.language as al


VEC = 8


@avelang.jit
def scale_kernel(
    a_ptr: al.Pointer(al.bf16),
    c_ptr: al.Pointer(al.bf16),
    s: al.f32,
    M: al.i32,
    N: al.i32,
):
    """Vectorized matrix-scalar multiply: C = A * s.

    Each thread loads 8 bf16 elements via raw_buffer_load_x4,
    multiplies by the scalar in registers, and stores via
    raw_buffer_store_x4.  The 1D grid-stride loop covers the
    full (M x N) tensor.
    """
    s_bf16 = al.convert(s, al.bf16)
    one = al.convert(1, al.i32)
    two = al.convert(2, al.i32)
    zero = al.convert(0, al.i32)

    tid = al.block_id(0) * al.block_dim(0) + al.thread_id(0)
    stride = al.block_dim(0) * al.grid_dim(0) * VEC
    total = M * N

    # 1D tensor views for flat indexing
    layout_1d = al.make_layout((total,), (one,))
    a_1d = al.make_tensor(a_ptr, al.bf16, layout_1d)
    c_1d = al.make_tensor(c_ptr, al.bf16, layout_1d)

    rsrc_a = al.amdgpu.make_rsrc(a_1d, al.convert(total * 2, al.i32))
    rsrc_c = al.amdgpu.make_rsrc(c_1d, al.convert(total * 2, al.i32))

    # Grid-stride loop: each iteration handles 8 bf16 elements
    for base in al.range(tid * VEC, total, stride):
        byte_off = base * two

        # Vectorized load: 16 bytes = 8 bf16 values
        data = al.amdgpu.raw_buffer_load_x4(rsrc_a, byte_off, 0, 0)
        bf16_data = al.view(data, al.Tensor((VEC,), al.bf16))

        # Scalar multiply by s in registers
        r0 = bf16_data[0] * s_bf16
        r1 = bf16_data[1] * s_bf16
        r2 = bf16_data[2] * s_bf16
        r3 = bf16_data[3] * s_bf16
        r4 = bf16_data[4] * s_bf16
        r5 = bf16_data[5] * s_bf16
        r6 = bf16_data[6] * s_bf16
        r7 = bf16_data[7] * s_bf16

        # Pack 8 bf16 → 4 u32 for vectorized store
        result_bf16 = al.make_local((VEC,), al.bf16)
        result_bf16[0] = r0
        result_bf16[1] = r1
        result_bf16[2] = r2
        result_bf16[3] = r3
        result_bf16[4] = r4
        result_bf16[5] = r5
        result_bf16[6] = r6
        result_bf16[7] = r7

        packed = al.view(result_bf16, al.Tensor((4,), al.u32))
        al.amdgpu.raw_buffer_store_x4(packed, rsrc_c, byte_off, 0, 0)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, s):
        if A.dtype != torch.bfloat16:
            A = A.to(torch.bfloat16)
        A = A.contiguous()
        M_val, N_val = A.shape

        BLOCK_SIZE = 256
        total_elems = M_val * N_val
        grid = (total_elems + BLOCK_SIZE * VEC - 1) // (BLOCK_SIZE * VEC)

        C = torch.empty_like(A)
        s_val = float(s)

        scale_kernel[lambda: ((grid, 1, 1), (BLOCK_SIZE, 1, 1))](
            A.data_ptr(),
            C.data_ptr(),
            s_val,
            M_val,
            N_val,
        )

        return C
