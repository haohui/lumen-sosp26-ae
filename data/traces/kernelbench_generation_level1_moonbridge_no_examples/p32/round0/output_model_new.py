import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def hardtanh_kernel(
    in_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    N: al.i32,
):
    tid = al.block_id(0) * al.block_dim(0) + al.thread_id(0)
    stride = al.block_dim(0) * al.grid_dim(0)

    in_layout = al.make_layout((N,), (1,))
    inp = al.make_tensor(in_ptr, al.bf16, in_layout)
    out_layout = al.make_layout((N,), (1,))
    out = al.make_tensor(out_ptr, al.bf16, out_layout)

    in_rsrc = al.amdgpu.make_rsrc(inp, N * 2)
    out_rsrc = al.amdgpu.make_rsrc(out, N * 2)

    neg_one = al.convert(-1.0, al.f32)
    one = al.convert(1.0, al.f32)

    vec_count = N // 8

    for i in al.range(tid, vec_count, stride):
        vindex = i * 16
        data = al.amdgpu.raw_buffer_load_x4(in_rsrc, vindex, 0, 0)
        vec = al.view(data, al.Tensor((8,), al.bf16))

        v0 = al.convert(vec[0], al.f32)
        if v0 < neg_one:
            v0 = neg_one
        elif v0 > one:
            v0 = one
        vec[0] = al.convert(v0, al.bf16)

        v1 = al.convert(vec[1], al.f32)
        if v1 < neg_one:
            v1 = neg_one
        elif v1 > one:
            v1 = one
        vec[1] = al.convert(v1, al.bf16)

        v2 = al.convert(vec[2], al.f32)
        if v2 < neg_one:
            v2 = neg_one
        elif v2 > one:
            v2 = one
        vec[2] = al.convert(v2, al.bf16)

        v3 = al.convert(vec[3], al.f32)
        if v3 < neg_one:
            v3 = neg_one
        elif v3 > one:
            v3 = one
        vec[3] = al.convert(v3, al.bf16)

        v4 = al.convert(vec[4], al.f32)
        if v4 < neg_one:
            v4 = neg_one
        elif v4 > one:
            v4 = one
        vec[4] = al.convert(v4, al.bf16)

        v5 = al.convert(vec[5], al.f32)
        if v5 < neg_one:
            v5 = neg_one
        elif v5 > one:
            v5 = one
        vec[5] = al.convert(v5, al.bf16)

        v6 = al.convert(vec[6], al.f32)
        if v6 < neg_one:
            v6 = neg_one
        elif v6 > one:
            v6 = one
        vec[6] = al.convert(v6, al.bf16)

        v7 = al.convert(vec[7], al.f32)
        if v7 < neg_one:
            v7 = neg_one
        elif v7 > one:
            v7 = one
        vec[7] = al.convert(v7, al.bf16)

        data_out = al.view(vec, al.Tensor((4,), al.i32))
        al.amdgpu.raw_buffer_store_x4(data_out, out_rsrc, vindex, 0, 0)


def avelang_hardtanh(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Input must be on CUDA/HIP device."

    x_contig = x.contiguous()
    N = x_contig.numel()

    out = torch.empty_like(x_contig)

    BLOCK_SIZE = 256
    num_sms = 304
    grid_size = num_sms * 16

    hardtanh_kernel[lambda: ((grid_size, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_contig, out, N
    )
    return out


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_hardtanh(x)
