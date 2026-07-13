import struct
import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def triplet_loss_kernel(
    anchor_ptr: al.Pointer(al.bf16),
    positive_ptr: al.Pointer(al.bf16),
    negative_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    dim: al.i32,
    batch_size: al.i32,
    margin_bits: al.i32,
):
    sample_id = al.block_id(0)
    tid = al.thread_id(0)
    bdim = al.block_dim(0)

    row_layout = al.make_layout((dim,), (1,))
    anchor = al.make_tensor(anchor_ptr, al.bf16, row_layout)
    positive = al.make_tensor(positive_ptr, al.bf16, row_layout)
    negative = al.make_tensor(negative_ptr, al.bf16, row_layout)

    base_idx = sample_id * dim

    d_ap = al.convert(0.0, al.f32)
    d_an = al.convert(0.0, al.f32)

    for j in al.range(tid, dim, bdim):
        a = al.convert(anchor[base_idx + j], al.f32)
        p = al.convert(positive[base_idx + j], al.f32)
        n = al.convert(negative[base_idx + j], al.f32)
        diff_ap = a - p
        diff_an = a - n
        d_ap = d_ap + diff_ap * diff_ap
        d_an = d_an + diff_an * diff_an

    s_ap = al.make_shared((256,), al.f32)
    s_an = al.make_shared((256,), al.f32)
    s_ap[tid] = d_ap
    s_an[tid] = d_an
    al.syncthreads()

    if tid < 128:
        s_ap[tid] = s_ap[tid] + s_ap[tid + 128]
        s_an[tid] = s_an[tid] + s_an[tid + 128]
    al.syncthreads()
    if tid < 64:
        s_ap[tid] = s_ap[tid] + s_ap[tid + 64]
        s_an[tid] = s_an[tid] + s_an[tid + 64]
    al.syncthreads()
    if tid < 32:
        s_ap[tid] = s_ap[tid] + s_ap[tid + 32]
        s_an[tid] = s_an[tid] + s_an[tid + 32]
    al.syncthreads()
    if tid < 16:
        s_ap[tid] = s_ap[tid] + s_ap[tid + 16]
        s_an[tid] = s_an[tid] + s_an[tid + 16]
    al.syncthreads()
    if tid < 8:
        s_ap[tid] = s_ap[tid] + s_ap[tid + 8]
        s_an[tid] = s_an[tid] + s_an[tid + 8]
    al.syncthreads()
    if tid < 4:
        s_ap[tid] = s_ap[tid] + s_ap[tid + 4]
        s_an[tid] = s_an[tid] + s_an[tid + 4]
    al.syncthreads()
    if tid < 2:
        s_ap[tid] = s_ap[tid] + s_ap[tid + 2]
        s_an[tid] = s_an[tid] + s_an[tid + 2]
    al.syncthreads()
    if tid < 1:
        s_ap[tid] = s_ap[tid] + s_ap[tid + 1]
        s_an[tid] = s_an[tid] + s_an[tid + 1]
    al.syncthreads()

    if tid == 0:
        margin_val = al.bitcast(margin_bits, al.f32)
        eps = al.convert(1e-6, al.f32)
        d_ap_norm = al.sqrt(s_ap[0] + eps)
        d_an_norm = al.sqrt(s_an[0] + eps)
        loss = d_ap_norm - d_an_norm + margin_val
        zero = al.convert(0.0, al.f32)
        if loss < zero:
            loss = zero
        out_layout = al.make_layout((batch_size,), (1,))
        out = al.make_tensor(out_ptr, al.bf16, out_layout)
        out[sample_id] = al.convert(loss, al.bf16)


@avelang.jit
def mean_reduce_kernel(
    in_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    batch_size: al.i32,
):
    tid = al.thread_id(0)
    bdim = al.block_dim(0)

    in_layout = al.make_layout((batch_size,), (1,))
    inp = al.make_tensor(in_ptr, al.bf16, in_layout)

    acc = al.convert(0.0, al.f32)
    for i in al.range(tid, batch_size, bdim):
        acc = acc + al.convert(inp[i], al.f32)

    s = al.make_shared((256,), al.f32)
    s[tid] = acc
    al.syncthreads()

    if tid < 128:
        s[tid] = s[tid] + s[tid + 128]
    al.syncthreads()
    if tid < 64:
        s[tid] = s[tid] + s[tid + 64]
    al.syncthreads()
    if tid < 32:
        s[tid] = s[tid] + s[tid + 32]
    al.syncthreads()
    if tid < 16:
        s[tid] = s[tid] + s[tid + 16]
    al.syncthreads()
    if tid < 8:
        s[tid] = s[tid] + s[tid + 8]
    al.syncthreads()
    if tid < 4:
        s[tid] = s[tid] + s[tid + 4]
    al.syncthreads()
    if tid < 2:
        s[tid] = s[tid] + s[tid + 2]
    al.syncthreads()
    if tid < 1:
        s[tid] = s[tid] + s[tid + 1]
    al.syncthreads()

    if tid == 0:
        mean = s[0] / al.convert(batch_size, al.f32)
        out_layout = al.make_layout((1,), (1,))
        out = al.make_tensor(out_ptr, al.bf16, out_layout)
        out[0] = al.convert(mean, al.bf16)


def avelang_triplet_margin_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    batch_size, dim = anchor.shape

    anchor_bf16 = anchor.to(torch.bfloat16).contiguous()
    positive_bf16 = positive.to(torch.bfloat16).contiguous()
    negative_bf16 = negative.to(torch.bfloat16).contiguous()

    margin_bits = struct.unpack('<i', struct.pack('<f', float(margin)))[0]

    losses = torch.empty(batch_size, dtype=torch.bfloat16, device=anchor.device)

    grid = (batch_size, 1, 1)
    block = (256, 1, 1)
    triplet_loss_kernel[lambda: (grid, block)](
        anchor_bf16.data_ptr(),
        positive_bf16.data_ptr(),
        negative_bf16.data_ptr(),
        losses.data_ptr(),
        dim,
        batch_size,
        margin_bits,
    )

    result = torch.empty(1, dtype=torch.bfloat16, device=anchor.device)
    mean_reduce_kernel[lambda: ((1, 1, 1), (256, 1, 1))](
        losses.data_ptr(),
        result.data_ptr(),
        batch_size,
    )

    return result.reshape(())


class ModelNew(nn.Module):
    def __init__(self, margin=1.0):
        super(ModelNew, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        return avelang_triplet_margin_loss(anchor, positive, negative, self.margin)
