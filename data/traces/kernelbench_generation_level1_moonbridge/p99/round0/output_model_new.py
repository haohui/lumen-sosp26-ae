import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE: al.constexpr = 256
_MARGIN = 1.0


@avelang.jit
def triplet_reduce_kernel(
    anchor_ptr: al.Pointer(al.bf16),
    positive_ptr: al.Pointer(al.bf16),
    negative_ptr: al.Pointer(al.bf16),
    dist_ap_ptr: al.Pointer(al.f32),
    dist_an_ptr: al.Pointer(al.f32),
    batch_size: al.i32,
    feat_dim: al.i32,
):
    tid = al.thread_id(0)
    sample_idx = al.block_id(0)

    if sample_idx < batch_size:
        smem_ap = al.make_shared((BLOCK_SIZE,), al.f32)
        smem_an = al.make_shared((BLOCK_SIZE,), al.f32)

        layout_in = al.make_layout((batch_size, feat_dim), (feat_dim, 1))
        anchor = al.make_tensor(anchor_ptr, al.bf16, layout_in)
        positive = al.make_tensor(positive_ptr, al.bf16, layout_in)
        negative = al.make_tensor(negative_ptr, al.bf16, layout_in)

        local_ap = al.convert(0.0, al.f32)
        local_an = al.convert(0.0, al.f32)

        for i in al.range(tid, feat_dim, BLOCK_SIZE):
            a_val = al.convert(anchor[sample_idx, i], al.f32)
            p_val = al.convert(positive[sample_idx, i], al.f32)
            n_val = al.convert(negative[sample_idx, i], al.f32)
            diff_ap = a_val - p_val
            diff_an = a_val - n_val
            local_ap = local_ap + diff_ap * diff_ap
            local_an = local_an + diff_an * diff_an

        smem_ap[tid] = local_ap
        smem_an[tid] = local_an
        al.syncthreads()

        if tid < 128:
            smem_ap[tid] = smem_ap[tid] + smem_ap[tid + 128]
            smem_an[tid] = smem_an[tid] + smem_an[tid + 128]
        al.syncthreads()
        if tid < 64:
            smem_ap[tid] = smem_ap[tid] + smem_ap[tid + 64]
            smem_an[tid] = smem_an[tid] + smem_an[tid + 64]
        al.syncthreads()
        if tid < 32:
            smem_ap[tid] = smem_ap[tid] + smem_ap[tid + 32]
            smem_an[tid] = smem_an[tid] + smem_an[tid + 32]
        al.syncthreads()
        if tid < 16:
            smem_ap[tid] = smem_ap[tid] + smem_ap[tid + 16]
            smem_an[tid] = smem_an[tid] + smem_an[tid + 16]
        al.syncthreads()
        if tid < 8:
            smem_ap[tid] = smem_ap[tid] + smem_ap[tid + 8]
            smem_an[tid] = smem_an[tid] + smem_an[tid + 8]
        al.syncthreads()
        if tid < 4:
            smem_ap[tid] = smem_ap[tid] + smem_ap[tid + 4]
            smem_an[tid] = smem_an[tid] + smem_an[tid + 4]
        al.syncthreads()
        if tid < 2:
            smem_ap[tid] = smem_ap[tid] + smem_ap[tid + 2]
            smem_an[tid] = smem_an[tid] + smem_an[tid + 2]
        al.syncthreads()
        if tid < 1:
            smem_ap[tid] = smem_ap[tid] + smem_ap[tid + 1]
            smem_an[tid] = smem_an[tid] + smem_an[tid + 1]

        if tid == 0:
            layout_out = al.make_layout((batch_size,), (1,))
            dap = al.make_tensor(dist_ap_ptr, al.f32, layout_out)
            dan = al.make_tensor(dist_an_ptr, al.f32, layout_out)
            dap[sample_idx] = smem_ap[0]
            dan[sample_idx] = smem_an[0]


@avelang.jit
def triplet_loss_kernel(
    dist_ap_ptr: al.Pointer(al.f32),
    dist_an_ptr: al.Pointer(al.f32),
    output_ptr: al.Pointer(al.f32),
    batch_size: al.i32,
):
    tid = al.thread_id(0)

    margin_val = al.convert(_MARGIN, al.f32)

    layout_in = al.make_layout((batch_size,), (1,))
    dap_sq = al.make_tensor(dist_ap_ptr, al.f32, layout_in)
    dan_sq = al.make_tensor(dist_an_ptr, al.f32, layout_in)

    local_sum = al.convert(0.0, al.f32)

    for i in al.range(tid, batch_size, BLOCK_SIZE):
        d_ap = al.sqrt(dap_sq[i])
        d_an = al.sqrt(dan_sq[i])
        loss = d_ap - d_an + margin_val
        zero = al.convert(0.0, al.f32)
        if loss < zero:
            loss = zero
        local_sum = local_sum + loss

    smem = al.make_shared((BLOCK_SIZE,), al.f32)
    smem[tid] = local_sum
    al.syncthreads()

    if tid < 128:
        smem[tid] = smem[tid] + smem[tid + 128]
    al.syncthreads()
    if tid < 64:
        smem[tid] = smem[tid] + smem[tid + 64]
    al.syncthreads()
    if tid < 32:
        smem[tid] = smem[tid] + smem[tid + 32]
    al.syncthreads()
    if tid < 16:
        smem[tid] = smem[tid] + smem[tid + 16]
    al.syncthreads()
    if tid < 8:
        smem[tid] = smem[tid] + smem[tid + 8]
    al.syncthreads()
    if tid < 4:
        smem[tid] = smem[tid] + smem[tid + 4]
    al.syncthreads()
    if tid < 2:
        smem[tid] = smem[tid] + smem[tid + 2]
    al.syncthreads()
    if tid < 1:
        smem[tid] = smem[tid] + smem[tid + 1]

    if tid == 0:
        batch_f32 = al.convert(batch_size, al.f32)
        result = smem[0] / batch_f32
        layout_out = al.make_layout((1,), (1,))
        out = al.make_tensor(output_ptr, al.f32, layout_out)
        out[0] = result


batch_size = 32768
input_shape = (8192,)
dim = 1


def get_inputs():
    scale = torch.rand(())
    return [
        torch.rand(batch_size, *input_shape) * scale,
        torch.rand(batch_size, *input_shape),
        torch.rand(batch_size, *input_shape),
    ]


def get_init_inputs():
    return [1.0]


def avelang_triplet_margin_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    global _MARGIN
    _MARGIN = margin

    assert anchor.is_cuda and positive.is_cuda and negative.is_cuda, (
        "Tensors must be on CUDA/HIP device."
    )
    assert anchor.dtype == positive.dtype == negative.dtype, (
        "All input tensors must have the same dtype."
    )

    b = anchor.shape[0]
    fdim = anchor.shape[1]

    anchor_bf16 = anchor.contiguous().to(torch.bfloat16)
    positive_bf16 = positive.contiguous().to(torch.bfloat16)
    negative_bf16 = negative.contiguous().to(torch.bfloat16)

    dist_ap = torch.empty((b,), dtype=torch.float32, device=anchor.device)
    dist_an = torch.empty((b,), dtype=torch.float32, device=anchor.device)

    triplet_reduce_kernel[lambda: ((b, 1, 1), (BLOCK_SIZE, 1, 1))](
        anchor_bf16,
        positive_bf16,
        negative_bf16,
        dist_ap,
        dist_an,
        b,
        fdim,
    )

    output = torch.empty((1,), dtype=torch.float32, device=anchor.device)

    triplet_loss_kernel[lambda: ((1, 1, 1), (BLOCK_SIZE, 1, 1))](
        dist_ap,
        dist_an,
        output,
        b,
    )

    return output.squeeze().to(anchor.dtype)


class ModelNew(nn.Module):
    """
    Optimized model that computes Triplet Margin Loss using AveLang DSL.
    """

    def __init__(self, margin: float = 1.0):
        super(ModelNew, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        return avelang_triplet_margin_loss(
            anchor, positive, negative, self.margin
        )
