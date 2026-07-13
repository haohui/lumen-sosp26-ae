import torch
import torch.nn as nn
import avelang
import avelang.language as al

IC = 64
KS = 3
K_DIM = IC * KS  # 192

POS_TILE = 32
OC_TILE = 128
POS_TILES_PER_BLOCK = 8
THREADS = 256
ELEMS_PER_THREAD = (POS_TILE * OC_TILE) // THREADS  # 16
SHM_W_SIZE = OC_TILE * K_DIM   # 24576
SHM_X_SIZE = POS_TILE * K_DIM  # 6144
OUT_PER_BLOCK = POS_TILE * POS_TILES_PER_BLOCK  # 256


@avelang.jit
def conv1d_bf16_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.u32,
    OC: al.u32,
    L: al.u32,
    stride: al.u32,
    dilation: al.u32,
    OUT_LEN: al.u32,
):
    tid = al.thread_id(0)
    pos_chunk = al.block_id(0)
    batch = al.block_id(1)

    if batch >= B:
        return

    pos_start_base = pos_chunk * OUT_PER_BLOCK

    x_mem = al.make_tensor(x_ptr, al.bf16, al.make_layout((B * IC * L,), (1,)))
    w_mem = al.make_tensor(w_ptr, al.bf16, al.make_layout((OC * K_DIM,), (1,)))
    out_mem = al.make_tensor(out_ptr, al.bf16, al.make_layout((B * OC * OUT_LEN,), (1,)))

    # Load all weights into shared memory once
    shm_w = al.make_shared((SHM_W_SIZE,), al.bf16)
    for idx in al.range(tid, SHM_W_SIZE, THREADS):
        shm_w[idx] = w_mem[idx]
    al.syncthreads()

    shm_x = al.make_shared((SHM_X_SIZE,), al.bf16)

    # Process tile 0
    pos0 = pos_start_base
    if pos0 < OUT_LEN:
        ap0 = OUT_LEN - pos0
        if ap0 > POS_TILE:
            ap0 = POS_TILE
        for idx in al.range(tid, SHM_X_SIZE, THREADS):
            pl = idx // K_DIM
            k = idx % K_DIM
            ic_ch = k // KS
            kw = k % KS
            if pl < ap0:
                shm_x[idx] = x_mem[batch * (IC * L) + ic_ch * L + (pos0 + pl) * stride + kw * dilation]
        al.syncthreads()
        for elem in al.range(ELEMS_PER_THREAD):
            gid = tid + elem * THREADS
            if gid < OC_TILE * ap0:
                oc_local = gid // ap0
                p_local = gid % ap0
                acc = al.convert(0.0, al.f32)
                for k in al.range(K_DIM):
                    w_val = al.convert(shm_w[oc_local * K_DIM + k], al.f32)
                    x_val = al.convert(shm_x[p_local * K_DIM + k], al.f32)
                    acc = acc + w_val * x_val
                out_mem[batch * (OC * OUT_LEN) + oc_local * OUT_LEN + (pos0 + p_local)] = al.convert(acc, al.bf16)
        al.syncthreads()

    # Process tile 1
    pos1 = pos_start_base + POS_TILE
    if pos1 < OUT_LEN:
        ap1 = OUT_LEN - pos1
        if ap1 > POS_TILE:
            ap1 = POS_TILE
        for idx in al.range(tid, SHM_X_SIZE, THREADS):
            pl = idx // K_DIM
            k = idx % K_DIM
            ic_ch = k // KS
            kw = k % KS
            if pl < ap1:
                shm_x[idx] = x_mem[batch * (IC * L) + ic_ch * L + (pos1 + pl) * stride + kw * dilation]
        al.syncthreads()
        for elem in al.range(ELEMS_PER_THREAD):
            gid = tid + elem * THREADS
            if gid < OC_TILE * ap1:
                oc_local = gid // ap1
                p_local = gid % ap1
                acc = al.convert(0.0, al.f32)
                for k in al.range(K_DIM):
                    w_val = al.convert(shm_w[oc_local * K_DIM + k], al.f32)
                    x_val = al.convert(shm_x[p_local * K_DIM + k], al.f32)
                    acc = acc + w_val * x_val
                out_mem[batch * (OC * OUT_LEN) + oc_local * OUT_LEN + (pos1 + p_local)] = al.convert(acc, al.bf16)
        al.syncthreads()

    # Process tile 2
    pos2 = pos_start_base + 2 * POS_TILE
    if pos2 < OUT_LEN:
        ap2 = OUT_LEN - pos2
        if ap2 > POS_TILE:
            ap2 = POS_TILE
        for idx in al.range(tid, SHM_X_SIZE, THREADS):
            pl = idx // K_DIM
            k = idx % K_DIM
            ic_ch = k // KS
            kw = k % KS
            if pl < ap2:
                shm_x[idx] = x_mem[batch * (IC * L) + ic_ch * L + (pos2 + pl) * stride + kw * dilation]
        al.syncthreads()
        for elem in al.range(ELEMS_PER_THREAD):
            gid = tid + elem * THREADS
            if gid < OC_TILE * ap2:
                oc_local = gid // ap2
                p_local = gid % ap2
                acc = al.convert(0.0, al.f32)
                for k in al.range(K_DIM):
                    w_val = al.convert(shm_w[oc_local * K_DIM + k], al.f32)
                    x_val = al.convert(shm_x[p_local * K_DIM + k], al.f32)
                    acc = acc + w_val * x_val
                out_mem[batch * (OC * OUT_LEN) + oc_local * OUT_LEN + (pos2 + p_local)] = al.convert(acc, al.bf16)
        al.syncthreads()

    # Process tile 3
    pos3 = pos_start_base + 3 * POS_TILE
    if pos3 < OUT_LEN:
        ap3 = OUT_LEN - pos3
        if ap3 > POS_TILE:
            ap3 = POS_TILE
        for idx in al.range(tid, SHM_X_SIZE, THREADS):
            pl = idx // K_DIM
            k = idx % K_DIM
            ic_ch = k // KS
            kw = k % KS
            if pl < ap3:
                shm_x[idx] = x_mem[batch * (IC * L) + ic_ch * L + (pos3 + pl) * stride + kw * dilation]
        al.syncthreads()
        for elem in al.range(ELEMS_PER_THREAD):
            gid = tid + elem * THREADS
            if gid < OC_TILE * ap3:
                oc_local = gid // ap3
                p_local = gid % ap3
                acc = al.convert(0.0, al.f32)
                for k in al.range(K_DIM):
                    w_val = al.convert(shm_w[oc_local * K_DIM + k], al.f32)
                    x_val = al.convert(shm_x[p_local * K_DIM + k], al.f32)
                    acc = acc + w_val * x_val
                out_mem[batch * (OC * OUT_LEN) + oc_local * OUT_LEN + (pos3 + p_local)] = al.convert(acc, al.bf16)
        al.syncthreads()

    # Process tile 4
    pos4 = pos_start_base + 4 * POS_TILE
    if pos4 < OUT_LEN:
        ap4 = OUT_LEN - pos4
        if ap4 > POS_TILE:
            ap4 = POS_TILE
        for idx in al.range(tid, SHM_X_SIZE, THREADS):
            pl = idx // K_DIM
            k = idx % K_DIM
            ic_ch = k // KS
            kw = k % KS
            if pl < ap4:
                shm_x[idx] = x_mem[batch * (IC * L) + ic_ch * L + (pos4 + pl) * stride + kw * dilation]
        al.syncthreads()
        for elem in al.range(ELEMS_PER_THREAD):
            gid = tid + elem * THREADS
            if gid < OC_TILE * ap4:
                oc_local = gid // ap4
                p_local = gid % ap4
                acc = al.convert(0.0, al.f32)
                for k in al.range(K_DIM):
                    w_val = al.convert(shm_w[oc_local * K_DIM + k], al.f32)
                    x_val = al.convert(shm_x[p_local * K_DIM + k], al.f32)
                    acc = acc + w_val * x_val
                out_mem[batch * (OC * OUT_LEN) + oc_local * OUT_LEN + (pos4 + p_local)] = al.convert(acc, al.bf16)
        al.syncthreads()

    # Process tile 5
    pos5 = pos_start_base + 5 * POS_TILE
    if pos5 < OUT_LEN:
        ap5 = OUT_LEN - pos5
        if ap5 > POS_TILE:
            ap5 = POS_TILE
        for idx in al.range(tid, SHM_X_SIZE, THREADS):
            pl = idx // K_DIM
            k = idx % K_DIM
            ic_ch = k // KS
            kw = k % KS
            if pl < ap5:
                shm_x[idx] = x_mem[batch * (IC * L) + ic_ch * L + (pos5 + pl) * stride + kw * dilation]
        al.syncthreads()
        for elem in al.range(ELEMS_PER_THREAD):
            gid = tid + elem * THREADS
            if gid < OC_TILE * ap5:
                oc_local = gid // ap5
                p_local = gid % ap5
                acc = al.convert(0.0, al.f32)
                for k in al.range(K_DIM):
                    w_val = al.convert(shm_w[oc_local * K_DIM + k], al.f32)
                    x_val = al.convert(shm_x[p_local * K_DIM + k], al.f32)
                    acc = acc + w_val * x_val
                out_mem[batch * (OC * OUT_LEN) + oc_local * OUT_LEN + (pos5 + p_local)] = al.convert(acc, al.bf16)
        al.syncthreads()

    # Process tile 6
    pos6 = pos_start_base + 6 * POS_TILE
    if pos6 < OUT_LEN:
        ap6 = OUT_LEN - pos6
        if ap6 > POS_TILE:
            ap6 = POS_TILE
        for idx in al.range(tid, SHM_X_SIZE, THREADS):
            pl = idx // K_DIM
            k = idx % K_DIM
            ic_ch = k // KS
            kw = k % KS
            if pl < ap6:
                shm_x[idx] = x_mem[batch * (IC * L) + ic_ch * L + (pos6 + pl) * stride + kw * dilation]
        al.syncthreads()
        for elem in al.range(ELEMS_PER_THREAD):
            gid = tid + elem * THREADS
            if gid < OC_TILE * ap6:
                oc_local = gid // ap6
                p_local = gid % ap6
                acc = al.convert(0.0, al.f32)
                for k in al.range(K_DIM):
                    w_val = al.convert(shm_w[oc_local * K_DIM + k], al.f32)
                    x_val = al.convert(shm_x[p_local * K_DIM + k], al.f32)
                    acc = acc + w_val * x_val
                out_mem[batch * (OC * OUT_LEN) + oc_local * OUT_LEN + (pos6 + p_local)] = al.convert(acc, al.bf16)
        al.syncthreads()

    # Process tile 7
    pos7 = pos_start_base + 7 * POS_TILE
    if pos7 < OUT_LEN:
        ap7 = OUT_LEN - pos7
        if ap7 > POS_TILE:
            ap7 = POS_TILE
        for idx in al.range(tid, SHM_X_SIZE, THREADS):
            pl = idx // K_DIM
            k = idx % K_DIM
            ic_ch = k // KS
            kw = k % KS
            if pl < ap7:
                shm_x[idx] = x_mem[batch * (IC * L) + ic_ch * L + (pos7 + pl) * stride + kw * dilation]
        al.syncthreads()
        for elem in al.range(ELEMS_PER_THREAD):
            gid = tid + elem * THREADS
            if gid < OC_TILE * ap7:
                oc_local = gid // ap7
                p_local = gid % ap7
                acc = al.convert(0.0, al.f32)
                for k in al.range(K_DIM):
                    w_val = al.convert(shm_w[oc_local * K_DIM + k], al.f32)
                    x_val = al.convert(shm_x[p_local * K_DIM + k], al.f32)
                    acc = acc + w_val * x_val
                out_mem[batch * (OC * OUT_LEN) + oc_local * OUT_LEN + (pos7 + p_local)] = al.convert(acc, al.bf16)
        al.syncthreads()


def avelang_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    stride: int,
    dilation: int,
) -> torch.Tensor:
    assert x.is_cuda, "Input must be on CUDA/HIP device."
    input_dtype = x.dtype
    x_bf16 = x.contiguous().to(dtype=torch.bfloat16)
    w_bf16 = weight.to(device=x_bf16.device, dtype=torch.bfloat16).contiguous()
    B_val, IC_val, L_val = x_bf16.shape
    OC_val, w_IC, KS_val = w_bf16.shape

    OUT_LEN_val = (L_val - dilation * (KS_val - 1) - 1) // stride + 1
    out_bf16 = torch.empty((B_val, OC_val, OUT_LEN_val), device=x_bf16.device, dtype=torch.bfloat16)

    num_pos_chunks = (OUT_LEN_val + OUT_PER_BLOCK - 1) // OUT_PER_BLOCK

    conv1d_bf16_kernel[lambda: ((num_pos_chunks, B_val, 1), (THREADS, 1, 1))](
        x_bf16, w_bf16, out_bf16,
        B_val, OC_val, L_val, stride, dilation, OUT_LEN_val,
    )

    return out_bf16.to(dtype=input_dtype)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.dilation = dilation
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if bias:
            raise NotImplementedError("Bias is not supported in the AveLang kernel.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_conv1d(x, self.weight, self.stride, self.dilation)


batch_size = 64
in_channels = 64
out_channels = 128
kernel_size = 3
length = 524280
stride = 3
dilation = 4


def get_inputs():
    x = torch.rand(batch_size, in_channels, length)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, dilation]
