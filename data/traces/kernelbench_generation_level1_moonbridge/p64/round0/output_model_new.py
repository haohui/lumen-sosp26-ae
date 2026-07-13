import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def conv_transpose1d_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    l_in: al.i32,
    l_out: al.i32,
):
    tid = al.thread_id(0)
    block_l = al.block_id(0)
    block_oc = al.block_id(1)
    batch_group = al.block_id(2)

    l_start = block_l * 64
    oc_start = block_oc * 64

    lid = tid % 64
    gid = tid // 64

    # Shared memory for input tile: 128 channels x 66 positions
    shm_input = al.make_shared((128, 66), al.bf16)

    # Build global memory views
    x_memref = al.make_tensor(x_ptr, al.bf16, al.make_layout((64 * 128 * l_in,), (1,)))
    w_memref = al.make_tensor(w_ptr, al.bf16, al.make_layout((128 * 128 * 3,), (1,)))
    out_memref = al.make_tensor(out_ptr, al.bf16, al.make_layout((64 * 128 * l_out,), (1,)))

    in_batch_stride = 128 * l_in
    out_batch_stride = 128 * l_out

    # Accumulators: 16 FP32 values per thread (gid 0..3, each handles 16 positions)
    acc = al.make_local((16,), al.f32)

    for bi in al.range(4):
        batch_idx = batch_group * 4 + bi

        # Zero-init shared memory
        for idx in al.range(tid, 128 * 66, 256):
            ic = idx // 66
            po = idx - ic * 66
            shm_input[ic, po] = al.convert(0.0, al.bf16)
        al.syncthreads()

        # Load input tile into shared memory
        batch_base = batch_idx * in_batch_stride
        for idx in al.range(tid, 128 * 66, 256):
            ic = idx // 66
            po = idx - ic * 66
            pos = l_start + po - 2
            if pos >= 0:
                if pos < l_in:
                    goff = batch_base + ic * l_in + pos
                    shm_input[ic, po] = x_memref[goff]
        al.syncthreads()

        # Init accumulators
        for e in al.range(16):
            acc[e] = al.convert(0.0, al.f32)

        # Accumulate over input channels and kernel positions
        l_off_base = gid * 16
        for ic in al.range(128):
            w_base = ic * 384 + (oc_start + lid) * 3
            w0 = al.convert(w_memref[w_base + 0], al.f32)
            w1 = al.convert(w_memref[w_base + 1], al.f32)
            w2 = al.convert(w_memref[w_base + 2], al.f32)

            p0 = l_off_base + 2
            p1 = l_off_base + 1
            p2 = l_off_base + 0

            i00 = al.convert(shm_input[ic, p0 + 0], al.f32)
            i01 = al.convert(shm_input[ic, p1 + 0], al.f32)
            i02 = al.convert(shm_input[ic, p2 + 0], al.f32)
            acc[0] = acc[0] + i00 * w0 + i01 * w1 + i02 * w2

            i10 = al.convert(shm_input[ic, p0 + 1], al.f32)
            i11 = al.convert(shm_input[ic, p1 + 1], al.f32)
            i12 = al.convert(shm_input[ic, p2 + 1], al.f32)
            acc[1] = acc[1] + i10 * w0 + i11 * w1 + i12 * w2

            i20 = al.convert(shm_input[ic, p0 + 2], al.f32)
            i21 = al.convert(shm_input[ic, p1 + 2], al.f32)
            i22 = al.convert(shm_input[ic, p2 + 2], al.f32)
            acc[2] = acc[2] + i20 * w0 + i21 * w1 + i22 * w2

            i30 = al.convert(shm_input[ic, p0 + 3], al.f32)
            i31 = al.convert(shm_input[ic, p1 + 3], al.f32)
            i32 = al.convert(shm_input[ic, p2 + 3], al.f32)
            acc[3] = acc[3] + i30 * w0 + i31 * w1 + i32 * w2

            i40 = al.convert(shm_input[ic, p0 + 4], al.f32)
            i41 = al.convert(shm_input[ic, p1 + 4], al.f32)
            i42 = al.convert(shm_input[ic, p2 + 4], al.f32)
            acc[4] = acc[4] + i40 * w0 + i41 * w1 + i42 * w2

            i50 = al.convert(shm_input[ic, p0 + 5], al.f32)
            i51 = al.convert(shm_input[ic, p1 + 5], al.f32)
            i52 = al.convert(shm_input[ic, p2 + 5], al.f32)
            acc[5] = acc[5] + i50 * w0 + i51 * w1 + i52 * w2

            i60 = al.convert(shm_input[ic, p0 + 6], al.f32)
            i61 = al.convert(shm_input[ic, p1 + 6], al.f32)
            i62 = al.convert(shm_input[ic, p2 + 6], al.f32)
            acc[6] = acc[6] + i60 * w0 + i61 * w1 + i62 * w2

            i70 = al.convert(shm_input[ic, p0 + 7], al.f32)
            i71 = al.convert(shm_input[ic, p1 + 7], al.f32)
            i72 = al.convert(shm_input[ic, p2 + 7], al.f32)
            acc[7] = acc[7] + i70 * w0 + i71 * w1 + i72 * w2

            i80 = al.convert(shm_input[ic, p0 + 8], al.f32)
            i81 = al.convert(shm_input[ic, p1 + 8], al.f32)
            i82 = al.convert(shm_input[ic, p2 + 8], al.f32)
            acc[8] = acc[8] + i80 * w0 + i81 * w1 + i82 * w2

            i90 = al.convert(shm_input[ic, p0 + 9], al.f32)
            i91 = al.convert(shm_input[ic, p1 + 9], al.f32)
            i92 = al.convert(shm_input[ic, p2 + 9], al.f32)
            acc[9] = acc[9] + i90 * w0 + i91 * w1 + i92 * w2

            ia0 = al.convert(shm_input[ic, p0 + 10], al.f32)
            ia1 = al.convert(shm_input[ic, p1 + 10], al.f32)
            ia2 = al.convert(shm_input[ic, p2 + 10], al.f32)
            acc[10] = acc[10] + ia0 * w0 + ia1 * w1 + ia2 * w2

            ib0 = al.convert(shm_input[ic, p0 + 11], al.f32)
            ib1 = al.convert(shm_input[ic, p1 + 11], al.f32)
            ib2 = al.convert(shm_input[ic, p2 + 11], al.f32)
            acc[11] = acc[11] + ib0 * w0 + ib1 * w1 + ib2 * w2

            ic0 = al.convert(shm_input[ic, p0 + 12], al.f32)
            ic1 = al.convert(shm_input[ic, p1 + 12], al.f32)
            ic2 = al.convert(shm_input[ic, p2 + 12], al.f32)
            acc[12] = acc[12] + ic0 * w0 + ic1 * w1 + ic2 * w2

            id0 = al.convert(shm_input[ic, p0 + 13], al.f32)
            id1 = al.convert(shm_input[ic, p1 + 13], al.f32)
            id2 = al.convert(shm_input[ic, p2 + 13], al.f32)
            acc[13] = acc[13] + id0 * w0 + id1 * w1 + id2 * w2

            ie0 = al.convert(shm_input[ic, p0 + 14], al.f32)
            ie1 = al.convert(shm_input[ic, p1 + 14], al.f32)
            ie2 = al.convert(shm_input[ic, p2 + 14], al.f32)
            acc[14] = acc[14] + ie0 * w0 + ie1 * w1 + ie2 * w2

            if0 = al.convert(shm_input[ic, p0 + 15], al.f32)
            if1 = al.convert(shm_input[ic, p1 + 15], al.f32)
            if2 = al.convert(shm_input[ic, p2 + 15], al.f32)
            acc[15] = acc[15] + if0 * w0 + if1 * w1 + if2 * w2

        al.syncthreads()

        # Write back results
        out_base = batch_idx * out_batch_stride
        l0 = l_start + gid * 16
        for e in al.range(16):
            l_idx = l0 + e
            if l_idx < l_out:
                oc_idx = oc_start + lid
                out_idx = out_base + oc_idx * l_out + l_idx
                out_memref[out_idx] = al.convert(acc[e], al.bf16)

        al.syncthreads()


def _to_bf16_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_conv_transpose1d(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = _to_bf16_contiguous(x)
    w_bf16 = _to_bf16_contiguous(weight)

    batch_size, c_in, l_in = x_bf16.shape
    w_c_in, c_out, kernel = w_bf16.shape

    l_out = l_in + kernel - 1

    out = torch.empty((batch_size, c_out, l_out), device=x_bf16.device, dtype=torch.bfloat16)

    l_blocks = (l_out + 63) // 64
    oc_blocks = (c_out + 63) // 64
    batch_groups = batch_size // 4
    grid = (l_blocks, oc_blocks, batch_groups)

    conv_transpose1d_kernel[lambda: (grid, (256, 1, 1))](
        x_bf16, w_bf16, out, l_in, l_out
    )
    return out


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.conv1d_transpose = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding,
            groups=groups, bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.conv1d_transpose.weight
        return avelang_conv_transpose1d(x, w)


def get_inputs():
    x = torch.rand(64, 128, 65536)
    return [x]


def get_init_inputs():
    return [128, 128, 3]
