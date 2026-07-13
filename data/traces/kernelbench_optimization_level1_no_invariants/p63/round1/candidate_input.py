import torch
import torch.nn as nn
import avelang
import avelang.language as al

_C_H_W = 16777216
_H_W = 1048576
_OC_OH_OW = 133693696
_OH_OW = 1044484
_C_KH_KW = 144
_KH_KW = 9


@avelang.jit
def conv2d_mfma_kernel(
    X_ptr: al.Pointer(al.bf16),
    W_ptr: al.Pointer(al.bf16),
    Y_ptr: al.Pointer(al.f32),
):
    x_layout = al.make_layout(
        (16, 16, 1024, 1024),
        (_C_H_W, _H_W, 1024, 1),
    )
    X = al.make_tensor(X_ptr, al.bf16, x_layout)

    w_layout = al.make_layout(
        (128, 16, 3, 3),
        (_C_KH_KW, _KH_KW, 3, 1),
    )
    Wt = al.make_tensor(W_ptr, al.bf16, w_layout)

    y_layout = al.make_layout(
        (16, 128, 1022, 1022),
        (_OC_OH_OW, _OH_OW, 1022, 1),
    )
    Y = al.make_tensor(Y_ptr, al.f32, y_layout)

    m_block = al.block_id(0)
    n_block = al.block_id(1)

    m_start = m_block * 32
    n_start = n_block * 32

    tid = al.thread_id(0)

    a_local = al.make_local((4,), al.bf16)
    b_local = al.make_local((4,), al.bf16)
    c_local = al.make_local((16,), al.f32)

    for i in al.range(16):
        c_local[i] = al.convert(0.0, al.f32)

    for k_step in al.range(18):
        k_start = k_step * 8

        # Load A fragment: 32x8 bf16, 256 elements, 4 per thread
        for mi in al.range(32):
            m_idx = m_start + mi
            n_idx = m_idx // _OH_OW
            spat = m_idx % _OH_OW
            h_out = spat // 1022
            w_out = spat % 1022

            for ki in al.range(8):
                flat_a = mi * 8 + ki
                k_idx = k_start + ki
                ic = k_idx // 9
                ksp = k_idx % 9
                kh = ksp // 3
                kw = ksp % 3
                if flat_a % 64 == tid:
                    a_local[flat_a // 64] = X[n_idx, ic, h_out + kh, w_out + kw]

        # Load B fragment: 8x32 bf16, 256 elements, 4 per thread
        for ki in al.range(8):
            k_idx = k_start + ki
            ic = k_idx // 9
            ksp = k_idx % 9
            kh = ksp // 3
            kw = ksp % 3

            for ni in al.range(32):
                flat_b = ki * 32 + ni
                oc = n_start + ni
                if flat_b % 64 == tid:
                    b_local[flat_b // 64] = Wt[oc, ic, kh, kw]

        a_vec = al.view(a_local, al.Tensor((2,), al.u32))
        b_vec = al.view(b_local, al.Tensor((2,), al.u32))
        c_vec = al.view(c_local, al.Tensor((16,), al.f32))

        d_vec = al.amdgpu.mfma_32x32x8_bf16_f32(a_vec, b_vec, c_vec)
        c_local = al.view(d_vec, al.Tensor((16,), al.f32))

    # Store C fragment: 32x32 f32, 1024 elements, 16 per thread
    for mi in al.range(32):
        m_idx = m_start + mi
        n_idx = m_idx // _OH_OW
        spat = m_idx % _OH_OW
        h_out = spat // 1022
        w_out = spat % 1022

        for ni in al.range(32):
            flat_c = mi * 32 + ni
            oc = n_start + ni
            if flat_c % 64 == tid:
                Y[n_idx, oc, h_out, w_out] = c_local[flat_c // 64]


def _launch():
    return ((522242, 4, 1), (64, 1, 1))


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            (kernel_size, kernel_size),
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if tuple(x.shape) != (16, 16, 1024, 1024):
            raise RuntimeError(
                "This fused kernel only supports input shape (16, 16, 1024, 1024)."
            )

        x_bf16 = x.to(dtype=torch.bfloat16).contiguous()
        w_bf16 = self.conv2d.weight.to(
            device=x.device, dtype=torch.bfloat16
        ).contiguous()

        y = torch.empty(
            (16, 128, 1022, 1022), device=x.device, dtype=torch.float32
        )

        conv2d_mfma_kernel[_launch](x_bf16, w_bf16, y)

        return y.to(dtype=x.dtype) if x.dtype != torch.float32 else y
