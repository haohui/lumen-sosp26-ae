import torch
import torch.nn as nn
import torch.nn.functional as F
import avelang
import avelang.language as al

BLOCK_M = 32
BLOCK_N = 32
BLOCK_K = 16


@avelang.jit
def depthwise_conv_kernel(
    A_ptr: al.Pointer(al.bf16),
    B_ptr: al.Pointer(al.bf16),
    C_ptr: al.Pointer(al.f32),
    C_dim: al.u32,
    M_dim: al.u32,
    K_dim: al.u32,
):
    A_bf16 = al.make_tensor(A_ptr, al.bf16, al.make_layout((C_dim, M_dim, K_dim), (M_dim * K_dim, K_dim, 1)))
    B_bf16 = al.make_tensor(B_ptr, al.bf16, al.make_layout((C_dim, 32, K_dim), (32 * K_dim, K_dim, 1)))
    C = al.make_tensor(C_ptr, al.f32, al.make_layout((C_dim, M_dim, 32), (M_dim * 32, 32, 1)))

    channel = al.block_id(2)

    k_vecs = K_dim >> 3
    packed_row_stride = K_dim >> 1

    A_vec = al.view(A_bf16, al.i32, al.make_layout((C_dim, M_dim, k_vecs, 4), (M_dim * k_vecs * 4, k_vecs * 4, 4, 1)))
    B_vec = al.view(B_bf16, al.i32, al.make_layout((C_dim, 32, k_vecs, 4), (32 * k_vecs * 4, k_vecs * 4, 4, 1)))
    C_vec = al.view(C, al.i32, al.make_layout((C_dim, M_dim, 8, 4), (M_dim * 32, 32, 4, 1)))

    lane = al.thread_id(0)
    lane_col = lane & 31
    lane_group = lane >> 5

    block_m = al.block_id(1) * BLOCK_M

    a_smem = al.make_shared((BLOCK_M * (BLOCK_K >> 3), BLOCK_K >> 2), al.i32)
    b_smem = al.make_shared((BLOCK_N * (BLOCK_K >> 3), BLOCK_K >> 2), al.i32)
    c_smem = al.make_shared((BLOCK_M, BLOCK_N), al.f32)
    c_smem_vec = al.view(c_smem, al.i32, al.make_layout((BLOCK_M, BLOCK_N >> 2, 4), (BLOCK_N, 4, 1)))

    acc = al.full((16,), 0.0, al.f32)

    for kt in al.range(K_dim // BLOCK_K):
        k_vec = kt * 2 + lane_group

        a_smem[lane] = A_vec[channel, block_m + lane_col, k_vec]
        b_smem[lane] = B_vec[channel, lane_col, k_vec]

        al.syncthreads()

        a_words = a_smem[lane]
        b_words = b_smem[lane]
        a_frag = al.view(a_words, al.Tensor((2, 2, 1), al.u32))
        b_frag = al.view(b_words, al.Tensor((2, 2, 1), al.u32))

        acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag[0], a_frag[0], acc)
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag[1], a_frag[1], acc)

        al.syncthreads()

    for r in al.range(16):
        row_offset = ((r >> 2) << 3) + lane_group * 4 + (r & 3)
        c_smem[lane_col, row_offset] = acc[r]

    al.syncthreads()

    store_row = lane >> 1
    store_vec_base = (lane & 1) * (BLOCK_N >> 3)

    for v in al.range(BLOCK_N >> 3):
        C_vec[channel, block_m + store_row, store_vec_base + v] = (
            c_smem_vec[store_row, store_vec_base + v]
        )


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size_h: int,
        kernel_size_w: int,
        stride_h: int = 1,
        stride_w: int = 1,
        padding_h: int = 0,
        padding_w: int = 0,
        dilation_h: int = 1,
        dilation_w: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.conv2d = nn.Conv2d(
            in_channels,
            in_channels,
            (kernel_size_h, kernel_size_w),
            stride=(stride_h, stride_w),
            padding=(padding_h, padding_w),
            dilation=(dilation_h, dilation_w),
            groups=in_channels,
            bias=bias,
        )
        self._in_channels = in_channels
        self._kernel_size_h = kernel_size_h
        self._kernel_size_w = kernel_size_w
        self._stride_h = stride_h
        self._stride_w = stride_w
        self._padding_h = padding_h
        self._padding_w = padding_w
        self._dilation_h = dilation_h
        self._dilation_w = dilation_w
        self._groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C_in, H_in, W_in = x.shape
        OH = (H_in + 2 * self._padding_h - self._dilation_h * (self._kernel_size_h - 1) - 1) // self._stride_h + 1
        OW = (W_in + 2 * self._padding_w - self._dilation_w * (self._kernel_size_w - 1) - 1) // self._stride_w + 1
        KH = self._kernel_size_h
        KW = self._kernel_size_w
        K_per_ch = KH * KW

        w = self.conv2d.weight
        x_bf16 = x.contiguous().to(torch.bfloat16)
        w_bf16 = w.contiguous().to(torch.bfloat16)

        x_unfold = F.unfold(x_bf16, (KH, KW),
                            dilation=(self._dilation_h, self._dilation_w),
                            padding=(self._padding_h, self._padding_w),
                            stride=(self._stride_h, self._stride_w))
        M_spatial = B * OH * OW

        K_pad = ((K_per_ch + BLOCK_K - 1) // BLOCK_K) * BLOCK_K

        A_all = x_unfold.view(B, C_in, K_per_ch, OH * OW).permute(1, 0, 3, 2).reshape(C_in, M_spatial, K_per_ch).contiguous()
        A_pad = torch.nn.functional.pad(A_all, (0, K_pad - K_per_ch)).contiguous()

        W_2d = w_bf16.reshape(C_in, 1, K_per_ch)
        W_pad = torch.nn.functional.pad(W_2d, (0, K_pad - K_per_ch))
        B_repl = W_pad.expand(C_in, 32, K_pad).contiguous()

        C_out = torch.empty((C_in, M_spatial, 32), device=x.device, dtype=torch.float32)

        grid_m = (M_spatial + BLOCK_M - 1) // BLOCK_M
        depthwise_conv_kernel[lambda: ((1, grid_m, C_in), (64, 1, 1))](
            A_pad.data_ptr(), B_repl.data_ptr(), C_out.data_ptr(),
            C_in, M_spatial, K_pad,
        )

        y_flat = C_out[:, :, 0]
        y_out = y_flat.view(C_in, B, OH, OW).permute(1, 0, 2, 3).contiguous()
        return y_out.to(x.dtype)
