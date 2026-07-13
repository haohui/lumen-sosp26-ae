import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE = 256


@avelang.jit
def conv_transpose3d_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    k_idx_ptr: al.Pointer(al.i32),
    k_counts_ptr: al.Pointer(al.i32),
    N: al.i32,
    C_in: al.i32,
    C_out: al.i32,
    D: al.i32,
    H: al.i32,
    W: al.i32,
    D_out: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    stride: al.i32,
    padding: al.i32,
    dilation: al.i32,
    K: al.i32,
    max_valid: al.i32,
    has_bias: al.i32,
    WEIGHT_ELEMS: al.constexpr,
):
    tid = al.thread_id(0)
    batch_idx = al.block_id(0)
    oc = al.block_id(1)
    spatial_tile = al.block_id(2)

    spatial_idx = spatial_tile * BLOCK_SIZE + tid
    spatial_total = D_out * H_out * W_out

    if batch_idx < N:
        if oc < C_out:
            if spatial_idx < spatial_total:
                # Cooperative load of weight for this output channel into shared memory
                shm_weight = al.make_shared((WEIGHT_ELEMS,), al.bf16)
                w_flat_layout = al.make_layout(
                    (C_out * WEIGHT_ELEMS,), (al.convert(1, al.i32),)
                )
                w_flat = al.make_tensor(weight_ptr, al.bf16, w_flat_layout)
                w_base = oc * WEIGHT_ELEMS
                for i in al.range(tid, WEIGHT_ELEMS, BLOCK_SIZE):
                    shm_weight[i] = w_flat[w_base + i]
                al.syncthreads()

                d_out = spatial_idx // (H_out * W_out)
                rem_sp = spatial_idx - d_out * (H_out * W_out)
                h_out = rem_sp // W_out
                w_out = rem_sp - h_out * W_out

                rd = d_out + padding
                rd = rd - (rd // stride) * stride
                rh = h_out + padding
                rh = rh - (rh // stride) * stride
                rw = w_out + padding
                rw = rw - (rw // stride) * stride

                residue_idx = (rd * stride + rh) * stride + rw

                k_counts_layout = al.make_layout(
                    (stride * stride * stride,), (al.convert(1, al.i32),)
                )
                k_counts = al.make_tensor(k_counts_ptr, al.i32, k_counts_layout)

                num_valid_k = al.convert(0, al.i32)
                zero_i32 = al.convert(0, al.i32)
                one_i32 = al.convert(1, al.i32)
                if residue_idx >= zero_i32:
                    if residue_idx < stride * stride * stride:
                        num_valid_k = k_counts[residue_idx]

                if num_valid_k:
                    k_idx_layout = al.make_layout(
                        (stride * stride * stride, max_valid),
                        (max_valid, al.convert(1, al.i32)),
                    )
                    k_indices = al.make_tensor(k_idx_ptr, al.i32, k_idx_layout)

                    input_layout = al.make_layout(
                        (N, C_in, D, H, W),
                        (C_in * D * H * W, D * H * W, H * W, W, al.convert(1, al.i32)),
                    )
                    input_t = al.make_tensor(input_ptr, al.bf16, input_layout)

                    output_layout = al.make_layout(
                        (N, C_out, D_out, H_out, W_out),
                        (
                            C_out * D_out * H_out * W_out,
                            D_out * H_out * W_out,
                            H_out * W_out,
                            W_out,
                            al.convert(1, al.i32),
                        ),
                    )
                    output_t = al.make_tensor(output_ptr, al.bf16, output_layout)

                    acc = al.convert(0.0, al.f32)
                    K2 = K * K

                    for valid_i in al.range(num_valid_k):
                        flat_k = k_indices[residue_idx, valid_i]
                        kd = flat_k // K2
                        rem_k = flat_k - kd * K2
                        kh = rem_k // K
                        kw = rem_k - kh * K

                        d_in = (d_out + padding - dilation * kd) // stride
                        h_in = (h_out + padding - dilation * kh) // stride
                        w_in = (w_out + padding - dilation * kw) // stride

                        d_ok = al.convert(0, al.i32)
                        if d_in >= zero_i32:
                            if d_in < D:
                                d_ok = one_i32
                        h_ok = al.convert(0, al.i32)
                        if h_in >= zero_i32:
                            if h_in < H:
                                h_ok = one_i32
                        w_ok = al.convert(0, al.i32)
                        if w_in >= zero_i32:
                            if w_in < W:
                                w_ok = one_i32

                        all_ok = al.convert(0, al.i32)
                        if d_ok:
                            if h_ok:
                                if w_ok:
                                    all_ok = one_i32

                        if all_ok:
                            for ic in al.range(C_in):
                                in_val = al.convert(
                                    input_t[batch_idx, ic, d_in, h_in, w_in], al.f32
                                )
                                w_idx = ic * K * K2 + kd * K2 + kh * K + kw
                                w_val = al.convert(shm_weight[w_idx], al.f32)
                                acc = acc + in_val * w_val

                    if has_bias:
                        bias_layout = al.make_layout(
                            (C_out,), (al.convert(1, al.i32),)
                        )
                        bias_t = al.make_tensor(bias_ptr, al.bf16, bias_layout)
                        b_val = al.convert(bias_t[oc], al.f32)
                        acc = acc + b_val

                    output_t[batch_idx, oc, d_out, h_out, w_out] = al.convert(
                        acc, al.bf16
                    )


def _build_stride_tables(stride: int, padding: int, dilation: int, K: int):
    """Precompute valid kernel flat-indices for each stride-remainder pattern."""
    stride_cubed = stride * stride * stride
    K3 = K * K * K

    valid_counts_list = [0] * stride_cubed
    k_indices_flat = []

    for rd in range(stride):
        for rh in range(stride):
            for rw in range(stride):
                residue_idx = (rd * stride + rh) * stride + rw
                count = 0
                row = [0] * K3
                for kd in range(K):
                    if (dilation * kd) % stride != rd:
                        continue
                    for kh in range(K):
                        if (dilation * kh) % stride != rh:
                            continue
                        for kw in range(K):
                            if (dilation * kw) % stride != rw:
                                continue
                            row[count] = kd * K * K + kh * K + kw
                            count += 1
                valid_counts_list[residue_idx] = count
                k_indices_flat.extend(row)

    k_counts = torch.tensor(valid_counts_list, dtype=torch.int32)
    k_indices = torch.tensor(k_indices_flat, dtype=torch.int32).reshape(stride_cubed, K3)
    max_valid = int(k_counts.max().item())
    return k_indices, k_counts, max_valid


def _prepare_bf16(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_conv_transpose3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    k_indices: torch.Tensor,
    k_counts: torch.Tensor,
    max_valid: int,
    stride: int,
    padding: int,
    dilation: int,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = _prepare_bf16(x)

    weight_reordered = weight.permute(1, 0, 2, 3, 4).contiguous()
    weight_bf16 = _prepare_bf16(weight_reordered)

    N, C_in, D, H, W = x_bf16.shape
    C_out, C_in_w, K_d, K_h, K_w = weight_bf16.shape

    K = K_d
    weight_elems = C_in * K * K * K
    K_eff = dilation * (K - 1) + 1

    D_out = (D - 1) * stride - 2 * padding + K_eff
    H_out = (H - 1) * stride - 2 * padding + K_eff
    W_out = (W - 1) * stride - 2 * padding + K_eff

    has_bias_flag = 1 if bias is not None else 0
    if bias is None:
        bias_bf16 = torch.zeros(C_out, dtype=torch.bfloat16, device=x_bf16.device)
    else:
        bias_bf16 = _prepare_bf16(bias)

    if D_out <= 0 or H_out <= 0 or W_out <= 0:
        out_shape = (N, C_out, max(D_out, 0), max(H_out, 0), max(W_out, 0))
        return torch.zeros(out_shape, dtype=x.dtype, device=x.device)

    out = torch.empty(
        (N, C_out, D_out, H_out, W_out),
        dtype=torch.bfloat16,
        device=x_bf16.device,
    )

    spatial_total = D_out * H_out * W_out
    num_spatial_tiles = (spatial_total + BLOCK_SIZE - 1) // BLOCK_SIZE

    grid = (N, C_out, num_spatial_tiles)

    conv_transpose3d_kernel[lambda: (grid, (BLOCK_SIZE, 1, 1))](
        x_bf16,
        weight_bf16,
        bias_bf16,
        out,
        k_indices,
        k_counts,
        N,
        C_in,
        C_out,
        D,
        H,
        W,
        D_out,
        H_out,
        W_out,
        stride,
        padding,
        dilation,
        K,
        max_valid,
        has_bias_flag,
        WEIGHT_ELEMS=weight_elems,
    )

    return out.to(x.dtype)


# Test code
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = 3
depth = 16
height = 32
width = 32
stride = 2
padding = 1
dilation = 2


def get_inputs():
    x = torch.rand(batch_size, in_channels, depth, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]


class ModelNew(nn.Module):
    """
    Performs a 3D transposed convolution operation with square input and square kernel,
    and supports padding, dilation, and stride. Optimized with AveLang DSL kernels
    using stride-remainder pre-filtering and shared-memory weight caching.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.use_bias = bias

        ref = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size, kernel_size),
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.weight = nn.Parameter(ref.weight.data.clone())
        if bias:
            self.bias = nn.Parameter(ref.bias.data.clone())
        else:
            self.bias = None

        k_indices, k_counts, max_valid = _build_stride_tables(
            stride, padding, dilation, kernel_size
        )
        self.register_buffer("_k_indices", k_indices, persistent=False)
        self.register_buffer("_k_counts", k_counts, persistent=False)
        self._max_valid = max_valid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_conv_transpose3d(
            x, self.weight, self.bias,
            self._k_indices, self._k_counts, self._max_valid,
            self.stride, self.padding, self.dilation,
        )
