import torch
import torch.nn as nn
import avelang
import avelang.language as al

THREADS = 256
ELTS_PER_THREAD = 4
TILE_SIZE = THREADS * ELTS_PER_THREAD


@avelang.jit
def conv_transpose3d_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C_in: al.i32,
    C_out: al.i32,
    D: al.i32,
    H: al.i32,
    W: al.i32,
    D_out: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    K: al.i32,
    total_elems: al.i32,
    in_spatial: al.i32,
    out_spatial: al.i32,
    w_per_oc: al.i32,
    w_K2: al.i32,
    w_C_out_K3: al.i32,
    num_blocks: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    in_flat = al.make_tensor(
        input_ptr, al.bf16, al.make_layout((N * C_in * in_spatial,), (1,))
    )
    out_flat = al.make_tensor(
        output_ptr, al.bf16, al.make_layout((N * C_out * out_spatial,), (1,))
    )
    w_flat = al.make_tensor(
        weight_ptr, al.bf16, al.make_layout((C_in * C_out * w_per_oc,), (1,))
    )

    zero_f32 = al.convert(0.0, al.f32)

    base_gid = bid * TILE_SIZE + tid

    for e in al.range(ELTS_PER_THREAD):
        gid = base_gid + e * THREADS
        if gid < total_elems:
            tmp = gid
            ow = tmp % W_out
            tmp = tmp // W_out
            oh = tmp % H_out
            tmp = tmp // H_out
            od = tmp % D_out
            tmp = tmp // D_out
            oc = tmp % C_out
            n = tmp // C_out

            acc = zero_f32
            in_n_base = n * C_in * in_spatial
            w_oc_base = oc * w_per_oc
            out_n_oc_base = n * C_out * out_spatial + oc * out_spatial
            out_spatial_off = od * H_out * W_out + oh * W_out + ow

            for ic in al.range(C_in):
                in_nic_base = in_n_base + ic * in_spatial
                w_ic_oc_base = ic * w_C_out_K3 + w_oc_base
                for kd in al.range(K):
                    id_val = od - kd
                    if id_val >= 0:
                        if id_val < D:
                            in_kd_off = id_val * H * W
                            w_kd_off = kd * w_K2
                            for kh in al.range(K):
                                ih_val = oh - kh
                                if ih_val >= 0:
                                    if ih_val < H:
                                        in_kh_off = in_kd_off + ih_val * W
                                        w_kh_off = w_kd_off + kh * K
                                        for kw in al.range(K):
                                            iw_val = ow - kw
                                            if iw_val >= 0:
                                                if iw_val < W:
                                                    in_off = (
                                                        in_nic_base
                                                        + in_kh_off
                                                        + iw_val
                                                    )
                                                    w_off = (
                                                        w_ic_oc_base
                                                        + w_kh_off
                                                        + kw
                                                    )
                                                    in_val = al.convert(
                                                        in_flat[in_off], al.f32
                                                    )
                                                    w_val = al.convert(
                                                        w_flat[w_off], al.f32
                                                    )
                                                    acc = acc + in_val * w_val

            out_flat[out_n_oc_base + out_spatial_off] = al.convert(acc, al.bf16)


def avelang_conv_transpose3d(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    if not x.is_cuda or not weight.is_cuda:
        raise RuntimeError("Tensors must be on CUDA/HIP device.")

    x_bf16 = x.contiguous().to(dtype=torch.bfloat16)
    weight_bf16 = weight.contiguous().to(dtype=torch.bfloat16)

    N, C_in, D, H, W = x_bf16.shape
    C_in_w, C_out, K, _, _ = weight_bf16.shape

    if C_in_w != C_in:
        raise ValueError(
            f"Weight C_in {C_in_w} does not match input C_in {C_in}"
        )

    D_out = (D - 1) * 1 - 2 * 0 + K + 0
    H_out = (H - 1) * 1 - 2 * 0 + K + 0
    W_out = (W - 1) * 1 - 2 * 0 + K + 0

    total_out = N * C_out * D_out * H_out * W_out

    in_spatial = int(D * H * W)
    out_spatial = int(D_out * H_out * W_out)
    w_per_oc = int(K * K * K)
    w_K2 = int(K * K)
    w_C_out_K3 = int(C_out * K * K * K)

    out = torch.empty(
        (N, C_out, D_out, H_out, W_out),
        device=x_bf16.device,
        dtype=torch.bfloat16,
    )

    num_blocks = (total_out + TILE_SIZE - 1) // TILE_SIZE
    grid = (num_blocks, 1, 1)

    conv_transpose3d_kernel[lambda: (grid, (THREADS, 1, 1))](
        x_bf16,
        weight_bf16,
        out,
        N,
        C_in,
        C_out,
        D,
        H,
        W,
        D_out,
        H_out,
        W_out,
        K,
        total_out,
        in_spatial,
        out_spatial,
        w_per_oc,
        w_K2,
        w_C_out_K3,
        num_blocks,
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
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

        self.conv_transpose3d = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size, kernel_size),
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.conv_transpose3d.weight
        bias_tensor = self.conv_transpose3d.bias
        if (
            self.stride != 1
            or self.padding != 0
            or self.output_padding != 0
            or self.groups != 1
        ):
            return torch.nn.functional.conv_transpose3d(
                x,
                weight,
                bias_tensor,
                stride=self.stride,
                padding=self.padding,
                output_padding=self.output_padding,
                groups=self.groups,
            )
        result = avelang_conv_transpose3d(x, weight)
        if bias_tensor is not None:
            result = result + bias_tensor.view(1, self.out_channels, 1, 1, 1)
        return result


def get_inputs():
    batch_size = 8
    in_channels = 48
    depth = 64
    height = 64
    width = 64
    x = torch.rand(batch_size, in_channels, depth, height, width)
    return [x]


def get_init_inputs():
    return [48, 48, 3]
