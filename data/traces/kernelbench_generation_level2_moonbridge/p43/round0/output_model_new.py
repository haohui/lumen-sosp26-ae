import torch
import torch.nn as nn
import avelang
import avelang.language as al

C_OUT_BLOCK = 8
D_BLOCK = 4
H_BLOCK = 8
W_BLOCK = 8
C_IN_TILE = 8
KERNEL_SIZE = 3
THREADS = 256

IN_D_EXTENT = D_BLOCK + KERNEL_SIZE - 1
IN_H_EXTENT = H_BLOCK + KERNEL_SIZE - 1
IN_W_EXTENT = W_BLOCK + KERNEL_SIZE - 1
SHM_IN_VOLS = IN_D_EXTENT * IN_H_EXTENT * IN_W_EXTENT
SHM_IN_SIZE = C_IN_TILE * SHM_IN_VOLS
SHM_WT_SIZE = C_OUT_BLOCK * C_IN_TILE * KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE

MP_THREADS = 256
LSE_THREADS = 256


@avelang.jit
def conv3d_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32, C_in: al.i32, C_out: al.i32,
    D: al.i32, H: al.i32, W: al.i32,
    num_w_blocks: al.i32, num_h_blocks: al.i32,
    num_d_blocks: al.i32, num_c_out_blocks: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    w_block = bid % num_w_blocks
    tmp = bid // num_w_blocks
    h_block = tmp % num_h_blocks
    tmp = tmp // num_h_blocks
    d_block = tmp % num_d_blocks
    tmp = tmp // num_d_blocks
    c_out_block = tmp % num_c_out_blocks
    b_idx = tmp // num_c_out_blocks

    if b_idx >= B:
        return

    w_start = w_block * W_BLOCK
    h_start = h_block * H_BLOCK
    d_start = d_block * D_BLOCK
    c_out_start = c_out_block * C_OUT_BLOCK

    td_off = tid // (H_BLOCK * W_BLOCK)
    tr = tid % (H_BLOCK * W_BLOCK)
    th_off = tr // W_BLOCK
    tw_off = tr % W_BLOCK

    x = al.make_tensor(x_ptr, al.bf16,
        al.make_layout((B, C_in, D, H, W),
                       (C_in * D * H * W, D * H * W, H * W, W, 1)))
    w = al.make_tensor(w_ptr, al.bf16,
        al.make_layout((C_out, C_in, 3, 3, 3), (C_in * 27, 27, 9, 3, 1)))
    bias = al.make_tensor(bias_ptr, al.bf16, al.make_layout((C_out,), (1,)))
    out = al.make_tensor(out_ptr, al.bf16,
        al.make_layout((B, C_out, D, H, W),
                       (C_out * D * H * W, D * H * W, H * W, W, 1)))

    shm_in = al.make_shared((C_IN_TILE, IN_D_EXTENT, IN_H_EXTENT, IN_W_EXTENT), al.bf16)
    shm_wt = al.make_shared((SHM_WT_SIZE,), al.bf16)

    acc = al.make_local((C_OUT_BLOCK,), al.f32)
    for i in al.range(C_OUT_BLOCK):
        acc[i] = al.convert(0.0, al.f32)

    c_in_tiles = C_in // C_IN_TILE
    zero_bf16 = al.convert(0.0, al.bf16)
    hw_ext = IN_H_EXTENT * IN_W_EXTENT

    for ci_tile in al.range(c_in_tiles):
        c_in_start = ci_tile * C_IN_TILE

        for idx in al.range(tid, SHM_IN_SIZE, THREADS):
            ci_local = idx // SHM_IN_VOLS
            lr = idx % SHM_IN_VOLS
            ld = lr // hw_ext
            lr2 = lr % hw_ext
            lh = lr2 // IN_W_EXTENT
            lw = lr2 % IN_W_EXTENT
            dg = d_start + ld - 1
            hg = h_start + lh - 1
            wg = w_start + lw - 1
            cg = c_in_start + ci_local
            if dg >= 0 and dg < D and hg >= 0 and hg < H and wg >= 0 and wg < W:
                shm_in[ci_local, ld, lh, lw] = x[b_idx, cg, dg, hg, wg]
            else:
                shm_in[ci_local, ld, lh, lw] = zero_bf16

        for idx in al.range(tid, SHM_WT_SIZE, THREADS):
            c_wt = C_IN_TILE * 27
            wt_co = idx // c_wt
            wr = idx % c_wt
            wt_ci = wr // 27
            wr2 = wr % 27
            wkd = wr2 // 9
            wr3 = wr2 % 9
            wkh = wr3 // 3
            wkw = wr3 % 3
            shm_wt[idx] = w[c_out_start + wt_co, c_in_start + wt_ci, wkd, wkh, wkw]

        al.syncthreads()

        wsc = C_IN_TILE * 27

        for kd in al.range(3):
            for kh in al.range(3):
                for kw in al.range(3):
                    for ci_l in al.range(C_IN_TILE):
                        iv = al.convert(shm_in[ci_l, td_off + kd, th_off + kh, tw_off + kw], al.f32)
                        wb = ci_l * 27 + kd * 9 + kh * 3 + kw
                        for e in al.range(C_OUT_BLOCK):
                            wv = al.convert(shm_wt[e * wsc + wb], al.f32)
                            acc[e] = acc[e] + iv * wv

        al.syncthreads()

    for e in al.range(C_OUT_BLOCK):
        dg = d_start + td_off
        hg = h_start + th_off
        wg = w_start + tw_off
        if dg < D and hg < H and wg < W:
            r = acc[e] + al.convert(bias[c_out_start + e], al.f32)
            out[b_idx, c_out_start + e, dg, hg, wg] = al.convert(r, al.bf16)


@avelang.jit
def maxpool3d_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32, C: al.i32,
    Di: al.i32, Hi: al.i32, Wi: al.i32,
    Do: al.i32, Ho: al.i32, Wo: al.i32,
    total: al.i32,
):
    gid = al.block_id(0) * MP_THREADS + al.thread_id(0)
    if gid >= total:
        return
    sp = Do * Ho * Wo
    bc = C * sp
    bi = gid // bc
    r = gid % bc
    ci = r // sp
    r = r % sp
    di = r // (Ho * Wo)
    r = r % (Ho * Wo)
    hi = r // Wo
    wi = r % Wo
    d0, h0, w0 = di * 2, hi * 2, wi * 2
    x = al.make_tensor(x_ptr, al.bf16,
        al.make_layout((B, C, Di, Hi, Wi),
                       (C * Di * Hi * Wi, Di * Hi * Wi, Hi * Wi, Wi, 1)))
    o = al.make_tensor(out_ptr, al.bf16,
        al.make_layout((B, C, Do, Ho, Wo),
                       (C * Do * Ho * Wo, Do * Ho * Wo, Ho * Wo, Wo, 1)))
    v0 = x[bi, ci, d0, h0, w0]
    v1 = x[bi, ci, d0, h0, w0 + 1]
    v2 = x[bi, ci, d0, h0 + 1, w0]
    v3 = x[bi, ci, d0, h0 + 1, w0 + 1]
    v4 = x[bi, ci, d0 + 1, h0, w0]
    v5 = x[bi, ci, d0 + 1, h0, w0 + 1]
    v6 = x[bi, ci, d0 + 1, h0 + 1, w0]
    v7 = x[bi, ci, d0 + 1, h0 + 1, w0 + 1]
    m0 = v0 if v0 > v1 else v1
    m1 = v2 if v2 > v3 else v3
    m2 = v4 if v4 > v5 else v5
    m3 = v6 if v6 > v7 else v7
    m0 = m0 if m0 > m1 else m1
    m1 = m2 if m2 > m3 else m3
    o[bi, ci, di, hi, wi] = m0 if m0 > m1 else m1


@avelang.jit
def logsumexp_relu_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32, C: al.i32, D: al.i32, H: al.i32, W: al.i32,
    total: al.i32,
):
    gid = al.block_id(0) * LSE_THREADS + al.thread_id(0)
    if gid >= total:
        return
    sp = D * H * W
    bi = gid // sp
    r = gid % sp
    di = r // (H * W)
    r = r % (H * W)
    hi = r // W
    wi = r % W
    x = al.make_tensor(x_ptr, al.bf16,
        al.make_layout((B, C, D, H, W),
                       (C * D * H * W, D * H * W, H * W, W, 1)))
    o = al.make_tensor(out_ptr, al.bf16,
        al.make_layout((B, D, H, W), (D * H * W, H * W, W, 1)))
    neg_inf = al.convert(-1e30, al.f32)
    zero_f = al.convert(0.0, al.f32)
    mx = neg_inf
    for c in al.range(C):
        v = al.convert(x[bi, c, di, hi, wi], al.f32)
        mx = v if v > mx else mx
    s = al.convert(0.0, al.f32)
    for c in al.range(C):
        v = al.convert(x[bi, c, di, hi, wi], al.f32)
        s = s + al.exp(v - mx)
    result = mx + al.log(s)
    if result < zero_f:
        result = zero_f
    o[bi, di, hi, wi] = al.convert(result, al.bf16)


def _bf16(t):
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_conv3d(x, weight, bias):
    xb, wb, bb = _bf16(x), _bf16(weight), _bf16(bias)
    B, Ci, D, H, W = xb.shape
    Co = wb.shape[0]
    nw = (W + W_BLOCK - 1) // W_BLOCK
    nh = (H + H_BLOCK - 1) // H_BLOCK
    nd = (D + D_BLOCK - 1) // D_BLOCK
    nc = (Co + C_OUT_BLOCK - 1) // C_OUT_BLOCK
    o = torch.empty((B, Co, D, H, W), device=xb.device, dtype=torch.bfloat16)
    conv3d_kernel[lambda: ((B * nc * nd * nh * nw, 1, 1), (THREADS, 1, 1))](
        xb, wb, bb, o, B, Ci, Co, D, H, W, nw, nh, nd, nc)
    return o


def avelang_maxpool3d(x):
    xb = _bf16(x)
    B, C, Di, Hi, Wi = xb.shape
    Do, Ho, Wo = Di // 2, Hi // 2, Wi // 2
    t = B * C * Do * Ho * Wo
    nb = (t + MP_THREADS - 1) // MP_THREADS
    o = torch.empty((B, C, Do, Ho, Wo), device=xb.device, dtype=torch.bfloat16)
    maxpool3d_kernel[lambda: ((nb, 1, 1), (MP_THREADS, 1, 1))](
        xb, o, B, C, Di, Hi, Wi, Do, Ho, Wo, t)
    return o


def avelang_logsumexp_relu(x):
    xb = _bf16(x)
    B, C, D, H, W = xb.shape
    t = B * D * H * W
    nb = (t + LSE_THREADS - 1) // LSE_THREADS
    o = torch.empty((B, D, H, W), device=xb.device, dtype=torch.bfloat16)
    logsumexp_relu_kernel[lambda: ((nb, 1, 1), (LSE_THREADS, 1, 1))](
        xb, o, B, C, D, H, W, t)
    return o


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / (fan_in**0.5) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        orig = x.dtype
        x = avelang_conv3d(x, self.weight, self.bias)
        x = avelang_maxpool3d(x)
        x = avelang_logsumexp_relu(x)
        return x.unsqueeze(1).to(dtype=orig)


batch_size = 4
in_channels = 32
out_channels = 64
depth, height, width = 32, 128, 128
kernel_size = 3
stride = 1
padding = 1


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]
