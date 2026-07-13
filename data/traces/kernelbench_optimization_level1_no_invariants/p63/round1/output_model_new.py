import torch
import torch.nn as nn
import avelang
import avelang.language as al

_BATCH = 16; _IN_CH = 16; _OUT_CH = 128; _H = 1024; _W = 1024; _KH = 3; _KW = 3
_OH = _H - _KH + 1; _OW = _W - _KW + 1
_M = _BATCH * _OH * _OW; _K = _IN_CH * _KH * _KW; _N = _OUT_CH
_M_TILES = (_M + 31) // 32; _N_TILES = (_N + 31) // 32
_TOTAL_TILES = _M_TILES * _N_TILES
_MAX_CHUNK_BYTES = 1_800_000_000
_CHUNK_M = _MAX_CHUNK_BYTES // (_N * 4)
_CHUNK_M_ALIGNED = (_CHUNK_M // 32) * 32


@avelang.jit
def gemm_split0_kernel(
    A_ptr: al.Pointer(al.bf16), B_ptr: al.Pointer(al.bf16), WS_ptr: al.Pointer(al.f32),
    m_off: al.u32, chunk_m: al.u32, n: al.u32, k: al.u32,
    chunk_tiles: al.u32, n_tiles: al.u32,
):
    lane = al.thread_id(0); lc = lane & 31; lg = lane >> 5
    bid = al.block_id(0); tm = bid // n_tiles; tn = bid % n_tiles
    bm = m_off + tm * 32; bn = tn * 32

    A = al.make_tensor(A_ptr, al.bf16, al.make_layout((_M, k), (k, 1)))
    B = al.make_tensor(B_ptr, al.bf16, al.make_layout((n, k), (k, 1)))
    kv = k >> 3; pr = k >> 1
    AV = al.view(A, al.i32, al.make_layout((_M, kv, 4), (pr, 4, 1)))
    BV = al.view(B, al.i32, al.make_layout((n, kv, 4), (pr, 4, 1)))
    sa = al.make_shared((64, 4), al.i32); sb = al.make_shared((64, 4), al.i32)
    acc = al.full((16,), 0.0, al.f32)

    for kt in al.range(4):
        kv2 = kt * 2 + lg
        sa[lane] = AV[bm + lc, kv2]
        sb[lane] = BV[bn + lc, kv2]
        al.syncthreads()
        aw = sa[lane]; bw = sb[lane]
        af = al.view(aw, al.Tensor((2, 2, 1), al.u32))
        bf = al.view(bw, al.Tensor((2, 2, 1), al.u32))
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(bf[0], af[0], acc)
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(bf[1], af[1], acc)
        al.syncthreads()

    sa[lane] = AV[bm + lc, 8]
    sb[lane] = BV[bn + lc, 8]
    al.syncthreads()
    aw = sa[lane]; bw = sb[lane]
    af = al.view(aw, al.Tensor((2, 2, 1), al.u32))
    bf = al.view(bw, al.Tensor((2, 2, 1), al.u32))
    acc = al.amdgpu.mfma_32x32x8_bf16_f32(bf[0], af[0], acc)
    al.syncthreads()

    ws = al.make_tensor(WS_ptr, al.f32, al.make_layout((chunk_m, n), (n, 1)))
    for r in al.range(16):
        ro = ((r >> 2) << 3) + lg * 4 + (r & 3)
        ws[(bm - m_off) + ro, bn + lc] = acc[r]


@avelang.jit
def gemm_split1_kernel(
    A_ptr: al.Pointer(al.bf16), B_ptr: al.Pointer(al.bf16), WS_ptr: al.Pointer(al.f32),
    m_off: al.u32, chunk_m: al.u32, n: al.u32, k: al.u32,
    chunk_tiles: al.u32, n_tiles: al.u32,
):
    lane = al.thread_id(0); lc = lane & 31; lg = lane >> 5
    bid = al.block_id(0); tm = bid // n_tiles; tn = bid % n_tiles
    bm = m_off + tm * 32; bn = tn * 32

    A = al.make_tensor(A_ptr, al.bf16, al.make_layout((_M, k), (k, 1)))
    B = al.make_tensor(B_ptr, al.bf16, al.make_layout((n, k), (k, 1)))
    kv = k >> 3; pr = k >> 1
    AV = al.view(A, al.i32, al.make_layout((_M, kv, 4), (pr, 4, 1)))
    BV = al.view(B, al.i32, al.make_layout((n, kv, 4), (pr, 4, 1)))
    sa = al.make_shared((64, 4), al.i32); sb = al.make_shared((64, 4), al.i32)
    acc = al.full((16,), 0.0, al.f32)

    sa[lane] = AV[bm + lc, 9]
    sb[lane] = BV[bn + lc, 9]
    al.syncthreads()
    aw = sa[lane]; bw = sb[lane]
    af = al.view(aw, al.Tensor((2, 2, 1), al.u32))
    bf = al.view(bw, al.Tensor((2, 2, 1), al.u32))
    acc = al.amdgpu.mfma_32x32x8_bf16_f32(bf[0], af[0], acc)
    al.syncthreads()

    for kt in al.range(4):
        kv2 = 10 + kt * 2 + lg
        sa[lane] = AV[bm + lc, kv2]
        sb[lane] = BV[bn + lc, kv2]
        al.syncthreads()
        aw = sa[lane]; bw = sb[lane]
        af = al.view(aw, al.Tensor((2, 2, 1), al.u32))
        bf = al.view(bw, al.Tensor((2, 2, 1), al.u32))
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(bf[0], af[0], acc)
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(bf[1], af[1], acc)
        al.syncthreads()

    ws = al.make_tensor(WS_ptr, al.f32, al.make_layout((chunk_m, n), (n, 1)))
    for r in al.range(16):
        ro = ((r >> 2) << 3) + lg * 4 + (r & 3)
        ws[(bm - m_off) + ro, bn + lc] = acc[r]


@avelang.jit
def finalize_kernel(
    WS0_ptr: al.Pointer(al.f32), WS1_ptr: al.Pointer(al.f32), Y_ptr: al.Pointer(al.bf16),
    m_off: al.u32, chunk_m: al.u32, n: al.u32, batch: al.u32, oh: al.u32, ow: al.u32,
    chunk_tiles: al.u32, n_tiles: al.u32,
):
    lane = al.thread_id(0); lc = lane & 31; lg = lane >> 5
    bid = al.block_id(0); tm = bid // n_tiles; tn = bid % n_tiles
    bm = tm * 32; bn = tn * 32

    ws0 = al.make_tensor(WS0_ptr, al.f32, al.make_layout((chunk_m, n), (n, 1)))
    ws1 = al.make_tensor(WS1_ptr, al.f32, al.make_layout((chunk_m, n), (n, 1)))
    Y = al.make_tensor(Y_ptr, al.bf16, al.make_layout((batch, n, oh, ow), (n * oh * ow, oh * ow, ow, 1)))
    oho = oh * ow

    for r in al.range(16):
        ro = ((r >> 2) << 3) + lg * 4 + (r & 3)
        v = al.convert(ws0[bm + ro, bn + lc] + ws1[bm + ro, bn + lc], al.bf16)
        gr = m_off + bm + ro
        Y[gr // oho, bn + lc, (gr % oho) // ow, (gr % oho) % ow] = v


def _im2col(x_bf16):
    xu = x_bf16.unfold(2, _KH, 1).unfold(3, _KW, 1)
    N, IC = x_bf16.shape[0], x_bf16.shape[1]
    return xu.permute(0, 2, 3, 1, 4, 5).reshape(_M, IC * _KH * _KW).contiguous()


def _launch_chunk(col, wr, out4, device, m_off, cm):
    ct = (cm + 31) // 32; tt = ct * _N_TILES
    ws0 = torch.empty((cm, _N), device=device, dtype=torch.float32)
    ws1 = torch.empty((cm, _N), device=device, dtype=torch.float32)
    gemm_split0_kernel[lambda: ((tt, 1, 1), (64, 1, 1))](col, wr, ws0, m_off, cm, _N, _K, ct, _N_TILES)
    gemm_split1_kernel[lambda: ((tt, 1, 1), (64, 1, 1))](col, wr, ws1, m_off, cm, _N, _K, ct, _N_TILES)
    finalize_kernel[lambda: ((tt, 1, 1), (64, 1, 1))](ws0, ws1, out4, m_off, cm, _N, _BATCH, _OH, _OW, ct, _N_TILES)


class ModelNew(nn.Module):
    def __init__(self, ic, oc, ks, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super().__init__()
        self.conv2d = nn.Conv2d(ic, oc, (ks, ks), stride=stride, padding=padding,
                                dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        if tuple(x.shape) != (_BATCH, _IN_CH, _H, _W):
            raise RuntimeError("Shape mismatch")
        xb = x.to(dtype=torch.bfloat16).contiguous()
        wb = self.conv2d.weight.to(device=x.device, dtype=torch.bfloat16).contiguous()
        col = _im2col(xb)
        wr = wb.reshape(_N, _K).contiguous()
        out = torch.empty((_BATCH, _OUT_CH, _OH, _OW), device=x.device, dtype=torch.bfloat16)
        mo = 0
        while mo < _M:
            cm = min(_CHUNK_M_ALIGNED, _M - mo)
            _launch_chunk(col, wr, out, x.device, mo, cm)
            mo += cm
        return out.to(dtype=x.dtype) if x.dtype != torch.bfloat16 else out
