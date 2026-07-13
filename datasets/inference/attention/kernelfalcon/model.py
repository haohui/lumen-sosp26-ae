import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_D": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_D": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_D": 128}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_D": 128}, num_warps=8, num_stages=1),
    ],
    key=["S"],
)
@triton.jit
def _fused_causal_attention_opt(
    q_ptr, k_ptr, v_ptr, o_ptr,
    B, HQ, HKV, S, D,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_bh = tl.program_id(axis=1)

    b = pid_bh // HQ
    hq = pid_bh % HQ
    hkv = tl.where(HKV == 1, 0, hq)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = q_ptr + b * stride_qb + hq * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q_mask = (offs_m[:, None] < S) & (offs_d[None, :] < D)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.bfloat16)

    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    # use exp2: exp(x) = exp2(x * log2(e))
    log2e = 1.4426950408889634
    qk_scale = sm_scale * log2e

    n_start = 0
    while n_start < S:
        offs_n = n_start + tl.arange(0, BLOCK_N)

        k_ptrs = k_ptr + b * stride_kb + hkv * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        v_ptrs = v_ptr + b * stride_vb + hkv * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd

        kv_mask = (offs_n[:, None] < S) & (offs_d[None, :] < D)
        k = tl.load(k_ptrs, mask=kv_mask, other=0.0).to(tl.bfloat16)
        v = tl.load(v_ptrs, mask=kv_mask, other=0.0).to(tl.bfloat16)

        qk = tl.dot(q, tl.trans(k)) * qk_scale  # already in log2 domain scaling

        causal = offs_m[:, None] >= offs_n[None, :]
        valid = (offs_m[:, None] < S) & (offs_n[None, :] < S)
        qk = tl.where(causal & valid, qk, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.math.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)

        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc)

        l_i = l_i * alpha + l_ij
        m_i = m_ij

        n_start += BLOCK_N

    out = acc / l_i[:, None]

    o_ptrs = o_ptr + b * stride_ob + hq * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    o_mask = (offs_m[:, None] < S) & (offs_d[None, :] < D)
    tl.store(o_ptrs, out.to(o_ptr.dtype.element_ty), mask=o_mask)


def kernel_function(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    # validation only (no compute in wrapper)
    assert isinstance(Q, torch.Tensor) and isinstance(K, torch.Tensor) and isinstance(V, torch.Tensor)
    assert Q.is_cuda and K.is_cuda and V.is_cuda, "All inputs must be on GPU device"
    assert Q.dtype == torch.bfloat16 and K.dtype == torch.bfloat16 and V.dtype == torch.bfloat16
    assert Q.ndim == 4 and K.ndim == 4 and V.ndim == 4

    B, S, HQ, D = Q.shape
    BK, SK, HKV, DK = K.shape
    BV, SV, HKV2, DV = V.shape

    assert B == BK == BV
    assert S == SK == SV
    assert D == DK == DV
    assert HKV == HKV2
    assert HKV == 1 or HKV == HQ
    assert D == 128

    Q_bhsd = Q.permute(0, 2, 1, 3).contiguous()
    K_bhsd = K.permute(0, 2, 1, 3).contiguous()
    V_bhsd = V.permute(0, 2, 1, 3).contiguous()
    O_bhsd = torch.empty_like(Q_bhsd)
    sm_scale = 1.0 / (D ** 0.5)

    grid = (triton.cdiv(S, 128), B * HQ)

    _fused_causal_attention_opt[grid](
        Q_bhsd, K_bhsd, V_bhsd, O_bhsd,
        B, HQ, HKV, S, D,
        Q_bhsd.stride(0), Q_bhsd.stride(1), Q_bhsd.stride(2), Q_bhsd.stride(3),
        K_bhsd.stride(0), K_bhsd.stride(1), K_bhsd.stride(2), K_bhsd.stride(3),
        V_bhsd.stride(0), V_bhsd.stride(1), V_bhsd.stride(2), V_bhsd.stride(3),
        O_bhsd.stride(0), O_bhsd.stride(1), O_bhsd.stride(2), O_bhsd.stride(3),
        sm_scale,
    )
    return O_bhsd.permute(0, 2, 1, 3).contiguous()
