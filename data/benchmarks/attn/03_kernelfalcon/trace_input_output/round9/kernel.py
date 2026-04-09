import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_D": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_D": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_D": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_D": 128}, num_warps=8, num_stages=3),
    ],
    key=["S"],
)
@triton.jit
def _fused_causal_attention_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    B, HQ, HKV, S, D,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    b = pid_bh // HQ
    hq = pid_bh % HQ
    hkv = hq % HKV  # supports HKV == 1 (MQA) and HKV == HQ

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = q_ptr + b * stride_qb + hq * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q_mask = (offs_m[:, None] < S) & (offs_d[None, :] < D)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    n_start = 0
    while n_start < S:
        offs_n = n_start + tl.arange(0, BLOCK_N)

        k_ptrs = k_ptr + b * stride_kb + hkv * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        k_mask = (offs_n[:, None] < S) & (offs_d[None, :] < D)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32, input_precision="ieee")
        qk = qk * scale

        causal = offs_m[:, None] >= offs_n[None, :]
        valid = (offs_m[:, None] < S) & (offs_n[None, :] < S)
        qk = tl.where(causal & valid, qk, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)

        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha + l_ij

        v_ptrs = v_ptr + b * stride_vb + hkv * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        v_mask = (offs_n[:, None] < S) & (offs_d[None, :] < D)
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        acc = tl.dot(p.to(tl.bfloat16), v, acc=acc, out_dtype=tl.float32, input_precision="ieee")
        m_i = m_ij
        n_start += BLOCK_N

    out = acc / l_i[:, None]

    o_ptrs = o_ptr + b * stride_ob + hq * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    o_mask = (offs_m[:, None] < S) & (offs_d[None, :] < D)
    tl.store(o_ptrs, out.to(o_ptr.dtype.element_ty), mask=o_mask)


def kernel_function(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    # Validation only (no compute in wrapper)
    assert isinstance(Q, torch.Tensor) and isinstance(K, torch.Tensor) and isinstance(V, torch.Tensor)
    assert Q.device.type == "cuda" and K.device.type == "cuda" and V.device.type == "cuda"
    assert Q.dtype == torch.bfloat16 and K.dtype == torch.bfloat16 and V.dtype == torch.bfloat16
    assert Q.ndim == 4 and K.ndim == 4 and V.ndim == 4

    B, HQ, S, D = Q.shape
    BK, HKV, SK, DK = K.shape
    BV, HKV2, SV, DV = V.shape

    assert B == BK == BV
    assert S == SK == SV
    assert D == DK == DV == 128
    assert HKV == HKV2
    assert HKV == 1 or HKV == HQ

    O = torch.empty_like(Q)
    scale = 1.0 / (D ** 0.5)

    grid = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B * HQ)

    _fused_causal_attention_kernel[grid](
        Q, K, V, O,
        B, HQ, HKV, S, D,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        scale,
    )
    return O